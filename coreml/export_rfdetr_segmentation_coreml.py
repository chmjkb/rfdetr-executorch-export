"""
Export RF-DETR Seg Nano (instance segmentation) to CoreML-backed ExecuTorch .pte

Requirements:
    pip install rfdetr executorch coremltools

Usage:
    python export_rfdetr_segmentation_coreml.py
    python export_rfdetr_segmentation_coreml.py --output my_model.pte
    python export_rfdetr_segmentation_coreml.py --compute-unit cpu_and_ne

Output tensor layout (3 tensors matching iOS runner expectations):
    [0] boxes_xyxy     [1, Q, 4]    float32  xyxy in pixel coords (0..resolution)
    [1] scores_labels  [1, Q, 2]    float32  stacked [score, label_id] per query
    [2] masks          [1, Q, 78, 78] float32  mask logits — apply sigmoid + threshold on device

Default settings (best from autoresearch + ALL compute unit):
    model      : RFDETRSegNano
    resolution : 312px  (native)
    backend    : CoreML fp32 + int8 weight quantization
    compute    : ALL  (ANE + GPU, ~2x faster than CPU_AND_NE on iPhone)
    delegation : 99.7%
    size       : ~30 MB
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor


# ---------------------------------------------------------------------------
# Patches — make RF-DETR compatible with torch.export
# ---------------------------------------------------------------------------

def _precompute_proposals_validity(spatial_shapes_list):
    valid_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes_list):
        gx = (np.arange(W_, dtype=np.float64) + 0.5) / W_
        gy = (np.arange(H_, dtype=np.float64) + 0.5) / H_
        gy_grid, gx_grid = np.meshgrid(gy, gx, indexing="ij")
        w = h = 0.05 * (2.0 ** lvl)
        props = np.stack([gx_grid, gy_grid,
                          np.full((H_, W_), w, dtype=np.float64),
                          np.full((H_, W_), h, dtype=np.float64)], axis=-1)
        valid_list.append(((props > 0.01) & (props < 0.99)).all(axis=-1).reshape(-1))
    valid_np = np.concatenate(valid_list, axis=0)
    return bool(valid_np.all()), valid_np


def _gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes,
                                   unsigmoid=True, spatial_shapes_list=None):
    N_, S_, C_ = memory.shape
    proposals = []
    _cur = 0
    _iter = spatial_shapes_list if spatial_shapes_list is not None else spatial_shapes
    if spatial_shapes_list is not None:
        all_valid, valid_np = _precompute_proposals_validity(spatial_shapes_list)
    else:
        all_valid, valid_np = False, None
    for lvl, (H_, W_) in enumerate(_iter):
        if memory_padding_mask is not None:
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        else:
            valid_H = torch.full((N_,), H_, dtype=torch.long, device=memory.device)
            valid_W = torch.full((N_,), W_, dtype=torch.long, device=memory.device)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H_, dtype=torch.float32, device=memory.device),
            torch.arange(W_, dtype=torch.float32, device=memory.device),
            indexing="ij")
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_
    output_proposals = torch.cat(proposals, 1)
    if all_valid:
        output_proposals_valid = None
    else:
        if valid_np is not None:
            output_proposals_valid = torch.as_tensor(
                valid_np[:, None], dtype=torch.bool, device=memory.device).unsqueeze(0)
        else:
            output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        if output_proposals_valid is not None:
            output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        if output_proposals_valid is not None:
            output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))
    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    if output_proposals_valid is not None:
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


def _transformer_forward(self, srcs, masks, pos_embeds, refpoint_embed, query_feat):
    masks = None
    src_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], []
    for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
        bs, c, h, w = src.shape
        spatial_shapes.append((h, w))
        src = src.flatten(2).transpose(1, 2)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        lvl_pos_embed_flatten.append(pos_embed)
        src_flatten.append(src)
    memory = torch.cat(src_flatten, 1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    spatial_shapes_list = list(spatial_shapes)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=memory.device)
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    if self.two_stage:
        output_memory, output_proposals = _gen_encoder_output_proposals(
            memory, None, spatial_shapes,
            unsigmoid=not self.bbox_reparam, spatial_shapes_list=spatial_shapes_list)
        refpoint_embed_ts, memory_ts, boxes_ts = [], [], []
        group_detr = self.group_detr if self.training else 1
        for g_idx in range(group_detr):
            output_memory_gidx = self.enc_output_norm[g_idx](self.enc_output[g_idx](output_memory))
            enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](output_memory_gidx)
            if self.bbox_reparam:
                enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](output_memory_gidx)
                enc_outputs_coord_cxcy_gidx = (enc_outputs_coord_delta_gidx[..., :2] * output_proposals[..., 2:]
                                                + output_proposals[..., :2])
                enc_outputs_coord_wh_gidx = (enc_outputs_coord_delta_gidx[..., 2:].exp() * output_proposals[..., 2:])
                enc_outputs_coord_unselected_gidx = torch.concat(
                    [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx], dim=-1)
            else:
                enc_outputs_coord_unselected_gidx = (self.enc_out_bbox_embed[g_idx](output_memory_gidx) + output_proposals)
            topk = min(self.num_queries, enc_outputs_class_unselected_gidx.shape[-2])
            topk_proposals_gidx = torch.topk(enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1)[1]
            refpoint_embed_gidx_undetach = torch.gather(
                enc_outputs_coord_unselected_gidx, 1,
                topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4))
            refpoint_embed_gidx = refpoint_embed_gidx_undetach.detach()
            tgt_undetach_gidx = torch.gather(
                output_memory_gidx, 1,
                topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, self.d_model))
            refpoint_embed_ts.append(refpoint_embed_gidx)
            memory_ts.append(tgt_undetach_gidx)
            boxes_ts.append(refpoint_embed_gidx_undetach)
        refpoint_embed_ts = torch.cat(refpoint_embed_ts, dim=1)
        memory_ts = torch.cat(memory_ts, dim=1)
        boxes_ts = torch.cat(boxes_ts, dim=1)
    if self.dec_layers > 0:
        tgt = query_feat.unsqueeze(0).repeat(bs, 1, 1)
        refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)
        if self.two_stage:
            ts_len = refpoint_embed_ts.shape[-2]
            refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :]
            refpoint_embed_subset = refpoint_embed[..., ts_len:, :]
            if self.bbox_reparam:
                refpoint_embed_cxcy = (refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
                                       + refpoint_embed_ts[..., :2])
                refpoint_embed_wh = refpoint_embed_ts_subset[..., 2:].exp() * refpoint_embed_ts[..., 2:]
                refpoint_embed_ts_subset = torch.concat([refpoint_embed_cxcy, refpoint_embed_wh], dim=-1)
            else:
                refpoint_embed_ts_subset = refpoint_embed_ts_subset + refpoint_embed_ts
            refpoint_embed = torch.concat([refpoint_embed_ts_subset, refpoint_embed_subset], dim=-2)
        hs, references = self.decoder(
            tgt, memory, memory_key_padding_mask=None,
            pos=lvl_pos_embed_flatten, refpoints_unsigmoid=refpoint_embed,
            level_start_index=level_start_index, spatial_shapes=spatial_shapes,
            valid_ratios=None, spatial_shapes_list=spatial_shapes_list)
    else:
        hs, references = None, None
    if self.two_stage:
        return hs, references, memory_ts, (boxes_ts if self.bbox_reparam else boxes_ts.sigmoid())
    return hs, references, None, None


def _transformer_decoder_forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                                   tgt_key_padding_mask=None, memory_key_padding_mask=None,
                                   pos=None, refpoints_unsigmoid=None, level_start_index=None,
                                   spatial_shapes=None, valid_ratios=None, spatial_shapes_list=None):
    output = tgt
    intermediate = []
    hs_refpoints_unsigmoid = [refpoints_unsigmoid]

    def get_reference(refpoints):
        obj_center = refpoints[..., :4]
        if self._export:
            from rfdetr.models.transformer import gen_sineembed_for_position
            query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model / 2)
            refpoints_input = obj_center[:, :, None]
        else:
            refpoints_input = obj_center[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            from rfdetr.models.transformer import gen_sineembed_for_position
            query_sine_embed = gen_sineembed_for_position(refpoints_input[:, :, 0, :], self.d_model / 2)
        return obj_center, refpoints_input, self.ref_point_head(query_sine_embed), query_sine_embed

    if self.lite_refpoint_refine:
        obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(
            refpoints_unsigmoid if self.bbox_reparam else refpoints_unsigmoid.sigmoid())

    for layer_id, layer in enumerate(self.layers):
        if not self.lite_refpoint_refine:
            obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(
                refpoints_unsigmoid if self.bbox_reparam else refpoints_unsigmoid.sigmoid())
        output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask,
                       memory_key_padding_mask=memory_key_padding_mask,
                       pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                       is_first=(layer_id == 0), reference_points=refpoints_input,
                       spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                       spatial_shapes_list=spatial_shapes_list)
        if not self.lite_refpoint_refine:
            new_refpoints_unsigmoid = self.refpoints_refine(refpoints_unsigmoid, self.bbox_embed(output))
            if layer_id != self.num_layers - 1:
                hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
            refpoints_unsigmoid = new_refpoints_unsigmoid.detach()
        if self.return_intermediate:
            intermediate.append(self.norm(output))

    if self.norm is not None:
        output = self.norm(output)
        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(output)

    if self.return_intermediate:
        if self._export:
            hs = intermediate[-1]
            ref = hs_refpoints_unsigmoid[-1] if self.bbox_embed is not None else refpoints_unsigmoid
            return hs, ref
        if self.bbox_embed is not None:
            return [torch.stack(intermediate), torch.stack(hs_refpoints_unsigmoid)]
        else:
            return [torch.stack(intermediate), refpoints_unsigmoid.unsqueeze(0)]
    return output.unsqueeze(0), refpoints_unsigmoid.unsqueeze(0)


def _decoder_layer_forward_post(self, tgt, memory, tgt_mask=None, memory_mask=None,
                                 tgt_key_padding_mask=None, memory_key_padding_mask=None,
                                 pos=None, query_pos=None, query_sine_embed=None,
                                 is_first=False, reference_points=None, spatial_shapes=None,
                                 level_start_index=None, spatial_shapes_list=None):
    bs, num_queries, _ = tgt.shape
    q = k = tgt + query_pos
    tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
                          key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]
    tgt = self.norm1(tgt + self.dropout1(tgt2))
    tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, memory,
                           spatial_shapes, level_start_index, memory_key_padding_mask,
                           spatial_shapes_list=spatial_shapes_list)
    tgt = self.norm2(tgt + self.dropout2(tgt2))
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = self.norm3(tgt + self.dropout3(tgt2))
    return tgt


def _decoder_layer_forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                            tgt_key_padding_mask=None, memory_key_padding_mask=None,
                            pos=None, query_pos=None, query_sine_embed=None,
                            is_first=False, reference_points=None, spatial_shapes=None,
                            level_start_index=None, spatial_shapes_list=None):
    return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask,
                             memory_key_padding_mask, pos, query_pos, query_sine_embed,
                             is_first, reference_points, spatial_shapes, level_start_index,
                             spatial_shapes_list=spatial_shapes_list)


def _ms_deform_attn_forward(self, query, reference_points, input_flatten,
                             input_spatial_shapes, input_level_start_index,
                             input_padding_mask=None, spatial_shapes_list=None):
    N, Len_q, _ = query.shape
    N, Len_in, _ = input_flatten.shape
    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))
    sampling_offsets = self.sampling_offsets(query).view(
        N, Len_q, self.n_heads, self.n_levels * self.n_points, 2)
    attention_weights = self.attention_weights(query).view(
        N, Len_q, self.n_heads, self.n_levels * self.n_points)
    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        off_norm_exp = (offset_normalizer.unsqueeze(1).expand(-1, self.n_points, -1)
                        .reshape(self.n_levels * self.n_points, 2))
        ref_pts_exp = (reference_points.unsqueeze(3).expand(-1, -1, -1, self.n_points, -1)
                       .reshape(N, Len_q, self.n_levels * self.n_points, 2).unsqueeze(2))
        sampling_locations = ref_pts_exp + sampling_offsets / off_norm_exp[None, None, None, :, :]
    elif reference_points.shape[-1] == 4:
        ref_center = (reference_points[..., :2].unsqueeze(3).expand(-1, -1, -1, self.n_points, -1)
                      .reshape(N, Len_q, self.n_levels * self.n_points, 2).unsqueeze(2))
        ref_wh = (reference_points[..., 2:].unsqueeze(3).expand(-1, -1, -1, self.n_points, -1)
                  .reshape(N, Len_q, self.n_levels * self.n_points, 2).unsqueeze(2))
        sampling_locations = ref_center + sampling_offsets / self.n_points * ref_wh * 0.5
    else:
        raise ValueError(f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}")
    attention_weights = F.softmax(attention_weights, -1)
    value = value.transpose(1, 2).contiguous().view(N, self.n_heads, self.d_model // self.n_heads, Len_in)
    output = _ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations,
                                          attention_weights, spatial_shapes_list=spatial_shapes_list)
    return self.output_proj(output)


def _ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations,
                                  attention_weights, spatial_shapes_list=None):
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, _, LP, _ = sampling_locations.shape
    _iter = spatial_shapes_list if spatial_shapes_list is not None else value_spatial_shapes
    L = len(_iter)
    P = LP // L
    value_list = value.split([H * W for H, W in _iter], dim=3)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H, W) in enumerate(_iter):
        value_l_ = value_list[lid_].view(B * n_heads, head_dim, H, W)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_ * P:(lid_ + 1) * P, :].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode="bilinear",
                                          padding_mode="zeros", align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(B * n_heads, 1, Len_q, L * P)
    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_value_list * attention_weights).sum(-1).view(B, n_heads * head_dim, Len_q)
    return output.transpose(1, 2).contiguous()


def apply_patches():
    import transformers.pytorch_utils as pu
    if not hasattr(pu, "find_pruneable_heads_and_indices"):
        def _stub(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            return heads, torch.arange(len(mask))[mask].long()
        pu.find_pruneable_heads_and_indices = _stub

    import rfdetr.models.transformer as T
    import rfdetr.models.ops.modules.ms_deform_attn as DA
    import rfdetr.models.ops.functions.ms_deform_attn_func as DAF

    T.gen_encoder_output_proposals = _gen_encoder_output_proposals
    T.Transformer.forward = _transformer_forward
    T.TransformerDecoder.forward = _transformer_decoder_forward
    T.TransformerDecoderLayer.forward = _decoder_layer_forward
    T.TransformerDecoderLayer.forward_post = _decoder_layer_forward_post
    DA.MSDeformAttn.forward = _ms_deform_attn_forward
    DAF.ms_deform_attn_core_pytorch = _ms_deform_attn_core_pytorch
    DA.ms_deform_attn_core_pytorch = _ms_deform_attn_core_pytorch


# ---------------------------------------------------------------------------
# Segmentation wrapper
# ---------------------------------------------------------------------------

class RFDETRSegmentationWrapper(nn.Module):
    """Outputs 3 tensors matching iOS runner expectations:
      - boxes_xyxy    [1, Q, 4]      xyxy in pixel coords
      - scores_labels [1, Q, 2]      stacked [score, label_id]
      - masks         [1, Q, 78, 78] mask logits (sigmoid + threshold on device)
    """

    def __init__(self, model: nn.Module, resolution: int):
        super().__init__()
        self.model = model
        self.resolution = resolution

    def forward(self, x: torch.Tensor):
        pred_boxes, pred_logits, pred_masks = self.model(x)
        scores, labels = pred_logits[0].sigmoid().max(dim=-1)
        boxes = pred_boxes[0]
        cx, cy, w, h = boxes.unbind(-1)
        w = w.clamp(min=0.0)
        h = h.clamp(min=0.0)
        x1 = (cx - 0.5 * w) * self.resolution
        y1 = (cy - 0.5 * h) * self.resolution
        x2 = (cx + 0.5 * w) * self.resolution
        y2 = (cy + 0.5 * h) * self.resolution
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).unsqueeze(0)           # [1, Q, 4]
        scores_labels = torch.stack([scores, labels.float()], dim=-1).unsqueeze(0) # [1, Q, 2]
        return boxes_xyxy, scores_labels, pred_masks                                # [1, Q, 78, 78]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(out_path: str, compute_unit: str = "all"):
    apply_patches()

    from rfdetr import RFDETRSegNano
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
    from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
    from executorch.devtools.backend_debug import get_delegation_info
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from tabulate import tabulate
    import coremltools as ct

    cu_map = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }

    print("Loading RFDETRSegNano...")
    model = RFDETRSegNano()
    actual = model.model.model.cpu()
    actual.eval()
    actual.export()
    resolution = model.model.resolution
    print(f"Resolution: {resolution}px")

    wrapper = RFDETRSegmentationWrapper(actual, resolution)
    wrapper.eval()

    print("Exporting with torch.export...")
    example = (torch.randn(1, 3, resolution, resolution),)
    exported = torch.export.export(wrapper, args=example, strict=False)

    print(f"Lowering to CoreML (fp32 + int8 weights, compute_unit={compute_unit})...")
    t0 = time.perf_counter()

    op_linear_quantizer_config = {"mode": "linear_symmetric", "dtype": "int8", "granularity": "per_channel"}
    compile_specs = CoreMLBackend.generate_compile_specs(
        compute_unit=cu_map[compute_unit],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        op_linear_quantizer_config=op_linear_quantizer_config,
    )
    partitioners = [CoreMLPartitioner(compile_specs=compile_specs, lower_full_graph=False)]
    compile_config = EdgeCompileConfig(_check_ir_validity=False)

    program = to_edge_transform_and_lower(
        exported,
        transform_passes=[AddmmToLinearTransform()],
        partitioner=partitioners,
        compile_config=compile_config,
    ).to_executorch()
    export_time = time.perf_counter() - t0

    graph_module = program.exported_program().graph_module
    delegation_info = get_delegation_info(graph_module)
    print(delegation_info.get_summary())
    df = delegation_info.get_operator_delegation_dataframe()
    print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
    total = delegation_info.num_delegated_nodes + delegation_info.num_non_delegated_nodes
    delegated = delegation_info.num_delegated_nodes
    print(f"\nDelegation: {delegated} / {total} nodes ({delegated/max(total,1)*100:.1f}%)")
    print(f"Export completed in {export_time:.1f}s")

    with open(out_path, "wb") as f:
        f.write(program.buffer)
    import os
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSaved {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RF-DETR Seg Nano segmentation to CoreML .pte")
    parser.add_argument("--output", type=str, default="rfdetr_segmentation_coreml.pte",
                        help="Output .pte path")
    parser.add_argument("--compute-unit", type=str, default="all",
                        choices=["all", "cpu_and_ne", "cpu_only"],
                        help="CoreML compute unit (default: all)")
    args = parser.parse_args()
    export(args.output, compute_unit=args.compute_unit)
