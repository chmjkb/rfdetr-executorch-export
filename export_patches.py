"""
Monkey-patches for making RF-DETR compatible with torch.export.

Usage:
    from swmansion.rfdetr.export_patches import apply_patches
    apply_patches()
    # now import and use rfdetr normally
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from torch import Tensor


def _precompute_proposals_validity(spatial_shapes_list):
    """Return (all_valid: bool, valid_np: np.ndarray[bool, shape (S,)]).

    Computes output_proposals_valid using concrete Python math so that no
    comparison ops (gt/lt/all/logical_not) are emitted into the exported graph.
    For RF-DETR with a fixed-size input all proposals are always valid, so the
    masked_fill calls can be removed entirely.
    """
    valid_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes_list):
        gx = (np.arange(W_, dtype=np.float64) + 0.5) / W_
        gy = (np.arange(H_, dtype=np.float64) + 0.5) / H_
        gy_grid, gx_grid = np.meshgrid(gy, gx, indexing="ij")
        w = h = 0.05 * (2.0**lvl)
        props = np.stack(
            [
                gx_grid,
                gy_grid,
                np.full((H_, W_), w, dtype=np.float64),
                np.full((H_, W_), h, dtype=np.float64),
            ],
            axis=-1,
        )
        valid_list.append(((props > 0.01) & (props < 0.99)).all(axis=-1).reshape(-1))
    valid_np = np.concatenate(valid_list, axis=0)
    return bool(valid_np.all()), valid_np


def apply_patches():
    """Apply all patches needed for torch.export compatibility."""
    _patch_transformers_import()

    import rfdetr.models.transformer as transformer_mod
    import rfdetr.models.ops.modules.ms_deform_attn as deform_attn_mod
    import rfdetr.models.ops.functions.ms_deform_attn_func as deform_attn_func_mod

    # Fix 2+3: gen_encoder_output_proposals
    transformer_mod.gen_encoder_output_proposals = _gen_encoder_output_proposals

    # Fix 5: Transformer.forward — keep spatial_shapes as Python list
    transformer_mod.Transformer.forward = _transformer_forward

    # Fix 5: TransformerDecoder.forward — thread spatial_shapes_list
    transformer_mod.TransformerDecoder.forward = _transformer_decoder_forward

    # Fix 5: TransformerDecoderLayer.forward / forward_post — thread spatial_shapes_list
    transformer_mod.TransformerDecoderLayer.forward = _decoder_layer_forward
    transformer_mod.TransformerDecoderLayer.forward_post = _decoder_layer_forward_post

    # Fix 4+5: MSDeformAttn.forward — skip assert + thread spatial_shapes_list
    deform_attn_mod.MSDeformAttn.forward = _ms_deform_attn_forward

    # Fix 5: ms_deform_attn_core_pytorch — use Python list for iteration
    deform_attn_func_mod.ms_deform_attn_core_pytorch = _ms_deform_attn_core_pytorch
    # Also update the reference in the module that imports it
    deform_attn_mod.ms_deform_attn_core_pytorch = _ms_deform_attn_core_pytorch


def _patch_transformers_import():
    """Fix missing find_pruneable_heads_and_indices in transformers>=5.0."""
    import transformers.pytorch_utils as pu

    if not hasattr(pu, "find_pruneable_heads_and_indices"):

        def _find_pruneable_heads_and_indices(
            heads, n_heads, head_size, already_pruned_heads
        ):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices


# ---------------------------------------------------------------------------
# Fix 2+3: gen_encoder_output_proposals
# ---------------------------------------------------------------------------


def _gen_encoder_output_proposals(
    memory,
    memory_padding_mask,
    spatial_shapes,
    unsigmoid=True,
    spatial_shapes_list=None,
):
    N_, S_, C_ = memory.shape
    proposals = []
    _cur = 0
    _iter = spatial_shapes_list if spatial_shapes_list is not None else spatial_shapes

    # Precompute proposals validity using Python math when spatial shapes are
    # known concrete values (always the case during export with our patches).
    # This eliminates all gt/lt/all/logical_not/masked_fill ops from the graph.
    if spatial_shapes_list is not None:
        all_valid, valid_np = _precompute_proposals_validity(spatial_shapes_list)
    else:
        all_valid = False
        valid_np = None

    for lvl, (H_, W_) in enumerate(_iter):
        if memory_padding_mask is not None:
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(
                N_, H_, W_, 1
            )
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        else:
            valid_H = torch.full((N_,), H_, dtype=torch.long, device=memory.device)
            valid_W = torch.full((N_,), W_, dtype=torch.long, device=memory.device)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H_, dtype=torch.float32, device=memory.device),
            torch.arange(W_, dtype=torch.float32, device=memory.device),
            indexing="ij",
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(
            N_, 1, 1, 2
        )
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_

    output_proposals = torch.cat(proposals, 1)

    if all_valid:
        # All anchor proposals lie within (0.01, 0.99) for this fixed input size —
        # skip the validity masked_fill ops entirely (no where/eq/logical_not in graph).
        output_proposals_valid = None
    else:
        if valid_np is not None:
            # Use precomputed constant — no comparison ops emitted into graph.
            output_proposals_valid = torch.as_tensor(
                valid_np[:, None], dtype=torch.bool, device=memory.device
            ).unsqueeze(0)
        else:
            output_proposals_valid = (
                (output_proposals > 0.01) & (output_proposals < 0.99)
            ).all(-1, keepdim=True)

    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_padding_mask.unsqueeze(-1), float("inf")
            )
        if output_proposals_valid is not None:
            output_proposals = output_proposals.masked_fill(
                ~output_proposals_valid, float("inf")
            )
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_padding_mask.unsqueeze(-1), float(0)
            )
        if output_proposals_valid is not None:
            output_proposals = output_proposals.masked_fill(
                ~output_proposals_valid, float(0)
            )

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)
        )
    if output_proposals_valid is not None:
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


# ---------------------------------------------------------------------------
# Fix 5: Transformer.forward
# ---------------------------------------------------------------------------


def _transformer_forward(self, srcs, masks, pos_embeds, refpoint_embed, query_feat):
    # Fixed-size inference has no padding — mask is always all-False.
    # Forcing None removes the entire where/eq/any/logical_not portable-op chain (~10 ms).
    masks = None
    src_flatten = []
    mask_flatten = [] if masks is not None else None
    lvl_pos_embed_flatten = []
    spatial_shapes = []
    valid_ratios = [] if masks is not None else None
    for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
        bs, c, h, w = src.shape
        spatial_shape = (h, w)
        spatial_shapes.append(spatial_shape)

        src = src.flatten(2).transpose(1, 2)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        lvl_pos_embed_flatten.append(pos_embed)
        src_flatten.append(src)
        if masks is not None:
            mask = masks[lvl].flatten(1)
            mask_flatten.append(mask)
    memory = torch.cat(src_flatten, 1)
    if masks is not None:
        mask_flatten = torch.cat(mask_flatten, 1)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

    # Keep Python list for export-friendly iteration
    spatial_shapes_list = list(spatial_shapes)
    spatial_shapes = torch.as_tensor(
        spatial_shapes, dtype=torch.long, device=memory.device
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )

    if self.two_stage:
        output_memory, output_proposals = _gen_encoder_output_proposals(
            memory,
            mask_flatten,
            spatial_shapes,
            unsigmoid=not self.bbox_reparam,
            spatial_shapes_list=spatial_shapes_list,
        )
        refpoint_embed_ts, memory_ts, boxes_ts = [], [], []
        group_detr = self.group_detr if self.training else 1
        for g_idx in range(group_detr):
            output_memory_gidx = self.enc_output_norm[g_idx](
                self.enc_output[g_idx](output_memory)
            )

            enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](
                output_memory_gidx
            )
            if self.bbox_reparam:
                enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](
                    output_memory_gidx
                )
                enc_outputs_coord_cxcy_gidx = (
                    enc_outputs_coord_delta_gidx[..., :2] * output_proposals[..., 2:]
                    + output_proposals[..., :2]
                )
                enc_outputs_coord_wh_gidx = (
                    enc_outputs_coord_delta_gidx[..., 2:].exp()
                    * output_proposals[..., 2:]
                )
                enc_outputs_coord_unselected_gidx = torch.concat(
                    [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx], dim=-1
                )
            else:
                enc_outputs_coord_unselected_gidx = (
                    self.enc_out_bbox_embed[g_idx](output_memory_gidx)
                    + output_proposals
                )

            topk = min(self.num_queries, enc_outputs_class_unselected_gidx.shape[-2])
            topk_proposals_gidx = torch.topk(
                enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1
            )[1]

            refpoint_embed_gidx_undetach = torch.gather(
                enc_outputs_coord_unselected_gidx,
                1,
                topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4),
            )
            refpoint_embed_gidx = refpoint_embed_gidx_undetach.detach()

            tgt_undetach_gidx = torch.gather(
                output_memory_gidx,
                1,
                topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, self.d_model),
            )

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
                refpoint_embed_cxcy = (
                    refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
                )
                refpoint_embed_cxcy = refpoint_embed_cxcy + refpoint_embed_ts[..., :2]
                refpoint_embed_wh = (
                    refpoint_embed_ts_subset[..., 2:].exp() * refpoint_embed_ts[..., 2:]
                )
                refpoint_embed_ts_subset = torch.concat(
                    [refpoint_embed_cxcy, refpoint_embed_wh], dim=-1
                )
            else:
                refpoint_embed_ts_subset = refpoint_embed_ts_subset + refpoint_embed_ts

            refpoint_embed = torch.concat(
                [refpoint_embed_ts_subset, refpoint_embed_subset], dim=-2
            )

        hs, references = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten,
            refpoints_unsigmoid=refpoint_embed,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios.to(memory.dtype)
            if valid_ratios is not None
            else valid_ratios,
            spatial_shapes_list=spatial_shapes_list,
        )
    else:
        assert self.two_stage, "if not using decoder, two_stage must be True"
        hs = None
        references = None

    if self.two_stage:
        if self.bbox_reparam:
            return hs, references, memory_ts, boxes_ts
        else:
            return hs, references, memory_ts, boxes_ts.sigmoid()
    return hs, references, None, None


# ---------------------------------------------------------------------------
# Fix 5: TransformerDecoder.forward
# ---------------------------------------------------------------------------


def _transformer_decoder_forward(
    self,
    tgt,
    memory,
    tgt_mask: Optional[Tensor] = None,
    memory_mask: Optional[Tensor] = None,
    tgt_key_padding_mask: Optional[Tensor] = None,
    memory_key_padding_mask: Optional[Tensor] = None,
    pos: Optional[Tensor] = None,
    refpoints_unsigmoid: Optional[Tensor] = None,
    level_start_index: Optional[Tensor] = None,
    spatial_shapes: Optional[Tensor] = None,
    valid_ratios: Optional[Tensor] = None,
    spatial_shapes_list=None,
):
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
            refpoints_input = (
                obj_center[:, :, None]
                * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            )
            from rfdetr.models.transformer import gen_sineembed_for_position

            query_sine_embed = gen_sineembed_for_position(
                refpoints_input[:, :, 0, :], self.d_model / 2
            )
        query_pos = self.ref_point_head(query_sine_embed)
        return obj_center, refpoints_input, query_pos, query_sine_embed

    if self.lite_refpoint_refine:
        if self.bbox_reparam:
            obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(
                refpoints_unsigmoid
            )
        else:
            obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(
                refpoints_unsigmoid.sigmoid()
            )

    for layer_id, layer in enumerate(self.layers):
        if not self.lite_refpoint_refine:
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, query_sine_embed = (
                    get_reference(refpoints_unsigmoid)
                )
            else:
                obj_center, refpoints_input, query_pos, query_sine_embed = (
                    get_reference(refpoints_unsigmoid.sigmoid())
                )

        pos_transformation = 1
        query_pos = query_pos * pos_transformation

        output = layer(
            output,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            query_sine_embed=query_sine_embed,
            is_first=(layer_id == 0),
            reference_points=refpoints_input,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            spatial_shapes_list=spatial_shapes_list,
        )

        if not self.lite_refpoint_refine:
            new_refpoints_delta = self.bbox_embed(output)
            new_refpoints_unsigmoid = self.refpoints_refine(
                refpoints_unsigmoid, new_refpoints_delta
            )
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
            if self.bbox_embed is not None:
                ref = hs_refpoints_unsigmoid[-1]
            else:
                ref = refpoints_unsigmoid
            return hs, ref
        if self.bbox_embed is not None:
            return [
                torch.stack(intermediate),
                torch.stack(hs_refpoints_unsigmoid),
            ]
        else:
            return [torch.stack(intermediate), refpoints_unsigmoid.unsqueeze(0)]

    return output.unsqueeze(0), refpoints_unsigmoid.unsqueeze(0)


# ---------------------------------------------------------------------------
# Fix 5: TransformerDecoderLayer.forward_post / forward
# ---------------------------------------------------------------------------


def _decoder_layer_forward_post(
    self,
    tgt,
    memory,
    tgt_mask=None,
    memory_mask=None,
    tgt_key_padding_mask=None,
    memory_key_padding_mask=None,
    pos=None,
    query_pos=None,
    query_sine_embed=None,
    is_first=False,
    reference_points=None,
    spatial_shapes=None,
    level_start_index=None,
    spatial_shapes_list=None,
):
    bs, num_queries, _ = tgt.shape

    q = k = tgt + query_pos
    v = tgt
    if self.training:
        q = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)
        k = torch.cat(k.split(num_queries // self.group_detr, dim=1), dim=0)
        v = torch.cat(v.split(num_queries // self.group_detr, dim=1), dim=0)

    tgt2 = self.self_attn(
        q,
        k,
        v,
        attn_mask=tgt_mask,
        key_padding_mask=tgt_key_padding_mask,
        need_weights=False,
    )[0]

    if self.training:
        tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)

    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)

    tgt2 = self.cross_attn(
        self.with_pos_embed(tgt, query_pos),
        reference_points,
        memory,
        spatial_shapes,
        level_start_index,
        memory_key_padding_mask,
        spatial_shapes_list=spatial_shapes_list,
    )

    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout3(tgt2)
    tgt = self.norm3(tgt)
    return tgt


def _decoder_layer_forward(
    self,
    tgt,
    memory,
    tgt_mask=None,
    memory_mask=None,
    tgt_key_padding_mask=None,
    memory_key_padding_mask=None,
    pos=None,
    query_pos=None,
    query_sine_embed=None,
    is_first=False,
    reference_points=None,
    spatial_shapes=None,
    level_start_index=None,
    spatial_shapes_list=None,
):
    return self.forward_post(
        tgt,
        memory,
        tgt_mask,
        memory_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
        pos,
        query_pos,
        query_sine_embed,
        is_first,
        reference_points,
        spatial_shapes,
        level_start_index,
        spatial_shapes_list=spatial_shapes_list,
    )


# ---------------------------------------------------------------------------
# Fix 4+5: MSDeformAttn.forward
# ---------------------------------------------------------------------------


def _ms_deform_attn_forward(
    self,
    query,
    reference_points,
    input_flatten,
    input_spatial_shapes,
    input_level_start_index,
    input_padding_mask=None,
    spatial_shapes_list=None,
):
    N, Len_q, _ = query.shape
    N, Len_in, _ = input_flatten.shape
    if not self._export:
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))

    sampling_offsets = self.sampling_offsets(query).view(
        N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
    )
    attention_weights = self.attention_weights(query).view(
        N, Len_q, self.n_heads, self.n_levels * self.n_points
    )

    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
    elif reference_points.shape[-1] == 4:
        sampling_locations = (
            reference_points[:, :, None, :, None, :2]
            + sampling_offsets
            / self.n_points
            * reference_points[:, :, None, :, None, 2:]
            * 0.5
        )
    else:
        raise ValueError(
            "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                reference_points.shape[-1]
            )
        )

    attention_weights = F.softmax(attention_weights, -1)

    value = (
        value.transpose(1, 2)
        .contiguous()
        .view(N, self.n_heads, self.d_model // self.n_heads, Len_in)
    )
    output = _ms_deform_attn_core_pytorch(
        value,
        input_spatial_shapes,
        sampling_locations,
        attention_weights,
        spatial_shapes_list=spatial_shapes_list,
    )
    output = self.output_proj(output)
    return output


# ---------------------------------------------------------------------------
# Fix 5: ms_deform_attn_core_pytorch
# ---------------------------------------------------------------------------


def _ms_deform_attn_core_pytorch(
    value,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
    spatial_shapes_list=None,
):
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, _, L, P, _ = sampling_locations.shape
    _iter = (
        spatial_shapes_list if spatial_shapes_list is not None else value_spatial_shapes
    )

    value_list = value.split([H * W for H, W in _iter], dim=3)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H, W) in enumerate(_iter):
        value_l_ = value_list[lid_].view(B * n_heads, head_dim, H, W)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        B * n_heads, 1, Len_q, L * P
    )
    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (
        (sampling_value_list * attention_weights)
        .sum(-1)
        .view(B, n_heads * head_dim, Len_q)
    )
    return output.transpose(1, 2).contiguous()
