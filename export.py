"""
Export RF-DETR Nano to an ExecuTorch .pte model.

Modes
-----
detection    (default)  RFDETRNano   → model_det.pte
                        outputs: boxes_xyxy [Q,4], scores [Q], labels [Q]  (fp32)

segmentation            RFDETRSegNano → model_seg.pte
                        outputs: pred_boxes [1,Q,4], pred_logits [1,Q,91], pred_masks [1,Q,78,78]

Usage
-----
    PATH=".venv/bin:$PATH" python -m swmansion.rfdetr.export
    PATH=".venv/bin:$PATH" python -m swmansion.rfdetr.export --segmentation
    PATH=".venv/bin:$PATH" python -m swmansion.rfdetr.export --output my_model.pte
"""

import argparse

import torch
import torch.nn as nn

from .export_patches import apply_patches

apply_patches()

from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from tabulate import tabulate


class RFDETRDetectionWrapper(nn.Module):
    """Wraps RF-DETR to output (boxes_xyxy [Q,4], scores [Q], labels [Q]).

    Boxes are xyxy in model pixel coordinates (0 to resolution).
    Scores are sigmoid probabilities in [0, 1].
    Labels are float32 COCO class IDs.
    """

    def __init__(self, model: nn.Module, resolution: int):
        super().__init__()
        self.model = model
        self.resolution = resolution

    def forward(self, x: torch.Tensor):
        pred_boxes, pred_logits = self.model(x)
        scores, labels = pred_logits[0].sigmoid().max(dim=-1)
        boxes = pred_boxes[0]
        cxy = boxes[..., :2]
        wh = boxes[..., 2:].clamp(min=0.0)
        half_wh = 0.5 * wh
        bboxes = torch.cat([cxy - half_wh, cxy + half_wh], dim=-1) * self.resolution
        return bboxes, scores, labels.float()


class RFDETRSegmentationWrapper(nn.Module):
    """Wraps RF-DETR Segmentation to output (bboxes [1,Q,4], scores [1,Q,2], mask_logits [1,Q,H,W]).

    Boxes are xyxy in model pixel coordinates (0 to resolution).
    Scores are [max_score, class_id], max_score is post-sigmoid.
    Mask logits are pre-sigmoid.
    """

    def __init__(self, model: nn.Module, resolution: int):
        super().__init__()
        self.model = model
        self.resolution = resolution

    def forward(self, x: torch.Tensor):
        # pred_boxes: [1, Q, 4], pred_logits: [1, Q, 91], pred_masks: [1, Q, H, W]
        pred_boxes, pred_logits, pred_masks = self.model(x)

        # Bbox: cxcywh -> xyxy via slicing (no unbind/stack roundtrip)
        cxy = pred_boxes[..., :2]
        wh = pred_boxes[..., 2:].clamp(min=0.0)
        half_wh = 0.5 * wh
        bboxes = torch.cat([cxy - half_wh, cxy + half_wh], dim=-1) * self.resolution

        # Scores: [1, Q, 2] containing [max_score, class_id]
        probs = pred_logits.sigmoid()
        max_scores, class_ids = probs.max(dim=-1)
        scores = torch.stack([max_scores, class_ids.float()], dim=-1)

        return bboxes, scores, pred_masks


def _lower(exported, out_path):
    program = to_edge_transform_and_lower(
        exported,
        transform_passes=[AddmmToLinearTransform()],
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default],
        ),
    ).to_executorch()

    graph_module = program.exported_program().graph_module
    delegation_info = get_delegation_info(graph_module)
    print(delegation_info.get_summary())
    df = delegation_info.get_operator_delegation_dataframe()
    print(tabulate(df, headers="keys", tablefmt="fancy_grid"))

    with open(out_path, "wb") as f:
        f.write(program.buffer)
    print(f"\nSaved {out_path}")


def export_detection(out_path: str = "model_det.pte"):
    from rfdetr import RFDETRNano  # type: ignore

    model = RFDETRNano()
    actual = model.model.model.cpu()  # type: ignore
    actual.eval()
    actual.export()
    resolution = model.model.resolution  # type: ignore

    wrapper = RFDETRDetectionWrapper(actual, resolution)
    wrapper.eval()

    example = (torch.randn(1, 3, resolution, resolution),)
    with torch.inference_mode():
        exported = torch.export.export(wrapper, args=example, strict=False)

    _lower(exported, out_path)


def export_segmentation(out_path: str = "model_seg.pte"):
    from rfdetr import RFDETRSegNano  # type: ignore

    model = RFDETRSegNano()
    actual = model.model.model.cpu()  # type: ignore
    actual.eval()
    actual.export()
    resolution = model.model.resolution  # type: ignore

    wrapper = RFDETRSegmentationWrapper(actual, resolution)
    wrapper.eval()

    example = (torch.randn(1, 3, resolution, resolution),)
    with torch.inference_mode():
        exported = torch.export.export(wrapper, args=example, strict=False)

    _lower(exported, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export RF-DETR Nano to ExecuTorch .pte"
    )
    parser.add_argument(
        "--segmentation",
        action="store_true",
        help="Export segmentation model (RFDETRSegNano) instead of detection",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pte path (default: model_det.pte or model_seg.pte)",
    )
    args = parser.parse_args()

    if args.segmentation:
        out = args.output or "model_seg.pte"
        export_segmentation(out)
    else:
        out = args.output or "model_det.pte"
        export_detection(out)
