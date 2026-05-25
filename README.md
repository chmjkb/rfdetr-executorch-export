
# RF-DETR ExecuTorch Export

Exports RF-DETR Nano to a `.pte` (ExecuTorch) model delegated to XNNPACK.

## Dependencies

| Dependency | Notes |
|---|---|
| `executorch` | built with `XNNPACK=ON` |
| `rfdetr` | the model package |
| `torch` | ≥2.3 |
| `flatc` | XNNPACK serialization — must be on `PATH` (lives at `.venv/bin/flatc`) |
| `tabulate`, `numpy` | display and precomputation |

Always prefix the command with `PATH=".venv/bin:$PATH"` so `flatc` is found.

## Usage

```bash
# Detection — RFDETRNano → model_det.pte
PATH=".venv/bin:$PATH" python -m swmansion.rfdetr.export

# Segmentation — RFDETRSegNano → model_seg.pte
PATH=".venv/bin:$PATH" python -m swmansion.rfdetr.export --segmentation

# Custom output path
PATH=".venv/bin:$PATH" python -m swmansion.rfdetr.export --output my_model.pte
```

## Outputs

| Model | File | Outputs |
|---|---|---|
| Detection | `model_det.pte` | `boxes_xyxy [Q,4]`, `scores [Q]`, `labels [Q]` — fp32, pixel coords |
| Segmentation | `model_seg.pte` | `bboxes [1,Q,4]`, `scores [1,Q,2]`, `mask_logits [1,Q,78,78]` |

Detection boxes are in model pixel coordinates (0 to resolution, default 384). Scale by `orig_w/resolution` and `orig_h/resolution` to get pixel coords in the original image.

## How it works

1. **Load model** — `RFDETRNano` / `RFDETRSegNano`, move to CPU, call `.export()` to switch to export mode.
2. **Apply patches** (`export_patches.py`) — monkey-patches the deformable attention and transformer to be compatible with `torch.export` (see below).
3. **Export** — `torch.export.export` with a random fp32 example input.
4. **Lower to XNNPACK** — `to_edge_transform_and_lower` with:
   - `AddmmToLinearTransform` converts `aten.addmm` → `aten.linear` so XNNPACK can capture HuggingFace-style encoder layers.
   - `XnnpackPartitioner` delegates compute-heavy ops (GEMM, conv, elementwise) to XNNPACK.
   - `_core_aten_ops_exception_list=[aten.linear]` prevents the edge compiler from decomposing linears back to addmm before partitioning.
5. **Serialize** — writes the `.pte` flatbuffer.

## Patches (`export_patches.py`)

RF-DETR uses deformable attention with dynamic spatial shapes, which breaks `torch.export` in several ways. The patches fix this without touching the model weights or changing numerical outputs.

| Patch | Problem solved |
|---|---|
| `_patch_transformers_import` | `transformers>=5.0` removed `find_pruneable_heads_and_indices`; rfdetr imports it |
| `_transformer_forward` (`masks=None`) | Padding masks are all-False for fixed-size input — forcing `None` removes ~10 ms of `where`/`any`/`logical_not` portable ops |
| `spatial_shapes_list` threading | `torch.export` traces with `FakeTensor`; iterating over or calling `.item()` on a `FakeTensor` fails. A Python list of concrete `(H, W)` tuples is threaded alongside the tensor version |
| `_ms_deform_attn_forward` assert removed | The shape assertion `(spatial_shapes[:, 0] * ...).sum() == Len_in` is data-dependent and fails during tracing |
| `_gen_encoder_output_proposals` | Original code uses `spatial_shapes` tensor for iteration (breaks tracing) and computes anchor validity via `gt/lt/all` (emits unnecessary portable ops). Replaced with Python-side precomputation — for any reasonable fixed input size all proposals are valid so the `masked_fill` ops are skipped entirely |
