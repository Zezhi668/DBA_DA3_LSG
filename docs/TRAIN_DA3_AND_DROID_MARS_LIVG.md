# Training DA3 Metric Head for MARS-LIVG

This file records the current DA3-only training path for MARS-LIVG. The old DROID-SLAM fine-tuning notes were removed because the practical workflow is to keep the frontend checkpoint fixed and adapt DA3Metric-Large as a metric depth prior first.

## Local Paths

Use these paths unless the machine layout changes:

```text
DPT-LSG repo:          /home/server/VINGS_work/DPT-LSG
DA3 source checkout:   /home/server/VINGS_work/Depth-Anything-3
MARS-LIVG root:        /media/server/yzz_disk/Dataset_sx/MARS-LIVG
Base DA3 model:        /home/server/VINGS_work/DPT-LSG/ckpts/DA3Metric-Large
Stage 1 checkpoint:    /home/server/VINGS_work/DPT-LSG/ckpts/da3_mars_metric_head.pth
Old Stage 2 checkpoint: /home/server/VINGS_work/DPT-LSG/ckpts/da3_mars_hkairport_amtown01_head.pth
New Stage 2 checkpoint: /home/server/VINGS_work/DPT-LSG/ckpts/da3_mars_hkairport_amtown01_stage2e3_head.pth
```

The final MARS configs should point to:

```yaml
da3:
  enabled: true
  model_name: da3metric-large
  checkpoint_path: /home/server/VINGS_work/DPT-LSG/ckpts/da3_mars_hkairport_amtown01_stage2e3_head.pth
```

## Environment

Do not install DA3 into Python 3.13. The DA3 package metadata rejects that environment. Use Python 3.10 or 3.11:

```bash
cd /home/server/VINGS_work
conda create -n da3_train python=3.10 -y
conda activate da3_train
```

Install DA3 from the source checkout:

```bash
cd /home/server/VINGS_work/Depth-Anything-3
pip install xformers "torch>=2" torchvision
pip install -e .
```

If editable install is not needed, the DPT-LSG training script can import DA3 directly through:

```bash
--source-dir /home/server/VINGS_work/Depth-Anything-3/src
```

## Training Data

The DA3 metric head is trained from sparse Livox LiDAR depth projected into the undistorted and cropped MARS-LIVG RGB frames.

| Split | Role | TUM directory | Rosbag | Calibration | RGB frames | Frames with LiDAR depth |
| --- | --- | --- | --- | --- | ---: | ---: |
| `HKairport_GNSS01` | Stage 1 adaptation | `/media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/HKairport_GNSS01_trimmed` | `/media/server/yzz_disk/Dataset_sx/MARS-LIVG/HKairport_GNSS01.bag` | `cali/HK_GNSS(airport & island).yaml` | 6,772 | 6,436 |
| `AMtown01` | Stage 2 continuation | `/media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/AMtown01_trimmed` | `/media/server/yzz_disk/Dataset_sx/MARS-LIVG/AMtown01.bag` | `cali/AMtown.yaml` | 12,372 | 12,145 |

The total supervised set contains 18,581 RGB frames with sparse depth labels. The labels are intentionally stored as `sparse_npz`, not dense full-frame arrays.

## Export TUM Data

Skip this step if the TUM folders already exist.

HKairport_GNSS01 was exported with the trimmed time interval:

```bash
cd /home/server/VINGS_work/DPT-LSG

python scripts/datasets/mars_livg_rosbag_to_tum.py \
  --sequence HKairport_GNSS01 \
  --output-dir /media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/HKairport_GNSS01_trimmed \
  --start-timestamp 1698217498.07 \
  --end-timestamp 1698218175.24 \
  --undistort \
  --crop-undistorted \
  --gt-mode nearest \
  --overwrite
```

AMtown01 uses the same TUM export script. If reproducing the exact existing `AMtown01_trimmed` folder, use the same trimming interval that produced that folder. If exporting from scratch without a trim requirement, omit the start/end timestamps:

```bash
cd /home/server/VINGS_work/DPT-LSG

python scripts/datasets/mars_livg_rosbag_to_tum.py \
  --sequence AMtown01 \
  --output-dir /media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/AMtown01_trimmed \
  --undistort \
  --crop-undistorted \
  --gt-mode nearest \
  --overwrite
```

Expected TUM output:

```text
rgb/
rgb.txt
camera_info.yaml
groundtruth.txt
export_info.json
```

## Export Sparse LiDAR Depth

Run the sparse depth exporter for both training splits.

Stage 1 labels:

```bash
cd /home/server/VINGS_work/DPT-LSG

python scripts/datasets/mars_livg_lidar_depth_export.py \
  --sequence HKairport_GNSS01 \
  --tum-dir /media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/HKairport_GNSS01_trimmed \
  --output-dir /media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/HKairport_GNSS01_trimmed/lidar_depth_training \
  --start-timestamp 1698217498.07 \
  --end-timestamp 1698218175.24 \
  --depth-format sparse_npz \
  --progress-interval 100 \
  --overwrite
```

Stage 2 labels:

```bash
cd /home/server/VINGS_work/DPT-LSG

python scripts/datasets/mars_livg_lidar_depth_export.py \
  --sequence AMtown01 \
  --bag-path /media/server/yzz_disk/Dataset_sx/MARS-LIVG/AMtown01.bag \
  --tum-dir /media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/AMtown01_trimmed \
  --output-dir /media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/AMtown01_trimmed/lidar_depth_training \
  --depth-format sparse_npz \
  --progress-interval 100 \
  --overwrite
```

Expected label output:

```text
lidar_depth_training/depth_sparse/*.npz
lidar_depth_training/mask/*.png
lidar_depth_training/manifest.jsonl
lidar_depth_training/summary.json
```

Each manifest record stores the RGB path, sparse depth path, mask path, timestamp pair, image size, valid pixel count, and per-frame intrinsics in standard `[fx, fy, cx, cy]` order.

## Intrinsics Check

Before training, verify that the TUM export and sparse-depth manifests agree:

```bash
cd /home/server/VINGS_work/DPT-LSG

python scripts/datasets/check_mars_intrinsics.py \
  /media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/HKairport_GNSS01_trimmed/lidar_depth_training/manifest.jsonl \
  /media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/AMtown01_trimmed/lidar_depth_training/manifest.jsonl
```

The manifest uses standard `[fx, fy, cx, cy]`. The runtime SLAM YAMLs use this repo's older convention:

```yaml
fu = fy
fv = fx
cu = cy
cv = cx
```

Do not mix those conventions when editing config files.

## Training Script

The local trainer is:

```text
scripts/da3/train_mars_metric_head.py
```

Default behavior:

- Loads `DepthAnything3.from_pretrained(...)`.
- Freezes the whole DA3 model.
- Unfreezes only `model.model.head`.
- Trains on sparse LiDAR pixels through a masked metric depth loss.
- Stores the full state dict plus optimizer/scaler metadata.

The loss is:

```text
mean(|log(pred) - log(gt)|) + 0.2 * mean(|pred - gt| / gt)
```

Only pixels passing the sparse mask, finite checks, and depth range checks contribute to the loss.

## Stage 1: HKairport_GNSS01, 20k Steps

This is the initial MARS-LIVG adaptation from DA3Metric-Large.

```bash
cd /home/server/VINGS_work/DPT-LSG
conda activate da3_train

python scripts/da3/train_mars_metric_head.py \
  --manifest /media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/HKairport_GNSS01_trimmed/lidar_depth_training/manifest.jsonl \
  --source-dir /home/server/VINGS_work/Depth-Anything-3/src \
  --pretrained /home/server/VINGS_work/DPT-LSG/ckpts/DA3Metric-Large \
  --output /home/server/VINGS_work/DPT-LSG/ckpts/da3_mars_metric_head.pth \
  --image-size 448 560 \
  --batch-size 2 \
  --workers 4 \
  --steps 20000 \
  --lr 1e-5 \
  --min-depth 0.5 \
  --max-depth 300.0 \
  --precision fp16 \
  --save-every 500
```

Important outputs:

| Checkpoint | Meaning |
| --- | --- |
| `ckpts/da3_mars_metric_head.pth` | Final Stage 1 checkpoint |
| `ckpts/da3_mars_metric_head_step20000.pth` | Explicit Stage 1 final snapshot |
| `ckpts/da3_mars_metric_head_step*.pth` | Intermediate snapshots every 500 steps |

## Stage 2: AMtown01, 3 Epochs

This is the missing continuation stage. It starts from the HKairport checkpoint and continues training the same metric head on AMtown01. It is not a fresh restart from DA3Metric-Large.

AMtown01 has 12,145 usable sparse-depth records. With `--batch-size 2` and `drop_last=True`, one epoch is 6,072 optimizer steps, so `--epochs 3` runs 18,216 steps. This replaces the earlier 8k-step continuation, which covered only about 1.3 epochs.

```bash
cd /home/server/VINGS_work/DPT-LSG
conda activate da3_train

python scripts/da3/train_mars_metric_head.py \
  --manifest /media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/AMtown01_trimmed/lidar_depth_training/manifest.jsonl \
  --source-dir /home/server/VINGS_work/Depth-Anything-3/src \
  --pretrained /home/server/VINGS_work/DPT-LSG/ckpts/DA3Metric-Large \
  --init-checkpoint /home/server/VINGS_work/DPT-LSG/ckpts/da3_mars_metric_head_step20000.pth \
  --output /home/server/VINGS_work/DPT-LSG/ckpts/da3_mars_hkairport_amtown01_stage2e3_head.pth \
  --image-size 448 560 \
  --batch-size 2 \
  --workers 4 \
  --epochs 3 \
  --lr 3e-6 \
  --min-depth 0.5 \
  --max-depth 120.0 \
  --precision fp16 \
  --save-every 2000
```

Important outputs:

| Checkpoint | Meaning |
| --- | --- |
| `ckpts/da3_mars_hkairport_amtown01_stage2e3_head.pth` | Final HKairport_GNSS01 + AMtown01 3-epoch checkpoint |
| `ckpts/da3_mars_hkairport_amtown01_stage2e3_head_step18000.pth` | Late Stage 2 snapshot near the final epoch |
| `ckpts/da3_mars_hkairport_amtown01_stage2e3_head_step*.pth` | Intermediate snapshots every 2000 steps |

Stage 2 uses a lower learning rate because the model is already adapted to MARS-LIVG after Stage 1. The AMtown continuation broadens the aerial-domain depth prior without aggressively overwriting the airport adaptation.

## Runtime Use

Use the final checkpoint for AMtown and AMvalley-style runs:

```yaml
da3:
  enabled: True
  model_id: depth-anything/DA3METRIC-LARGE
  model_name: da3metric-large
  source_dir: /home/server/VINGS_work/Depth-Anything-3/src
  model_dir: /home/server/VINGS_work/DPT-LSG/ckpts/DA3Metric-Large
  checkpoint_path: /home/server/VINGS_work/DPT-LSG/ckpts/da3_mars_hkairport_amtown01_stage2e3_head.pth
  precision: float16
  process_res: 560
  process_res_method: upper_bound_resize
  async: True
  canonical_focal_scale: True
  depth_scale: 1.0
  min_depth: 0.5
  max_depth: 300.0
  replace_existing_depth: True
```

The current safe depth-prior anchoring policy is scale-only:

```yaml
anchor_mode: scale
align_min_pixels: 512
align_trim: 0.1
align_min_scale: 0.5
align_max_scale: 2.0
align_min_shift: 0.0
align_max_shift: 0.0
align_blend: 0.35
frontend_refine_updates: 0
```

This avoids aggressive affine depth shifts during tracking and mapping.

## Checkpoint Evaluation

Use the sparse-depth evaluator to compare checkpoints against the manifest labels:

```bash
cd /home/server/VINGS_work/DPT-LSG

python scripts/da3/evaluate_mars_metric_head.py \
  --manifest /media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/AMtown01_trimmed/lidar_depth_training/manifest.jsonl \
  --checkpoint /home/server/VINGS_work/DPT-LSG/ckpts/da3_mars_metric_head_step20000.pth \
  --checkpoint /home/server/VINGS_work/DPT-LSG/ckpts/da3_mars_hkairport_amtown01_stage2e3_head.pth \
  --batch-size 2 \
  --workers 4 \
  --max-depth 120.0 \
  --output /home/server/output/DPT-LSG_output/da3_eval_amtown_checkpoint_compare.json
```

For paper-quality metric-depth numbers, use raw non-aligned metrics. The evaluator's `--median-align` option is diagnostic only.

## Current Default

The current default training recipe is:

```text
Stage 1: HKairport_GNSS01 sparse LiDAR labels, 20,000 steps, lr 1e-5
Stage 2: AMtown01 sparse LiDAR labels, 3 epochs / 18,216 steps, lr 3e-6
Trainable: DA3 metric head only
Frozen: DA3 backbone
Runtime checkpoint after validation: ckpts/da3_mars_hkairport_amtown01_stage2e3_head.pth
```
