# DBA_DA3_LSG

DBA_DA3_LSG is a local research branch derived from the original VINGS-Mono
codebase. It focuses on a DBA/DROID-style frontend with DA3 metric-depth prior
integration and large-scale Gaussian mapping utilities.

This branch is based on the VINGS-Mono open-source project and includes
optimizations for UAV scenarios as part of an undergraduate project.

For the original project description, baseline setup, dataset preparation,
paper citation, and inherited design context, refer to the original README:

- Original project page: https://vings-mono.github.io/
- Original code: https://github.com/Fudan-MAGIC-Lab/VINGS-Mono
- Original paper: https://arxiv.org/abs/2501.08286

## What Is Included

### Environment

- `env_setup.sh`: reproducible environment setup script. The default conda
  environment is `vings_isolated`.
- `set_env.sh`: older local setup helper kept for compatibility.
- `requirements.txt`: Python package requirements for the uploaded project.
- `backup/environment.yaml`, `backup/piplist.txt`, `backup/setenv_vo.sh`: saved
  environment references from the original/local setup.

### Main Entry Points

- `scripts/run.py`: single-process tracking and mapping entry point.
- `scripts/run_tracking.py`: tracking-side entry point.
- `scripts/run_mapping.py`: Gaussian mapping-side entry point.
- `scripts/run_multiprocess.py`: two-process tracking/mapping entry point.
- `scripts/run_multiprocess_novis.py`: two-process tracking/mapping entry point
  without visualization.
- `scripts/run_mobile.py`, `scripts/run_multiprocess_mobile.py`: mobile/server
  workflow entry points.
- `scripts/evaluate.py`: evaluation entry point.

### DBA / DA3 / Frontend Code

- `scripts/frontend/`: DBA/DROID-style frontend, motion filtering, covisible
  graph, depth video, and DA3 depth-prior integration.
- `scripts/frontend/da3_depth_prior.py`: keyframe DA3 metric-depth prior module.
- `scripts/da3/`: DA3 metric-head training and evaluation scripts.
- `scripts/metric/`: metric-depth model wrappers.
- `scripts/frontend_vo/`: visual odometry and factor-graph frontend modules.
- `scripts/frontend_mast3r/`: MASt3R-SLAM frontend integration.
- `scripts/loop/`: loop detection, loop model, and loop rectification.
- `scripts/submap/`: submap management.

### Gaussian Mapping

- `scripts/gaussian/`: Gaussian model, optimization utilities, rendering and
  visualization helpers, loss utilities, and camera/data helpers.
- `scripts/storage/`: map and runtime storage management.
- `scripts/dynamic/`: dynamic-scene helper utilities.
- `scripts/vings_utils/`: shared geometry, middleware, Sim3, memory, and utility
  functions.

### Dataset Utilities

- `scripts/datasets/`: dataset readers and conversion utilities.
- Included support covers Hierarchical 3DGS, RTG hotel, KITTI, KITTI360, Waymo,
  MARS-LIVG, R3live, SEU-BEV, UAV image folders, TUM RGB-D, Replica, ScanNet,
  TartanAir, Bonn, BundleFusion, MegaNeRF, UrbanScene3D, and custom/local
  formats.

### Configurations

- `configs/hierarchical/`: SmallCity and MASt3R SmallCity configs.
- `configs/rtg/`: RTG hotel config.
- `configs/kitti/sync/`: KITTI sync configs.
- `configs/kitti360/unsync/`: KITTI360 unsync configs.
- `configs/waymo/`: Waymo scene configs and `nerfslam` variants.
- `configs/MARS-LIVG/`: MARS-LIVG configs.
- `configs/R3live/`: R3live config.
- `configs/SEU-BEV/`: SEU-BEV configs.
- `configs/UAV_dataset/`: UAV dataset config.

### Documentation

- `docs/PREPARE_DATA.md`: inherited dataset preparation notes.
- `docs/CODEBASE_PIPELINE_FLOWCHART.md`: codebase pipeline flowchart.
- `docs/DA3_METRIC_DEPTH_PRIOR_INTEGRATION_FLOWCHART.md`: DA3 integration flow.
- `docs/DPT_LSG_DA3_DROID_FRONTEND_DESIGN.md`: DA3-anchored DROID frontend
  design.
- `docs/TRAIN_DA3_AND_DROID_MARS_LIVG.md`: DA3/DROID training notes for
  MARS-LIVG.
- `docs/DA3_MARS_LIVG_TRAINING_CHAPTER.md`: DA3 MARS-LIVG training chapter.
- `docs/EVALUATOR_SIM3_ALIGNMENT.md`: Sim3 evaluator alignment notes.
- `docs/PAPER_EXPERIMENT_SETTINGS_CONSERVATIVE.md`: conservative experiment
  settings.
- `docs/SUBMAP_MECHANISM_FLOWCHART.md`: submap mechanism flowchart.
- `docs/MAST3R_LSG_PATENT_DRAFT.md`: MASt3R-LSG patent draft.
- `docs/SUBMAP_MECHANISM_ADAPTIVE_GAUSSIAN_MONOCULAR_SLAM_PATENT_DRAFT.md`:
  adaptive Gaussian monocular SLAM submap patent draft.
- `docs/UAV_MAST3R_SUBMAP_GAUSSIAN_SLAM_PATENT_DRAFT.md`: UAV MASt3R submap
  Gaussian SLAM patent draft.
- `docs/logo.png`: inherited project logo image.

### Native Extensions and Third-Party Code

- `dbaf/`: local DBAF CUDA extension source and vendored Eigen/LieTorch code.
- `submodules/diff-surfel-rasterization`: diff-surfel rasterization submodule.
- `submodules/gtsam`: GTSAM submodule.
- `submodules/dbaf`: DBAF submodule.
- `submodules/metric_modules`: metric-depth module submodule.
- `submodules/dbef/thirdparty/lietorch`: LieTorch submodule.
- `submodules/dbef/thirdparty/eigen`: Eigen submodule.

## Environment Setup

Clone with submodules:

```bash
git clone --recursive https://github.com/Zezhi668/DBA_DA3_LSG.git
cd DBA_DA3_LSG
```

Create or reuse the default conda environment:

```bash
bash env_setup.sh
conda activate vings_isolated
```

Useful overrides:

```bash
CONDA_BIN=/path/to/conda bash env_setup.sh
ENV_NAME=vings_isolated bash env_setup.sh
INSTALL_DA3=1 DA3_PATH=/path/to/Depth-Anything-3 bash env_setup.sh
```

The script installs the Torch/CUDA 11.8 stack, project requirements, DBAF, and
GTSAM by default. DA3 source installation is optional because Depth-Anything-3
is not vendored in this repository.

## Not Included

Large local runtime artifacts are intentionally ignored and are not part of the
GitHub repository:

- `ckpts/`
- root-level `datasets/` and `data/`
- `output/`, `outputs/`, `results/`, `logs/`, `wandb/`
- Python caches and local build metadata

See `NOT_UPLOADED_THIS_TIME.md` for the exact local checkpoint list observed
when this repository was prepared.

## Multi-GPU Status

This project does not currently implement true multi-GPU parallel execution
through `torch.distributed`, `DistributedDataParallel`, or `DataParallel`.

What exists:

- The multiprocess scripts split tracking and mapping into separate Python
  processes.
- Config files expose `device.tracker` and `device.mapper` fields.

Current limitation:

- Most configs set both tracker and mapper to `cuda:0`.
- Several frontend/utility modules hard-code `cuda:0`.
- There is no distributed synchronization or model sharding path.

Practical answer: it can run multiple processes, but it should be treated as a
single-GPU project unless the device handling is audited and patched for a
specific multi-GPU layout.
