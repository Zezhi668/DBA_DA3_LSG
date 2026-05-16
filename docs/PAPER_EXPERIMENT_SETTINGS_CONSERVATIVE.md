# Experiment Settings

This section describes the final conservative configuration used for the main
DPT-LSG experiments. Ablation settings are not included here; the goal of this
section is to document the final system configuration used for the reported
MARS-LIVG and SEU-BEV results.

## Experimental Setup

All experiments use a monocular RGB input only. No IMU, GPS, RTK, LiDAR, or
other test-time sensor measurements are used by the SLAM pipeline. The system is
run in visual odometry mode with DROID-SLAM as the tracking frontend, DA3Metric
as the monocular metric depth prior, a submap-based Gaussian mapper, and a
LightGlue-based loop candidate validator. The same conservative algorithmic
configuration is used for AMvalley01-03, HKisland01-03, and SEU-BEV 01/04/05;
only the dataset path and camera calibration are changed between sequences.

The experiments are executed with `use_metric: True`, `use_loop: True`, and
`submap.enabled: True`. Pose refinement, sky masking, and dynamic-object masking
are disabled. Tracking and mapping are both run on `cuda:0`. Runtime memory is
recorded with the built-in memory monitor at a 1 s interval.

## Test Sequences

| Dataset | Sequence | Config file | Input root |
|---|---:|---|---|
| MARS-LIVG | AMvalley01 | `configs/MARS-LIVG/AMvalley01_tum.yaml` | `/media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/AMvalley01_trimmed` |
| MARS-LIVG | AMvalley02 | `configs/MARS-LIVG/AMvalley02_tum.yaml` | `/media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/AMvalley02_trimmed` |
| MARS-LIVG | AMvalley03 | `configs/MARS-LIVG/AMvalley03_con.yaml` | `/media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/AMvalley03_trimmed` |
| MARS-LIVG | HKisland01 | `configs/MARS-LIVG/HKisland01_con.yaml` | `/media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/HKisland01_trimmed` |
| MARS-LIVG | HKisland02 | `configs/MARS-LIVG/HKisland02_con.yaml` | `/media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/HKisland02_trimmed` |
| MARS-LIVG | HKisland03 | `configs/MARS-LIVG/HKisland03_con.yaml` | `/media/server/yzz_disk/Dataset_sx/MARS-LIVG/tum/HKisland03_trimmed` |
| SEU-BEV | 01 | `configs/SEU-BEV/seu_bev_01_con.yaml` | `/media/server/yzz_disk/Dataset_sx/SEU-BEV/01extracted_images` |
| SEU-BEV | 04 | `configs/SEU-BEV/seu_bev_04_con.yaml` | `/media/server/yzz_disk/Dataset_sx/SEU-BEV/04extracted_images` |
| SEU-BEV | 05 | `configs/SEU-BEV/seu_bev_05_con.yaml` | `/media/server/yzz_disk/Dataset_sx/SEU-BEV/05extracted_images` |

## Camera Calibration

The repository stores camera intrinsics with the convention
`fu = fy`, `fv = fx`, `cu = cy`, and `cv = cx`. The following table reports the
values in that convention.

| Sequence group | H | W | fu | fv | cu | cv | Distortion handling |
|---|---:|---:|---:|---:|---:|---:|---|
| AMvalley01 | 2047 | 2447 | 1449.7943 | 1452.5300 | 1049.8915 | 1182.9197 | Not used |
| AMvalley02/03 | 2047 | 2447 | 1449.0828 | 1451.9288 | 1049.3727 | 1182.4370 | Not used |
| HKisland01/02/03 | 2047 | 2447 | 1448.7804 | 1450.1049 | 1046.5909 | 1178.6275 | Not used |
| SEU-BEV 01/04/05 | 1024 | 1280 | 880.5116 | 879.5616 | 521.0695 | 625.8222 | Undistortion enabled |

For SEU-BEV, the distortion coefficients are
`[-0.095725, 0.089338, -0.002196, -0.000740, 0.000000]`. The camera-LiDAR
extrinsic matrix from the SEU-BEV calibration file is kept in the config for
completeness, but the reported monocular SLAM experiments do not use LiDAR at
test time.

## Common Conservative Configuration

| Component | Setting |
|---|---|
| Frontend checkpoint | `/home/server/VINGS_work/DPT-LSG/ckpts/droid.pth` |
| Frontend input size | `[344, 616]` |
| Frontend buffer | `80` |
| Warm-up frames | `8` |
| Keyframe threshold | `3.0` |
| Frontend window | `5` |
| Active window | `12` |
| Maximum frontend factors | `48` |
| Translation threshold | `0.5` |
| Middleware max depth | `80 m` |
| DA3 model | `depth-anything/DA3METRIC-LARGE` |
| DA3 checkpoint | `/home/server/VINGS_work/DPT-LSG/ckpts/da3_mars_lvig_hkairport_amtown01_head.pth` |
| DA3 precision | `float16` |
| DA3 process resolution | `560`, upper-bound resize |
| DA3 execution | Asynchronous, `max_pending: 2` |
| DA3 valid depth range | `[0.5 m, 80 m]` |
| DA3 alignment | Scale-only anchoring, trimmed fit ratio `0.1` |
| DA3 scale clamp | `[0.75, 1.33]` |
| DA3 blend weight | `0.15` |
| DA3 replacement rule | Replace existing depth only when the fitted scale is sane |
| Submap state | Enabled |
| Submap max keyframes | `360` |
| Submap max translation | `400 m` |
| Submap overlap | `12` keyframes |
| Submap loop interval | Every `6` keyframes |
| Loop min separation | `60` keyframes |
| Retrieval candidates | Top `2` submaps and top `2` keyframes |
| Loop validation | At least `40` matches, error threshold `0.16` |
| Seam bridge | Enabled with strict scale and translation gates |
| Seam bridge gates | Scale `[0.9, 1.12]`, translation `< 35 m` |
| Loop depth gates | Candidate max depth `8 m`, final max depth `15 m` |
| LightGlue input | `344 x 616`, ONNX width `512` |
| Sim(3) RANSAC | `256` iterations, inlier threshold `0.75`, minimum inliers `48` |
| Gaussian optimization | `50` iterations, `8` keyframes |
| Gaussian ADC threshold | `0.98` |
| Mapping losses | RGB `1.0`, depth `1.0`, alpha `1.0`, normal `0.1`, distortion `0.0` |

## Rationale for the Conservative Settings

The final configuration is intentionally conservative because UAV monocular
SLAM is sensitive to fast viewpoint change, motion blur, water/sky regions,
large depth range, and repeated or weakly textured structures. These factors
can make aggressive depth replacement or loose loop constraints harmful even
when they improve some individual frames.

First, DA3 is used as a metric depth prior, but its influence is bounded. The
depth range is capped at 80 m and the fitted scale is restricted to `[0.75,
1.33]`. DA3 is allowed to replace the current depth only when the scale fitting
is valid, and the final depth is blended with a small weight of `0.15`. This
keeps useful learned depth priors for mapping while avoiding large pose or map
distortions caused by unreliable depth estimates in water, sky, or highly
out-of-domain regions.

Second, the submap mechanism is configured to reduce memory pressure while
preserving enough overlap for cross-submap consistency. Each submap keeps up to
360 keyframes and shares 12 keyframes with neighboring submaps. Loop checking is
performed every 6 keyframes, but loop acceptance is gated by a minimum number
of matches and a geometric error threshold. The seam bridge is also restricted
by tight scale and translation limits. This design favors fewer false
connections over more aggressive loop closures, which is important for UAV
sequences where false Sim(3) links can damage the global trajectory and the
Gaussian map.

Third, the mapper uses stable Gaussian settings across all datasets. We keep
the alpha loss at `1.0` and the adaptive density-control accumulation threshold
at `0.98` to maintain surface completeness while preventing excessive Gaussian
growth. The same frontend resolution `[344, 616]` is used for all experiments
to balance tracking robustness, DA3 runtime, loop matching cost, and GPU memory
usage.

Overall, the reported configuration is selected to make DPT-LSG stable across
different UAV scenes rather than to maximize performance on a single sequence.
It improves map quality and controls memory usage while keeping the tracking
frontend close to the original DROID-SLAM behavior.

## Paper-Ready Paragraph

In the main experiments, all methods are evaluated on monocular RGB sequences
without using IMU, GPS, RTK, or LiDAR measurements at test time. We evaluate
DPT-LSG on six MARS-LIVG sequences, including AMvalley01-03 and HKisland01-03,
and additionally test the same configuration on SEU-BEV sequences 01, 04, and
05. For all sequences, the system runs in visual odometry mode with DROID-SLAM
as the tracking frontend, DA3Metric-Large as the monocular metric depth prior,
a submap-based Gaussian mapper, and LightGlue-based loop validation. The same
conservative hyperparameters are used across all datasets; only the dataset
path and camera calibration are changed.

The frontend image size is fixed to 344 x 616, with an active local window of
12 frames and at most 48 frontend factors. The DA3 depth prior is evaluated in
half precision with an input processing resolution of 560. To avoid unstable
depth injection, DA3 depth is capped to 80 m, aligned by scale-only fitting,
clamped to a scale range of 0.75 to 1.33, and blended with the existing depth
using a weight of 0.15. DA3 replaces existing depth only when the fitted scale
is considered reliable. The submap mapper is enabled with a maximum of 360
keyframes per submap and 12 overlapping keyframes between adjacent submaps.
Loop and seam constraints are accepted only under conservative geometric
validation, requiring at least 40 loop matches, an error threshold of 0.16, and
strict seam scale and translation limits. For Gaussian mapping, we use 50
optimization iterations with RGB, depth, and alpha loss weights all set to 1.0,
normal loss weight set to 0.1, and adaptive density-control accumulation
threshold set to 0.98.

These settings are chosen to improve robustness on UAV videos. UAV sequences
contain fast camera motion, large viewpoint changes, wide depth ranges, and
weakly textured regions such as water or sky. Therefore, aggressive DA3 depth
replacement or loose loop closure can introduce incorrect scale or pose
constraints. The conservative configuration keeps the learned depth prior useful
for mapping while limiting its ability to corrupt tracking, and uses submaps to
control memory usage while preserving enough overlap for global consistency.
