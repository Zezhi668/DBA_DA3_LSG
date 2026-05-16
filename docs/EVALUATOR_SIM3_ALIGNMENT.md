# Evaluator Sim(3) Alignment

This note explains the trajectory alignment used by:

```bash
python -m scripts.evaluate configs/MARS-LIVG/AMvalley_tum.yaml
```

The short version: tracking ATE is computed after a global Umeyama Sim(3) alignment from the predicted camera centers to the reference trajectory positions. This is the default because `scripts.evaluate` sets `--alignment sim3`.

## What It Aligns

The evaluator reads the predicted camera-to-world poses from the selected run output:

```text
<run>/droid_c2w_new/*.txt
```

If `droid_c2w_new` is absent, it falls back to:

```text
<run>/droid_c2w/*.txt
```

Each pose file is loaded as a `4 x 4` matrix. The numeric filename stem is treated as the predicted frame id.

The ground truth is loaded through the configured dataset module. For MARS-LIVG TUM configs, this is:

```yaml
dataset:
  module: datasets.tumrgbd
```

`datasets.tumrgbd` reads `groundtruth.txt` or `pose.txt`, converts the TUM translation/quaternion records to `c2w` matrices, and associates each ground-truth timestamp to the nearest RGB frame id. The evaluator then matches predicted frame ids to ground-truth frame ids and extracts only the translation column:

```text
predicted point: p_i = predicted_c2w_i[:3, 3]
reference point: g_i = groundtruth_c2w_i[:3, 3]
```

Rotations are loaded but are not part of the current tracking ATE metric.

## The Sim(3) Problem

For matched positions, the evaluator solves one global least-squares similarity transform:

```text
g_i ~= s R p_i + t
```

where:

```text
s: positive global scale
R: 3 x 3 rotation matrix
t: 3D translation
```

The optimized objective is:

```text
minimize sum_i ||s R p_i + t - g_i||^2
```

This is the classic Umeyama alignment. In code, `scripts/evaluate.py`:

1. Centers both point clouds by subtracting their means.
2. Forms the covariance between reference and predicted centered points.
3. Runs SVD on the covariance.
4. Builds a proper rotation and rejects reflection by flipping the last axis when needed.
5. Estimates scale from the singular values and predicted point variance.
6. Computes translation from the two centroids.
7. Applies `p_i_aligned = s R p_i + t`.

The per-frame ATE is then:

```text
ATE_i = ||p_i_aligned - g_i||
```

The summary reports mean, RMSE, median, std, min, max, and frame count.

## Why This Helps With GPS/IMU Versus Camera Frames

The SLAM output is a monocular camera trajectory. The MARS-LIVG reference trajectory may come from a drone GPS/IMU frame, whose origin, axes, and metric convention do not necessarily match the camera trajectory. A global Sim(3) alignment absorbs:

- arbitrary world-origin differences,
- global axis-frame rotation differences,
- global translation offsets,
- monocular scale ambiguity,
- constant trajectory-frame convention differences.

That is why the default evaluator is useful when the goal is to judge trajectory shape and drift rather than to verify the exact camera-to-IMU/GPS extrinsic calibration.

There is an important limitation: a real camera-to-IMU lever arm is attached to the drone body. If the GPS/IMU position is the body origin and the SLAM trajectory is the camera center, then the offset in world coordinates changes with drone attitude:

```text
camera_world = body_world + R_world_body * t_body_camera
```

A single global Sim(3) cannot exactly remove that time-varying lever-arm term. In large UAV trajectories the lever arm is usually small compared with the path length, so the residual is often acceptable for tracking ATE. If the goal is calibration-quality evaluation, generate a camera-center ground truth trajectory with the real body-camera extrinsic and evaluate that instead.

## Alignment Modes

The evaluator supports three tracking modes:

```bash
--alignment sim3
```

Default. Solves global scale, rotation, and translation. This is the normal monocular SLAM reporting mode.

```bash
--alignment se3
```

Solves only global rotation and translation, with scale fixed to `1.0`. Use this when the predicted trajectory is already metric and you want to penalize scale error.

```bash
--alignment none
```

Applies no trajectory alignment. Use this only when predicted and reference poses are already in the same world frame, origin, orientation, and scale.

## What It Does Not Do

The Sim(3) alignment is post-processing for tracking metrics only.

It does not:

- change the saved SLAM trajectory,
- feed back into tracking or mapping,
- align individual frames independently,
- evaluate orientation error,
- fix local drift or trajectory deformation,
- affect mapping PSNR, SSIM, or LPIPS,
- replace LiDAR-camera extrinsics used to export DA3 sparse depth labels.

It is also separate from the runtime/submap Sim(3) machinery used for loop correction. The evaluator alignment is a one-shot global fit after the run has finished.

## Outputs

For a normal run, the evaluator writes:

```text
<run>/evaluation/summary.json
<run>/evaluation/tracking_metrics.csv
<run>/evaluation/tracking_plots.png
<run>/evaluation/mapping_metrics.csv
<run>/evaluation/mapping_plots.png
```

`summary.json` records the selected alignment mode under:

```json
"tracking": {
  "alignment": "sim3"
}
```

`tracking_metrics.csv` contains the per-frame aligned translation error.

## Recommended Commands

Default AMvalley evaluation with Sim(3) tracking ATE:

```bash
cd /home/server/VINGS_work/DPT-LSG

python -m scripts.evaluate configs/MARS-LIVG/AMvalley_tum.yaml
```

Evaluate a specific run directory:

```bash
python -m scripts.evaluate configs/MARS-LIVG/AMvalley_tum.yaml \
  --output-dir /home/server/output/DPT-LSG_output/<run_name>
```

Metric-scale check without scale alignment:

```bash
python -m scripts.evaluate configs/MARS-LIVG/AMvalley_tum.yaml \
  --alignment se3
```

Strict same-frame check:

```bash
python -m scripts.evaluate configs/MARS-LIVG/AMvalley_tum.yaml \
  --alignment none
```

## Suggested Reporting Wording

Use wording like:

> Tracking ATE is reported after global Sim(3) Umeyama alignment between the predicted monocular camera centers and the reference trajectory positions. This removes global origin, orientation, and scale differences between the SLAM camera trajectory and the GPS/IMU-derived reference frame, while still penalizing residual local drift and trajectory-shape error.

If the reference trajectory is GPS/IMU-center rather than camera-center ground truth, also state:

> The alignment does not explicitly model a time-varying camera-to-IMU lever arm; this residual is assumed small relative to the UAV trajectory scale.
