# A DA3-Anchored DROID Frontend Tracker for DPT-LSG

## Abstract

To solve the unstable scale and incomplete dense geometry problems that appear when a DROID-SLAM-style monocular frontend is used as the source of a Gaussian mapping backend, the author refers to the frontend design of Splat-SLAM and inspects its DSPO mechanism. Splat-SLAM shows that DROID-style dense bundle adjustment can be strengthened by coupling the optimized multi-view disparity with a monocular depth prior, then passing a proxy depth map to the Gaussian mapper. Based on this observation, the author designs a DA3-anchored DROID frontend mechanism for DPT-LSG. Instead of using the relative DPT depth adopted by Splat-SLAM, DPT-LSG uses Depth Anything 3 Metric Large as the monocular depth prior, aligns it to the current DROID depth state when necessary, injects it into the existing DROID disparity-prior path, and lets the original dense bundle adjustment kernel refine poses and disparities. The resulting frontend produces keyframe RGB, dense depth, depth uncertainty, camera pose, intrinsics, and global keyframe identity for the Gaussian backend.

## Keywords

DPT-LSG, DROID-SLAM, Splat-SLAM, DSPO, Depth Anything 3, monocular depth prior, dense bundle adjustment, Gaussian SLAM frontend

## 1. Introduction

To solve the problem that a pure recurrent optical-flow frontend can produce locally smooth but scale-weak monocular geometry, the author starts from the DROID-SLAM tracking pipeline already used by DPT-LSG and inspects how Splat-SLAM reinforces a similar tracker. The key reference is not Splat-SLAM's Gaussian renderer itself, but its frontend idea: tracking should not only optimize camera poses from dense optical-flow correspondences, but should also use a monocular depth prior to repair weak or inaccurate disparity regions.

Focusing on large-scale monocular Gaussian SLAM scenarios, the author treats the frontend as the geometry source for downstream mapping. In this setting, a frontend error is not limited to trajectory drift. It also becomes a mapping error, because the backend initializes and optimizes Gaussians from the frontend depth and pose packets. Therefore, the frontend is designed to answer a practical question: how can a DROID-SLAM pipeline retain its dense optical-flow BA advantages while receiving stronger metric depth guidance before the backend starts Gaussian optimization?

Based on the repository implementation, the author designs the frontend around five operations:

1. Preserve the DROID-SLAM-style recurrent optical-flow tracker and covisible factor graph.
2. Detect keyframes using motion and graph distance criteria rather than sending every frame to mapping.
3. Predict a DA3 metric depth prior only for useful keyframes and for the initialized warmup window.
4. Convert the DA3 depth into DROID-compatible inverse-disparity anchors through a DSPO-inspired alignment step.
5. Package the optimized keyframe state into the backend format: RGB, depth, depth covariance, pose, intrinsics, timestamp, and global keyframe id.

## 2. Splat-SLAM Frontend Design

To solve RGB-only Gaussian SLAM's weak reconstruction problem, Splat-SLAM combines frame-to-frame tracking, global trajectory optimization, and a deformable 3D Gaussian map. Its frontend begins from a recurrent dense optical-flow tracker and performs local bundle adjustment over a factor graph. The graph nodes store keyframe pose and dense disparity, while the edges store optical-flow correspondences between keyframes. When the mean flow between the current frame and the last keyframe exceeds a threshold, the frame is promoted to a new keyframe.

Splat-SLAM's frontend is important because it does not treat monocular depth as a standalone map. It uses monocular depth as a constraint inside tracking. The tracking section of the paper defines a DSPO layer, namely Disparity, Scale and Pose Optimization. The DSPO layer alternates two objectives:

1. Dense Bundle Adjustment, which optimizes keyframe poses and dense disparities by minimizing the reprojection error induced by predicted optical flow.
2. Monocular-depth prior optimization, which uses a pretrained relative depth model and fits per-frame scale and shift parameters so that monocular depth can constrain inaccurate disparity regions.

In simplified notation, Splat-SLAM's first objective is the DROID-style DBA objective:

```text
min_{pose, disparity} sum_{(i,j) in E} || flow_ij - project(pose_i, pose_j, disparity_i, K) ||^2_weight
```

The second objective introduces the relative monocular depth prior:

```text
min_{d_high, theta, gamma}
    flow reprojection error using d_high
  + alpha_1 * || d_high - (theta * inv(D_mono) + gamma) ||^2
  + alpha_2 * || d_low  - (theta * inv(D_mono) + gamma) ||^2
```

Here, `theta` and `gamma` absorb the unknown scale and shift of the relative monocular depth. Low-error disparities are used to stabilize the scale-shift fit, while high-error disparities are allowed to move toward the monocular-depth prior. Splat-SLAM alternates the DBA objective and this depth-prior objective to avoid optimizing pose, disparity, scale, and shift in one ambiguous joint system.

To decide where monocular depth should help, Splat-SLAM inspects multi-view consistency. A depth value is considered more reliable when the 3D point reconstructed from one keyframe remains consistent after being checked against other keyframes. The final proxy depth sent to mapping is therefore not simply the monocular prediction. It is a fused depth map:

```text
D_proxy(u, v) = D_multi_view(u, v),                 if multi-view depth is valid
D_proxy(u, v) = theta * D_mono(u, v) + gamma,       otherwise
```

This frontend design gives Splat-SLAM two practical advantages. First, tracking remains grounded in dense multi-view geometry from DROID-style BA. Second, the mapper receives dense proxy depth even when multi-view geometry is missing or unstable. The paper then uses this proxy depth to initialize and optimize a deformable Gaussian map.

## 3. Design Motivation for DPT-LSG

To solve the same frontend-to-backend geometry handoff problem in DPT-LSG, the author refers to Splat-SLAM's DSPO idea but does not copy it directly. The reason is that the depth prior is different. Splat-SLAM uses a relative monocular depth estimator and must optimize per-frame scale and shift inside the DSPO layer. DPT-LSG uses DA3 Metric Large, which already predicts a canonical metric depth proxy when intrinsics are available. Therefore, the main problem changes from "how to recover relative depth scale" to "how to inject a metric depth prior into DROID without breaking the existing CUDA BA pipeline."

After inspecting the repository, the author locates the DROID-style tracking core in `scripts/frontend/dbaf.py`, `scripts/frontend/dbaf_frontend.py`, `scripts/frontend/covisible_graph.py`, and `scripts/frontend/depth_video.py`. The tracker already stores frame images, poses, disparities, sensor disparities, intrinsics, learned feature maps, recurrent context features, and uncertainty buffers. It also already calls the DROID backend BA interface with `disps_sens`, a field that the BA kernel can use as an inverse-depth measurement prior. This existing interface becomes the most economical insertion point for DA3.

Focusing on compatibility and runtime stability, the author designs a DA3 depth-prior module in `scripts/frontend/da3_depth_prior.py` instead of rewriting the DROID optimizer. The module borrows the DSPO alignment logic from Splat-SLAM, but applies it outside the BA kernel. The result is a practical DSPO-inspired strategy:

```text
RGB keyframe
  -> DA3 metric depth prediction
  -> optional focal and affine alignment
  -> inverse-depth anchor in video.disps_sens
  -> DROID dense BA refines pose and disparity
  -> middleware packages optimized geometry for backend
```

## 4. DPT-LSG Frontend Architecture

### 4.1 Frame Filtering and Feature Encoding

To solve the cost problem of applying expensive tracking and DA3 inference to every RGB frame, the author keeps the DROID-SLAM motion filtering design. `MotionFilter.track` normalizes the input image, extracts learned feature maps through the DROID network, estimates a one-step optical-flow update against the last accepted frame, and appends the frame to `DepthVideo` only if the mean motion magnitude exceeds `frontend.filter_thresh`.

This design means the frontend does not use DA3 as a frame-by-frame dense depth sensor. It first asks whether the frame contributes enough motion and covisibility. Only accepted frames are placed into the DROID buffer, and only initialized keyframes are later considered for DA3 anchoring.

### 4.2 DROID Covisible Graph and Local Dense BA

To solve the local pose and disparity estimation problem, the author keeps the DROID covisible graph. `DBAFusionFrontend` creates a `CovisibleGraph`, adds neighborhood factors during initialization, and later adds proximity factors inside a sliding frontend window. Each graph update performs three steps:

1. Reproject current disparities and poses into connected frames.
2. Use the DROID recurrent update operator to refine target flow, confidence weight, damping, and upsampling mask.
3. Call `DepthVideo.ba`, which passes targets, weights, damping, graph edges, poses, disparities, intrinsics, and `disps_sens` to the DROID backend.

The important design point is that DPT-LSG keeps DROID as the optimizer of record. DA3 does not replace the optical-flow graph and does not directly output poses. It only modifies the depth anchor that the DROID optimizer already understands.

### 4.3 DA3 Depth Prior Module

To solve the weak or drifting disparity prior problem, the author adds `DroidDA3DepthPrior`. The module is keyframe-only and lazy-loaded. When the frontend finishes warmup, it can seed the initialized window; when a new keyframe is accepted, it can submit that keyframe for DA3 prediction. The default configuration enables asynchronous execution with a bounded pending queue, so the frontend does not wait indefinitely for depth inference.

DA3 receives the keyframe RGB image and resized camera intrinsics. The predictor calls the Depth Anything 3 API, resizes the prediction back to the frontend image size if needed, applies the DA3 canonical focal conversion when configured, and returns a dense metric depth proxy. The code expresses the intended metric form as:

```text
r_da3(u) = d_i(u) - 1 / (s_f * D_DA3_i(u) + eps)
```

where `d_i` is DROID inverse disparity, `D_DA3_i` is the DA3 depth prediction, and `s_f` is the optional focal conversion. For DA3 Metric Large, the repository uses the focal factor `focal / 300`.

### 4.4 DSPO-Inspired Anchor Alignment

To solve the mismatch between a fresh DA3 prediction and the current DROID state, the author borrows the scale-shift idea from Splat-SLAM's DSPO. The implementation fits DA3 depth to the current DROID depth before converting it into inverse disparity. This is controlled by `da3.anchor_mode`, whose default configuration uses `affine`.

The alignment is practical rather than symbolic DSPO. Splat-SLAM optimizes `theta` and `gamma` as part of an alternating objective because its depth is relative. DPT-LSG estimates a robust affine anchor outside BA:

```text
D_anchor = scale * D_DA3 + shift
```

The fitting process uses only pixels where both DA3 depth and DROID depth are finite and within the configured depth range. It rejects invalid ratios, trims outliers, clamps scale and shift, optionally blends the fit, and falls back to the raw DA3 metric depth when too few pixels are valid. This design keeps the DROID BA kernel unchanged while still preserving the central DSPO idea: monocular depth should be aligned to the current multi-view tracking state before it constrains disparity.

### 4.5 Injection Into DROID BA

To realize the depth-prior constraint without rewriting CUDA kernels, the author writes the aligned DA3 depth into `DepthVideo.disps_sens`. The aligned dense depth is converted to inverse disparity, downsampled at the same `3::8, 3::8` grid used by the DROID buffers, and stored as:

```text
video.disps_sens[index] = 1 / D_anchor_downsampled
video.disps[index] = video.disps_sens[index] where the prior is valid
```

During graph update, `DepthVideo.ba_raw` calls:

```text
droid_backends.ba(poses, disps, intrinsics, disps_sens,
                  target, weight, eta, ii, jj, t0, t1, ...)
```

This is the main implementation bridge. The DA3 module provides a sensor-like inverse-depth prior, and the existing DROID backend uses it during dense bundle adjustment. The author therefore realizes a DSPO-inspired frontend by combining DA3's metric depth with DROID's native disparity-prior channel.

## 5. Differences From Splat-SLAM

To solve DPT-LSG's specific large-scale Gaussian mapping need, the author intentionally changes several parts of the Splat-SLAM frontend idea.

| Design Aspect | Splat-SLAM | DPT-LSG Frontend |
| --- | --- | --- |
| Monocular prior | Relative DPT or Omnidata-style depth | DA3 Metric Large depth |
| Scale handling | Per-frame `theta` and `gamma` optimized in DSPO | Optional focal conversion plus robust external scale or affine alignment |
| Optimization form | Alternates DBA and monocular-depth prior objective | Injects DA3 as `disps_sens` and reuses DROID BA |
| Depth use | Builds proxy depth from valid multi-view depth plus monocular completion | Anchors DROID disparity with DA3, then exports optimized dense depth |
| Runtime strategy | Tracking and mapping are interleaved in the Splat-SLAM pipeline | DA3 prediction is lazy, keyframe-only, and optionally asynchronous |
| Backend handoff | Proxy depth guides Gaussian initialization and deformation | Middleware exports RGB, optimized depth, covariance, pose, intrinsics, and global keyframe id |

The main improvement is that DA3 reduces the burden of relative-depth scale recovery. Since DA3 Metric Large can consume intrinsics and produce metric depth, DPT-LSG does not need to treat the monocular prior as purely scale-free. However, the author still keeps DSPO-style affine alignment because the frontend's optimized DROID depth is the state that the rest of the tracker trusts. This hybrid choice avoids two bad extremes: blindly trusting DA3 over multi-view tracking, or ignoring monocular depth when DROID disparity becomes weak.

The second improvement is implementation compatibility. Splat-SLAM describes DSPO as an optimization layer with alternating objectives. DPT-LSG realizes the same design intent through the existing `disps_sens` interface. This makes the depth prior easy to enable or disable from configuration and avoids destabilizing the DROID backend.

The third improvement is backend awareness. The frontend is designed with the Gaussian mapper's input contract in mind. It does not only output poses. It outputs dense depth, uncertainty, and keyframe identity, which are needed for Gaussian initialization, pruning, optimization, and later loop or submap correction.

## 6. Backend Interface

To link the frontend to the backend without over-coupling the two modules, the author uses the middleware packet generated in `scripts/vings_utils/middleware_utils.py`. The default DROID/DA3 path calls `judge_and_package_v3` when a new keyframe is added.

The frontend gives the backend the following fields:

| Field | Meaning |
| --- | --- |
| `images` | Keyframe RGB images in mapping resolution |
| `depths` | Dense depth from optimized frontend disparity, with invalid pixels zeroed |
| `depths_cov` | Depth uncertainty propagated from the frontend BA covariance buffer |
| `poses` | Camera-to-world keyframe poses |
| `intrinsic` | Resized camera intrinsics and image size |
| `viz_out_idx_to_f_idx` | Original frame timestamps or frame indices |
| `global_kf_id` | Stable global keyframe identity used by mapping, loop, and submap modules |
| `pixel_mask` | Valid pixel mask used by downstream mapping logic |
| `valid_localkf_id` | Local keyframe ids that allow backend corrections to be written back to the tracker |

Before packaging, the middleware filters invalid depth using `middleware.max_depth` and an uncertainty gate based on per-frame median covariance. This design gives the backend a conservative geometry packet. The backend does not need to know whether a depth value was originally produced by pure DROID, DA3-anchored DROID, or later loop correction. It only consumes the standardized keyframe batch.

`GaussianModel` then uses this packet to initialize and update Gaussian primitives. The author keeps this backend description brief because the frontend contribution is the production of a stronger packet, not a new Gaussian optimizer. The important boundary is:

```text
Frontend responsibility:
    robust keyframe selection, pose, dense depth, uncertainty, keyframe id

Backend responsibility:
    Gaussian creation, rendering loss, pruning, densification, storage, loop/submap correction
```

## 7. Method Summary

To solve monocular scale and dense geometry weakness in DROID-SLAM-based tracking, the author refers to Splat-SLAM's DSPO layer and designs a DA3-anchored depth-prior mechanism. The author first keeps the DROID recurrent optical-flow frontend because it provides dense correspondences and a mature local BA path. Then, after inspecting the DROID BA interface, the author identifies `disps_sens` as the correct insertion point for depth priors. Finally, the author adapts Splat-SLAM's scale-shift reasoning to DA3 by fitting an optional robust affine depth anchor before writing the prior into the DROID buffer.

The final frontend works as follows:

1. RGB frames enter the DROID motion filter.
2. Frames with enough motion are appended to `DepthVideo`.
3. The frontend initializes a covisible graph and performs DROID dense BA.
4. When the initialized window or a new keyframe is available, DA3 predicts metric depth.
5. The DA3 depth is optionally focal-scaled and affine-aligned to current DROID depth.
6. The aligned depth is converted to inverse disparity and written into `disps_sens`.
7. Subsequent DROID BA updates refine poses and disparities under the DA3 anchor.
8. Middleware exports optimized RGB-depth-pose-uncertainty packets to the Gaussian backend.

## 8. Discussion

The DPT-LSG frontend should be understood as a design adaptation rather than a direct reimplementation of Splat-SLAM. Splat-SLAM's contribution is the insight that a DROID-style tracker benefits from a monocular depth prior when the depth is aligned through DSPO. DPT-LSG changes the prior model to DA3 and changes the implementation strategy to fit the existing DROID backend. This is why the frontend is not introduced as a new tracker from scratch. It is a carefully inserted depth-prior mechanism that lets the existing tracker produce stronger geometry for the backend.

The design also leaves several limitations. The DA3 prediction can still be wrong in reflective, dynamic, sky, or very long-range regions, so DPT-LSG keeps depth range checks, robust alignment, covariance filtering, and backend invalid-depth masking. The asynchronous DA3 path improves runtime behavior, but a delayed prior may only affect later BA updates. The current design also relies on the DROID backend's existing `disps_sens` treatment, so any deeper DSPO variant that optimizes DA3 scale, shift, pose, and disparity in one custom solver would require changing the optimizer itself.

## 9. Conclusion

To solve the frontend geometry weakness that affects monocular Gaussian SLAM, the author borrows the DSPO principle from Splat-SLAM and redesigns it around DA3 and the DPT-LSG codebase. Splat-SLAM demonstrates that monocular depth should be aligned to multi-view tracking before being trusted by mapping. DPT-LSG realizes this principle by predicting DA3 metric depth for keyframes, aligning it to DROID depth when needed, injecting it through `disps_sens`, and exporting the refined depth, pose, uncertainty, and keyframe identity to the Gaussian backend. The resulting frontend remains DROID-compatible while providing a stronger geometry source for large-scale Gaussian SLAM.

## References

[1] Erik Sandstrom et al. "Splat-SLAM: Globally Optimized RGB-only SLAM with 3D Gaussians." arXiv:2405.16544, 2024. Source inspected from `/home/server/Documents/Papers/Priors/Splat-SLAM.pdf`.

[2] Zachary Teed and Jia Deng. "DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras." NeurIPS, 2021.

[3] DPT-LSG repository implementation, especially `scripts/frontend/dbaf.py`, `scripts/frontend/dbaf_frontend.py`, `scripts/frontend/covisible_graph.py`, `scripts/frontend/depth_video.py`, `scripts/frontend/da3_depth_prior.py`, and `scripts/vings_utils/middleware_utils.py`.

[4] Depth Anything 3 Metric Large integration as configured in `configs/MARS-LIVG/HKairport03_tum.yaml` and implemented in `scripts/frontend/da3_depth_prior.py`.
