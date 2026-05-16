# 1. Title of the Invention

METHOD AND SYSTEM FOR LARGE-SCALE MONOCULAR SLAM WITH ADAPTIVE GAUSSIAN MAP BASED ON SUBMAP MECHANISM

Applicant: `[To be completed]`

Inventors: `[To be completed]`

Application Type: Invention

Note: This is a technical patent draft prepared from the present `DPT-LSG` project context. Applicant information, jurisdiction-specific formatting, claim-scope adjustment, and legal language normalization should be finalized with patent counsel.

## 2. Abstract

The present invention relates to monocular visual simultaneous localization and mapping, dense three-dimensional reconstruction, robotics, and augmented reality, and discloses a method and system for large-scale monocular SLAM with an adaptive Gaussian map based on a submap mechanism. Existing monocular dense SLAM systems suffer from poor tracking robustness in aggressive camera motion, low-texture scenes, and motion blur, tight coupling between front-end and back-end modules, severe long-term scale drift, insufficient large-scale scalability, and weak global consistency after loop closure. To solve these problems, the invention provides a decoupled end-to-end pipeline comprising a hybrid monocular dense geometric prior front-end, a standardized uncertainty-aware keyframe middleware, an adaptive Gaussian mapping back-end, and a submap-based global Sim(3) correction module. The hybrid front-end integrates a pre-trained MASt3R model into a DROID-SLAM tracking framework, replaces the original SfM-dependent geometric prior path with a pre-trained dense geometric prior, and retains the native output format of the tracking framework so that dense point maps, confidence maps, and flow-compatible motion states can be seamlessly consumed by downstream modules. The middleware converts front-end confidence into depth uncertainty covariance and packages standardized keyframe data packets. The mapping back-end performs online Gaussian initialization, RGB-depth-normal joint optimization, stability-aware pruning and densification, and GPU/CPU hierarchical storage. The submap module divides a global map into active and frozen submaps, performs submap-level coarse retrieval and keyframe-level fine retrieval, validates cross-submap loop candidates by depth-assisted Sim(3) estimation, and optimizes a global Sim(3) pose graph to synchronously correct front-end pose caches and Gaussian map assets. The invention improves tracking robustness, scale consistency, mapping accuracy, and real-time scalability in large-scale monocular scenes.

## 3. Claims

1. A method for large-scale monocular simultaneous localization and mapping with an adaptive Gaussian map based on a submap mechanism, characterized by comprising:

   S1: acquiring a monocular RGB image sequence and a corresponding camera intrinsic matrix, preprocessing the monocular image, and obtaining a dense point map, a confidence map, and an optical flow field by means of a hybrid dense geometric prior front-end that integrates a pre-trained MASt3R model into a DROID-SLAM tracking framework while retaining a native output format of the DROID-SLAM tracking framework;

   S2: converting the dense point map and the confidence map into a depth map and an uncertainty covariance according to a Sim(3) pose state of the image sequence, performing local pose estimation by factor-graph optimization, and packaging a standardized keyframe data packet;

   S3: initializing and online optimizing an adaptive Gaussian scene map according to the standardized keyframe data packet, performing stability control, pruning, and densification on Gaussian primitives, and implementing GPU/CPU hierarchical storage according to camera-pose distance;

   S4: dividing a global Gaussian map into active submaps and frozen submaps, generating submap descriptors according to keyframe features, and retrieving cross-submap loop-closure candidates by descriptor similarity; and

   S5: performing depth-assisted submap-level Sim(3) similarity-transformation estimation and multi-stage validation on the loop-closure candidates, constructing and optimizing a global Sim(3) pose graph after validation, and feeding an obtained correction transformation back to a front-end pose cache and full Gaussian map assets to obtain a globally consistent camera trajectory and dense scene map.

2. The method according to claim 1, characterized in that the hybrid dense geometric prior front-end in step S1 satisfies:

   $$
   K_{t}^{\prime }=
   \left[
   \begin{array}{ccc}
   s_{x}f_{x} & 0 & s_{x}c_{x}+\Delta _{x}\\
   0 & s_{y}f_{y} & s_{y}c_{y}+\Delta _{y}\\
   0 & 0 & 1
   \end{array}
   \right],
   $$

   and

   $$
   \left(P_{t}, C_{t}, F_{t}\right)=\mathcal{F}_{\theta}\left(I_{t}', I_{t-1}', K_{t}'\right),
   $$

   wherein $P_t$ is the dense point map, $C_t$ is the confidence map, $F_t$ is the optical flow field, and $\mathcal{F}_{\theta}$ is the hybrid front-end network with pre-trained geometric prior weights, and wherein a compatibility mapping further satisfies

   $$
   \mathcal{O}_{t}^{hyb}=\mathcal{T}_{comp}\left(P_{t},C_{t},F_{t}\right),
   \qquad
   \operatorname{layout}\!\left(\mathcal{O}_{t}^{hyb}\right)=
   \operatorname{layout}\!\left(\mathcal{O}_{t}^{native}\right),
   $$

   so that an output tensor layout of the hybrid front-end is consistent with the native output tensor layout of the tracking framework.

3. The method according to claim 1, characterized in that step S2 satisfies:

   $$
   T_{t}^{sim3}=
   \left[
   \begin{array}{cc}
   s_{t} R_{t} & t_{t} \\
   0 & 1
   \end{array}
   \right],
   \qquad
   s_{t}=
   \left|
   \det\left(T_{t,1:3,1:3}^{sim3}\right)
   \right|^{1/3},
   $$

   $$
   D_{t}(u)=s_{t} Z_{t}(u),
   \qquad
   \Sigma_{t}(u)=
   \operatorname{clip}
   \left(
   \frac{\alpha}{\max \left(C_{t}(u), \varepsilon\right)} s_{t}^{2},
   \Sigma_{min},
   \Sigma_{max}
   \right),
   $$

   $$
   M_{t}(u)=
   \mathbf{1}
   \left[
   D_{min}<D_{t}(u)<D_{max}
   \wedge
   \Sigma_{t}(u)\leq \tau_{\Sigma}
   \wedge
   D_{t}(u), \Sigma_{t}(u)\in \mathbb{R}
   \right],
   $$

   and

   $$
   \mathcal{X}_{t}^{*}=
   \arg \min _{\mathcal{X}_{t}}
   \left(
   \sum_{(i, j) \in \mathcal{E}_{t}^{temp }}
   \rho\left(\left\| r_{i j}^{ray }\left(\mathcal{X}_{t}\right)\right\| _{W_{i j}}^{2}\right)
   +
   \sum_{(i, j) \in \mathcal{E}_{t}^{flow }}
   \rho\left(\left\| r_{i j}^{flow }\left(\mathcal{X}_{t}\right)\right\| _{W_{i j}}^{2}\right)
   +
   \sum_{(i, k) \in \mathcal{E}_{t}^{relocal }}
   \rho\left(\left\| r_{i k}^{relocal }\left(\mathcal{X}_{t}\right)\right\| _{W_{i k}}^{2}\right)
   \right),
   $$

   wherein the standardized keyframe data packet comprises image, depth, covariance, pose, mask, timestamp, global keyframe index, and intrinsic parameters.

4. The method according to claim 1, characterized in that the adaptive Gaussian scene map in step S3 is represented by

   $$
   \mathcal{G}_{t}=
   \left\{
   \gamma_{n}
   \right\}_{n=1}^{N_{t}},
   \qquad
   \gamma_{n}=
   \left(
   \mu_{n}, q_{n}, a_{n}, o_{n}, c_{n}, \kappa_{n}, b_{n}
   \right),
   $$

   and is optimized by a joint loss

   $$
   \mathcal {L}_{t}=
   \lambda _{rgb}\mathcal {L}_{rgb}+
   \lambda _{d}\mathcal {L}_{d}+
   \lambda _{a}\mathcal {L}_{a}+
   \lambda _{n}\mathcal {L}_{n},
   $$

   wherein

   $$
   \mathcal {L}_{rgb}=
   \sum_{u}M_{t}(u)\left\| \hat{I}_{t}(u)-I_{t}'(u)\right\| _{1},
   $$

   $$
   \mathcal {L}_{d}=
   \sum_{u}M_{t}(u)\frac{\left| \hat{D}_{t}(u)-D_{t}(u)\right| }{\Sigma _{t}(u)+\varepsilon },
   $$

   $$
   \mathcal {L}_{a}=
   \sum_{u}M_{t}(u)\left| \hat{A}_{t}(u)-1\right|,
   $$

   $$
   \mathcal {L}_{n}=
   \sum_{u}M_{t}(u)\left(1-\left\langle \hat{N}_{t}(u),N_{t}^{surf}(u)\right\rangle \right).
   $$

5. The method according to claim 1, characterized in that Gaussian stability scoring and hierarchical memory management in step S3 satisfy:

   $$
   \ell _{n}^{imp}\gets \ell _{n}^{imp}+s_{n}^{imp},
   \qquad
   \ell _{n}^{err}\gets \max(\ell _{n}^{err},s_{n}^{err}),
   $$

   and

   $$
   \delta _{i}=
   \left\|
   \operatorname{trans}
   \left(
   (T_{t}^{c2w}) ^{-1}T_{i}^{c2w}
   \right)
   \right\| _{2},
   \qquad
   \mathcal {G}_{t}^{gpu}=
   \{
   \gamma _{n}\mid
   \delta _{\kappa _{n}}\leq \tau _{mem}
   \},
   $$

   $$
   \mathcal {G}_{t}^{cpu}=
   \{
   \gamma _{n}\mid
   \delta _{\kappa _{n}}>\tau _{mem}
   \},
   $$

   so that near-field Gaussian primitives are retained in GPU memory and far-field Gaussian primitives are stored in CPU memory.

6. The method according to claim 1, characterized in that submap division and frozen snapshotting in step S4 satisfy:

   $$
   S_{k}=
   \left(
   \mathcal{V}_{k},
   \mathcal{G}_{k},
   \Omega_{k},
   \overline{\phi}_{k}
   \right),
   $$

   and a submap division trigger condition satisfies

   $$
   \left|\mathcal{V}_{k}\right| \geq N_{max }
   \vee
   \sum_{i=2}^{\left|\mathcal{V}_{k}\right|}
   \left\| p_{i}-p_{i-1}\right\| _{2} \geq L_{max },
   $$

   wherein $\Omega_k$ is a frozen submap snapshot storing Gaussian state, keyframe association, and submap metadata.

7. The method according to claim 1, characterized in that the two-stage cross-submap loop-closure retrieval in step S4 satisfies:

   $$
   \phi _{i}=
   \operatorname{norm}
   \left(
   \left[
   \operatorname{vec}(P(I_{i})),
   \operatorname{vec}(\nabla _{x}\overline {P}(I_{i})),
   \operatorname{vec}(\nabla _{y}\overline {P}(I_{i}))
   \right]
   \right),
   \qquad
   \overline{\phi}_{k}=
   \operatorname{norm}
   \left(
   \frac{1}{\left|\mathcal{V}_{k}\right|}
   \sum_{i \in \mathcal{V}_{k}} \phi_{i}
   \right),
   $$

   $$
   s_{ret }(q, k)=\phi_{q}^{\top} \overline{\phi}_{k},
   \qquad
   s_{ret }(q, j)=\phi_{q}^{\top} \phi_{j},
   $$

   so that a coarse submap-level retrieval stage and a fine keyframe-level retrieval stage are sequentially executed.

8. The method according to claim 1, characterized in that the loop-closure validation and full-link correction feedback in step S5 satisfy:

   $$
   \left(R_{qr}, t_{qr}, s_{qr}\right)=
   \arg \min _{R \in SO(3), t \in \mathbb{R}^{3}, s>0}
   \sum_{m \in \mathcal{I}}
   \left\|
   x_{q}^{(m)}-\left(s R x_{r}^{(m)}+t\right)
   \right\| _{2}^{2},
   $$

   and a four-stage validation rule satisfies

   $$
   \Omega _{val}=
   \left\{
   u\mid
   \hat{A}_{q}(u)>\tau _{acc}
   \wedge
   \hat{D}_{q}(u)>0
   \right\},
   $$

   $$
   e_{photo}=
   \frac{1}{|\Omega _{val}|}
   \sum_{u\in \Omega _{val}}
   \left|
   \hat{I}_{q}^{gray}(u)-I_{q}^{gray}(u)
   \right|,
   $$

   $$
   \chi _{loop}(q,r)=
   \mathbf{1}
   \left[
   |\mathcal{I}|\geq N_{min}
   \wedge
   |\Omega _{val}|\geq A_{min}
   \wedge
   e_{photo}\leq \tau _{photo}
   \wedge
   \left\|
   \hat{p}_{q}-p_{q}
   \right\| _{2}\leq \tau _{jump}
   \right],
   $$

   and

   $$
   \{ S_{i}^{*}\} =
   \arg \operatorname* {min}_{\left\{ S_{i}\right\} }
   \sum _{(i,j)\in \mathcal {E}_{adj}\cup \mathcal {E}_{ov}\cup \mathcal {E}_{loop}}
   \left\|
   \log \left( Z_{ij}^{-1}S_{i}^{-1}S_{j}\right)
   \right\| _{\Lambda _{ij}}^{2},
   $$

   with Gaussian correction satisfying

   $$
   \mu _{n}\leftarrow s_{\Delta _{i}}R_{\Delta _{i}}\mu _{n}+t_{\Delta _{i}},
   \qquad
   q_{n}\leftarrow R_{\Delta _{i}}\otimes q_{n},
   \qquad
   a_{n}\leftarrow a_{n}+\log s_{\Delta _{i}}.
   $$

9. A system for large-scale monocular simultaneous localization and mapping with an adaptive Gaussian map based on a submap mechanism, characterized by comprising:

   a hybrid monocular dense geometric prior front-end module configured to perform step S1 of claim 1;

   a local tracking and standardized keyframe packaging module configured to perform step S2 of claim 1;

   a Gaussian mapping and hierarchical memory management module configured to perform step S3 of claim 1; and

   a submap management and global Sim(3) correction module configured to perform steps S4-S5 of claim 1.

10. A non-transitory computer-readable storage medium storing computer instructions which, when executed by a processor, cause the processor to perform the method according to any one of claims 1-8.

## 4. Technical Field

The present invention belongs to the technical field of monocular visual simultaneous localization and mapping (SLAM), dense three-dimensional scene reconstruction, differentiable scene representation, autonomous robotics, and augmented reality, and particularly relates to a large-scale monocular SLAM method and system that employs a hybrid dense geometric prior front-end, a standardized uncertainty-aware keyframe middleware, an adaptive Gaussian mapping back-end, and a submap-based global Sim(3) correction mechanism.

## 5. Background Art

Monocular SLAM has been widely used in mobile robotics, autonomous navigation, digital-twin construction, embodied intelligence, and augmented reality. In recent years, Gaussian splatting has shown strong advantages in dense scene representation because it supports differentiable rendering, compact scene parametrization, and efficient online optimization. Accordingly, integrating monocular SLAM with Gaussian-based dense mapping has become an important technical direction for large-scale visual perception.

However, existing technologies still have several substantial limitations.

First, sparse-feature monocular SLAM methods cannot directly provide dense geometric maps. Vanilla DROID-SLAM type dense front-ends rely on SfM-dependent optical flow prediction and recurrent update alone, and therefore often suffer from tracking failure, poor scale consistency, and severe drift in aggressive motion, low-texture environments, and motion-blurred scenes. Standalone MASt3R-SLAM type front-ends provide strong dense geometric priors, but their output structures are not standardized for traditional dense Gaussian mapping back-ends, and seamless integration into a complete large-scale SLAM pipeline usually requires heavy interface modification.

Second, existing dense monocular SLAM systems generally couple the tracking front-end and the mapping back-end too tightly. Confidence produced by the front-end is usually not propagated as uncertainty into the dense mapping process, so unreliable depth regions are not explicitly down-weighted during map optimization, which reduces dense reconstruction accuracy and robustness.

Third, VINGS-Mono uses 6-DoF SE(3) Pose(3) PnP loop closure. Such a rigid-body loop-closure model has no scale optimization dimension and therefore cannot fundamentally solve long-term scale drift in monocular SLAM. In addition, its global single-frame retrieval strategy causes rapid growth of retrieval cost in large-scale scenes.

Fourth, existing Gaussian SLAM schemes and LiDAR-oriented submap mechanisms do not provide a complete large-scale monocular solution that jointly addresses adaptive Gaussian control, frozen dense submap management, cross-submap retrieval, and scale-aware global correction. As a result, Gaussian primitives may grow uncontrollably, memory usage becomes unstable, and long-term large-scale operation becomes difficult to maintain.

Therefore, there remains a need for a large-scale monocular SLAM method that simultaneously improves front-end tracking robustness, preserves dense geometric prior compatibility with dense mapping, propagates uncertainty explicitly, controls Gaussian-map growth, reduces large-scale loop-retrieval complexity, and corrects accumulated scale drift by submap-level Sim(3) global optimization.

## 6. Summary of the Invention

### 6.1 Technical Problems to be Solved

The invention aims to solve the following technical problems.

1. Existing monocular dense front-ends cannot simultaneously achieve robust tracking under aggressive motion, low texture, and motion blur, and still remain compatible with downstream dense mapping interfaces.
2. Existing dense monocular SLAM systems lack a standardized mechanism for propagating front-end confidence into back-end uncertainty-aware dense mapping, resulting in tight front-end/back-end coupling.
3. Existing rigid six-DoF loop-closure formulations cannot optimize monocular scale, so long-term scale drift remains unsolved.
4. Existing large-scale dense SLAM schemes lack efficient submap retrieval, adaptive Gaussian-map growth control, and full-link correction feedback to all mapping assets.

### 6.2 Technical Solution

To solve the above technical problems, the invention provides a method and system for large-scale monocular SLAM with an adaptive Gaussian map based on a submap mechanism.

In step S1, a hybrid monocular dense geometric prior front-end is employed. A pre-trained MASt3R model is integrated into a DROID-SLAM tracking framework so as to replace the original SfM-dependent geometric prior path with a pre-trained dense geometric prior, while preserving the native output format required by the tracking framework. The front-end outputs a dense point map, a confidence map, and a flow-compatible motion field for robust local tracking.

In step S2, the point map and confidence map are converted into depth and uncertainty covariance according to a Sim(3) state. Local pose estimation is performed by a factor graph jointly constrained by temporal geometric residuals, flow residuals, and relocalization residuals. The optimized tracking state is then converted into a standardized keyframe data packet, thereby realizing decoupling between the tracking front-end and the mapping back-end.

In step S3, the standardized keyframe data packet drives adaptive Gaussian-map initialization and online optimization. The mapping back-end performs differentiable rendering and joint RGB-depth-normal optimization, executes stability-aware pruning and densification, and manages Gaussian primitives by a GPU/CPU hierarchical storage rule based on pose distance.

In step S4, the global dense map is divided into active submaps and frozen submaps. The system computes keyframe descriptors and submap descriptors, then retrieves cross-submap loop candidates by a two-stage mechanism including submap-level coarse retrieval and keyframe-level fine retrieval.

In step S5, the system performs depth-assisted submap-level Sim(3) estimation, applies multi-stage validation to loop candidates, constructs a global Sim(3) pose graph, and feeds the optimized correction back to the front-end pose cache, online GPU Gaussian primitives, CPU-stored Gaussian primitives, and frozen submap snapshots.

### 6.3 Advantageous Effects

Compared with existing technologies, the invention provides the following advantageous effects.

1. The hybrid MASt3R-DROID front-end combines pre-trained geometric prior robustness with recurrent dense tracking continuity, thereby improving tracking robustness, tracking continuity, and scale consistency in aggressive motion, low-texture, and motion-blurred scenes.
2. The standardized uncertainty-aware keyframe middleware realizes complete decoupling of the tracking front-end and mapping back-end while preserving lossless propagation of geometric confidence into dense mapping uncertainty.
3. The submap-level 7-DoF Sim(3) optimization fundamentally solves long-term monocular scale drift that cannot be corrected by rigid six-DoF loop-closure formulations.
4. The two-stage retrieval mechanism reduces the computational burden of large-scale loop closure and improves real-time performance for long trajectories.
5. The adaptive Gaussian optimization and hierarchical memory management mechanism suppress uncontrolled growth of Gaussian primitives and improve dense mapping robustness in large-scale scenes.
6. The full-link correction feedback mechanism synchronously corrects front-end pose caches and all Gaussian-map assets, thereby avoiding post-loop map dislocation and improving global consistency of both trajectory and dense map.

## 7. Brief Description of the Drawings

Fig. 1 is an overall flow chart of the proposed large-scale monocular SLAM method.

Fig. 2 is a schematic diagram of the hybrid MASt3R-DROID front-end architecture.

Fig. 3 is a schematic diagram of the standardized keyframe middleware and uncertainty propagation process.

Fig. 4 is a schematic diagram of submap division, cross-submap loop closure validation, and the global Sim(3) correction mechanism.

### Fig. 1. Overall Flow Chart

```mermaid
flowchart TB
    A["Monocular RGB Sequence"] --> B["Hybrid Monocular Dense Geometric Prior Front-End"]
    B --> C["Standardized Uncertainty-Aware Keyframe Middleware"]
    C --> D["Adaptive Gaussian Mapping and Hierarchical Memory Management"]
    D --> E["Submap Division and Two-Stage Cross-Submap Retrieval"]
    E --> F["Submap-Level Sim3 Validation and Global Sim3 Pose Graph Optimization"]
    F --> G["Globally Consistent Trajectory and Dense Gaussian Scene Map"]
    F -. correction feedback .-> B
    F -. correction feedback .-> D
```

### Fig. 2. Hybrid MASt3R-DROID Front-End Architecture

```mermaid
flowchart LR
    A["Current/Previous RGB Frames"] --> B["Image Resize + Intrinsic Scaling"]
    B --> C["Tracking Feature Branch"]
    B --> D["Pre-Trained Geometric Prior Branch"]
    C --> E["Feature / Motion Fusion Adapter"]
    D --> E
    E --> F["Dense Point Map + Confidence Map + Flow-Compatible Motion State"]
    F --> G["Local Factor Graph Tracking"]
```

### Fig. 3. Standardized Keyframe Middleware and Uncertainty Propagation

```mermaid
flowchart LR
    A["Hybrid Front-End Output"] --> B["Sim3 Pose Decomposition"]
    A --> C["Point Map to Depth Conversion"]
    A --> D["Confidence to Covariance Conversion"]
    B --> E["Validity Filtering and Effective Pixel Mask"]
    C --> E
    D --> E
    E --> F["Standardized Keyframe Data Packet"]
    F --> G["Gaussian Mapping Back-End"]
```

### Fig. 4. Submap Division, Loop Validation, and Sim(3) Correction

```mermaid
flowchart TB
    A["Active Gaussian Map"] --> B["Submap Trigger Detection"]
    B --> C["Frozen Submap Snapshot"]
    C --> D["Submap-Level Coarse Retrieval"]
    D --> E["Keyframe-Level Fine Retrieval"]
    E --> F["Depth-Assisted Sim3 Estimation"]
    F --> G["Four-Stage Loop Validation"]
    G --> H["Global Sim3 Pose Graph Optimization"]
    H --> I["Correction of Pose Cache + GPU Gaussians + CPU Gaussians + Frozen Snapshots"]
```

## 8. Detailed Description of the Embodiments

The invention is further described below in conjunction with preferred embodiments and mathematical expressions. The embodiments are used to explain the invention and shall not be construed as limiting the protection scope of the claims.

### 8.1 S1: Hybrid Monocular Dense Geometric Prior Front-End

In the present invention, a monocular RGB sequence $\{I_t\}_{t=1}^{T}$ is first preprocessed by image resizing and cropping. The camera intrinsic matrix is synchronously adjusted so that the projected geometry remains consistent after image scaling. The scaled intrinsic matrix is:

$$
K_{t}^{\prime }=
\left[
\begin{array} {ccc}
{s_{x}f_{x}} & {0} & {s_{x}c_{x}+\Delta _{x}}\\
{0} & {s_{y}f_{y}} & {s_{y}c_{y}+\Delta _{y}}\\
{0} & {0} & {1}
\end{array}
\right].
$$

The front-end adopts a hybrid dense geometric prior architecture. A recurrent dense tracking framework provides local motion-state propagation and factor-graph-compatible tracking state, while a pre-trained MASt3R geometric prior branch provides dense global geometric cues. The hybrid front-end inference is expressed as:

$$
\left(P_{t}, C_{t}, F_{t}\right)=\mathcal{F}_{\theta}\left(I_{t}', I_{t-1}', K_{t}'\right),
$$

where $P_t$ is a dense point map, $C_t$ is a confidence map, $F_t$ is an optical flow field or flow-compatible motion field, and $\mathcal{F}_{\theta}$ denotes the hybrid front-end with pre-trained parameters.

In one preferred embodiment, the front-end may be written as a dual-branch fusion model:

$$
\Phi _{t}^{trk}=E_{trk}\left(I_{t-1}',I_{t}'\right),
\qquad
\Phi _{t}^{geo}=E_{geo}\left(I_{t-1}',I_{t}',K_{t}'\right),
$$

$$
\left(P_{t}, C_{t}, F_{t}\right)=
\mathcal{U}
\left(
\Phi _{t}^{trk},
\Phi _{t}^{geo},
\operatorname{Corr}_{t}
\right),
$$

where $\Phi_t^{trk}$ is a tracking feature state, $\Phi_t^{geo}$ is a pre-trained dense geometric prior state, and $\operatorname{Corr}_t$ is a frame-to-frame correlation state.

The crucial technical point is not merely to fuse two models, but to preserve downstream compatibility. Therefore, the output state is transformed by:

$$
\mathcal{O}_{t}^{hyb}=
\mathcal{T}_{comp}\left(P_{t},C_{t},F_{t}\right),
\qquad
\operatorname{layout}\!\left(\mathcal{O}_{t}^{hyb}\right)=
\operatorname{layout}\!\left(\mathcal{O}_{t}^{native}\right).
$$

Thus, the hybrid front-end retains the native output structure of the dense tracking framework and can be consumed by subsequent tracking, buffering, and mapping modules without heavy pipeline rewriting. Compared with purely recurrent dense tracking or a standalone dense-prior front-end, this architecture simultaneously improves tracking robustness, preserves tracking continuity, and maintains better scale stability under aggressive motion, low-texture, and motion-blurred conditions.

### 8.2 S2: Local Pose Estimation and Standardized Keyframe Packet Generation

The pose state of a current frame is expressed in Sim(3) form:

$$
T_{t}^{sim 3}=
\left[
\begin{array}{cc}
s_{t} R_{t} & t_{t} \\
0 & 1
\end{array}
\right],
\qquad
s_{t}=
\left|
\det\left(T_{t, 1: 3,1: 3}^{sim 3}\right)
\right|^{1 / 3}.
$$

The point map is converted into depth by using the scale term:

$$
D_{t}(u)=s_{t} Z_{t}(u).
$$

The confidence map is propagated as an uncertainty covariance:

$$
\Sigma_{t}(u)=
\operatorname{clip}
\left(
\frac{\alpha}{\max \left(C_{t}(u), \varepsilon\right)} s_{t}^{2},
\Sigma_{min },
\Sigma_{max }
\right).
$$

An effective pixel mask is then constructed as:

$$
M_{t}(u)=
\mathbf{1}
\left[
D_{min }<D_{t}(u)<D_{max }
\wedge
\Sigma_{t}(u) \leq \tau_{\Sigma}
\wedge
D_{t}(u), \Sigma_{t}(u) \in \mathbb{R}
\right].
$$

The local pose state is optimized on a factor graph jointly constrained by temporal geometric residuals, flow residuals, and relocalization residuals:

$$
\mathcal{X}_{t}^{*}=
\arg \min _{\mathcal{X}_{t}}
\left(
\sum_{(i, j) \in \mathcal{E}_{t}^{temp }}
\rho\left(\left\| r_{i j}^{ray }\left(\mathcal{X}_{t}\right)\right\| _{W_{i j}}^{2}\right)
+
\sum_{(i, j) \in \mathcal{E}_{t}^{flow }}
\rho\left(\left\| r_{i j}^{flow }\left(\mathcal{X}_{t}\right)\right\| _{W_{i j}}^{2}\right)
+
\sum_{(i, k) \in \mathcal{E}_{t}^{relocal }}
\rho\left(\left\| r_{i k}^{relocal }\left(\mathcal{X}_{t}\right)\right\| _{W_{i k}}^{2}\right)
\right).
$$

Here, $r_{ij}^{ray}$ denotes a geometric ray residual, $r_{ij}^{flow}$ denotes an optical-flow consistency residual, $r_{ik}^{relocal}$ denotes a relocalization residual, and $\rho(\cdot)$ denotes a robust kernel.

After optimization, the system generates a standardized keyframe data packet:

$$
\mathcal{B}_{t}=
\left\{
I_{t}',
D_{t},
\Sigma_{t},
M_{t},
T_{t}^{c2w},
\tau_{t},
g_{t},
K_{t}'
\right\}.
$$

This standardized packet is a core inventive step. It completely decouples the front-end and back-end while preserving geometric confidence in the form of uncertainty covariance. The mapping back-end therefore receives dense geometry and uncertainty in a standardized interface, rather than being hard-coded to a specific tracking implementation.

### 8.3 S3: Adaptive Gaussian Map Optimization and Hierarchical Memory Management

The dense map is represented by adaptive Gaussian primitives:

$$
\mathcal{G}_{t}=\left\{\gamma_{n}\right\}_{n=1}^{N_{t}},
\qquad
\gamma_{n}=\left(\mu_{n}, q_{n}, a_{n}, o_{n}, c_{n}, \kappa_{n}, b_{n}\right),
$$

where $\mu_n$ is the Gaussian center, $q_n$ is the rotation parameter, $a_n$ is the scale parameter, $o_n$ is opacity, $c_n$ is color, $\kappa_n$ is an ownership keyframe index, and $b_n$ is a birth keyframe index.

For each valid pixel, a Gaussian center may be initialized by back-projection:

$$
x_{t}(u)=D_{t}(u)K_{t}^{'-1}\tilde{u},
\qquad
\mu _{t}(u)=R_{t}x_{t}(u)+t_{t},
$$

where $\tilde{u}=[u,v,1]^{\top}$.

During online optimization, the Gaussian map is rendered into the current view and optimized by an RGB-Depth-Normal joint loss:

$$
\mathcal {L}_{t}=
\lambda _{rgb}\mathcal {L}_{rgb}+
\lambda _{d}\mathcal {L}_{d}+
\lambda _{a}\mathcal {L}_{a}+
\lambda _{n}\mathcal {L}_{n}.
$$

The sub-losses are written as:

$$
\mathcal {L}_{rgb}=
\sum_{u}M_{t}(u)\left\| \hat{I}_{t}(u)-I_{t}'(u)\right\| _{1},
$$

$$
\mathcal {L}_{d}=
\sum_{u}M_{t}(u)\frac{\left| \hat{D}_{t}(u)-D_{t}(u)\right| }{\Sigma _{t}(u)+\varepsilon },
$$

$$
\mathcal {L}_{a}=
\sum_{u}M_{t}(u)\left| \hat{A}_{t}(u)-1\right|,
$$

$$
\mathcal {L}_{n}=
\sum_{u}M_{t}(u)\left(1-\left\langle \hat{N}_{t}(u),N_{t}^{surf}(u)\right\rangle \right).
$$

The invention further introduces adaptive Gaussian stability control. A Gaussian importance score and Gaussian error score are updated as:

$$
\ell _{n}^{imp}\gets \ell _{n}^{imp}+s_{n}^{imp},
\qquad
\ell _{n}^{err}\gets \operatorname* {max}(\ell _{n}^{err},s_{n}^{err}).
$$

In one embodiment, pruning and densification are determined by:

$$
\mathcal{P}_{t}=
\left\{
n \mid
\ell_{n}^{imp}\in [\tau_{p}^{min},\tau_{p}^{max}]
\wedge
\ell_{n}^{err}\leq \tau_{e}
\right\},
$$

$$
\mathcal{D}_{t}=
\left\{
n \mid
\ell_{n}^{err}>\tau_{d}
\vee
\hat{A}_{t}(u)<\tau_{a}
\right\}.
$$

For large-scale operation, the invention uses hierarchical memory management:

$$
\delta _{i}=
\left\|
\operatorname{trans}
\left(
(T_{t}^{c2w}) ^{-1}T_{i}^{c2w}
\right)
\right\| _{2},
$$

$$
\mathcal {G}_{t}^{gpu}=
\{ \gamma _{n} \mid \delta _{\kappa _{n}}\leq \tau _{mem}\},
\qquad
\mathcal {G}_{t}^{cpu}=
\{ \gamma _{n} \mid \delta _{\kappa _{n}}>\tau _{mem}\}.
$$

Accordingly, the number of active Gaussian primitives is controlled while preserving local rendering fidelity. This solves the uncontrolled growth problem that appears in large-scale Gaussian mapping.

### 8.4 S4: Submap Division and Cross-Submap Loop Closure Candidate Retrieval

The global map is divided into submaps:

$$
S_{k}=
\left(
\mathcal{V}_{k},
\mathcal{G}_{k},
\Omega_{k},
\overline{\phi}_{k}
\right),
$$

where $\mathcal{V}_k$ is a keyframe set, $\mathcal{G}_k$ is a Gaussian set, $\Omega_k$ is a frozen snapshot state, and $\overline{\phi}_k$ is a submap descriptor.

The invention triggers submap division when either a keyframe-count condition or a travel-distance condition is satisfied:

$$
\left|\mathcal{V}_{k}\right| \geq N_{max }
\vee
\sum_{i=2}^{\left|\mathcal{V}_{k}\right|}
\left\| p_{i}-p_{i-1}\right\| _{2} \geq L_{max }.
$$

Once the trigger condition is met, the active submap is frozen into a snapshot:

$$
\Omega_{k}=
\left\{
\mathcal{G}_{k}^{gpu},
\mathcal{G}_{k}^{cpu},
\mathcal{A}_{k}
\right\},
$$

where $\mathcal{A}_k$ stores keyframe association, descriptor information, and submap metadata.

For cross-submap loop closure, the invention employs two-stage retrieval. The keyframe descriptor is:

$$
\phi _{i}=
\operatorname{norm}
\left(
\left[
\operatorname{vec}(P(I_{i})),
\operatorname{vec}(\nabla _{x}\overline {P}(I_{i})),
\operatorname{vec}(\nabla _{y}\overline {P}(I_{i}))
\right]
\right),
$$

and the submap descriptor is:

$$
\overline{\phi}_{k}=
\operatorname{norm}
\left(
\frac{1}{\left|\mathcal{V}_{k}\right|}
\sum_{i \in \mathcal{V}_{k}} \phi_{i}
\right).
$$

The two-stage retrieval scores are:

$$
s_{ret }(q, k)=\phi_{q}^{\top} \overline{\phi}_{k},
\qquad
s_{ret }(q, j)=\phi_{q}^{\top} \phi_{j}.
$$

Thus, the system first retrieves a limited number of candidate submaps and then retrieves a limited number of candidate keyframes inside those submaps. The retrieval complexity is reduced from a global single-frame search to a coarse-to-fine search that scales approximately as $O(M+K)$ instead of $O(N)$, where $M$ is the number of selected submaps and $K$ is the number of selected keyframes.

### 8.5 S5: Submap-Level Sim(3) Loop Closure Validation, Global Optimization and Full-Link Correction Feedback

This step is the key inventive step of the present invention. Unlike rigid six-DoF loop-closure formulations that correct only rotation and translation, the present invention estimates a seven-DoF Sim(3) transformation and therefore simultaneously corrects rotation, translation, and scale drift in monocular SLAM.

For a retrieved query-reference pair, matched three-dimensional points are reconstructed from depth, and a Sim(3) similarity transformation is estimated by:

$$
\left(R_{qr}, t_{qr}, s_{qr}\right)=
\arg \min _{R \in S O(3), t \in \mathbb{R}^{3}, s>0}
\sum_{m \in \mathcal{I}}
\left\| x_{q}^{(m)}-\left(s R x_{r}^{(m)}+t\right)\right\| _{2}^{2}.
$$

To avoid false loop insertions, the invention performs four-stage loop validation.

Stage 1 is correspondence sufficiency:

$$
|\mathcal{I}|\geq N_{min}.
$$

Stage 2 is accumulation-valid area checking:

$$
\Omega _{val}=
\left\{
u\mid
\hat{A}_{q}(u)>\tau _{acc}
\wedge
\hat{D}_{q}(u)>0
\right\},
\qquad
|\Omega _{val}|\geq A_{min}.
$$

Stage 3 is rendered-versus-observed photometric consistency:

$$
e_{photo}=
\frac{1}{|\Omega _{val}|}
\sum_{u\in \Omega _{val}}
\left|
\hat{I}_{q}^{gray}(u)-I_{q}^{gray}(u)
\right|
\leq \tau _{photo}.
$$

Stage 4 is pose-jump consistency:

$$
\left\|
\hat{p}_{q}-p_{q}
\right\| _{2}\leq \tau _{jump}.
$$

The complete loop validation rule is:

$$
\chi _{loop}(q,r)=
\mathbf{1}
\left[
|\mathcal{I}|\geq N_{min}
\wedge
|\Omega _{val}|\geq A_{min}
\wedge
e_{photo}\leq \tau _{photo}
\wedge
\left\|
\hat{p}_{q}-p_{q}
\right\| _{2}\leq \tau _{jump}
\right].
$$

Only validated loop constraints are inserted into a global Sim(3) pose graph:

$$
\{ S_{i}^{*}\} =
\arg \operatorname* {min}_{\left\{ S_{i}\right\} }
\sum _{(i,j)\in \mathcal {E}_{adj}\cup \mathcal {E}_{ov}\cup \mathcal {E}_{loop}}
\left\|
\log \left( Z_{ij}^{-1}S_{i}^{-1}S_{j}\right)
\right\| _{\Lambda _{ij}}^{2}.
$$

Let the optimized correction of node $i$ be:

$$
\Delta _{i}=S_{i}^{*}(S_{i}^{old})^{-1}.
$$

The front-end pose cache is corrected by:

$$
T_{i}^{c2w}\leftarrow \Pi _{SE(3)}\left(\Delta _{i}T_{i}^{c2w}\right),
$$

and Gaussian primitives belonging to the corrected submap or corrected keyframe are updated by:

$$
\mu _{n}\leftarrow s_{\Delta _{i}}R_{\Delta _{i}}\mu _{n}+t_{\Delta _{i}},
\qquad
q_{n}\leftarrow R_{\Delta _{i}}\otimes q_{n},
\qquad
a_{n}\leftarrow a_{n}+\log s_{\Delta _{i}}.
$$

The same correction is synchronously applied to online GPU Gaussian primitives, CPU-stored Gaussian primitives, and frozen submap snapshots. Therefore, the invention does not perform a symbolic trajectory correction only. Instead, it realizes full-link global consistency correction across the tracking state, the online dense map, the hierarchical memory state, and the frozen submap assets.
