import copy
import math
import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SCRIPT_ROOT)
GTSAM_PYTHON_ROOT = os.path.join(REPO_ROOT, "submodules", "gtsam", "build", "python")
if os.path.isdir(GTSAM_PYTHON_ROOT) and GTSAM_PYTHON_ROOT not in sys.path:
    sys.path.insert(0, GTSAM_PYTHON_ROOT)

import gtsam

from gaussian.gaussian_model import GaussianModel
from gaussian.vis_utils import save_ply
from loop.loop_detect import LoopDetector
from storage.storage_manage import StorageManager
from vings_utils.gtsam_utils import matrix_to_tq
from vings_utils.pytorch3d_utils import R2q, q2R
from vings_utils.sim3_utils import (
    Sim3Transform,
    numerical_jacobian,
    pose3_matrix_to_sim3,
    sim3_compose,
    sim3_exp,
    sim3_inverse,
    sim3_log,
    sim3_to_pose3_matrix,
)


def _clone_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def _merge_attr(
    primary_tensor: torch.Tensor,
    secondary_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    primary = _clone_cpu(primary_tensor)
    if secondary_tensor is None or secondary_tensor.numel() == 0:
        return primary
    secondary = _clone_cpu(secondary_tensor)
    if primary.numel() == 0:
        return secondary
    return torch.cat((primary, secondary), dim=0)


def _relative_pose(c2w_src: torch.Tensor, c2w_dst: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(c2w_src) @ c2w_dst


def _pose3_from_matrix(matrix: torch.Tensor) -> gtsam.Pose3:
    matrix_np = matrix.detach().cpu().numpy()
    return gtsam.Pose3(gtsam.Rot3(matrix_np[:3, :3]), matrix_np[:3, 3])


def _matrix_from_pose3(pose: gtsam.Pose3, device: torch.device) -> torch.Tensor:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = pose.rotation().matrix().astype(np.float32)
    matrix[:3, 3] = np.asarray(pose.translation(), dtype=np.float32)
    return torch.tensor(matrix, dtype=torch.float32, device=device)


def _intrinsic_to_cpu_dict(intrinsic: Dict[str, object]) -> Dict[str, float]:
    intrinsic_cpu = {}
    for key, value in intrinsic.items():
        if torch.is_tensor(value):
            intrinsic_cpu[key] = float(value.detach().cpu().item())
        else:
            intrinsic_cpu[key] = float(value)
    return intrinsic_cpu


def _same_pose(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5) -> bool:
    return torch.allclose(a, b, atol=atol, rtol=0.0)


def _same_sim3(a: Sim3Transform, b: Sim3Transform, atol: float = 1e-5) -> bool:
    return (
        np.allclose(a.R, b.R, atol=atol, rtol=0.0)
        and np.allclose(a.t, b.t, atol=atol, rtol=0.0)
        and abs(math.log(max(float(a.s), 1e-12)) - math.log(max(float(b.s), 1e-12))) <= atol
    )


@dataclass
class Sim3Edge:
    src: int
    dst: int
    edge_type: str
    relative_sim3: Sim3Transform
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class KeyframeRecord:
    global_kf_id: int
    timestamp: float
    c2w: torch.Tensor
    intrinsic: Dict[str, float]
    image_hw3: torch.Tensor
    depth_hw: torch.Tensor
    descriptor: torch.Tensor
    owner_submap_id: int
    submap_ids: Set[int] = field(default_factory=set)
    loop_metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class SubmapRecord:
    submap_id: int
    keyframe_ids: List[int] = field(default_factory=list)
    overlap_kf_ids: List[int] = field(default_factory=list)
    altitude_anchor_z: Optional[float] = None
    frozen: bool = False
    snapshot_path: Optional[str] = None
    descriptor: Optional[torch.Tensor] = None
    loop_edges: List[Tuple[int, int]] = field(default_factory=list)


class GlobalKeyframeDatabase:
    def __init__(self, cfg: Dict[str, object]):
        self.cfg = cfg
        self.records: Dict[int, KeyframeRecord] = {}
        self._submap_to_keyframes: Dict[int, Set[int]] = {}
        self._sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        self._sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)

    def _compute_descriptor(self, image_hw3: torch.Tensor) -> torch.Tensor:
        descriptor_size = int(self.cfg.get("descriptor_size", 12))
        chw = image_hw3.detach().cpu().permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        pooled = F.interpolate(
            chw,
            size=(descriptor_size, descriptor_size),
            mode="bilinear",
            align_corners=False,
        )
        gray = pooled.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self._sobel_x, padding=1)
        grad_y = F.conv2d(gray, self._sobel_y, padding=1)
        descriptor = torch.cat((pooled.reshape(1, -1), grad_x.reshape(1, -1), grad_y.reshape(1, -1)), dim=1)
        descriptor = F.normalize(descriptor, dim=1).squeeze(0)
        return descriptor

    def add_membership(
        self,
        submap_id: int,
        global_kf_id: int,
        timestamp: float,
        c2w: torch.Tensor,
        image_hw3: torch.Tensor,
        depth_hw: torch.Tensor,
        intrinsic: Dict[str, object],
    ) -> None:
        descriptor = self._compute_descriptor(image_hw3)
        intrinsic_cpu = _intrinsic_to_cpu_dict(intrinsic)
        image_cpu = _clone_cpu(image_hw3)
        if image_cpu.dtype.is_floating_point:
            image_cpu = torch.clamp(image_cpu, 0.0, 1.0).mul(255.0).round().to(torch.uint8)
        else:
            image_cpu = image_cpu.to(torch.uint8)
        depth_cpu = depth_hw.detach().cpu().to(torch.float32)
        if depth_cpu.ndim == 3 and depth_cpu.shape[-1] == 1:
            depth_cpu = depth_cpu.squeeze(-1)
        elif depth_cpu.ndim == 3 and depth_cpu.shape[0] == 1:
            depth_cpu = depth_cpu.squeeze(0)
        self._submap_to_keyframes.setdefault(submap_id, set()).add(global_kf_id)
        if global_kf_id not in self.records:
            self.records[global_kf_id] = KeyframeRecord(
                global_kf_id=global_kf_id,
                timestamp=float(timestamp),
                c2w=_clone_cpu(c2w),
                intrinsic=intrinsic_cpu,
                image_hw3=image_cpu.clone(),
                depth_hw=depth_cpu.clone(),
                descriptor=descriptor,
                owner_submap_id=submap_id,
                submap_ids={submap_id},
                loop_metadata={"attempts": 0, "successes": 0, "partners": []},
            )
            return

        record = self.records[global_kf_id]
        record.timestamp = float(timestamp)
        record.c2w = _clone_cpu(c2w)
        record.intrinsic = intrinsic_cpu
        record.image_hw3 = image_cpu.clone()
        record.depth_hw = depth_cpu.clone()
        record.descriptor = descriptor
        record.submap_ids.add(submap_id)

    def submap_keyframes(self, submap_id: int) -> Set[int]:
        return self._submap_to_keyframes.get(submap_id, set())

    def update_pose(self, global_kf_id: int, c2w: torch.Tensor) -> None:
        if global_kf_id in self.records:
            self.records[global_kf_id].c2w = _clone_cpu(c2w)

    def add_existing_membership(self, submap_id: int, global_kf_id: int) -> None:
        self._submap_to_keyframes.setdefault(submap_id, set()).add(int(global_kf_id))
        if global_kf_id in self.records:
            self.records[global_kf_id].submap_ids.add(int(submap_id))

    def update_loop_metadata(
        self,
        query_kf_id: int,
        ref_kf_id: int,
        success: bool,
        score: float,
    ) -> None:
        for global_kf_id, partner in ((query_kf_id, ref_kf_id), (ref_kf_id, query_kf_id)):
            if global_kf_id not in self.records:
                continue
            metadata = self.records[global_kf_id].loop_metadata
            metadata["attempts"] = int(metadata.get("attempts", 0)) + 1
            metadata["last_score"] = float(score)
            metadata["last_partner"] = int(partner)
            if success:
                metadata["successes"] = int(metadata.get("successes", 0)) + 1
                partners = list(metadata.get("partners", []))
                if partner not in partners:
                    partners.append(int(partner))
                metadata["partners"] = partners

    def query_candidates(
        self,
        query_image_hw3: torch.Tensor,
        active_submap_id: int,
        frozen_submaps: Dict[int, SubmapRecord],
    ) -> List[Tuple[int, int, float]]:
        if not frozen_submaps:
            return []

        query_descriptor = self._compute_descriptor(query_image_hw3)
        top_submaps = int(self.cfg.get("retrieval_top_submaps", 3))
        top_keyframes = int(self.cfg.get("retrieval_top_keyframes", 2))

        scored_submaps: List[Tuple[int, float]] = []
        for submap_id, submap in frozen_submaps.items():
            if submap_id == active_submap_id:
                continue
            if submap.descriptor is None:
                continue
            score = float(torch.dot(query_descriptor, submap.descriptor))
            scored_submaps.append((submap_id, score))
        scored_submaps.sort(key=lambda item: item[1], reverse=True)

        candidates: List[Tuple[int, int, float]] = []
        for submap_id, _ in scored_submaps[:top_submaps]:
            scored_keyframes: List[Tuple[int, float]] = []
            for global_kf_id in sorted(self.submap_keyframes(submap_id)):
                record = self.records.get(global_kf_id)
                if record is None or active_submap_id in record.submap_ids:
                    continue
                score = float(torch.dot(query_descriptor, record.descriptor))
                scored_keyframes.append((global_kf_id, score))
            scored_keyframes.sort(key=lambda item: item[1], reverse=True)
            for global_kf_id, score in scored_keyframes[:top_keyframes]:
                candidates.append((submap_id, global_kf_id, score))
        candidates.sort(key=lambda item: item[2], reverse=True)
        return candidates


class GlobalSim3Graph:
    def __init__(self, cfg: Dict[str, object], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.nodes: Dict[int, torch.Tensor] = {}
        self.node_sim3: Dict[int, Sim3Transform] = {}
        self.edges: Dict[Tuple[int, int, str], Sim3Edge] = {}
        self.altitude_cfg = dict(cfg.get("altitude", {}))
        self.altitude_anchors: Dict[int, float] = {}

    def upsert_node(self, global_kf_id: int, c2w: torch.Tensor, scale_hint: Optional[float] = None) -> None:
        global_kf_id = int(global_kf_id)
        c2w_cpu = _clone_cpu(c2w)
        self.nodes[global_kf_id] = c2w_cpu

        matrix_np = c2w_cpu.numpy()
        if global_kf_id in self.node_sim3:
            previous = self.node_sim3[global_kf_id]
            self.node_sim3[global_kf_id] = Sim3Transform(
                R=matrix_np[:3, :3].astype(np.float64),
                t=matrix_np[:3, 3].astype(np.float64),
                s=float(previous.s),
            )
            return

        self.node_sim3[global_kf_id] = Sim3Transform(
            R=matrix_np[:3, :3].astype(np.float64),
            t=matrix_np[:3, 3].astype(np.float64),
            s=float(1.0 if scale_hint is None else scale_hint),
        )

    def upsert_edge(
        self,
        src: int,
        dst: int,
        edge_type: str,
        relative_sim3: Sim3Transform,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        if src == dst:
            return
        key = (int(src), int(dst), edge_type)
        self.edges[key] = Sim3Edge(
            src=int(src),
            dst=int(dst),
            edge_type=edge_type,
            relative_sim3=Sim3Transform(
                R=np.asarray(relative_sim3.R, dtype=np.float64).copy(),
                t=np.asarray(relative_sim3.t, dtype=np.float64).copy(),
                s=float(relative_sim3.s),
            ),
            metadata=metadata or {},
        )

    def set_altitude_anchor(self, global_kf_id: int, anchor_z: float) -> None:
        self.altitude_anchors[int(global_kf_id)] = float(anchor_z)

    def _altitude_enabled(self) -> bool:
        return bool(
            self.altitude_cfg.get("enabled", False)
            and float(self.altitude_cfg.get("lambda_alt", 0.0)) > 0.0
        )

    def _edge_noise(self, edge_type: str) -> gtsam.noiseModel.Base:
        pose_graph_cfg = self.cfg.get("pose_graph", {})
        sigma_map = {
            "adjacency": pose_graph_cfg.get("adjacency_sigma", [0.08, 0.08, 0.08, 0.5, 0.5, 0.5, 0.02]),
            "overlap": pose_graph_cfg.get("overlap_sigma", [0.04, 0.04, 0.04, 0.25, 0.25, 0.25, 0.01]),
            "seam": pose_graph_cfg.get("seam_sigma", [0.05, 0.05, 0.05, 0.35, 0.35, 0.35, 0.04]),
            "loop": pose_graph_cfg.get("loop_sigma", [0.03, 0.03, 0.03, 0.2, 0.2, 0.2, 0.08]),
        }
        base = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sigma_map[edge_type], dtype=np.float64))
        if edge_type in {"loop", "seam"}:
            return gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber(1.345),
                base,
            )
        return base

    @staticmethod
    def _between_error(
        sim3_src_vec: np.ndarray,
        sim3_dst_vec: np.ndarray,
        measurement: Sim3Transform,
    ) -> np.ndarray:
        sim3_src = sim3_exp(sim3_src_vec)
        sim3_dst = sim3_exp(sim3_dst_vec)
        predicted = sim3_compose(sim3_inverse(sim3_src), sim3_dst)
        residual = sim3_compose(sim3_inverse(measurement), predicted)
        return sim3_log(residual).astype(np.float64)

    def _make_between_factor(self, edge: Sim3Edge) -> gtsam.CustomFactor:
        noise = self._edge_noise(edge.edge_type)
        measurement = edge.relative_sim3

        def error_func(this: gtsam.CustomFactor, values: gtsam.Values, H) -> np.ndarray:
            sim3_src_vec = values.atVector(this.keys()[0]).reshape(-1)
            sim3_dst_vec = values.atVector(this.keys()[1]).reshape(-1)
            error = self._between_error(sim3_src_vec, sim3_dst_vec, measurement)

            if H is not None:
                H[0] = numerical_jacobian(
                    lambda x: self._between_error(x, sim3_dst_vec, measurement),
                    sim3_src_vec,
                )
                H[1] = numerical_jacobian(
                    lambda x: self._between_error(sim3_src_vec, x, measurement),
                    sim3_dst_vec,
                )
            return error

        keys = gtsam.KeyVector()
        keys.append(edge.src)
        keys.append(edge.dst)
        return gtsam.CustomFactor(noise, keys, error_func)

    def _altitude_noise(self) -> gtsam.noiseModel.Base:
        lambda_alt = float(self.altitude_cfg.get("lambda_alt", 1.0))
        sigma = float(self.altitude_cfg.get("sigma", math.sqrt(1.0 / max(lambda_alt, 1e-12))))
        base = gtsam.noiseModel.Diagonal.Sigmas(np.asarray([sigma], dtype=np.float64))
        huber_delta = float(self.altitude_cfg.get("huber_delta", 0.0))
        if huber_delta > 0.0:
            return gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber(huber_delta),
                base,
            )
        return base

    @staticmethod
    def _altitude_error(sim3_vec: np.ndarray, anchor_z: float, axis: int) -> np.ndarray:
        sim3_state = sim3_exp(sim3_vec)
        return np.asarray([float(sim3_state.t[int(axis)] - anchor_z)], dtype=np.float64)

    def _make_altitude_factor(self, global_kf_id: int, anchor_z: float) -> gtsam.CustomFactor:
        noise = self._altitude_noise()
        axis = int(self.altitude_cfg.get("axis", 2))

        def error_func(this: gtsam.CustomFactor, values: gtsam.Values, H) -> np.ndarray:
            sim3_vec = values.atVector(this.keys()[0]).reshape(-1)
            error = self._altitude_error(sim3_vec, anchor_z, axis)
            if H is not None:
                H[0] = numerical_jacobian(
                    lambda x: self._altitude_error(x, anchor_z, axis),
                    sim3_vec,
                )
            return error

        keys = gtsam.KeyVector()
        keys.append(int(global_kf_id))
        return gtsam.CustomFactor(noise, keys, error_func)

    def optimize(self) -> Dict[int, Sim3Transform]:
        if len(self.node_sim3) < 2 or not self.edges:
            return {
                key: Sim3Transform(
                    R=value.R.copy(),
                    t=value.t.copy(),
                    s=float(value.s),
                )
                for key, value in self.node_sim3.items()
            }

        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        anchor_id = min(self.node_sim3.keys())
        anchor_sim3 = sim3_log(self.node_sim3[anchor_id]).astype(np.float64)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray([1e-6] * 7, dtype=np.float64))
        graph.add(gtsam.PriorFactorVector(anchor_id, anchor_sim3, prior_noise))

        for global_kf_id in sorted(self.node_sim3.keys()):
            initial_estimate.insert(global_kf_id, sim3_log(self.node_sim3[global_kf_id]).astype(np.float64))

        for edge in self.edges.values():
            graph.add(self._make_between_factor(edge))

        if self._altitude_enabled():
            for global_kf_id, anchor_z in sorted(self.altitude_anchors.items()):
                if global_kf_id in self.node_sim3:
                    graph.add(self._make_altitude_factor(global_kf_id, anchor_z))

        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR")
        result = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params).optimize()

        optimized: Dict[int, Sim3Transform] = {}
        for global_kf_id in sorted(self.node_sim3.keys()):
            optimized[global_kf_id] = sim3_exp(result.atVector(global_kf_id).reshape(-1))
        return optimized


class SubmapManager:
    def __init__(
        self,
        cfg: Dict[str, object],
        mapper: GaussianModel,
        storage_manager: Optional[StorageManager] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg["device"]["mapper"])
        raw_submap_cfg = copy.deepcopy(cfg.get("submap", {}))
        raw_submap_altitude_cfg = dict(raw_submap_cfg.get("altitude", {}))
        self.submap_cfg = self._with_defaults(raw_submap_cfg)
        top_altitude_cfg = dict(cfg.get("constant_altitude", {}))
        if top_altitude_cfg:
            if raw_submap_altitude_cfg:
                for key, value in top_altitude_cfg.items():
                    if key not in raw_submap_altitude_cfg:
                        self.submap_cfg["altitude"][key] = value
            else:
                self.submap_cfg["altitude"].update(top_altitude_cfg)
        looper_cfg = cfg.get("looper", {})
        if "validation_error_threshold" not in cfg.get("submap", {}):
            self.submap_cfg["validation_error_threshold"] = looper_cfg.get("is_loop_mse_threshold", self.submap_cfg["validation_error_threshold"])
        if "validation_min_matches" not in cfg.get("submap", {}):
            self.submap_cfg["validation_min_matches"] = looper_cfg.get("is_loop_min_match_num", self.submap_cfg["validation_min_matches"])
        self.mapper = mapper
        self.storage_manager = storage_manager
        self.storage_interval = int(self.submap_cfg.get("storage_interval", 10))
        self.use_storage_manager = self.storage_manager is not None

        self.db = GlobalKeyframeDatabase(self.submap_cfg)
        self.pose_graph = GlobalSim3Graph(self.submap_cfg, self.device)
        self.loop_detector = LoopDetector(cfg)

        self.submap_dir = os.path.join(cfg["output"]["save_dir"], self.submap_cfg.get("snapshot_subdir", "submaps"))
        os.makedirs(self.submap_dir, exist_ok=True)

        self.submaps: Dict[int, SubmapRecord] = {}
        self.active_submap_id = 0
        self._next_submap_id = 1
        self.submaps[self.active_submap_id] = SubmapRecord(submap_id=self.active_submap_id)

        self._cached_renderer_submap_id: Optional[int] = None
        self._cached_renderer: Optional[GaussianModel] = None
        self._last_intrinsic: Optional[Dict[str, float]] = None

    @staticmethod
    def _with_defaults(submap_cfg: Dict[str, object]) -> Dict[str, object]:
        submap_cfg.setdefault("enabled", False)
        submap_cfg.setdefault("descriptor_size", 12)
        submap_cfg.setdefault("max_keyframes", 60)
        submap_cfg.setdefault("max_translation", 80.0)
        submap_cfg.setdefault("overlap_keyframes", 4)
        submap_cfg.setdefault("loop_every", 3)
        submap_cfg.setdefault("loop_min_separation", 20)
        submap_cfg.setdefault("retrieval_top_submaps", 3)
        submap_cfg.setdefault("retrieval_top_keyframes", 2)
        submap_cfg.setdefault("validation_accum_threshold", 0.85)
        submap_cfg.setdefault("validation_error_threshold", 0.16)
        submap_cfg.setdefault("validation_min_matches", 40)
        submap_cfg.setdefault("pose_jump_threshold", 25.0)
        submap_cfg.setdefault("seam_bridge_enabled", False)
        submap_cfg.setdefault("seam_bridge_every", 12)
        submap_cfg.setdefault("seam_bridge_ref_keyframes", 8)
        submap_cfg.setdefault("seam_bridge_min_matches", 80)
        submap_cfg.setdefault("seam_bridge_min_inliers", 60)
        submap_cfg.setdefault("seam_bridge_min_scale", 0.8)
        submap_cfg.setdefault("seam_bridge_max_scale", 1.25)
        submap_cfg.setdefault("seam_bridge_max_translation", 60.0)
        submap_cfg.setdefault("bootstrap_min_valid_pixels", 256)
        submap_cfg.setdefault("storage_interval", 10)
        submap_cfg.setdefault("refine_affected_submaps", False)
        submap_cfg.setdefault("refine_iters", 10)
        submap_cfg.setdefault("snapshot_subdir", "submaps")
        submap_cfg.setdefault("pose_graph", {})
        submap_cfg.setdefault("altitude", {})
        altitude_cfg = submap_cfg["altitude"]
        altitude_cfg.setdefault("enabled", False)
        altitude_cfg.setdefault("lambda_alt", 0.0)
        altitude_cfg.setdefault("axis", 2)
        altitude_cfg.setdefault("huber_delta", 1.0)
        altitude_cfg.setdefault("anchor_mode", "submap_first")
        return submap_cfg

    @property
    def active_mapper(self) -> GaussianModel:
        return self.mapper

    @property
    def active_storage_manager(self) -> Optional[StorageManager]:
        return self.storage_manager

    def process(self, tracker, viz_out: Dict[str, torch.Tensor], frame_idx: int) -> None:
        self._last_intrinsic = _intrinsic_to_cpu_dict(viz_out["intrinsic"])
        self._register_keyframes(viz_out)
        altitude_anchor_z = self.submaps[self.active_submap_id].altitude_anchor_z
        if altitude_anchor_z is not None:
            viz_out["altitude_anchor_z"] = torch.tensor(
                altitude_anchor_z,
                dtype=torch.float32,
                device=self.device,
            )
        self.mapper.run(viz_out, True)

        if (
            self.use_storage_manager
            and self.mapper.initialized_state
            and (frame_idx + 1) % self.storage_interval == 0
        ):
            self.storage_manager.run(tracker, self.mapper, viz_out)
            torch.cuda.empty_cache()

        self._maybe_run_seam_bridge(tracker, viz_out)
        self._maybe_run_loop(tracker, viz_out)
        self._maybe_split_active_submap(tracker)

    def build_export_storage_view(self) -> Optional[SimpleNamespace]:
        merged: Dict[str, List[torch.Tensor]] = {
            "_xyz": [],
            "_rgb": [],
            "_scaling": [],
            "_rotation": [],
            "_opacity": [],
            "_globalkf_id": [],
        }

        for submap in self.submaps.values():
            if not submap.frozen or submap.snapshot_path is None:
                continue
            state = torch.load(submap.snapshot_path, map_location="cpu")
            state = self._filter_state_by_owner(state, submap.submap_id)
            for key in merged.keys():
                tensor = state.get(key)
                if tensor is not None and tensor.numel() > 0:
                    merged[key].append(tensor)

        if self.use_storage_manager and self.storage_manager is not None:
            if self.storage_manager._xyz.numel() > 0:
                filtered_storage = self._filter_storage_by_owner(self.storage_manager, self.active_submap_id)
                if filtered_storage is not None:
                    merged["_xyz"].append(filtered_storage._xyz)
                    merged["_rgb"].append(filtered_storage._rgb)
                    merged["_scaling"].append(filtered_storage._scaling)
                    merged["_rotation"].append(filtered_storage._rotation)
                    merged["_opacity"].append(filtered_storage._opacity)
                    merged["_globalkf_id"].append(filtered_storage._globalkf_id)

        if not any(values for values in merged.values()):
            return None

        export_state = {}
        for key, values in merged.items():
            if values:
                export_state[key] = torch.cat(values, dim=0)
            else:
                export_state[key] = torch.empty(0, dtype=torch.long if key == "_globalkf_id" else torch.float32)
        return SimpleNamespace(**export_state)

    def export_submap_plys(self, idx: int, save_mode: str = "2dgs") -> None:
        for submap in self.submaps.values():
            if submap.frozen and submap.snapshot_path is not None:
                state = torch.load(submap.snapshot_path, map_location="cpu")
                state = self._filter_state_by_owner(state, submap.submap_id)
                self._save_state_as_ply(state, idx, submap.submap_id, save_mode)

        active_state = self._filter_state_by_owner(
            {
                "_xyz": _clone_cpu(self.mapper._xyz),
                "_rgb": _clone_cpu(self.mapper._rgb),
                "_scaling": _clone_cpu(self.mapper._scaling),
                "_rotation": _clone_cpu(self.mapper._rotation),
                "_opacity": _clone_cpu(self.mapper._opacity),
                "_globalkf_id": _clone_cpu(self.mapper._globalkf_id),
                "_globalkf_max_scores": _clone_cpu(self.mapper._globalkf_max_scores),
                "_birth_globalkf_id": _clone_cpu(self.mapper._birth_globalkf_id),
            },
            self.active_submap_id,
        )
        active_storage = None
        if self.use_storage_manager and self.storage_manager is not None and self.storage_manager._xyz.numel() > 0:
            active_storage = self._filter_storage_by_owner(self.storage_manager, self.active_submap_id)
        self._save_state_as_ply(active_state, idx, self.active_submap_id, save_mode, storage_manager=active_storage)

    def _save_state_as_ply(
        self,
        state: Dict[str, torch.Tensor],
        idx: int,
        submap_id: int,
        save_mode: str,
        storage_manager: Optional[SimpleNamespace] = None,
    ) -> None:
        if state.get("_xyz") is None or state["_xyz"].numel() == 0:
            if storage_manager is None or storage_manager._xyz.numel() == 0:
                return
        export_model = SimpleNamespace(
            cfg=self.cfg,
            tfer=self.mapper.tfer,
            _xyz=state.get("_xyz", torch.empty(0, 3)),
            _rgb=state.get("_rgb", torch.empty(0, 3)),
            _scaling=state.get("_scaling", torch.empty(0, 2)),
            _rotation=state.get("_rotation", torch.empty(0, 4)),
            _opacity=state.get("_opacity", torch.empty(0, 1)),
        )
        save_ply(
            export_model,
            idx,
            save_mode=save_mode,
            storage_manager=storage_manager,
            filename=f"idx={idx}_submap_{submap_id:04d}_{save_mode}.ply",
        )

    def _owner_submap_id(self, global_kf_id: int, fallback_submap_id: int) -> int:
        record = self.db.records.get(int(global_kf_id))
        if record is None:
            return fallback_submap_id
        return int(record.owner_submap_id)

    def _resolve_owner_ids(
        self,
        globalkf_id: Optional[torch.Tensor],
        birth_globalkf_id: Optional[torch.Tensor] = None,
        globalkf_max_scores: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if globalkf_id is None or globalkf_id.numel() == 0:
            return globalkf_id

        resolved_ids = globalkf_id.detach().cpu().to(torch.long).clone()
        if birth_globalkf_id is None or birth_globalkf_id.numel() == 0:
            return resolved_ids

        birth_ids = birth_globalkf_id.detach().cpu().to(torch.long).reshape(-1)
        if birth_ids.shape[0] != resolved_ids.shape[0]:
            return resolved_ids

        unresolved_mask = torch.zeros_like(resolved_ids, dtype=torch.bool)
        if globalkf_max_scores is not None and globalkf_max_scores.numel() == resolved_ids.shape[0]:
            unresolved_mask |= globalkf_max_scores.detach().cpu().reshape(-1) <= 0

        missing_mask = torch.tensor(
            [int(global_kf_id) not in self.db.records for global_kf_id in resolved_ids.tolist()],
            dtype=torch.bool,
        )
        unresolved_mask |= missing_mask
        resolved_ids[unresolved_mask] = birth_ids[unresolved_mask]
        return resolved_ids

    def _filter_state_by_owner(self, state: Dict[str, torch.Tensor], owner_submap_id: int) -> Dict[str, torch.Tensor]:
        resolved_ids = self._resolve_owner_ids(
            state.get("_globalkf_id"),
            state.get("_birth_globalkf_id"),
            state.get("_globalkf_max_scores"),
        )
        if resolved_ids is None or resolved_ids.numel() == 0:
            return state
        mask = torch.tensor(
            [self._owner_submap_id(int(global_kf_id), owner_submap_id) == owner_submap_id for global_kf_id in resolved_ids.tolist()],
            dtype=torch.bool,
        )
        if mask.all():
            return state
        filtered = {}
        for key, value in state.items():
            if torch.is_tensor(value) and value.shape[0] == mask.shape[0]:
                filtered[key] = value[mask]
            else:
                filtered[key] = value
        return filtered

    def _filter_storage_by_owner(self, storage_manager: StorageManager, owner_submap_id: int) -> Optional[SimpleNamespace]:
        resolved_ids = self._resolve_owner_ids(
            storage_manager._globalkf_id,
            getattr(storage_manager, "_birth_globalkf_id", None),
            storage_manager._globalkf_max_scores,
        )
        if resolved_ids is None or resolved_ids.numel() == 0:
            return None
        mask = torch.tensor(
            [self._owner_submap_id(int(global_kf_id), owner_submap_id) == owner_submap_id for global_kf_id in resolved_ids.tolist()],
            dtype=torch.bool,
        )
        if mask.sum().item() == 0:
            return None
        return SimpleNamespace(
            _xyz=_clone_cpu(storage_manager._xyz[mask]),
            _rgb=_clone_cpu(storage_manager._rgb[mask]),
            _scaling=_clone_cpu(storage_manager._scaling[mask]),
            _rotation=_clone_cpu(storage_manager._rotation[mask]),
            _opacity=_clone_cpu(storage_manager._opacity[mask]),
            _globalkf_id=_clone_cpu(storage_manager._globalkf_id[mask]),
        )

    def _register_keyframes(self, viz_out: Dict[str, torch.Tensor]) -> None:
        current_submap = self.submaps[self.active_submap_id]
        images = viz_out["images"].detach().cpu()
        depths = viz_out["depths"].detach().cpu()
        poses = viz_out["poses"].detach().cpu()
        timestamps = viz_out["viz_out_idx_to_f_idx"].detach().cpu().tolist()
        global_kf_ids = viz_out["global_kf_id"].detach().cpu().tolist()
        altitude_cfg = self.submap_cfg.get("altitude", {})
        if bool(altitude_cfg.get("enabled", False)) and current_submap.altitude_anchor_z is None and len(global_kf_ids) > 0:
            axis = int(altitude_cfg.get("axis", 2))
            if altitude_cfg.get("fixed_z", None) is not None:
                current_submap.altitude_anchor_z = float(altitude_cfg["fixed_z"])
            elif str(altitude_cfg.get("anchor_mode", "submap_first")) in {"global_first", "world_first"} and self.pose_graph.altitude_anchors:
                first_anchor_id = min(self.pose_graph.altitude_anchors.keys())
                current_submap.altitude_anchor_z = float(self.pose_graph.altitude_anchors[first_anchor_id])
            elif str(altitude_cfg.get("anchor_mode", "submap_first")) in {"mean", "submap_mean", "batch_mean"}:
                current_submap.altitude_anchor_z = float(poses[:, axis, 3].mean().item())
            else:
                current_submap.altitude_anchor_z = float(poses[0, axis, 3].item())

        for idx, global_kf_id in enumerate(global_kf_ids):
            global_kf_id = int(global_kf_id)
            scale_hint = None
            if global_kf_id in self.pose_graph.node_sim3:
                scale_hint = self.pose_graph.node_sim3[global_kf_id].s
            elif idx > 0:
                prev_scale = self.pose_graph.node_sim3.get(int(global_kf_ids[idx - 1]))
                if prev_scale is not None:
                    scale_hint = prev_scale.s
            self.pose_graph.upsert_node(global_kf_id, poses[idx], scale_hint=scale_hint)
            self.db.add_membership(
                self.active_submap_id,
                global_kf_id,
                float(timestamps[idx]),
                poses[idx],
                images[idx],
                depths[idx],
                viz_out["intrinsic"],
            )
            if global_kf_id not in current_submap.keyframe_ids:
                current_submap.keyframe_ids.append(global_kf_id)
            if current_submap.altitude_anchor_z is not None:
                self.pose_graph.set_altitude_anchor(global_kf_id, current_submap.altitude_anchor_z)

        for idx in range(1, len(global_kf_ids)):
            src = int(global_kf_ids[idx - 1])
            dst = int(global_kf_ids[idx])
            self.pose_graph.upsert_edge(
                src,
                dst,
                "adjacency",
                pose3_matrix_to_sim3(_relative_pose(poses[idx - 1], poses[idx]).numpy()),
            )

        current_submap.descriptor = self._compute_submap_descriptor(current_submap.submap_id)

    def _compute_submap_descriptor(self, submap_id: int) -> Optional[torch.Tensor]:
        descriptors = []
        for global_kf_id in sorted(self.db.submap_keyframes(submap_id)):
            record = self.db.records.get(global_kf_id)
            if record is not None:
                descriptors.append(record.descriptor)
        if not descriptors:
            return None
        descriptor = torch.stack(descriptors, dim=0).mean(dim=0)
        return F.normalize(descriptor.unsqueeze(0), dim=1).squeeze(0)

    def _maybe_run_loop(self, tracker, viz_out: Dict[str, torch.Tensor]) -> None:
        if len(self.submaps) <= 1:
            return

        query_kf_id = int(viz_out["global_kf_id"][-1].item())
        if query_kf_id % int(self.submap_cfg["loop_every"]) != 0:
            return

        frozen_submaps = {sid: submap for sid, submap in self.submaps.items() if submap.frozen}
        query_image = viz_out["images"][-1].detach().cpu()
        candidates = self.db.query_candidates(query_image, self.active_submap_id, frozen_submaps)
        if not candidates:
            return

        for candidate_submap_id, ref_kf_id, retrieval_score in candidates:
            if abs(query_kf_id - ref_kf_id) < int(self.submap_cfg["loop_min_separation"]):
                continue
            validation = self._validate_loop_candidate(viz_out, query_kf_id, ref_kf_id, candidate_submap_id)
            self.db.update_loop_metadata(query_kf_id, ref_kf_id, validation["accepted"], validation["score"])
            if not validation["accepted"]:
                continue

            self.pose_graph.upsert_edge(
                ref_kf_id,
                query_kf_id,
                "loop",
                validation["loop_relative_sim3"],
                {
                    "retrieval_score": float(retrieval_score),
                    "validation_score": float(validation["score"]),
                    "match_count": int(validation["match_count"]),
                    "candidate_submap_id": int(candidate_submap_id),
                    "scale": float(validation["loop_relative_sim3"].s),
                },
            )
            frozen_submaps[candidate_submap_id].loop_edges.append((ref_kf_id, query_kf_id))
            print(
                f"Accepted cross-submap loop: ref_kf={ref_kf_id} "
                f"query_kf={query_kf_id} submap={candidate_submap_id} "
                f"score={validation['score']:.4f} matches={validation['match_count']} "
                f"scale={validation['loop_relative_sim3'].s:.4f}"
            )
            self._optimize_global_graph_and_correct(tracker)
            break

    @staticmethod
    def _record_image_chw(record: KeyframeRecord, device: torch.device) -> torch.Tensor:
        image = record.image_hw3.to(device)
        if image.dtype == torch.uint8:
            image = image.to(torch.float32) / 255.0
        else:
            image = image.to(torch.float32)
        return image.permute(2, 0, 1).contiguous()

    @staticmethod
    def _record_depth_chw(record: KeyframeRecord, device: torch.device) -> torch.Tensor:
        depth = record.depth_hw.to(device, dtype=torch.float32)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        elif depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth.permute(2, 0, 1)
        return depth.contiguous()

    def _maybe_run_seam_bridge(self, tracker, viz_out: Dict[str, torch.Tensor]) -> None:
        if not bool(self.submap_cfg.get("seam_bridge_enabled", True)):
            return
        if self.active_submap_id == 0 or len(self.submaps) <= 1:
            return

        query_kf_id = int(viz_out["global_kf_id"][-1].item())
        active_submap = self.submaps[self.active_submap_id]
        if query_kf_id in set(active_submap.overlap_kf_ids):
            return
        every = max(1, int(self.submap_cfg.get("seam_bridge_every", 1)))
        if query_kf_id % every != 0:
            return
        if any(edge[1] == query_kf_id for edge in active_submap.loop_edges):
            return

        frozen_submap_ids = [
            submap_id
            for submap_id, submap in self.submaps.items()
            if submap.frozen and submap.keyframe_ids and submap_id != self.active_submap_id
        ]
        if not frozen_submap_ids:
            return
        ref_submap_id = max(frozen_submap_ids)
        ref_submap = self.submaps[ref_submap_id]
        ref_count = max(1, int(self.submap_cfg.get("seam_bridge_ref_keyframes", 16)))
        ref_candidates = list(reversed(ref_submap.keyframe_ids[-ref_count:]))

        best_validation = None
        best_ref_kf_id = None
        for ref_kf_id in ref_candidates:
            if ref_kf_id == query_kf_id:
                continue
            validation = self._validate_seam_candidate(query_kf_id, int(ref_kf_id))
            if not validation["accepted"]:
                continue
            if best_validation is None or validation["match_count"] > best_validation["match_count"]:
                best_validation = validation
                best_ref_kf_id = int(ref_kf_id)

        if best_validation is None or best_ref_kf_id is None:
            return

        self.pose_graph.upsert_edge(
            best_ref_kf_id,
            query_kf_id,
            "seam",
            best_validation["relative_sim3"],
            {
                "match_count": int(best_validation["match_count"]),
                "scale": float(best_validation["relative_sim3"].s),
                "ref_submap_id": int(ref_submap_id),
                "query_submap_id": int(self.active_submap_id),
            },
        )
        active_submap.loop_edges.append((best_ref_kf_id, query_kf_id))
        print(
            f"Accepted cross-submap seam bridge: ref_kf={best_ref_kf_id} "
            f"query_kf={query_kf_id} ref_submap={ref_submap_id} "
            f"matches={best_validation['match_count']} "
            f"scale={best_validation['relative_sim3'].s:.4f}"
        )
        self._optimize_global_graph_and_correct(tracker)

    def _validate_seam_candidate(self, query_kf_id: int, ref_kf_id: int) -> Dict[str, object]:
        query_record = self.db.records.get(int(query_kf_id))
        ref_record = self.db.records.get(int(ref_kf_id))
        if query_record is None or ref_record is None:
            return {"accepted": False, "match_count": 0}

        query_image = self._record_image_chw(query_record, self.device)
        ref_image = self._record_image_chw(ref_record, self.device)
        query_depth = self._record_depth_chw(query_record, self.device)
        ref_depth = self._record_depth_chw(ref_record, self.device)

        vu_ref, vu_query = self.loop_detector.get_matches(ref_image, query_image)
        min_matches = int(self.submap_cfg.get("seam_bridge_min_matches", 25))
        if len(vu_ref) < min_matches:
            return {"accepted": False, "match_count": len(vu_ref)}

        vu_ref_tensor = torch.tensor(vu_ref, device=self.device, dtype=torch.long)
        vu_query_tensor = torch.tensor(vu_query, device=self.device, dtype=torch.long)
        previous_intrinsic = {
            "fu": float(self.mapper.tfer.fu),
            "fv": float(self.mapper.tfer.fv),
            "cu": float(self.mapper.tfer.cu),
            "cv": float(self.mapper.tfer.cv),
            "H": int(self.mapper.tfer.H),
            "W": int(self.mapper.tfer.W),
        }
        self._apply_intrinsic(self.mapper, ref_record.intrinsic)
        try:
            sim3_result = self.loop_detector.get_sim3(
                ref_depth,
                query_depth,
                vu_ref_tensor,
                vu_query_tensor,
                self.mapper.tfer,
            )
        finally:
            self._apply_intrinsic(self.mapper, previous_intrinsic)
        if sim3_result is None:
            return {"accepted": False, "match_count": len(vu_ref)}

        match_count = int(sim3_result["match_count"])
        if match_count < int(self.submap_cfg.get("seam_bridge_min_inliers", min_matches)):
            return {"accepted": False, "match_count": match_count}

        sim3_query_from_ref = Sim3Transform(
            R=np.asarray(sim3_result["R"], dtype=np.float64),
            t=np.asarray(sim3_result["t"], dtype=np.float64),
            s=float(sim3_result["s"]),
        )
        relative_sim3 = sim3_inverse(sim3_query_from_ref)
        scale = float(relative_sim3.s)
        if scale < float(self.submap_cfg.get("seam_bridge_min_scale", 0.5)):
            return {"accepted": False, "match_count": match_count}
        if scale > float(self.submap_cfg.get("seam_bridge_max_scale", 2.0)):
            return {"accepted": False, "match_count": match_count}
        if np.linalg.norm(relative_sim3.t.reshape(3)) > float(self.submap_cfg.get("seam_bridge_max_translation", 160.0)):
            return {"accepted": False, "match_count": match_count}

        return {
            "accepted": True,
            "match_count": match_count,
            "relative_sim3": relative_sim3,
        }

    def _validate_loop_candidate(
        self,
        viz_out: Dict[str, torch.Tensor],
        query_kf_id: int,
        ref_kf_id: int,
        candidate_submap_id: int,
    ) -> Dict[str, object]:
        renderer = self._preload_submap_renderer(candidate_submap_id)
        ref_record = self.db.records[ref_kf_id]
        ref_pose = ref_record.c2w.to(self.device)
        intrinsic = ref_record.intrinsic
        self._apply_intrinsic(renderer, intrinsic)

        with torch.no_grad():
            reference_render = renderer.render(torch.linalg.inv(ref_pose), intrinsic)
        rendered_ref_depth = reference_render["depth"].detach()
        ref_image = ref_record.image_hw3.to(self.device)
        if ref_image.dtype == torch.uint8:
            ref_image = ref_image.to(torch.float32) / 255.0
        else:
            ref_image = ref_image.to(torch.float32)
        ref_image = ref_image.permute(2, 0, 1).contiguous()
        ref_depth = ref_record.depth_hw.to(self.device)
        if ref_depth.ndim == 2:
            ref_depth = ref_depth.unsqueeze(0)
        if (
            ref_depth.numel() == 0
            or torch.isfinite(ref_depth).logical_and(ref_depth > 0).sum().item() < int(self.submap_cfg["validation_min_matches"])
        ):
            ref_depth = rendered_ref_depth
        query_image = viz_out["images"][-1].to(self.device).permute(2, 0, 1)
        query_depth = viz_out["depths"][-1].to(self.device)
        if query_depth.ndim == 3 and query_depth.shape[-1] == 1:
            query_depth = query_depth.permute(2, 0, 1)
        elif query_depth.ndim == 2:
            query_depth = query_depth.unsqueeze(0)

        vu_ref, vu_query = self.loop_detector.get_matches(ref_image, query_image)
        if len(vu_ref) < int(self.submap_cfg["validation_min_matches"]):
            return {"accepted": False, "score": 1.0, "match_count": len(vu_ref)}

        vu_ref_tensor = torch.tensor(vu_ref, device=self.device, dtype=torch.long)
        vu_query_tensor = torch.tensor(vu_query, device=self.device, dtype=torch.long)
        sim3_result = self.loop_detector.get_sim3(
            ref_depth,
            query_depth,
            vu_ref_tensor,
            vu_query_tensor,
            renderer.tfer,
        )
        if sim3_result is None:
            return {"accepted": False, "score": 1.0, "match_count": len(vu_ref)}
        sim3_query_from_ref = Sim3Transform(
            R=np.asarray(sim3_result["R"], dtype=np.float64),
            t=np.asarray(sim3_result["t"], dtype=np.float64),
            s=float(sim3_result["s"]),
        )
        sim3_ref_from_query = sim3_inverse(sim3_query_from_ref)
        sim3_world_from_ref = self.pose_graph.node_sim3.get(ref_kf_id, pose3_matrix_to_sim3(ref_pose.detach().cpu().numpy()))
        sim3_world_from_query = sim3_compose(sim3_world_from_ref, sim3_ref_from_query)
        pred_c2w_query = torch.tensor(
            sim3_to_pose3_matrix(sim3_world_from_query),
            dtype=torch.float32,
            device=self.device,
        )
        pred_w2c_query = torch.linalg.inv(pred_c2w_query)

        current_query_pose = self.pose_graph.nodes[query_kf_id].to(self.device)
        if torch.linalg.norm(pred_c2w_query[:3, 3] - current_query_pose[:3, 3]) > float(self.submap_cfg["pose_jump_threshold"]):
            return {"accepted": False, "score": 1.0, "match_count": len(vu_ref)}

        with torch.no_grad():
            pred_query = renderer.render(pred_w2c_query, intrinsic)
        valid_mask = (
            (pred_query["accum"].squeeze(0) > float(self.submap_cfg["validation_accum_threshold"]))
            & (pred_query["depth"].squeeze(0) > 0.0)
        )
        if valid_mask.sum().item() < 256:
            return {"accepted": False, "score": 1.0, "match_count": len(vu_ref)}

        query_gray = query_image.mean(dim=0)
        pred_gray = pred_query["rgb"].mean(dim=0)
        score = torch.abs(pred_gray - query_gray)[valid_mask].mean().item()
        accepted = score < float(self.submap_cfg["validation_error_threshold"])
        return {
            "accepted": accepted,
            "score": float(score),
            "match_count": int(sim3_result["match_count"]),
            "query_pose": pred_c2w_query.detach(),
            "loop_relative_sim3": sim3_ref_from_query,
        }

    def _optimize_global_graph_and_correct(self, tracker) -> None:
        old_sim3 = {
            global_kf_id: Sim3Transform(
                R=sim3_state.R.copy(),
                t=sim3_state.t.copy(),
                s=float(sim3_state.s),
            )
            for global_kf_id, sim3_state in self.pose_graph.node_sim3.items()
        }
        optimized = self.pose_graph.optimize()
        if not optimized:
            return

        affected_kf_ids = []
        optimized_pose3 = {}
        transforms: Dict[int, Sim3Transform] = {}
        for global_kf_id, new_sim3 in optimized.items():
            old_state = old_sim3.get(global_kf_id)
            if old_state is not None and _same_sim3(old_state, new_sim3):
                continue
            affected_kf_ids.append(global_kf_id)
            if old_state is None:
                delta = new_sim3
            else:
                delta = sim3_compose(new_sim3, sim3_inverse(old_state))
            transforms[global_kf_id] = delta
            optimized_pose3[global_kf_id] = torch.tensor(
                sim3_to_pose3_matrix(new_sim3),
                dtype=torch.float32,
                device=self.device,
            )

        if not affected_kf_ids:
            return

        for global_kf_id in affected_kf_ids:
            self.pose_graph.node_sim3[global_kf_id] = optimized[global_kf_id]
            self.pose_graph.nodes[global_kf_id] = optimized_pose3[global_kf_id].cpu()
            self.db.update_pose(global_kf_id, optimized_pose3[global_kf_id].cpu())

        self._update_tracker_poses(tracker, optimized_pose3)
        affected_submaps = self._apply_creator_corrections(transforms)

        if bool(self.submap_cfg["refine_affected_submaps"]):
            self._refine_affected_submaps(tracker, affected_submaps)

    def _update_tracker_poses(self, tracker, optimized: Dict[int, torch.Tensor]) -> None:
        for global_kf_id, c2w in optimized.items():
            w2c = torch.linalg.inv(c2w)
            tracker.frontend.video.poses_save[global_kf_id] = matrix_to_tq(w2c.unsqueeze(0)).cpu().squeeze(0)
            if hasattr(tracker, "local_to_global_bias"):
                local_kf_id = global_kf_id - tracker.local_to_global_bias
                if 0 <= local_kf_id < tracker.frontend.video.poses.shape[0]:
                    tracker.frontend.video.poses[local_kf_id] = matrix_to_tq(w2c.unsqueeze(0)).to(tracker.frontend.video.poses.device).squeeze(0)

    def _apply_creator_corrections(self, transforms: Dict[int, Sim3Transform]) -> Set[int]:
        affected_submaps: Set[int] = set()
        transform_keys = set(int(key) for key in transforms.keys())

        self._transform_live_gaussians(self.mapper, transforms)
        if self.storage_manager is not None:
            self._transform_storage_tier(self.storage_manager, transforms)

        for submap_id, submap in self.submaps.items():
            if submap_id == self.active_submap_id or not submap.frozen or submap.snapshot_path is None:
                continue
            if not transform_keys.intersection(self.db.submap_keyframes(submap_id)):
                continue
            state = torch.load(submap.snapshot_path, map_location="cpu")
            self._transform_snapshot_state(state, transforms)
            torch.save(state, submap.snapshot_path)
            affected_submaps.add(submap_id)

        if self._cached_renderer_submap_id in affected_submaps:
            self._drop_cached_renderer()

        if self.active_submap_id in affected_submaps or transform_keys.intersection(set(self.submaps[self.active_submap_id].keyframe_ids)):
            affected_submaps.add(self.active_submap_id)
        return affected_submaps

    def _transform_live_gaussians(self, mapper: GaussianModel, transforms: Dict[int, Sim3Transform]) -> None:
        if mapper._xyz.numel() == 0:
            return
        with torch.no_grad():
            self._transform_xyz_rotation_and_scaling(
                mapper._xyz,
                mapper._rotation,
                mapper._scaling,
                self._resolve_owner_ids(
                    mapper._globalkf_id,
                    getattr(mapper, "_birth_globalkf_id", None),
                    mapper._globalkf_max_scores,
                ),
                transforms,
                mapper.device,
            )

    def _transform_storage_tier(self, storage_manager: StorageManager, transforms: Dict[int, Sim3Transform]) -> None:
        if storage_manager._xyz.numel() == 0:
            return
        with torch.no_grad():
            self._transform_xyz_rotation_and_scaling(
                storage_manager._xyz,
                storage_manager._rotation,
                storage_manager._scaling,
                self._resolve_owner_ids(
                    storage_manager._globalkf_id,
                    getattr(storage_manager, "_birth_globalkf_id", None),
                    storage_manager._globalkf_max_scores,
                ),
                transforms,
                torch.device("cpu"),
            )

    def _transform_snapshot_state(self, state: Dict[str, torch.Tensor], transforms: Dict[int, Sim3Transform]) -> None:
        if state.get("_xyz") is None or state["_xyz"].numel() == 0:
            return
        self._transform_xyz_rotation_and_scaling(
            state["_xyz"],
            state["_rotation"],
            state["_scaling"],
            self._resolve_owner_ids(
                state.get("_globalkf_id"),
                state.get("_birth_globalkf_id"),
                state.get("_globalkf_max_scores"),
            ),
            transforms,
            torch.device("cpu"),
        )

    @staticmethod
    def _transform_xyz_rotation_and_scaling(
        xyz: torch.Tensor,
        rotation: torch.Tensor,
        scaling: torch.Tensor,
        creator_ids: torch.Tensor,
        transforms: Dict[int, Sim3Transform],
        device: torch.device,
    ) -> None:
        if creator_ids.numel() == 0:
            return
        creator_ids_cpu = creator_ids.detach().cpu()
        rotation_matrix = q2R(rotation.detach().cpu())
        new_rotation = rotation.detach().cpu().clone()
        new_xyz = xyz.detach().cpu().clone()
        new_scaling = scaling.detach().cpu().clone()

        for global_kf_id, transform in transforms.items():
            update_mask = creator_ids_cpu == int(global_kf_id)
            if not update_mask.any():
                continue
            rotation_delta = torch.tensor(transform.R, dtype=torch.float32)
            translation_delta = torch.tensor(transform.t, dtype=torch.float32)
            scale_delta = float(transform.s)

            new_xyz[update_mask] = scale_delta * (new_xyz[update_mask] @ rotation_delta.T) + translation_delta.unsqueeze(0)
            new_rotation[update_mask] = R2q(
                torch.matmul(rotation_delta.unsqueeze(0), rotation_matrix[update_mask])
            ).to(new_rotation.dtype)
            new_scaling[update_mask] = new_scaling[update_mask] + math.log(max(scale_delta, 1e-12))

        xyz.copy_(new_xyz.to(device))
        rotation.copy_(new_rotation.to(device))
        scaling.copy_(new_scaling.to(device))

    def _maybe_split_active_submap(self, tracker) -> None:
        current_submap = self.submaps[self.active_submap_id]
        if len(current_submap.keyframe_ids) < 2:
            return
        if not self._should_split_submap(current_submap):
            return
        if not self.mapper.initialized_state:
            print(
                f"Skipping submap split for submap {self.active_submap_id}: active Gaussian map is not initialized.",
                flush=True,
            )
            return

        self._freeze_active_submap()
        overlap = int(self.submap_cfg["overlap_keyframes"])
        overlap_kf_ids = current_submap.keyframe_ids[-overlap:] if overlap > 0 else []
        self._start_new_submap(tracker, overlap_kf_ids)

    def _should_split_submap(self, submap: SubmapRecord) -> bool:
        if len(submap.keyframe_ids) >= int(self.submap_cfg["max_keyframes"]):
            return True
        travel_distance = self._submap_travel_distance(submap)
        return travel_distance >= float(self.submap_cfg["max_translation"])

    def _submap_travel_distance(self, submap: SubmapRecord) -> float:
        if len(submap.keyframe_ids) < 2:
            return 0.0
        distance = 0.0
        for idx in range(1, len(submap.keyframe_ids)):
            prev_pose = self.pose_graph.nodes[submap.keyframe_ids[idx - 1]]
            curr_pose = self.pose_graph.nodes[submap.keyframe_ids[idx]]
            distance += torch.linalg.norm(curr_pose[:3, 3] - prev_pose[:3, 3]).item()
        return float(distance)

    def _freeze_active_submap(self) -> None:
        current_submap = self.submaps[self.active_submap_id]
        snapshot_state = self._snapshot_active_state()
        snapshot_path = os.path.join(self.submap_dir, f"submap_{self.active_submap_id:04d}.pt")
        torch.save(snapshot_state, snapshot_path)
        current_submap.snapshot_path = snapshot_path
        current_submap.frozen = True
        current_submap.descriptor = self._compute_submap_descriptor(self.active_submap_id)
        print(
            f"Froze submap {self.active_submap_id} with "
            f"{len(current_submap.keyframe_ids)} keyframes and "
            f"{snapshot_state['_xyz'].shape[0]} gaussians."
        )
        self._drop_cached_renderer()

    def _snapshot_active_state(self) -> Dict[str, torch.Tensor]:
        snapshot = {
            "_xyz": _merge_attr(self.mapper._xyz, self.storage_manager._xyz if self.storage_manager is not None else None),
            "_rgb": _merge_attr(self.mapper._rgb, self.storage_manager._rgb if self.storage_manager is not None else None),
            "_scaling": _merge_attr(self.mapper._scaling, self.storage_manager._scaling if self.storage_manager is not None else None),
            "_rotation": _merge_attr(self.mapper._rotation, self.storage_manager._rotation if self.storage_manager is not None else None),
            "_opacity": _merge_attr(self.mapper._opacity, self.storage_manager._opacity if self.storage_manager is not None else None),
            "_global_scores": _merge_attr(
                self.mapper._global_scores,
                self.storage_manager._global_scores if self.storage_manager is not None else None,
            ),
            "_local_scores": _merge_attr(
                self.mapper._local_scores,
                self.storage_manager._local_scores if self.storage_manager is not None else None,
            ),
            "_stable_mask": _merge_attr(
                self.mapper._stable_mask.to(torch.float32),
                self.storage_manager._stable_mask.to(torch.float32) if self.storage_manager is not None else None,
            ).to(torch.bool),
            "_globalkf_id": _merge_attr(
                self.mapper._globalkf_id,
                self.storage_manager._globalkf_id if self.storage_manager is not None else None,
            ).to(torch.long),
            "_globalkf_max_scores": _merge_attr(
                self.mapper._globalkf_max_scores,
                self.storage_manager._globalkf_max_scores if self.storage_manager is not None else None,
            ),
            "_birth_globalkf_id": _merge_attr(
                self.mapper._birth_globalkf_id,
                self.storage_manager._birth_globalkf_id if self.storage_manager is not None else None,
            ).to(torch.long),
        }
        return snapshot

    def _start_new_submap(self, tracker, overlap_kf_ids: Sequence[int]) -> None:
        new_submap_id = self._next_submap_id
        self._next_submap_id += 1
        self.active_submap_id = new_submap_id
        altitude_anchor_z = self._new_submap_altitude_anchor(overlap_kf_ids)
        self.submaps[new_submap_id] = SubmapRecord(
            submap_id=new_submap_id,
            keyframe_ids=list(overlap_kf_ids),
            overlap_kf_ids=list(overlap_kf_ids),
            altitude_anchor_z=altitude_anchor_z,
        )
        print(f"Starting submap {new_submap_id} from overlap keyframes {list(overlap_kf_ids)}.")
        for global_kf_id in overlap_kf_ids:
            self.db.add_existing_membership(new_submap_id, int(global_kf_id))
            if altitude_anchor_z is not None:
                self.pose_graph.set_altitude_anchor(int(global_kf_id), altitude_anchor_z)

        if len(overlap_kf_ids) >= 2:
            first_pose = self.pose_graph.nodes[int(overlap_kf_ids[0])]
            last_pose = self.pose_graph.nodes[int(overlap_kf_ids[-1])]
            self.pose_graph.upsert_edge(
                int(overlap_kf_ids[0]),
                int(overlap_kf_ids[-1]),
                "overlap",
                pose3_matrix_to_sim3(_relative_pose(first_pose, last_pose).numpy()),
            )

        mapper_cfg = copy.deepcopy(self.cfg)
        mapper_cfg["use_wandb"] = False
        self.mapper = GaussianModel(mapper_cfg)
        if self.use_storage_manager:
            self.storage_manager = StorageManager(self.cfg)
            if hasattr(tracker.frontend.video, "tstamp_save"):
                self.storage_manager.dataset_length = tracker.frontend.video.tstamp_save[-1] - tracker.frontend.video.tstamp_save[0]

        if overlap_kf_ids and self._last_intrinsic is not None:
            bootstrap_batch = self._build_batch_from_tracker(tracker, overlap_kf_ids)
            bootstrap_batch = self._filter_batch_by_valid_depth(
                bootstrap_batch,
                int(self.submap_cfg.get("bootstrap_min_valid_pixels", 256)),
            )
            if bootstrap_batch is None:
                print(
                    f"Skipping bootstrap for submap {new_submap_id}: overlap keyframes have no valid depth.",
                    flush=True,
                )
            else:
                bootstrap_batch["skip_viz"] = True
                self.mapper.run(bootstrap_batch, True)
        self.submaps[new_submap_id].descriptor = self._compute_submap_descriptor(new_submap_id)

    @staticmethod
    def _filter_batch_by_valid_depth(
        batch: Dict[str, torch.Tensor],
        min_valid_pixels: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        depths = batch.get("depths")
        if depths is None or not torch.is_tensor(depths) or depths.shape[0] == 0:
            return None
        valid_counts = (torch.isfinite(depths) & (depths > 0)).reshape(depths.shape[0], -1).sum(dim=1)
        keep = valid_counts >= int(min_valid_pixels)
        if not keep.any():
            return None
        filtered = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.shape[:1] == keep.shape:
                filtered[key] = value[keep]
            else:
                filtered[key] = value
        return filtered

    def _new_submap_altitude_anchor(self, overlap_kf_ids: Sequence[int]) -> Optional[float]:
        altitude_cfg = self.submap_cfg.get("altitude", {})
        if not bool(altitude_cfg.get("enabled", False)):
            return None
        if altitude_cfg.get("fixed_z", None) is not None:
            return float(altitude_cfg["fixed_z"])
        if str(altitude_cfg.get("anchor_mode", "submap_first")) in {"global_first", "world_first"} and self.pose_graph.altitude_anchors:
            return float(self.pose_graph.altitude_anchors[min(self.pose_graph.altitude_anchors.keys())])
        axis = int(altitude_cfg.get("axis", 2))
        overlap_z = [
            float(self.pose_graph.nodes[int(global_kf_id)][axis, 3].item())
            for global_kf_id in overlap_kf_ids
            if int(global_kf_id) in self.pose_graph.nodes
        ]
        if not overlap_z:
            return None
        if str(altitude_cfg.get("anchor_mode", "submap_first")) in {"mean", "submap_mean", "batch_mean"}:
            return float(np.mean(overlap_z))
        return float(overlap_z[0])

    def _altitude_anchor_for_keyframes(self, global_kf_ids: Sequence[int]) -> Optional[float]:
        altitude_cfg = self.submap_cfg.get("altitude", {})
        if not bool(altitude_cfg.get("enabled", False)):
            return None
        ids = set(int(global_kf_id) for global_kf_id in global_kf_ids)
        best_anchor = None
        best_overlap = -1
        for submap in self.submaps.values():
            if submap.altitude_anchor_z is None:
                continue
            overlap = len(ids.intersection(set(submap.keyframe_ids)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_anchor = float(submap.altitude_anchor_z)
        return best_anchor if best_overlap > 0 else None

    def _build_batch_from_tracker(self, tracker, global_kf_ids: Sequence[int]) -> Dict[str, torch.Tensor]:
        ids = torch.tensor(list(global_kf_ids), dtype=torch.long)
        images = tracker.frontend.video.images_up_save[ids][..., [2, 1, 0]].to(self.device)
        depths = 1.0 / (tracker.frontend.video.disps_up_save[ids].to(self.device) + 1e-6)
        depths = depths.unsqueeze(-1)
        depths_cov = tracker.frontend.video.depths_cov_up_save[ids].to(self.device).unsqueeze(-1)
        poses = torch.stack([self.pose_graph.nodes[int(global_kf_id)].to(self.device) for global_kf_id in global_kf_ids], dim=0)
        timestamps = tracker.frontend.video.tstamp_save[ids].to(self.device)

        cov_median = torch.tensor(
            np.median(depths_cov.detach().cpu().numpy().reshape(depths_cov.shape[0], -1), axis=1)[:, None, None, None],
            device=self.device,
        )
        zero_mask = torch.bitwise_or(
            depths > float(self.cfg["middleware"]["max_depth"]),
            depths_cov > float(self.cfg["middleware"]["cov_times"]) * cov_median,
        )
        depths[zero_mask] = 0.0
        images[depths.squeeze(-1) == 0.0] = 0.0

        batch = {
            "images": images,
            "depths": depths,
            "depths_cov": depths_cov,
            "poses": poses,
            "viz_out_idx_to_f_idx": timestamps,
            "global_kf_id": ids.to(self.device),
            "intrinsic": self._last_intrinsic,
        }
        altitude_anchor_z = self._altitude_anchor_for_keyframes(global_kf_ids)
        if altitude_anchor_z is not None:
            batch["altitude_anchor_z"] = torch.tensor(
                altitude_anchor_z,
                dtype=torch.float32,
                device=self.device,
            )
        return batch

    def _preload_submap_renderer(self, submap_id: int) -> GaussianModel:
        if self._cached_renderer_submap_id == submap_id and self._cached_renderer is not None:
            return self._cached_renderer

        self._drop_cached_renderer()
        submap = self.submaps[submap_id]
        if submap.snapshot_path is None:
            raise FileNotFoundError(f"Frozen submap snapshot is missing for submap {submap_id}.")

        renderer_cfg = copy.deepcopy(self.cfg)
        renderer_cfg["use_wandb"] = False
        renderer = GaussianModel(renderer_cfg)
        state = torch.load(submap.snapshot_path, map_location=self.device)
        renderer._xyz = nn.Parameter(state["_xyz"].to(self.device).requires_grad_(True))
        renderer._rgb = nn.Parameter(state["_rgb"].to(self.device).requires_grad_(True))
        renderer._scaling = nn.Parameter(state["_scaling"].to(self.device).requires_grad_(True))
        renderer._rotation = nn.Parameter(state["_rotation"].to(self.device).requires_grad_(True))
        renderer._opacity = nn.Parameter(state["_opacity"].to(self.device).requires_grad_(True))
        renderer._global_scores = state["_global_scores"].to(self.device)
        renderer._local_scores = state["_local_scores"].to(self.device)
        renderer._stable_mask = state["_stable_mask"].to(self.device)
        renderer._globalkf_id = state["_globalkf_id"].to(self.device)
        renderer._globalkf_max_scores = state["_globalkf_max_scores"].to(self.device)
        renderer._birth_globalkf_id = state.get("_birth_globalkf_id", state["_globalkf_id"]).to(self.device)
        renderer.initialized_state = True
        renderer.setup_optimizer()
        renderer.history_list = list(submap.keyframe_ids)

        anchor_kf_id = submap.keyframe_ids[0]
        self._apply_intrinsic(renderer, self.db.records[anchor_kf_id].intrinsic)
        self._cached_renderer_submap_id = submap_id
        self._cached_renderer = renderer
        return renderer

    @staticmethod
    def _apply_intrinsic(mapper: GaussianModel, intrinsic: Dict[str, float]) -> None:
        mapper.tfer.fu = intrinsic["fu"]
        mapper.tfer.fv = intrinsic["fv"]
        mapper.tfer.cu = intrinsic["cu"]
        mapper.tfer.cv = intrinsic["cv"]
        mapper.tfer.H = int(intrinsic["H"])
        mapper.tfer.W = int(intrinsic["W"])

    def _drop_cached_renderer(self) -> None:
        if self._cached_renderer is not None:
            del self._cached_renderer
            self._cached_renderer = None
            self._cached_renderer_submap_id = None
            torch.cuda.empty_cache()

    def _refine_affected_submaps(self, tracker, affected_submaps: Set[int]) -> None:
        if not affected_submaps:
            return
        refine_iters = int(self.submap_cfg["refine_iters"])
        for submap_id in affected_submaps:
            submap = self.submaps[submap_id]
            if submap_id == self.active_submap_id:
                batch = self._build_batch_from_tracker(tracker, submap.keyframe_ids)
                batch["skip_viz"] = True
                self.mapper.train_once(batch, refine_iters)
                continue

            if submap.snapshot_path is None:
                continue
            renderer = self._preload_submap_renderer(submap_id)
            batch = self._build_batch_from_tracker(tracker, submap.keyframe_ids)
            batch["skip_viz"] = True
            renderer.train_once(batch, refine_iters)
            torch.save(self._snapshot_renderer_state(renderer), submap.snapshot_path)
            submap.descriptor = self._compute_submap_descriptor(submap_id)
        self._drop_cached_renderer()

    @staticmethod
    def _snapshot_renderer_state(renderer: GaussianModel) -> Dict[str, torch.Tensor]:
        return {
            "_xyz": _clone_cpu(renderer._xyz),
            "_rgb": _clone_cpu(renderer._rgb),
            "_scaling": _clone_cpu(renderer._scaling),
            "_rotation": _clone_cpu(renderer._rotation),
            "_opacity": _clone_cpu(renderer._opacity),
            "_global_scores": _clone_cpu(renderer._global_scores),
            "_local_scores": _clone_cpu(renderer._local_scores),
            "_stable_mask": _clone_cpu(renderer._stable_mask),
            "_globalkf_id": _clone_cpu(renderer._globalkf_id),
            "_globalkf_max_scores": _clone_cpu(renderer._globalkf_max_scores),
            "_birth_globalkf_id": _clone_cpu(renderer._birth_globalkf_id),
        }
