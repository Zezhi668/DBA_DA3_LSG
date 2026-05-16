import copy
import importlib
import os
import sys
import threading

import torch
import torch.nn.functional as F
import lietorch

from vings_utils.gtsam_utils import matrix_to_tq


def _as_float_tensor(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _as_scalar(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        raise ValueError("Expected a scalar tensor.")
    return value


def _normalize_rgb(rgb_tensor):
    rgb = rgb_tensor.detach().to(dtype=torch.float32)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.ndim != 3:
        raise ValueError(f"Expected RGB tensor with shape [3, H, W], got {tuple(rgb.shape)}.")
    if rgb.shape[0] != 3:
        raise ValueError(f"Expected channel-first RGB tensor, got shape {tuple(rgb.shape)}.")
    if rgb.max().item() > 1.0:
        rgb = rgb / 255.0
    rgb = rgb.clamp(0.0, 1.0)
    return rgb.permute(1, 2, 0).contiguous().cpu().numpy()


class _FrontendState:
    def __init__(self):
        self.video = None
        self.new_frame_added = False
        self.all_imu = None
        self.all_stamp = None


class _LocalKeyframes:
    def __init__(self, use_calib):
        self.lock = threading.RLock()
        self.frames = []
        self.use_calib = use_calib
        self.K = None

    def __getitem__(self, idx):
        with self.lock:
            frame = self.frames[idx]
            if self.use_calib and self.K is not None:
                frame.K = self.K
            return frame

    def __setitem__(self, idx, value):
        with self.lock:
            if idx == len(self.frames):
                self.frames.append(value)
            else:
                self.frames[idx] = value

    def __len__(self):
        with self.lock:
            return len(self.frames)

    def append(self, value):
        with self.lock:
            self.frames.append(value)

    def pop_last(self):
        with self.lock:
            if self.frames:
                self.frames.pop()

    def last_keyframe(self):
        with self.lock:
            if not self.frames:
                return None
            frame = self.frames[-1]
            if self.use_calib and self.K is not None:
                frame.K = self.K
            return frame

    def update_T_WCs(self, T_WCs, idx):
        with self.lock:
            if torch.is_tensor(idx):
                idx_list = idx.detach().cpu().tolist()
            else:
                idx_list = list(idx)
            for offset, frame_idx in enumerate(idx_list):
                self.frames[int(frame_idx)].T_WC = lietorch.Sim3(T_WCs.data[offset : offset + 1].clone())

    def set_intrinsics(self, K):
        with self.lock:
            self.K = K.detach().clone()
            for frame in self.frames:
                frame.K = self.K


class _VideoBuffer:
    def __init__(self, cfg, height, width, buffer_size, save_buffer):
        device = cfg["device"]["tracker"]
        height_low = height // 8
        width_low = width // 8

        self.device = device
        self.ht = height
        self.wd = width
        self.height_dsf = height_low
        self.width_dsf = width_low
        self.buffer = buffer_size
        self.save_pkl = True
        self.upsample_flag = True

        self.tstamp = -torch.ones(buffer_size, device=device, dtype=torch.float64)
        self.images = torch.zeros(buffer_size, 3, height, width, device=device, dtype=torch.uint8)
        self.poses = torch.zeros(buffer_size, 7, device=device, dtype=torch.float32)
        self.poses[:, -1] = 1.0
        self.disps = torch.zeros(buffer_size, height_low, width_low, device=device, dtype=torch.float32)
        self.disps_up = torch.zeros(buffer_size, height, width, device=device, dtype=torch.float32)
        self.depths_cov = torch.zeros(buffer_size, height_low, width_low, device=device, dtype=torch.float32)
        self.depths_cov_up = torch.zeros(buffer_size, height, width, device=device, dtype=torch.float32)
        self.intrinsics = torch.zeros(buffer_size, 4, device=device, dtype=torch.float32)

        self.tstamp_save = -torch.ones(save_buffer, device="cpu", dtype=torch.float64)
        self.poses_save = torch.zeros(save_buffer, 7, device="cpu", dtype=torch.float32)
        self.poses_save[:, -1] = 1.0
        self.disps_save = torch.zeros(save_buffer, height_low, width_low, device="cpu", dtype=torch.float32)
        self.disps_up_save = torch.zeros(save_buffer, height, width, device="cpu", dtype=torch.float32)
        self.depths_cov_up_save = torch.zeros(save_buffer, height, width, device="cpu", dtype=torch.float32)
        self.images_save = torch.zeros(save_buffer, height_low, width_low, 3, device="cpu", dtype=torch.float32)
        self.images_up_save = torch.zeros(save_buffer, height, width, 3, device="cpu", dtype=torch.float32)

        self.count_save = 0
        self.count_save_bias = 0
        self.imu_enabled = False

    def _grow(self, tensor, new_first_dim, fill_value=0.0, last_quat_identity=False):
        new_shape = (new_first_dim,) + tuple(tensor.shape[1:])
        expanded = torch.full(
            new_shape,
            fill_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        expanded[: tensor.shape[0]] = tensor
        if last_quat_identity and expanded.shape[-1] == 7:
            expanded[:, -1] = 1.0
            expanded[: tensor.shape[0]] = tensor
        return expanded

    def ensure_save_capacity(self, min_size):
        current = self.tstamp_save.shape[0]
        if min_size <= current:
            return

        new_size = current
        while new_size < min_size:
            new_size *= 2

        self.tstamp_save = self._grow(self.tstamp_save, new_size, fill_value=-1.0)
        self.poses_save = self._grow(self.poses_save, new_size, fill_value=0.0, last_quat_identity=True)
        self.disps_save = self._grow(self.disps_save, new_size, fill_value=0.0)
        self.disps_up_save = self._grow(self.disps_up_save, new_size, fill_value=0.0)
        self.depths_cov_up_save = self._grow(self.depths_cov_up_save, new_size, fill_value=0.0)
        self.images_save = self._grow(self.images_save, new_size, fill_value=0.0)
        self.images_up_save = self._grow(self.images_up_save, new_size, fill_value=0.0)


class Mast3rSLAM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["device"]["tracker"]
        self.frontend = _FrontendState()
        self.visual_frontend = self.frontend
        self.video = None
        self.viz_out = None

        self.mast3r_cfg = cfg.get("mast3r_slam", {})
        self.repo_root = self.mast3r_cfg.get(
            "repo_root",
            os.environ.get("MAST3R_SLAM_ROOT", "/home/server/VINGS_work/MASt3R-SLAM"),
        )
        self.img_size = int(self.mast3r_cfg.get("img_size", 512))
        self.publish_window = int(self.mast3r_cfg.get("publish_window", 8))
        self.save_buffer = int(self.mast3r_cfg.get("save_buffer", 2500))
        self.enable_retrieval = bool(self.mast3r_cfg.get("enable_retrieval", False))
        self.depth_min = float(self.mast3r_cfg.get("depth_min", 1e-3))
        self.depth_eps = float(self.mast3r_cfg.get("depth_eps", 1e-6))
        self.cov_scale = float(self.mast3r_cfg.get("cov_scale", 0.05))
        self.cov_min = float(self.mast3r_cfg.get("cov_min", 1e-3))
        self.cov_max = float(self.mast3r_cfg.get("cov_max", 3e3))
        self.conf_eps = float(self.mast3r_cfg.get("conf_eps", 1e-6))
        self.max_depth = float(self.cfg["middleware"].get("max_depth", self.mast3r_cfg.get("max_depth", 35.0)))
        self.max_cov = float(self.cfg["middleware"].get("max_cov", self.cov_max))
        self.warned_about_reloc = False

        self._load_upstream_modules()
        self.runtime_config = self._load_runtime_config()
        self.use_calib = bool(self.runtime_config.get("use_calib", True))

        model_path = self.mast3r_cfg.get("model_path") or None
        retriever_path = self.mast3r_cfg.get("retriever_path") or None

        self.model = self.load_mast3r(path=model_path, device=self.device)
        self.keyframes = _LocalKeyframes(use_calib=self.use_calib)
        self.factor_graph = self.FactorGraph(self.model, self.keyframes, K=None, device=self.device)
        self.tracker_impl = self.FrameTracker(self.model, self.keyframes, self.device)

        self.retrieval_database = None
        if self.enable_retrieval:
            self.retrieval_database = self.load_retriever(
                self.model,
                retriever_path=retriever_path,
                device=self.device,
            )

        self.current_K = None
        self.keyframe_timestamps = []
        self.total_frames_seen = 0
        self.dataset_length = None

    def _install_instructions(self):
        return (
            f"MASt3R-SLAM is not ready at `{self.repo_root}`.\n"
            "You need to clone and build the upstream project before `mode: vo_mast3rslam` can run.\n"
            "Recommended setup:\n"
            f"  git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git {self.repo_root}\n"
            f"  cd {self.repo_root}\n"
            "  conda create -n mast3rslam python=3.11 -y\n"
            "  conda activate mast3rslam\n"
            "  pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install -e thirdparty/mast3r\n"
            "  pip install -e thirdparty/in3d\n"
            "  pip install --no-build-isolation -e .\n"
            "A separate Python environment is strongly recommended because MASt3R-SLAM targets a newer "
            "PyTorch stack than the legacy DBAF environment in DPT-LSG."
        )

    def _load_upstream_modules(self):
        if not os.path.isdir(self.repo_root):
            raise RuntimeError(self._install_instructions())

        candidate_paths = [
            self.repo_root,
            os.path.join(self.repo_root, "thirdparty", "mast3r"),
            os.path.join(self.repo_root, "thirdparty", "in3d"),
        ]
        for path in candidate_paths:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

        try:
            config_module = importlib.import_module("mast3r_slam.config")
            frame_module = importlib.import_module("mast3r_slam.frame")
            tracker_module = importlib.import_module("mast3r_slam.tracker")
            global_opt_module = importlib.import_module("mast3r_slam.global_opt")
            utils_module = importlib.import_module("mast3r_slam.mast3r_utils")
        except Exception as exc:
            raise RuntimeError(self._install_instructions()) from exc

        self.config_module = config_module
        self.load_config = config_module.load_config
        self.set_global_config = config_module.set_global_config

        self.Frame = frame_module.Frame
        self.FrameTracker = tracker_module.FrameTracker
        self.FactorGraph = global_opt_module.FactorGraph

        self.load_mast3r = utils_module.load_mast3r
        self.load_retriever = utils_module.load_retriever
        self.mast3r_inference_mono = utils_module.mast3r_inference_mono
        self.resize_img = utils_module.resize_img

    def _load_runtime_config(self):
        config_path = self.mast3r_cfg.get("config", "config/calib.yaml")
        if not os.path.isabs(config_path):
            config_path = os.path.join(self.repo_root, config_path)

        self.load_config(config_path)
        runtime_config = copy.deepcopy(dict(self.config_module.config))
        runtime_config["single_thread"] = bool(self.mast3r_cfg.get("single_thread", True))
        runtime_config["use_calib"] = bool(self.mast3r_cfg.get("use_calib", runtime_config.get("use_calib", True)))
        self.set_global_config(runtime_config)
        return runtime_config

    def _build_frame(self, data_packet, frame_id):
        rgb_np = _normalize_rgb(data_packet["rgb"])
        resized, transform = self.resize_img(
            rgb_np,
            self.img_size,
            return_transformation=True,
        )
        rgb = resized["img"].to(device=self.device)
        img_shape = torch.tensor(resized["true_shape"], device=self.device)
        img_true_shape = img_shape.clone()
        uimg = torch.from_numpy(resized["unnormalized_img"]).to(torch.float32) / 255.0

        downsample = int(self.runtime_config.get("dataset", {}).get("img_downsample", 1))
        if downsample > 1:
            uimg = uimg[::downsample, ::downsample]
            img_shape = img_shape // downsample

        if len(self.keyframes) == 0:
            pose = lietorch.Sim3.Identity(1, device=self.device, dtype=torch.float32)
        else:
            pose = lietorch.Sim3(self.keyframes.last_keyframe().T_WC.data.clone())

        frame = self.Frame(
            frame_id=frame_id,
            img=rgb,
            img_shape=img_shape,
            img_true_shape=img_true_shape,
            uimg=uimg,
            T_WC=pose,
        )

        resized_K = self._resize_intrinsics(data_packet["intrinsic"], transform).to(
            device=self.device,
            dtype=torch.float32,
        )
        frame._lsg_intrinsics = resized_K
        if self.use_calib:
            frame.K = resized_K
        return frame

    def _resize_intrinsics(self, intrinsics, transform):
        intrinsic = _as_float_tensor(intrinsics, device="cpu")
        fx, fy, cx, cy = intrinsic.tolist()
        scale_w, scale_h, crop_w, crop_h = transform

        fx = fx / scale_w
        fy = fy / scale_h
        cx = cx / scale_w - crop_w
        cy = cy / scale_h - crop_h

        K = torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        return K

    def _ensure_video(self, frame):
        if self.video is not None:
            return

        height, width = frame.uimg.shape[:2]
        buffer_size = int(self.cfg.get("frontend", {}).get("buffer", 80))
        self.video = _VideoBuffer(
            self.cfg,
            height=height,
            width=width,
            buffer_size=buffer_size,
            save_buffer=self.save_buffer,
        )
        self.frontend.video = self.video

    def _pose_to_c2w_and_scale(self, frame):
        sim3 = frame.T_WC.matrix().squeeze(0).to(torch.float32)
        scaled_rotation = sim3[:3, :3]
        scale = torch.linalg.det(scaled_rotation).abs().clamp_min(1e-8).pow(1.0 / 3.0)
        rotation = scaled_rotation / scale

        u, _, vh = torch.linalg.svd(rotation)
        rotation = u @ vh
        if torch.linalg.det(rotation) < 0:
            correction = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
            correction[-1, -1] = -1.0
            rotation = u @ correction @ vh

        c2w = torch.eye(4, device=sim3.device, dtype=sim3.dtype)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = sim3[:3, 3]
        return c2w, scale

    def _confidence_to_covariance(self, confidence, scale):
        covariance = self.cov_scale / confidence.clamp_min(self.conf_eps)
        covariance = covariance * (scale * scale)
        covariance = covariance.clamp(self.cov_min, self.cov_max)
        return covariance

    def _extract_keyframe_bundle(self, frame):
        c2w, scale = self._pose_to_c2w_and_scale(frame)
        rgb = frame.uimg.to(self.device, dtype=torch.float32)
        height, width = rgb.shape[:2]

        points = frame.X_canon.reshape(height, width, 3).to(self.device, dtype=torch.float32)
        confidence = frame.get_average_conf().reshape(height, width, 1).to(self.device, dtype=torch.float32)
        depth = points[..., 2:3] * scale
        depth_cov = self._confidence_to_covariance(confidence, scale)

        invalid = ~torch.isfinite(depth)
        invalid |= ~torch.isfinite(depth_cov)
        invalid |= depth <= self.depth_min
        invalid |= depth > self.max_depth
        invalid |= depth_cov > self.max_cov

        depth = depth.clone()
        depth_cov = depth_cov.clone()
        rgb = rgb.clone()

        depth[invalid] = 0.0
        depth_cov[invalid] = 0.0
        rgb[invalid.squeeze(-1)] = 0.0

        pixel_mask = ~invalid.squeeze(-1)
        disps_up = torch.where(depth.squeeze(-1) > self.depth_eps, 1.0 / depth.squeeze(-1), 0.0)
        depth_cov_up = depth_cov.squeeze(-1)

        disps_low = F.avg_pool2d(disps_up[None, None], kernel_size=8, stride=8).squeeze(0).squeeze(0)
        depth_cov_low = F.avg_pool2d(depth_cov_up[None, None], kernel_size=8, stride=8).squeeze(0).squeeze(0)
        rgb_low = F.avg_pool2d(
            rgb.permute(2, 0, 1)[None],
            kernel_size=8,
            stride=8,
        ).squeeze(0).permute(1, 2, 0)

        return {
            "rgb": rgb,
            "rgb_bgr": rgb[..., [2, 1, 0]].cpu(),
            "rgb_low_bgr": rgb_low[..., [2, 1, 0]].cpu(),
            "depth": depth,
            "depth_cov": depth_cov,
            "pixel_mask": pixel_mask,
            "c2w": c2w,
            "w2c_tq": matrix_to_tq(torch.linalg.inv(c2w).unsqueeze(0)).squeeze(0),
            "disps_up": disps_up,
            "disps_low": disps_low,
            "depth_cov_up": depth_cov_up,
            "depth_cov_low": depth_cov_low,
            "intrinsic": frame._lsg_intrinsics.to(self.device, dtype=torch.float32),
        }

    def _write_pose_only(self, global_kf_id):
        frame = self.keyframes[global_kf_id]
        c2w, _ = self._pose_to_c2w_and_scale(frame)
        self.video.ensure_save_capacity(global_kf_id + 1)
        self.video.poses_save[global_kf_id] = matrix_to_tq(torch.linalg.inv(c2w).unsqueeze(0)).squeeze(0).cpu()

    def _write_dense_history(self, global_kf_id):
        frame = self.keyframes[global_kf_id]
        bundle = self._extract_keyframe_bundle(frame)
        timestamp = self.keyframe_timestamps[global_kf_id]

        self.video.ensure_save_capacity(global_kf_id + 1)
        self.video.tstamp_save[global_kf_id] = float(timestamp)
        self.video.poses_save[global_kf_id] = bundle["w2c_tq"].cpu()
        self.video.disps_save[global_kf_id] = bundle["disps_low"].cpu()
        self.video.disps_up_save[global_kf_id] = bundle["disps_up"].cpu()
        self.video.depths_cov_up_save[global_kf_id] = bundle["depth_cov_up"].cpu()
        self.video.images_save[global_kf_id] = bundle["rgb_low_bgr"]
        self.video.images_up_save[global_kf_id] = bundle["rgb_bgr"]
        self.video.count_save = max(self.video.count_save, global_kf_id + 1)

    def _refresh_all_saved_poses(self):
        for global_kf_id in range(len(self.keyframes)):
            self._write_pose_only(global_kf_id)
        self.video.count_save = len(self.keyframes)

    def _sync_local_window(self):
        n_keyframes = len(self.keyframes)
        if n_keyframes == 0:
            return

        start = max(0, n_keyframes - self.video.buffer)
        recent_ids = list(range(start, n_keyframes))

        self.video.tstamp.fill_(-1)
        self.video.images.zero_()
        self.video.poses.zero_()
        self.video.poses[:, -1] = 1.0
        self.video.disps.zero_()
        self.video.disps_up.zero_()
        self.video.depths_cov.zero_()
        self.video.depths_cov_up.zero_()
        self.video.intrinsics.zero_()

        for local_idx, global_kf_id in enumerate(recent_ids):
            frame = self.keyframes[global_kf_id]
            bundle = self._extract_keyframe_bundle(frame)
            intrinsic = bundle["intrinsic"]

            self.video.tstamp[local_idx] = float(self.keyframe_timestamps[global_kf_id])
            self.video.images[local_idx] = (bundle["rgb"].permute(2, 0, 1) * 255.0).round().to(torch.uint8)
            self.video.poses[local_idx] = bundle["w2c_tq"]
            self.video.disps[local_idx] = bundle["disps_low"]
            self.video.disps_up[local_idx] = bundle["disps_up"]
            self.video.depths_cov[local_idx] = bundle["depth_cov_low"]
            self.video.depths_cov_up[local_idx] = bundle["depth_cov_up"]
            self.video.intrinsics[local_idx] = torch.tensor(
                [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]],
                device=self.device,
                dtype=torch.float32,
            )

    def _build_viz_out(self):
        n_keyframes = len(self.keyframes)
        if n_keyframes == 0:
            return None

        start = max(0, n_keyframes - self.publish_window)
        recent_ids = list(range(start, n_keyframes))

        images = []
        depths = []
        depths_cov = []
        poses = []
        pixel_masks = []
        timestamps = []
        global_ids = []

        for global_kf_id in recent_ids:
            frame = self.keyframes[global_kf_id]
            bundle = self._extract_keyframe_bundle(frame)
            if not torch.any(bundle["depth"] > 0):
                continue

            images.append(bundle["rgb"])
            depths.append(bundle["depth"])
            depths_cov.append(bundle["depth_cov"])
            poses.append(bundle["c2w"])
            pixel_masks.append(bundle["pixel_mask"])
            timestamps.append(float(self.keyframe_timestamps[global_kf_id]))
            global_ids.append(global_kf_id)

        if not images:
            return None

        intrinsic = self.keyframes[global_ids[-1]]._lsg_intrinsics.to(self.device, dtype=torch.float32)
        return {
            "images": torch.stack(images, dim=0),
            "depths": torch.stack(depths, dim=0),
            "depths_cov": torch.stack(depths_cov, dim=0),
            "poses": torch.stack(poses, dim=0),
            "pixel_mask": torch.stack(pixel_masks, dim=0),
            "viz_out_idx_to_f_idx": torch.tensor(timestamps, device=self.device, dtype=torch.float64),
            "global_kf_id": torch.tensor(global_ids, device=self.device, dtype=torch.long),
            "intrinsic": {
                "fv": intrinsic[0, 0],
                "fu": intrinsic[1, 1],
                "cv": intrinsic[0, 2],
                "cu": intrinsic[1, 2],
                "H": int(images[0].shape[0]),
                "W": int(images[0].shape[1]),
            },
        }

    def _finalize_new_keyframe(self, global_kf_id):
        if self.use_calib:
            self.keyframes.set_intrinsics(self.current_K)
            self.factor_graph.K = self.current_K

        self._refresh_all_saved_poses()
        for refresh_id in range(max(0, global_kf_id - self.publish_window + 1), global_kf_id + 1):
            self._write_dense_history(refresh_id)
        self._sync_local_window()

        self.viz_out = self._build_viz_out()
        self.frontend.new_frame_added = self.viz_out is not None

    def _optimize_new_keyframe(self, global_kf_id):
        frame = self.keyframes[global_kf_id]
        connected_ids = []
        if global_kf_id > 0:
            connected_ids.append(global_kf_id - 1)

        if self.retrieval_database is not None:
            retrieval_ids = self.retrieval_database.update(
                frame,
                add_after_query=True,
                k=self.runtime_config["retrieval"]["k"],
                min_thresh=self.runtime_config["retrieval"]["min_thresh"],
            )
            for retrieval_id in retrieval_ids:
                retrieval_id = int(retrieval_id)
                if retrieval_id != global_kf_id and retrieval_id not in connected_ids:
                    connected_ids.append(retrieval_id)

        if connected_ids:
            self.factor_graph.add_factors(
                connected_ids,
                [global_kf_id] * len(connected_ids),
                self.runtime_config["local_opt"]["min_match_frac"],
            )

        if self.use_calib:
            self.factor_graph.solve_GN_calib()
        else:
            self.factor_graph.solve_GN_rays()

        self._finalize_new_keyframe(global_kf_id)

    def _attempt_relocalization(self, frame, timestamp):
        if self.retrieval_database is None:
            if not self.warned_about_reloc:
                print(
                    "MASt3R-SLAM requested relocalization, but retrieval support is disabled. "
                    "Set `mast3r_slam.enable_retrieval: true` after the upstream repo is fully installed "
                    "if you want MASt3R's retrieval-based relocalization path."
                )
                self.warned_about_reloc = True
            self.tracker_impl.reset_idx_f2k()
            return False

        X_init, C_init = self.mast3r_inference_mono(self.model, frame)
        frame.update_pointmap(X_init, C_init)

        retrieval_ids = self.retrieval_database.update(
            frame,
            add_after_query=False,
            k=self.runtime_config["retrieval"]["k"],
            min_thresh=self.runtime_config["retrieval"]["min_thresh"],
        )
        retrieval_ids = [int(idx) for idx in retrieval_ids]
        if not retrieval_ids:
            self.tracker_impl.reset_idx_f2k()
            return False

        self.keyframes.append(frame)
        self.keyframe_timestamps.append(timestamp)
        new_global_id = len(self.keyframes) - 1

        success = self.factor_graph.add_factors(
            [new_global_id] * len(retrieval_ids),
            retrieval_ids,
            self.runtime_config["reloc"]["min_match_frac"],
            is_reloc=self.runtime_config["reloc"]["strict"],
        )
        if not success:
            self.keyframes.pop_last()
            self.keyframe_timestamps.pop()
            self.tracker_impl.reset_idx_f2k()
            return False

        self.retrieval_database.update(
            frame,
            add_after_query=True,
            k=self.runtime_config["retrieval"]["k"],
            min_thresh=self.runtime_config["retrieval"]["min_thresh"],
        )
        frame.T_WC = copy.deepcopy(self.keyframes[retrieval_ids[0]].T_WC)

        if self.use_calib:
            self.factor_graph.solve_GN_calib()
        else:
            self.factor_graph.solve_GN_rays()

        self._finalize_new_keyframe(new_global_id)
        return True

    def track(self, data_packet):
        with torch.no_grad():
            frame = self._build_frame(data_packet, self.total_frames_seen)
            timestamp = _as_scalar(data_packet["timestamp"])
            self.total_frames_seen += 1

            self.current_K = frame._lsg_intrinsics
            self._ensure_video(frame)

            self.frontend.new_frame_added = False
            self.viz_out = None

            if len(self.keyframes) == 0:
                X_init, C_init = self.mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(X_init, C_init)
                self.keyframes.append(frame)
                self.keyframe_timestamps.append(timestamp)
                self._finalize_new_keyframe(0)
                return

            add_new_kf, _, try_reloc = self.tracker_impl.track(frame)
            if try_reloc:
                self._attempt_relocalization(frame, timestamp)
                return

            if add_new_kf:
                self.keyframes.append(frame)
                self.keyframe_timestamps.append(timestamp)
                self._optimize_new_keyframe(len(self.keyframes) - 1)

    def save_pt_ckpt(self, save_path):
        if self.video is None:
            raise RuntimeError("No MASt3R-SLAM state has been created yet.")

        save_dict = {
            "frontend": {
                "video": {
                    "tstamp_save": self.video.tstamp_save,
                    "poses_save": self.video.poses_save,
                    "images_up_save": self.video.images_up_save,
                    "disps_up_save": self.video.disps_up_save,
                    "disps_save": self.video.disps_save,
                    "poses": self.video.poses,
                    "disps_up": self.video.disps_up,
                    "disps": self.video.disps,
                    "depths_cov_up_save": self.video.depths_cov_up_save,
                    "count_save": self.video.count_save,
                    "count_save_bias": self.video.count_save_bias,
                }
            },
            "keyframe_timestamps": self.keyframe_timestamps,
        }
        torch.save(save_dict, save_path)
