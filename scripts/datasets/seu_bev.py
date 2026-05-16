import glob
import os

import numpy as np
import torch
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation


def _numeric_sort_key(path: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


class SEUBEVDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.h_resize = int(cfg["frontend"]["image_size"][0])
        self.w_resize = int(cfg["frontend"]["image_size"][1])
        self.dataset_dir = cfg["dataset"]["root"]
        self.preload_rgbinfo()
        self.c2i = np.eye(4)
        self.intrinsic = None
        intrinsic_cfg = cfg.get("intrinsic", {})
        self.undistort = bool(intrinsic_cfg.get("undistort", False))
        distortion_coeffs = intrinsic_cfg.get(
            "distortion_coeffs", intrinsic_cfg.get("distCoeffs", None)
        )
        self.distortion_coeffs = None
        self.camera_matrix = None
        if self.undistort and distortion_coeffs is not None:
            self.distortion_coeffs = np.asarray(distortion_coeffs, dtype=np.float64).reshape(-1)
            self.camera_matrix = np.array(
                [
                    [intrinsic_cfg["fv"], 0.0, intrinsic_cfg["cv"]],
                    [0.0, intrinsic_cfg["fu"], intrinsic_cfg["cu"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        self.tqdm = tqdm(total=self.__len__())

    def __len__(self):
        return len(self.rgbinfo_dict["timestamp"])

    def preload_camtimestamp(self):
        return np.array(self.rgbinfo_dict["timestamp"]).reshape(-1, 1)

    def preload_imu(self):
        all_imu = np.zeros((len(self.rgbinfo_dict["timestamp"]), 7))
        all_imu[:, 0] = np.array(self.rgbinfo_dict["timestamp"])
        return all_imu

    def preload_rgbinfo(self):
        configured_extensions = self.cfg.get("dataset", {}).get("image_extensions")
        if configured_extensions:
            image_patterns = tuple(
                ext if ext.startswith("*") else f"*{ext}"
                for ext in configured_extensions
            )
        else:
            image_patterns = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
        rgb_files = []
        for pattern in image_patterns:
            rgb_files.extend(glob.glob(os.path.join(self.dataset_dir, pattern)))
        rgb_files = sorted(set(rgb_files), key=_numeric_sort_key)
        if not rgb_files:
            raise FileNotFoundError(
                f"No images found in {self.dataset_dir} matching extensions/patterns: {image_patterns}"
            )

        self.rgbinfo_dict = {
            "timestamp": list(range(len(rgb_files))),
            "filepath": rgb_files,
        }

    def _resolve_sequence_id(self):
        dataset_name = os.path.basename(os.path.normpath(self.dataset_dir))
        match = None
        for char in dataset_name:
            if char.isdigit():
                match = char if match is None else match + char
            elif match is not None:
                break
        if match is not None:
            return match.zfill(2)
        raise ValueError(
            f"Could not infer SEU-BEV sequence id from dataset root: {self.dataset_dir}"
        )

    def _resolve_gt_path(self):
        sequence_id = self._resolve_sequence_id()
        dataset_root = os.path.abspath(os.path.join(self.dataset_dir, os.pardir))
        candidates = [
            os.path.join(dataset_root, "ground_truth", f"{sequence_id}.txt"),
            os.path.join(dataset_root, str(int(sequence_id)), "ground_truth.txt"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"Could not find SEU-BEV ground truth for sequence {sequence_id}. "
            f"Tried: {candidates}"
        )

    def load_gt_dict(self):
        gt_path = self._resolve_gt_path()
        gt_data = np.loadtxt(gt_path, dtype=np.float64)
        if gt_data.ndim != 2 or gt_data.shape[1] < 8:
            raise ValueError(
                f"Unexpected SEU-BEV ground-truth format in {gt_path}. "
                f"Expected Nx8 [timestamp tx ty tz qx qy qz qw]."
            )

        gt_timestamps_sec = gt_data[:, 0]
        translations = gt_data[:, 1:4]
        quaternions_xyzw = gt_data[:, 4:8]

        c2ws = np.repeat(np.eye(4, dtype=np.float64)[None, ...], gt_data.shape[0], axis=0)
        c2ws[:, :3, :3] = Rotation.from_quat(quaternions_xyzw).as_matrix()
        c2ws[:, :3, 3] = translations

        if len(self.rgbinfo_dict["timestamp"]) <= 1:
            frame_ids = np.arange(gt_data.shape[0], dtype=np.float64)
        else:
            duration = max(float(gt_timestamps_sec[-1] - gt_timestamps_sec[0]), 1e-6)
            image_rate = (len(self.rgbinfo_dict["timestamp"]) - 1) / duration
            frame_ids = np.round((gt_timestamps_sec - gt_timestamps_sec[0]) * image_rate)
            frame_ids = np.clip(frame_ids, 0, len(self.rgbinfo_dict["timestamp"]) - 1).astype(np.int64)

        unique_ids, unique_indices = np.unique(frame_ids, return_index=True)
        c2ws = c2ws[unique_indices]

        return {
            "timestamps": unique_ids.astype(np.float64),
            "c2ws": c2ws.astype(np.float64),
        }

    def __getitem__(self, idx):
        resized_h = int(self.cfg["frontend"]["image_size"][0])
        resized_w = int(self.cfg["frontend"]["image_size"][1])
        rgb_raw = cv2.imread(self.rgbinfo_dict["filepath"][idx])
        if rgb_raw is None:
            raise FileNotFoundError(f"Could not read image: {self.rgbinfo_dict['filepath'][idx]}")
        if self.undistort and self.distortion_coeffs is not None:
            rgb_raw = cv2.undistort(rgb_raw, self.camera_matrix, self.distortion_coeffs)
        rgb = (
            torch.tensor(cv2.resize(rgb_raw, (resized_w, resized_h)))[..., [2, 1, 0]]
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.cfg["device"]["tracker"])
        )
        u_scale = resized_h / self.cfg["intrinsic"]["H"]
        v_scale = resized_w / self.cfg["intrinsic"]["W"]
        intrinsic = torch.tensor(
            [
                self.cfg["intrinsic"]["fv"] * v_scale,
                self.cfg["intrinsic"]["fu"] * u_scale,
                self.cfg["intrinsic"]["cv"] * v_scale,
                self.cfg["intrinsic"]["cu"] * u_scale,
            ],
            dtype=torch.float32,
            device=self.cfg["device"]["tracker"],
        )
        self.tqdm.update(1)
        return {
            "timestamp": self.rgbinfo_dict["timestamp"][idx],
            "rgb": rgb,
            "intrinsic": intrinsic,
        }


def get_dataset(config):
    return SEUBEVDataset(config)
