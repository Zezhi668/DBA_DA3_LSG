import glob
import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def _numeric_sort_key(path: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


class UAVImageFolderDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.h_resize = int(cfg["frontend"]["image_size"][0])
        self.w_resize = int(cfg["frontend"]["image_size"][1])
        self.dataset_dir = cfg["dataset"]["root"]
        self.rgb_strip = max(1, int(cfg["dataset"].get("rgb_strip", 1)))

        self.preload_rgbinfo()
        self.c2i = np.eye(4, dtype=np.float32)
        self.intrinsic = None
        self.tqdm = tqdm(total=self.__len__())

    def __len__(self):
        return len(self.rgbinfo_dict["timestamp"])

    def preload_camtimestamp(self):
        return np.array(self.rgbinfo_dict["timestamp"], dtype=np.float64).reshape(-1, 1)

    def preload_imu(self):
        all_imu = np.zeros((len(self.rgbinfo_dict["timestamp"]), 7), dtype=np.float64)
        all_imu[:, 0] = np.array(self.rgbinfo_dict["timestamp"], dtype=np.float64)
        return all_imu

    def preload_rgbinfo(self):
        configured_extensions = self.cfg.get("dataset", {}).get("image_extensions")
        if configured_extensions:
            image_patterns = tuple(
                ext if ext.startswith("*") else f"*{ext}" for ext in configured_extensions
            )
        else:
            image_patterns = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")

        rgb_files = []
        for pattern in image_patterns:
            rgb_files.extend(glob.glob(os.path.join(self.dataset_dir, pattern)))

        rgb_files = sorted(set(rgb_files), key=_numeric_sort_key)
        rgb_files = rgb_files[:: self.rgb_strip]

        if not rgb_files:
            raise FileNotFoundError(
                f"No images found under {self.dataset_dir}. "
                f"Checked extensions: {image_patterns}"
            )

        self.rgbinfo_dict = {
            "timestamp": list(range(len(rgb_files))),
            "filepath": rgb_files,
        }

    def __getitem__(self, idx):
        rgb_path = self.rgbinfo_dict["filepath"][idx]
        rgb_raw = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_raw is None:
            raise FileNotFoundError(f"Failed to read image: {rgb_path}")

        rgb = (
            torch.tensor(cv2.resize(rgb_raw, (self.w_resize, self.h_resize)))[..., [2, 1, 0]]
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.cfg["device"]["tracker"])
        )

        u_scale = self.h_resize / self.cfg["intrinsic"]["H"]
        v_scale = self.w_resize / self.cfg["intrinsic"]["W"]
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

    @staticmethod
    def _lla_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)

        a = 6378137.0
        e_sq = 6.69437999014e-3

        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        N = a / np.sqrt(1.0 - e_sq * sin_lat * sin_lat)
        x = (N + alt_m) * cos_lat * cos_lon
        y = (N + alt_m) * cos_lat * sin_lon
        z = (N * (1.0 - e_sq) + alt_m) * sin_lat
        return np.stack([x, y, z], axis=-1)

    @staticmethod
    def _ecef_to_enu(ecef_xyz: np.ndarray, ref_lat_deg: float, ref_lon_deg: float, ref_ecef_xyz: np.ndarray) -> np.ndarray:
        lat0 = np.deg2rad(ref_lat_deg)
        lon0 = np.deg2rad(ref_lon_deg)

        sin_lat = np.sin(lat0)
        cos_lat = np.cos(lat0)
        sin_lon = np.sin(lon0)
        cos_lon = np.cos(lon0)

        rot = np.array(
            [
                [-sin_lon, cos_lon, 0.0],
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
            ],
            dtype=np.float64,
        )
        return (ecef_xyz - ref_ecef_xyz[None, :]) @ rot.T

    def load_gt_dict(self):
        gt_pose_path = self.cfg["dataset"].get("gt_pose_path")
        if gt_pose_path is None:
            raise ValueError(
                "UAV image-folder dataset requires dataset.gt_pose_path for tracking evaluation."
            )

        gt_data = np.loadtxt(gt_pose_path, dtype=np.float64)
        if gt_data.ndim != 2 or gt_data.shape[1] < 8:
            raise ValueError(
                f"Unexpected UAV GT format in {gt_pose_path}. "
                "Expected Nx8 [timestamp lat lon alt qx qy qz qw]."
            )

        gt_timestamps_sec = gt_data[:, 0]
        lat_deg = gt_data[:, 1]
        lon_deg = gt_data[:, 2]
        alt_m = gt_data[:, 3]
        quaternions_xyzw = gt_data[:, 4:8]

        ecef_xyz = self._lla_to_ecef(lat_deg, lon_deg, alt_m)
        enu_xyz = self._ecef_to_enu(
            ecef_xyz,
            ref_lat_deg=float(lat_deg[0]),
            ref_lon_deg=float(lon_deg[0]),
            ref_ecef_xyz=ecef_xyz[0],
        )

        c2ws = np.repeat(np.eye(4, dtype=np.float64)[None, ...], gt_data.shape[0], axis=0)
        c2ws[:, :3, :3] = Rotation.from_quat(quaternions_xyzw).as_matrix()
        c2ws[:, :3, 3] = enu_xyz

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


def get_dataset(config):
    return UAVImageFolderDataset(config)
