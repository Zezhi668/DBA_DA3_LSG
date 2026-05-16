import glob
import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def _read_tum_list(file_path):
    entries = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            entries.append((float(parts[0]), parts[1]))
    return entries


def _read_tum_groundtruth(file_path):
    timestamps = []
    poses = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            timestamps.append(float(parts[0]))
            poses.append([float(value) for value in parts[1:8]])

    if not timestamps:
        raise ValueError(f"No valid TUM-format poses found in {file_path}")

    return np.asarray(timestamps, dtype=np.float64), np.asarray(poses, dtype=np.float64)


class TUMRGBDDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.h_resize = int(cfg["frontend"]["image_size"][0])
        self.w_resize = int(cfg["frontend"]["image_size"][1])
        self.dataset_dir = os.path.join(cfg["dataset"]["root"])
        self.preload_rgbinfo()
        self.c2i = np.eye(4)
        self.intrinsic = None
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
        rgb_txt = os.path.join(self.dataset_dir, "rgb.txt")
        if os.path.isfile(rgb_txt):
            entries = _read_tum_list(rgb_txt)
            rgb_files = [os.path.join(self.dataset_dir, rel_path) for _, rel_path in entries]
            source_timestamps = [timestamp for timestamp, _ in entries]
        else:
            rgb_files = sorted(glob.glob(os.path.join(self.dataset_dir, "rgb", "*.png")))
            if not rgb_files:
                rgb_files = sorted(glob.glob(os.path.join(self.dataset_dir, "rgb", "*.jpg")))
            source_timestamps = list(range(len(rgb_files)))

        if not rgb_files:
            raise FileNotFoundError(
                f"No RGB frames found in {self.dataset_dir}. "
                "Expected rgb.txt or rgb/*.png|jpg."
            )

        self.rgbinfo_dict = {
            "timestamp": list(range(len(rgb_files))),
            "source_timestamp": source_timestamps,
            "filepath": rgb_files,
        }

    def __getitem__(self, idx):
        resized_h = int(self.cfg["frontend"]["image_size"][0])
        resized_w = int(self.cfg["frontend"]["image_size"][1])
        rgb_raw = cv2.imread(self.rgbinfo_dict["filepath"][idx])
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
        data_packet = {}
        data_packet["timestamp"] = self.rgbinfo_dict["timestamp"][idx]
        data_packet["rgb"] = rgb
        data_packet["intrinsic"] = intrinsic
        self.tqdm.update(1)
        return data_packet

    def load_gt_dict(self):
        gt_candidates = [
            os.path.join(self.dataset_dir, "groundtruth.txt"),
            os.path.join(self.dataset_dir, "pose.txt"),
        ]
        gt_path = None
        for candidate in gt_candidates:
            if os.path.isfile(candidate):
                gt_path = candidate
                break

        if gt_path is None:
            raise FileNotFoundError(
                f"Could not find TUM ground truth in {self.dataset_dir}. "
                f"Tried: {gt_candidates}"
            )

        gt_timestamps, gt_poses = _read_tum_groundtruth(gt_path)
        translations = gt_poses[:, :3]
        quaternions_xyzw = gt_poses[:, 3:7]

        c2ws = np.repeat(np.eye(4, dtype=np.float64)[None, ...], gt_poses.shape[0], axis=0)
        c2ws[:, :3, :3] = Rotation.from_quat(quaternions_xyzw).as_matrix()
        c2ws[:, :3, 3] = translations

        rgb_source_timestamps = np.asarray(self.rgbinfo_dict["source_timestamp"], dtype=np.float64)
        frame_ids = []
        matched_indices = []
        for gt_idx, gt_timestamp in enumerate(gt_timestamps):
            nearest_rgb_idx = int(np.argmin(np.abs(rgb_source_timestamps - gt_timestamp)))
            frame_ids.append(nearest_rgb_idx)
            matched_indices.append(gt_idx)

        frame_ids = np.asarray(frame_ids, dtype=np.int64)
        matched_indices = np.asarray(matched_indices, dtype=np.int64)

        unique_ids, unique_pos = np.unique(frame_ids, return_index=True)
        matched_gt_indices = matched_indices[unique_pos]

        return {
            "timestamps": unique_ids.astype(np.float64),
            "c2ws": c2ws[matched_gt_indices].astype(np.float64),
        }


def get_dataset(config):
    return TUMRGBDDataset(config)
