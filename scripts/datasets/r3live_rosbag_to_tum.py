#!/usr/bin/env python3
"""Convert an R3live rosbag into a TUM-style RGB dataset.

This exporter writes:
  - rgb/*.png or rgb/*.jpg
  - rgb.txt
  - groundtruth.txt
  - calibration.txt
  - associations.txt
  - camera_info.yaml
  - export_info.json

It only exports RGB frames. IMU and LiDAR are intentionally ignored here.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rosbag
import yaml


DEFAULT_DATASET_ROOT = Path("/media/server/yzz_disk/Dataset_sx/R3live")
DEFAULT_IMAGE_TOPIC = "/camera/image_color/compressed"
DEFAULT_GROUND_TRUTH_DIR = DEFAULT_DATASET_ROOT / "hku_ground_truth" / "tum"


@dataclass
class Calibration:
    source_path: Path
    intrinsic_matrix_raw: np.ndarray
    intrinsic_matrix_scaled: np.ndarray
    distortion_coeffs: np.ndarray
    extrinsic_matrix: np.ndarray
    calib_image_width: int
    calib_image_height: int


@dataclass
class ProbeResult:
    header_timestamps: np.ndarray
    bag_timestamps: np.ndarray
    image_shape: Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an R3live rosbag as a TUM-style RGB dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root folder containing the R3live bags and calibration.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="hku_small",
        help="Sequence stem, for example hku_small or hku_campus_seq_03.",
    )
    parser.add_argument(
        "--bag-path",
        type=Path,
        default=None,
        help="Optional explicit rosbag path. Overrides --sequence lookup.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to <dataset-root>/tum/<sequence>.",
    )
    parser.add_argument(
        "--image-topic",
        type=str,
        default=DEFAULT_IMAGE_TOPIC,
        help="Compressed RGB image topic to export.",
    )
    parser.add_argument(
        "--timestamp-source",
        choices=("auto", "header", "bag"),
        default="auto",
        help=(
            "Timestamp source for rgb.txt. 'auto' scores header vs bag timestamps "
            "against the selected GT and chooses the better match."
        ),
    )
    parser.add_argument(
        "--image-ext",
        choices=("png", "jpg"),
        default="png",
        help="Image format for exported frames.",
    )
    parser.add_argument(
        "--undistort",
        action="store_true",
        help="Undistort images before saving them.",
    )
    parser.add_argument(
        "--crop-undistorted",
        action="store_true",
        help="Crop the undistorted output to the valid ROI returned by OpenCV.",
    )
    parser.add_argument(
        "--undistort-alpha",
        type=float,
        default=0.0,
        help=(
            "Alpha passed to cv2.getOptimalNewCameraMatrix when --undistort is used. "
            "0.0 removes black borders most aggressively, 1.0 keeps more FOV."
        ),
    )
    parser.add_argument(
        "--gt-path",
        type=Path,
        default=None,
        help="Optional explicit TUM ground-truth file.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_DIR,
        help="Directory that contains TUM-format GT txt files.",
    )
    parser.add_argument(
        "--calib-path",
        type=Path,
        default=None,
        help="Optional explicit calibration txt file.",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help=(
            "Optional calibration image size. If omitted, the exporter infers it "
            "from the calibration filename, for example camera_livox_cali640.txt."
        ),
    )
    parser.add_argument(
        "--gt-mode",
        choices=("copy", "nearest", "interpolate"),
        default="interpolate",
        help=(
            "How groundtruth.txt should be written. 'interpolate' is the default "
            "because some bags, such as hku_small, are subsequences of a larger GT."
        ),
    )
    parser.add_argument(
        "--max-gt-gap",
        type=float,
        default=0.1,
        help="Maximum allowed timestamp gap in seconds when associating or resampling GT.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output folder.",
    )
    return parser.parse_args()


def infer_bag_path(dataset_root: Path, sequence: str, bag_path: Optional[Path]) -> Path:
    if bag_path is not None:
        return bag_path
    return dataset_root / f"{sequence}.bag"


def infer_output_dir(dataset_root: Path, sequence: str, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        return output_dir
    return dataset_root / "tum" / sequence


def infer_calib_path(dataset_root: Path, calib_path: Optional[Path]) -> Path:
    if calib_path is not None:
        return calib_path
    candidate = dataset_root / "camera_livox_cali640.txt"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Could not find the default calibration file at {candidate}. "
        "Please pass --calib-path explicitly."
    )


def read_tum_groundtruth(gt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    timestamps: List[float] = []
    poses: List[List[float]] = []

    with gt_path.open("r", encoding="utf-8") as handle:
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
        raise ValueError(f"No valid TUM-format poses found in {gt_path}")

    return np.asarray(timestamps, dtype=np.float64), np.asarray(poses, dtype=np.float64)


def load_all_ground_truth_paths(ground_truth_dir: Path) -> List[Path]:
    if not ground_truth_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {ground_truth_dir}")
    valid_paths: List[Path] = []
    for path in sorted(ground_truth_dir.glob("*.txt")):
        if not path.is_file():
            continue
        try:
            read_tum_groundtruth(path)
        except Exception:
            continue
        valid_paths.append(path)
    return valid_paths


def choose_ground_truth_path(
    sequence: str,
    ground_truth_dir: Path,
    gt_path: Optional[Path],
    header_timestamps: np.ndarray,
    bag_timestamps: np.ndarray,
    max_gap: float,
) -> Path:
    if gt_path is not None:
        return gt_path

    all_gt_paths = load_all_ground_truth_paths(ground_truth_dir)
    exact_candidate = ground_truth_dir / f"{sequence}.txt"
    if exact_candidate.is_file():
        return exact_candidate

    best_path = None
    best_score = (-1.0, -1.0, -1)

    image_start = float(min(header_timestamps[0], bag_timestamps[0]))
    image_end = float(max(header_timestamps[-1], bag_timestamps[-1]))

    for candidate in all_gt_paths:
        gt_timestamps, _ = read_tum_groundtruth(candidate)
        gt_start = float(gt_timestamps[0])
        gt_end = float(gt_timestamps[-1])
        overlap = max(0.0, min(image_end, gt_end) - max(image_start, gt_start))

        header_score = count_nearest_matches(header_timestamps, gt_timestamps, max_gap)
        bag_score = count_nearest_matches(bag_timestamps, gt_timestamps, max_gap)
        match_score = max(header_score, bag_score)
        ranking = (overlap, float(match_score), -len(candidate.name))
        if ranking > best_score:
            best_score = ranking
            best_path = candidate

    if best_path is None or best_score[0] <= 0.0 and best_score[1] <= 0.0:
        raise FileNotFoundError(
            f"Could not infer a matching GT file for '{sequence}' from {ground_truth_dir}. "
            "Please pass --gt-path explicitly."
        )

    return best_path


def count_nearest_matches(
    image_timestamps: np.ndarray,
    gt_timestamps: np.ndarray,
    max_gap: float,
) -> int:
    count = 0
    for timestamp in image_timestamps:
        insert_idx = int(np.searchsorted(gt_timestamps, timestamp))
        candidates = []
        if insert_idx < len(gt_timestamps):
            candidates.append(abs(gt_timestamps[insert_idx] - timestamp))
        if insert_idx > 0:
            candidates.append(abs(gt_timestamps[insert_idx - 1] - timestamp))
        if candidates and min(candidates) <= max_gap:
            count += 1
    return count


def choose_timestamp_source(
    requested: str,
    header_timestamps: np.ndarray,
    bag_timestamps: np.ndarray,
    gt_timestamps: np.ndarray,
    max_gap: float,
) -> str:
    if requested in {"header", "bag"}:
        return requested

    header_matches = count_nearest_matches(header_timestamps, gt_timestamps, max_gap)
    bag_matches = count_nearest_matches(bag_timestamps, gt_timestamps, max_gap)
    return "header" if header_matches >= bag_matches else "bag"


def read_probe_info(
    bag_path: Path,
    image_topic: str,
) -> ProbeResult:
    header_timestamps: List[float] = []
    bag_timestamps: List[float] = []
    image_shape = None

    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, msg, bag_stamp in bag.read_messages(topics=[image_topic]):
            header_timestamps.append(msg.header.stamp.to_sec())
            bag_timestamps.append(bag_stamp.to_sec())

            if image_shape is None:
                encoded = np.frombuffer(msg.data, dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Failed to decode the first frame from {bag_path}")
                image_shape = image.shape

    if not header_timestamps or image_shape is None:
        raise RuntimeError(f"No RGB frames found on topic {image_topic} in {bag_path}")

    return ProbeResult(
        header_timestamps=np.asarray(header_timestamps, dtype=np.float64),
        bag_timestamps=np.asarray(bag_timestamps, dtype=np.float64),
        image_shape=image_shape,
    )


def infer_calibration_size(
    calib_path: Path,
    target_width: int,
    target_height: int,
    explicit_size: Optional[Sequence[int]],
) -> Tuple[int, int]:
    if explicit_size is not None:
        return int(explicit_size[0]), int(explicit_size[1])

    matches = re.findall(r"(\d+)", calib_path.stem)
    for match in reversed(matches):
        width = int(match)
        if width <= 0:
            continue
        height = int(round(width * target_height / target_width))
        if height > 0:
            return width, height

    return target_width, target_height


def load_calibration(
    calib_path: Path,
    target_width: int,
    target_height: int,
    explicit_size: Optional[Sequence[int]],
) -> Calibration:
    raw_lines = [
        line.strip()
        for line in calib_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    intrinsic_idx = raw_lines.index("Intrinsic:")
    raw_k = np.array(
        [
            [float(value) for value in raw_lines[intrinsic_idx + 1].split()],
            [float(value) for value in raw_lines[intrinsic_idx + 2].split()],
            [float(value) for value in raw_lines[intrinsic_idx + 3].split()],
        ],
        dtype=np.float64,
    )

    distortion_idx = raw_lines.index("Distortion:")
    distortion_text = raw_lines[distortion_idx + 1].strip().strip("[]")
    distortion = np.array(
        [float(value) for value in distortion_text.split(",") if value.strip()],
        dtype=np.float64,
    )

    extrinsic_idx = raw_lines.index("Extrinsic:")
    extrinsic_matrix = np.eye(4, dtype=np.float64)
    r_idx = next(
        index for index in range(extrinsic_idx, len(raw_lines)) if raw_lines[index] == "R:"
    )
    t_idx = next(
        index for index in range(extrinsic_idx, len(raw_lines)) if raw_lines[index].startswith("t:")
    )
    extrinsic_matrix[:3, :3] = np.array(
        [
            [float(value) for value in raw_lines[r_idx + 1].split()],
            [float(value) for value in raw_lines[r_idx + 2].split()],
            [float(value) for value in raw_lines[r_idx + 3].split()],
        ],
        dtype=np.float64,
    )
    extrinsic_matrix[:3, 3] = np.array(
        [float(value) for value in raw_lines[t_idx].split(":", 1)[1].split()],
        dtype=np.float64,
    )

    calib_width, calib_height = infer_calibration_size(
        calib_path=calib_path,
        target_width=target_width,
        target_height=target_height,
        explicit_size=explicit_size,
    )

    scale_x = target_width / float(calib_width)
    scale_y = target_height / float(calib_height)
    scaled_k = raw_k.copy()
    scaled_k[0, 0] *= scale_x
    scaled_k[0, 2] *= scale_x
    scaled_k[1, 1] *= scale_y
    scaled_k[1, 2] *= scale_y

    return Calibration(
        source_path=calib_path,
        intrinsic_matrix_raw=raw_k,
        intrinsic_matrix_scaled=scaled_k,
        distortion_coeffs=distortion,
        extrinsic_matrix=extrinsic_matrix,
        calib_image_width=calib_width,
        calib_image_height=calib_height,
    )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory '{output_dir}' already exists and is not empty. "
                f"Pass --overwrite to reuse it."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "rgb").mkdir(parents=True, exist_ok=True)


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion((1.0 - alpha) * q0 + alpha * q1)

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)

    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def interpolate_pose(
    left_ts: float,
    left_pose: np.ndarray,
    right_ts: float,
    right_pose: np.ndarray,
    target_ts: float,
) -> np.ndarray:
    if right_ts <= left_ts:
        return left_pose.copy()

    alpha = (target_ts - left_ts) / (right_ts - left_ts)
    alpha = min(1.0, max(0.0, alpha))

    interp = np.zeros(7, dtype=np.float64)
    interp[:3] = (1.0 - alpha) * left_pose[:3] + alpha * right_pose[:3]
    interp[3:] = normalize_quaternion(slerp(left_pose[3:], right_pose[3:], alpha))
    return interp


def resample_groundtruth(
    gt_timestamps: np.ndarray,
    gt_poses: np.ndarray,
    target_timestamps: Iterable[float],
    mode: str,
    max_gap: float,
) -> List[Tuple[float, np.ndarray, float]]:
    if mode not in {"nearest", "interpolate"}:
        raise ValueError(f"Unsupported GT resampling mode: {mode}")

    results: List[Tuple[float, np.ndarray, float]] = []

    for target_ts in target_timestamps:
        insert_idx = int(np.searchsorted(gt_timestamps, target_ts))

        if mode == "nearest":
            candidate_indices: List[int] = []
            if insert_idx > 0:
                candidate_indices.append(insert_idx - 1)
            if insert_idx < len(gt_timestamps):
                candidate_indices.append(insert_idx)
            if not candidate_indices:
                continue

            best_idx = min(candidate_indices, key=lambda idx: abs(gt_timestamps[idx] - target_ts))
            delta = abs(gt_timestamps[best_idx] - target_ts)
            if delta > max_gap:
                continue
            results.append((target_ts, gt_poses[best_idx].copy(), float(gt_timestamps[best_idx])))
            continue

        if insert_idx == 0 or insert_idx >= len(gt_timestamps):
            continue

        left_idx = insert_idx - 1
        right_idx = insert_idx
        left_gap = target_ts - gt_timestamps[left_idx]
        right_gap = gt_timestamps[right_idx] - target_ts
        if left_gap > max_gap or right_gap > max_gap:
            continue

        pose = interpolate_pose(
            gt_timestamps[left_idx],
            gt_poses[left_idx],
            gt_timestamps[right_idx],
            gt_poses[right_idx],
            target_ts,
        )
        results.append((target_ts, pose, target_ts))

    return results


def build_association_entries(
    rgb_entries: Sequence[Tuple[float, str]],
    gt_timestamps: np.ndarray,
    gt_poses: np.ndarray,
    max_gap: float,
) -> List[Tuple[float, str, float, np.ndarray]]:
    association_entries: List[Tuple[float, str, float, np.ndarray]] = []
    gt_matches = resample_groundtruth(
        gt_timestamps=gt_timestamps,
        gt_poses=gt_poses,
        target_timestamps=[timestamp for timestamp, _ in rgb_entries],
        mode="nearest",
        max_gap=max_gap,
    )

    gt_by_rgb_timestamp = {
        float(rgb_ts): (float(gt_ts), pose.copy()) for rgb_ts, pose, gt_ts in gt_matches
    }
    for rgb_ts, rgb_path in rgb_entries:
        match = gt_by_rgb_timestamp.get(float(rgb_ts))
        if match is None:
            continue
        gt_ts, pose = match
        association_entries.append((rgb_ts, rgb_path, gt_ts, pose))
    return association_entries


def write_rgb_list(output_path: Path, rgb_entries: Sequence[Tuple[float, str]]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# timestamp filename\n")
        for timestamp, rel_path in rgb_entries:
            handle.write(f"{timestamp:.9f} {rel_path}\n")


def write_groundtruth(output_path: Path, entries: Sequence[Tuple[float, np.ndarray]]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for timestamp, pose in entries:
            handle.write(
                f"{timestamp:.9f} "
                f"{pose[0]:.9f} {pose[1]:.9f} {pose[2]:.9f} "
                f"{pose[3]:.9f} {pose[4]:.9f} {pose[5]:.9f} {pose[6]:.9f}\n"
            )


def write_associations(
    output_path: Path,
    association_entries: Sequence[Tuple[float, str, float, np.ndarray]],
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# rgb_timestamp rgb_path gt_timestamp tx ty tz qx qy qz qw\n")
        for rgb_ts, rgb_path, gt_ts, pose in association_entries:
            handle.write(
                f"{rgb_ts:.9f} {rgb_path} {gt_ts:.9f} "
                f"{pose[0]:.9f} {pose[1]:.9f} {pose[2]:.9f} "
                f"{pose[3]:.9f} {pose[4]:.9f} {pose[5]:.9f} {pose[6]:.9f}\n"
            )


def write_calibration(output_path: Path, intrinsics: Sequence[float]) -> None:
    values = np.asarray(intrinsics, dtype=np.float64).reshape(1, 4)
    np.savetxt(output_path, values, fmt="%.9f")


def export_rgb_frames(
    bag_path: Path,
    image_topic: str,
    output_dir: Path,
    image_ext: str,
    calibration: Calibration,
    timestamp_source: str,
    undistort: bool,
    crop_undistorted: bool,
    undistort_alpha: float,
    max_frames: Optional[int],
) -> Tuple[List[Tuple[float, str]], List[float], dict]:
    rgb_dir = output_dir / "rgb"
    rgb_entries: List[Tuple[float, str]] = []

    intrinsic_matrix = calibration.intrinsic_matrix_scaled.copy()
    distortion_coeffs = calibration.distortion_coeffs.copy()
    output_intrinsics = [
        float(intrinsic_matrix[0, 0]),
        float(intrinsic_matrix[1, 1]),
        float(intrinsic_matrix[0, 2]),
        float(intrinsic_matrix[1, 2]),
    ]

    map1 = None
    map2 = None
    crop_roi: Optional[Tuple[int, int, int, int]] = None
    raw_shape = None
    written_shape = None

    with rosbag.Bag(str(bag_path), "r") as bag:
        print(f"Reading {bag_path}...")
        for index, (_, msg, bag_stamp) in enumerate(bag.read_messages(topics=[image_topic])):
            encoded = np.frombuffer(msg.data, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to decode compressed image #{index} from {image_topic}")

            if raw_shape is None:
                raw_shape = image.shape[:2]
                print(f"Detected raw resolution: {raw_shape[1]}x{raw_shape[0]}")
                if undistort:
                    new_k, roi = cv2.getOptimalNewCameraMatrix(
                        intrinsic_matrix,
                        distortion_coeffs,
                        (raw_shape[1], raw_shape[0]),
                        undistort_alpha,
                    )
                    map1, map2 = cv2.initUndistortRectifyMap(
                        intrinsic_matrix,
                        distortion_coeffs,
                        None,
                        new_k,
                        (raw_shape[1], raw_shape[0]),
                        cv2.CV_16SC2,
                    )
                    intrinsic_matrix = new_k
                    if crop_undistorted:
                        crop_roi = roi
                        intrinsic_matrix[0, 2] -= crop_roi[0]
                        intrinsic_matrix[1, 2] -= crop_roi[1]

                    output_intrinsics = [
                        float(intrinsic_matrix[0, 0]),
                        float(intrinsic_matrix[1, 1]),
                        float(intrinsic_matrix[0, 2]),
                        float(intrinsic_matrix[1, 2]),
                    ]

            if undistort:
                image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
                if crop_roi is not None and crop_roi[2] > 0 and crop_roi[3] > 0:
                    x0, y0, width, height = crop_roi
                    image = image[y0 : y0 + height, x0 : x0 + width]

            written_shape = image.shape[:2]

            frame_timestamp = (
                msg.header.stamp.to_sec() if timestamp_source == "header" else bag_stamp.to_sec()
            )
            relative_path = Path("rgb") / f"{index:06d}.{image_ext}"
            output_path = output_dir / relative_path

            if image_ext == "jpg":
                ok = cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                ok = cv2.imwrite(str(output_path), image)
            if not ok:
                raise RuntimeError(f"Failed to save image to {output_path}")

            rgb_entries.append((frame_timestamp, relative_path.as_posix()))

            if index % 500 == 0:
                print(f"Extracted {index} images...")

            if max_frames is not None and len(rgb_entries) >= max_frames:
                break

    if not rgb_entries:
        raise RuntimeError(f"No frames found on topic {image_topic} in {bag_path}")

    metadata = {
        "raw_height": int(raw_shape[0]),
        "raw_width": int(raw_shape[1]),
        "written_height": int(written_shape[0]),
        "written_width": int(written_shape[1]),
        "frame_count": len(rgb_entries),
    }
    return rgb_entries, output_intrinsics, metadata


def write_camera_info(
    output_path: Path,
    calibration: Calibration,
    export_intrinsics: Sequence[float],
    frame_metadata: dict,
    timestamp_source: str,
    gt_path: Path,
    undistort: bool,
    crop_undistorted: bool,
) -> None:
    payload = {
        "source_calibration": str(calibration.source_path),
        "source_ground_truth": str(gt_path),
        "timestamp_source": timestamp_source,
        "undistort": bool(undistort),
        "crop_undistorted": bool(crop_undistorted),
        "calib_image_size": [calibration.calib_image_width, calibration.calib_image_height],
        "camera_intrinsic_raw": calibration.intrinsic_matrix_raw.reshape(-1).tolist(),
        "camera_intrinsic_scaled": calibration.intrinsic_matrix_scaled.reshape(-1).tolist(),
        "camera_dist_coeffs_raw": calibration.distortion_coeffs.reshape(-1).tolist(),
        "camera_lidar_extrinsic": calibration.extrinsic_matrix.reshape(-1).tolist(),
        "calibration_txt_fx_fy_cx_cy": list(export_intrinsics),
        "raw_resolution": [frame_metadata["raw_width"], frame_metadata["raw_height"]],
        "written_resolution": [frame_metadata["written_width"], frame_metadata["written_height"]],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def main() -> None:
    args = parse_args()

    sequence = args.sequence
    bag_path = infer_bag_path(args.dataset_root, sequence, args.bag_path)
    if args.bag_path is not None:
        sequence = bag_path.stem

    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    probe = read_probe_info(bag_path, args.image_topic)
    gt_path = choose_ground_truth_path(
        sequence=sequence,
        ground_truth_dir=args.ground_truth_dir,
        gt_path=args.gt_path,
        header_timestamps=probe.header_timestamps,
        bag_timestamps=probe.bag_timestamps,
        max_gap=args.max_gt_gap,
    )
    gt_timestamps, gt_poses = read_tum_groundtruth(gt_path)

    timestamp_source = choose_timestamp_source(
        requested=args.timestamp_source,
        header_timestamps=probe.header_timestamps,
        bag_timestamps=probe.bag_timestamps,
        gt_timestamps=gt_timestamps,
        max_gap=args.max_gt_gap,
    )

    output_dir = infer_output_dir(args.dataset_root, sequence, args.output_dir)
    calib_path = infer_calib_path(args.dataset_root, args.calib_path)
    calibration = load_calibration(
        calib_path=calib_path,
        target_width=probe.image_shape[1],
        target_height=probe.image_shape[0],
        explicit_size=args.calib_size,
    )

    prepare_output_dir(output_dir, args.overwrite)

    print(f"Sequence: {sequence}")
    print(f"Image topic: {args.image_topic}")
    print(f"Timestamp source: {timestamp_source}")
    print(f"Calibration: {calib_path}")
    print(f"Ground truth: {gt_path}")
    print(f"Output: {output_dir}")

    rgb_entries, export_intrinsics, frame_metadata = export_rgb_frames(
        bag_path=bag_path,
        image_topic=args.image_topic,
        output_dir=output_dir,
        image_ext=args.image_ext,
        calibration=calibration,
        timestamp_source=timestamp_source,
        undistort=args.undistort,
        crop_undistorted=args.crop_undistorted,
        undistort_alpha=args.undistort_alpha,
        max_frames=args.max_frames,
    )

    write_rgb_list(output_dir / "rgb.txt", rgb_entries)
    write_calibration(output_dir / "calibration.txt", export_intrinsics)

    if args.gt_mode == "copy":
        shutil.copy2(gt_path, output_dir / "groundtruth.txt")
    else:
        shutil.copy2(gt_path, output_dir / "groundtruth_source.txt")
        gt_resampled = resample_groundtruth(
            gt_timestamps=gt_timestamps,
            gt_poses=gt_poses,
            target_timestamps=[timestamp for timestamp, _ in rgb_entries],
            mode=args.gt_mode,
            max_gap=args.max_gt_gap,
        )
        write_groundtruth(
            output_dir / "groundtruth.txt",
            [(timestamp, pose) for timestamp, pose, _ in gt_resampled],
        )

    association_entries = build_association_entries(
        rgb_entries=rgb_entries,
        gt_timestamps=gt_timestamps,
        gt_poses=gt_poses,
        max_gap=args.max_gt_gap,
    )
    write_associations(output_dir / "associations.txt", association_entries)

    write_camera_info(
        output_path=output_dir / "camera_info.yaml",
        calibration=calibration,
        export_intrinsics=export_intrinsics,
        frame_metadata=frame_metadata,
        timestamp_source=timestamp_source,
        gt_path=gt_path,
        undistort=args.undistort,
        crop_undistorted=args.crop_undistorted,
    )

    summary = {
        "sequence": sequence,
        "bag_path": str(bag_path),
        "image_topic": args.image_topic,
        "timestamp_source": timestamp_source,
        "output_dir": str(output_dir),
        "calibration_path": str(calib_path),
        "groundtruth_path": str(gt_path),
        "groundtruth_mode": args.gt_mode,
        "max_gt_gap_sec": args.max_gt_gap,
        "undistort": args.undistort,
        "crop_undistorted": args.crop_undistorted,
        "image_ext": args.image_ext,
        "frame_count": len(rgb_entries),
        "associated_gt_count": len(association_entries),
        "raw_resolution": [frame_metadata["raw_width"], frame_metadata["raw_height"]],
        "written_resolution": [frame_metadata["written_width"], frame_metadata["written_height"]],
        "calib_image_size": [calibration.calib_image_width, calibration.calib_image_height],
        "calibration_txt_fx_fy_cx_cy": export_intrinsics,
        "source_gt_count": int(len(gt_timestamps)),
    }
    with (output_dir / "export_info.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Finished. Exported {len(rgb_entries)} frames.")
    print(f"Associated {len(association_entries)} RGB frames with ground truth.")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
