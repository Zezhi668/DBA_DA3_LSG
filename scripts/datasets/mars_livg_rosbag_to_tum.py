#!/usr/bin/env python3
"""Convert a MARS-LIVG rosbag into a TUM-style RGB dataset.

This exporter writes:
  - rgb/*.png or rgb/*.jpg
  - rgb.txt
  - groundtruth.txt
  - calibration.txt
  - associations.txt
  - camera_info.yaml
  - export_info.json

It intentionally does not fabricate depth maps. The MARS-LIVG bags expose
compressed RGB images plus Livox LiDAR, so a full TUM-RGBD export would require
an additional LiDAR-to-image depth projection stage.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rosbag
import yaml


DEFAULT_DATASET_ROOT = Path("/media/server/yzz_disk/Dataset_sx/MARS-LIVG")
DEFAULT_IMAGE_TOPIC = "/left_camera/image/compressed"


@dataclass
class Calibration:
    source_path: Path
    intrinsic_matrix: np.ndarray
    distortion_coeffs: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a MARS-LIVG rosbag as a TUM-style RGB dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root folder that contains the MARS-LIVG bags, cali/, and ground_truth/.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="HKairport03",
        help="Sequence stem, for example HKairport03.",
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
        help="Compressed image topic to export.",
    )
    parser.add_argument(
        "--timestamp-source",
        choices=("bag", "header"),
        default="bag",
        help=(
            "Timestamp source for rgb.txt. For HKairport03 the bag timestamp aligns "
            "better with the provided ground truth than msg.header.stamp."
        ),
    )
    parser.add_argument(
        "--trim-timestamp-source",
        choices=("bag", "header"),
        default="bag",
        help=(
            "Timestamp source used to decide whether a frame is inside the trim window. "
            "Use 'bag' to match the DJI GPS/IMU/height plot script."
        ),
    )
    parser.add_argument(
        "--start-timestamp",
        type=float,
        default=None,
        help="Optional inclusive start timestamp for extraction.",
    )
    parser.add_argument(
        "--end-timestamp",
        type=float,
        default=None,
        help="Optional inclusive end timestamp for extraction.",
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
        help="Optional explicit TUM-format ground-truth file.",
    )
    parser.add_argument(
        "--calib-path",
        type=Path,
        default=None,
        help="Optional explicit calibration yaml file.",
    )
    parser.add_argument(
        "--gt-mode",
        choices=("copy", "nearest", "interpolate"),
        default="copy",
        help=(
            "How groundtruth.txt should be written. 'copy' preserves the source GT, "
            "'nearest' and 'interpolate' resample it to image timestamps."
        ),
    )
    parser.add_argument(
        "--max-gt-gap",
        type=float,
        default=0.25,
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


def normalize_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def infer_bag_path(dataset_root: Path, sequence: str, bag_path: Optional[Path]) -> Path:
    if bag_path is not None:
        return bag_path
    return dataset_root / f"{sequence}.bag"


def infer_calib_path(dataset_root: Path, sequence: str, calib_path: Optional[Path]) -> Path:
    if calib_path is not None:
        return calib_path

    sequence_key = normalize_key(sequence)
    if "gnss" in sequence_key and ("hkairport" in sequence_key or "hkisland" in sequence_key):
        return dataset_root / "cali" / "HK_GNSS(airport & island).yaml"
    if sequence_key.startswith("hkairport"):
        return dataset_root / "cali" / "HKairport.yaml"
    if sequence_key.startswith("hkisland"):
        return dataset_root / "cali" / "HKisland.yaml"
    if sequence_key.startswith("amtown"):
        return dataset_root / "cali" / "AMtown.yaml"
    if sequence_key.startswith("amvalley") or sequence_key.startswith("newvalley"):
        return dataset_root / "cali" / "AMvalley.yaml"
    if sequence_key.startswith("featurelessgnss"):
        return dataset_root / "cali" / "Featureless_GNSS.yaml"

    raise FileNotFoundError(
        f"Could not infer calibration yaml for sequence '{sequence}'. "
        f"Please pass --calib-path explicitly."
    )


def infer_gt_path(dataset_root: Path, sequence: str, gt_path: Optional[Path]) -> Path:
    if gt_path is not None:
        return gt_path

    gt_dir = dataset_root / "ground_truth" / "gt"
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")

    sequence_key = normalize_key(sequence)
    candidate_keys = {
        sequence_key,
        sequence_key.replace("gnss", ""),
        sequence_key.replace("newvalley", "amvalley"),
    }

    for file_path in sorted(gt_dir.glob("*.txt")):
        stem_key = normalize_key(file_path.stem)
        stem_key = stem_key[:-2] if stem_key.endswith("gt") else stem_key
        if stem_key in candidate_keys:
            return file_path

    raise FileNotFoundError(
        f"Could not infer ground-truth txt for sequence '{sequence}'. "
        f"Please pass --gt-path explicitly."
    )


def load_calibration(calib_path: Path) -> Calibration:
    with calib_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    intrinsic = np.asarray(data["camera_intrinsic"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(data["camera_dist_coeffs"], dtype=np.float64).reshape(-1)

    return Calibration(
        source_path=calib_path,
        intrinsic_matrix=intrinsic,
        distortion_coeffs=distortion,
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
            poses.append([float(x) for x in parts[1:8]])

    if not timestamps:
        raise ValueError(f"No valid TUM-format poses found in {gt_path}")

    return np.asarray(timestamps, dtype=np.float64), np.asarray(poses, dtype=np.float64)


def write_rgb_list(output_path: Path, rgb_entries: Sequence[Tuple[float, str]]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# timestamp filename\n")
        for timestamp, rel_path in rgb_entries:
            handle.write(f"{timestamp:.9f} {rel_path}\n")


def write_calibration(output_path: Path, intrinsics: Sequence[float]) -> None:
    values = np.asarray(intrinsics, dtype=np.float64).reshape(1, 4)
    np.savetxt(output_path, values, fmt="%.9f")


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
        results.append((target_ts, pose, float(gt_timestamps[left_idx])))

    return results


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


def timestamp_from_message(msg, bag_timestamp: float, source: str) -> float:
    if source == "bag":
        return bag_timestamp
    if hasattr(msg, "header"):
        return msg.header.stamp.to_sec()
    return bag_timestamp


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


def export_rgb_frames(
    bag_path: Path,
    image_topic: str,
    output_dir: Path,
    image_ext: str,
    calibration: Calibration,
    timestamp_source: str,
    trim_timestamp_source: str,
    start_timestamp: Optional[float],
    end_timestamp: Optional[float],
    undistort: bool,
    crop_undistorted: bool,
    undistort_alpha: float,
    max_frames: Optional[int],
) -> Tuple[List[Tuple[float, str]], List[float], dict]:
    rgb_dir = output_dir / "rgb"
    rgb_entries: List[Tuple[float, str]] = []

    intrinsic_matrix = calibration.intrinsic_matrix.copy()
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
            bag_time = bag_stamp.to_sec()
            trim_timestamp = timestamp_from_message(msg, bag_time, trim_timestamp_source)
            if start_timestamp is not None and trim_timestamp < start_timestamp:
                continue
            if end_timestamp is not None and trim_timestamp > end_timestamp:
                continue

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

            frame_timestamp = timestamp_from_message(msg, bag_time, timestamp_source)
            relative_path = Path("rgb") / f"{len(rgb_entries):06d}.{image_ext}"
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
    undistort: bool,
    crop_undistorted: bool,
    timestamp_source: str,
    trim_timestamp_source: str,
    start_timestamp: Optional[float],
    end_timestamp: Optional[float],
) -> None:
    payload = {
        "source_calibration": str(calibration.source_path),
        "timestamp_source": timestamp_source,
        "trim_timestamp_source": trim_timestamp_source,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "undistort": bool(undistort),
        "crop_undistorted": bool(crop_undistorted),
        "camera_intrinsic_raw": calibration.intrinsic_matrix.reshape(-1).tolist(),
        "camera_dist_coeffs_raw": calibration.distortion_coeffs.reshape(-1).tolist(),
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

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.dataset_root / "tum" / sequence

    calib_path = infer_calib_path(args.dataset_root, sequence, args.calib_path)
    gt_path = infer_gt_path(args.dataset_root, sequence, args.gt_path)

    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")
    if not calib_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    prepare_output_dir(output_dir, args.overwrite)
    calibration = load_calibration(calib_path)
    gt_timestamps, gt_poses = read_tum_groundtruth(gt_path)

    print(f"Sequence: {sequence}")
    print(f"Image topic: {args.image_topic}")
    print(f"Timestamp source: {args.timestamp_source}")
    print(f"Trim timestamp source: {args.trim_timestamp_source}")
    if args.start_timestamp is not None or args.end_timestamp is not None:
        print(f"Trim window: {args.start_timestamp} -> {args.end_timestamp}")
    print(f"Calibration: {calib_path}")
    print(f"Ground truth: {gt_path}")
    print(f"Output: {output_dir}")

    rgb_entries, export_intrinsics, frame_metadata = export_rgb_frames(
        bag_path=bag_path,
        image_topic=args.image_topic,
        output_dir=output_dir,
        image_ext=args.image_ext,
        calibration=calibration,
        timestamp_source=args.timestamp_source,
        trim_timestamp_source=args.trim_timestamp_source,
        start_timestamp=args.start_timestamp,
        end_timestamp=args.end_timestamp,
        undistort=args.undistort,
        crop_undistorted=args.crop_undistorted,
        undistort_alpha=args.undistort_alpha,
        max_frames=args.max_frames,
    )

    write_rgb_list(output_dir / "rgb.txt", rgb_entries)
    write_calibration(output_dir / "calibration.txt", export_intrinsics)

    association_targets = [timestamp for timestamp, _ in rgb_entries]
    association_entries = build_association_entries(
        rgb_entries=rgb_entries,
        gt_timestamps=gt_timestamps,
        gt_poses=gt_poses,
        max_gap=args.max_gt_gap,
    )

    if args.gt_mode == "copy":
        shutil.copy2(gt_path, output_dir / "groundtruth.txt")
    else:
        shutil.copy2(gt_path, output_dir / "groundtruth_source.txt")
        gt_resampled = resample_groundtruth(
            gt_timestamps=gt_timestamps,
            gt_poses=gt_poses,
            target_timestamps=association_targets,
            mode=args.gt_mode,
            max_gap=args.max_gt_gap,
        )
        write_groundtruth(
            output_dir / "groundtruth.txt",
            [(timestamp, pose) for timestamp, pose, _ in gt_resampled],
        )

    write_associations(output_dir / "associations.txt", association_entries)

    write_camera_info(
        output_path=output_dir / "camera_info.yaml",
        calibration=calibration,
        export_intrinsics=export_intrinsics,
        frame_metadata=frame_metadata,
        undistort=args.undistort,
        crop_undistorted=args.crop_undistorted,
        timestamp_source=args.timestamp_source,
        trim_timestamp_source=args.trim_timestamp_source,
        start_timestamp=args.start_timestamp,
        end_timestamp=args.end_timestamp,
    )

    summary = {
        "sequence": sequence,
        "bag_path": str(bag_path),
        "image_topic": args.image_topic,
        "timestamp_source": args.timestamp_source,
        "trim_timestamp_source": args.trim_timestamp_source,
        "start_timestamp": args.start_timestamp,
        "end_timestamp": args.end_timestamp,
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
