#!/usr/bin/env python3
"""Export sparse metric LiDAR depth labels for MARS-LIVG RGB frames.

The exporter uses the existing TUM-style RGB export as the image timeline and
projects the nearest `/livox/lidar` scan into each undistorted/cropped image.
Output is a JSONL manifest plus sparse `.npy` depth maps and mask PNGs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
from genpy import Time
import numpy as np
import rosbag
import yaml


DEFAULT_DATASET_ROOT = Path("/media/server/yzz_disk/Dataset_sx/MARS-LIVG")
DEFAULT_LIDAR_TOPIC = "/livox/lidar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MARS-LIVG sparse LiDAR depth labels.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sequence", type=str, default="HKisland03")
    parser.add_argument("--bag-path", type=Path, default=None)
    parser.add_argument("--tum-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--calib-path", type=Path, default=None)
    parser.add_argument("--lidar-topic", type=str, default=DEFAULT_LIDAR_TOPIC)
    parser.add_argument("--start-timestamp", type=float, default=None)
    parser.add_argument("--end-timestamp", type=float, default=None)
    parser.add_argument("--max-time-diff", type=float, default=0.06)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=300.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--depth-format",
        choices=("sparse_npz", "dense_npy"),
        default="sparse_npz",
        help="sparse_npz is the normal format. dense_npy writes one full float32 image per frame and is very large.",
    )
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def infer_calib_path(dataset_root: Path, sequence: str, calib_path: Optional[Path]) -> Path:
    if calib_path is not None:
        return calib_path
    key = normalize_key(sequence)
    if "gnss" in key and ("hkairport" in key or "hkisland" in key):
        return dataset_root / "cali" / "HK_GNSS(airport & island).yaml"
    if key.startswith("hkairport"):
        return dataset_root / "cali" / "HKairport.yaml"
    if key.startswith("hkisland"):
        return dataset_root / "cali" / "HKisland.yaml"
    if key.startswith("amtown"):
        return dataset_root / "cali" / "AMtown.yaml"
    if key.startswith("amvalley") or key.startswith("newvalley"):
        return dataset_root / "cali" / "AMvalley.yaml"
    if key.startswith("featurelessgnss"):
        return dataset_root / "cali" / "Featureless_GNSS.yaml"
    raise FileNotFoundError(f"Pass --calib-path for sequence {sequence}")


def infer_tum_dir(dataset_root: Path, sequence: str, tum_dir: Optional[Path]) -> Path:
    if tum_dir is not None:
        return tum_dir
    trimmed = dataset_root / "tum" / f"{sequence}_trimmed"
    if trimmed.is_dir():
        return trimmed
    return dataset_root / "tum" / sequence


def infer_bag_path(dataset_root: Path, sequence: str, bag_path: Optional[Path]) -> Path:
    if bag_path is not None:
        return bag_path

    candidates = [dataset_root / f"{sequence}.bag"]
    if sequence.endswith("_trimmed"):
        candidates.append(dataset_root / f"{sequence[:-len('_trimmed')]}.bag")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    candidate_text = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find rosbag for sequence "
        f"{sequence}. Tried:\n  {candidate_text}\n"
        "Pass --bag-path explicitly if the bag is stored elsewhere."
    )


def load_rgb_entries(
    tum_dir: Path,
    max_frames: Optional[int],
    start_timestamp: Optional[float],
    end_timestamp: Optional[float],
) -> Tuple[np.ndarray, list[str]]:
    rgb_txt = tum_dir / "rgb.txt"
    timestamps = []
    rel_paths = []
    with rgb_txt.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            timestamp = float(parts[0])
            if start_timestamp is not None and timestamp < start_timestamp:
                continue
            if end_timestamp is not None and timestamp > end_timestamp:
                continue
            timestamps.append(timestamp)
            rel_paths.append(parts[1])
            if max_frames is not None and len(timestamps) >= max_frames:
                break
    if not timestamps:
        raise ValueError(
            f"No RGB entries found in {rgb_txt} for "
            f"start={start_timestamp}, end={end_timestamp}"
        )
    return np.asarray(timestamps, dtype=np.float64), rel_paths


def load_calibration(calib_path: Path, tum_dir: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[float, float, float, float]]:
    with calib_path.open("r", encoding="utf-8") as handle:
        raw_calib = yaml.safe_load(handle)

    R = np.asarray(raw_calib["camera_ext_R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(raw_calib["camera_ext_t"], dtype=np.float64).reshape(3, 1)

    camera_info_path = tum_dir / "camera_info.yaml"
    if camera_info_path.is_file():
        with camera_info_path.open("r", encoding="utf-8") as handle:
            camera_info = yaml.safe_load(handle)
        fx, fy, cx, cy = [float(v) for v in camera_info["calibration_txt_fx_fy_cx_cy"]]
        width, height = [int(v) for v in camera_info["written_resolution"]]
    else:
        K_raw = np.asarray(raw_calib["camera_intrinsic"], dtype=np.float64).reshape(3, 3)
        fx, fy, cx, cy = float(K_raw[0, 0]), float(K_raw[1, 1]), float(K_raw[0, 2]), float(K_raw[1, 2])
        sample = next((tum_dir / "rgb").glob("*"))
        image = cv2.imread(str(sample))
        height, width = image.shape[:2]

    return R, t, (height, width), (fx, fy, cx, cy)


def livox_points(msg) -> np.ndarray:
    points = msg.points
    pts = np.empty((len(points), 3), dtype=np.float32)
    for i, point in enumerate(points):
        pts[i] = (point.x, point.y, point.z)
    return pts


def project_sparse_depth(
    points_lidar: np.ndarray,
    R_cam_lidar: np.ndarray,
    t_cam_lidar: np.ndarray,
    image_hw: Tuple[int, int],
    intrinsics: Tuple[float, float, float, float],
    min_depth: float,
    max_depth: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_hw
    fx, fy, cx, cy = intrinsics
    if points_lidar.size == 0:
        return (
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.float32),
        )

    points_lidar = points_lidar.astype(np.float32, copy=False)
    R_cam_lidar = R_cam_lidar.astype(np.float32, copy=False)
    t_cam_lidar = t_cam_lidar.reshape(1, 3).astype(np.float32, copy=False)
    points_cam = points_lidar @ R_cam_lidar.T + t_cam_lidar
    z = points_cam[:, 2]
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    points_cam = points_cam[valid]
    z = z[valid]
    if z.size == 0:
        return (
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.float32),
        )

    u = np.rint(fx * points_cam[:, 0] / z + cx).astype(np.int64)
    v = np.rint(fy * points_cam[:, 1] / z + cy).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], z[inside]
    if z.size == 0:
        return (
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.float32),
        )

    flat = v * width + u
    order = np.argsort(z)
    flat_sorted = flat[order]
    z_sorted = z[order].astype(np.float32)
    unique_flat, first = np.unique(flat_sorted, return_index=True)
    u_unique = (unique_flat % width).astype(np.uint16, copy=False)
    v_unique = (unique_flat // width).astype(np.uint16, copy=False)
    return u_unique, v_unique, z_sorted[first].astype(np.float32, copy=False)


def sparse_to_dense_depth(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    image_hw: Tuple[int, int],
) -> np.ndarray:
    height, width = image_hw
    depth_flat = np.zeros(height * width, dtype=np.float32)
    if z.size == 0:
        return depth_flat.reshape(height, width)
    flat = v.astype(np.int64) * width + u.astype(np.int64)
    order = np.argsort(z)
    flat_sorted = flat[order]
    z_sorted = z[order].astype(np.float32)
    unique_flat, first = np.unique(flat_sorted, return_index=True)
    depth_flat[unique_flat] = z_sorted[first]
    return depth_flat.reshape(height, width)


def write_mask(mask_path: Path, u: np.ndarray, v: np.ndarray, image_hw: Tuple[int, int]) -> None:
    height, width = image_hw
    mask = np.zeros((height, width), dtype=np.uint8)
    if u.size > 0:
        mask[v.astype(np.int64), u.astype(np.int64)] = 255
    cv2.imwrite(str(mask_path), mask)


def write_record(
    output_dir: Path,
    tum_dir: Path,
    rel_path: str,
    timestamp: float,
    lidar_timestamp: float,
    sparse_depth: Tuple[np.ndarray, np.ndarray, np.ndarray],
    image_hw: Tuple[int, int],
    intrinsics: Tuple[float, float, float, float],
    depth_format: str,
) -> Dict[str, object]:
    stem = Path(rel_path).stem
    u, v, z = sparse_depth
    depth_suffix = ".npy" if depth_format == "dense_npy" else ".npz"
    depth_path = output_dir / "depth_sparse" / f"{stem}{depth_suffix}"
    mask_path = output_dir / "mask" / f"{stem}.png"
    if depth_format == "dense_npy":
        depth = sparse_to_dense_depth(u, v, z, image_hw)
        np.save(depth_path, depth.astype(np.float32))
    else:
        np.savez_compressed(
            depth_path,
            u=u.astype(np.uint16, copy=False),
            v=v.astype(np.uint16, copy=False),
            depth=z.astype(np.float32, copy=False),
            height=np.asarray(image_hw[0], dtype=np.int32),
            width=np.asarray(image_hw[1], dtype=np.int32),
        )
    write_mask(mask_path, u, v, image_hw)
    return {
        "image": str((tum_dir / rel_path).resolve()),
        "depth": str(depth_path.resolve()),
        "depth_format": depth_format,
        "mask": str(mask_path.resolve()),
        "timestamp": timestamp,
        "lidar_timestamp": lidar_timestamp,
        "intrinsics_fx_fy_cx_cy": list(intrinsics),
        "height": int(image_hw[0]),
        "width": int(image_hw[1]),
        "valid_pixels": int(z.size),
    }


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    managed_paths = [
        output_dir / "depth_sparse",
        output_dir / "mask",
        output_dir / "manifest.jsonl",
        output_dir / "summary.json",
    ]
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty. Pass --overwrite.")
    if overwrite:
        for path in managed_paths:
            if path.is_dir():
                print(f"Removing previous exporter output: {path}", flush=True)
                shutil.rmtree(path)
            elif path.exists():
                print(f"Removing previous exporter output: {path}", flush=True)
                path.unlink()
    (output_dir / "depth_sparse").mkdir(parents=True, exist_ok=True)
    (output_dir / "mask").mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    sequence = args.sequence
    bag_path = infer_bag_path(args.dataset_root, sequence, args.bag_path)
    tum_dir = infer_tum_dir(args.dataset_root, sequence, args.tum_dir)
    calib_path = infer_calib_path(args.dataset_root, sequence, args.calib_path)
    output_dir = args.output_dir or (tum_dir / "lidar_depth_training")

    prepare_output_dir(output_dir, args.overwrite)

    image_times, rel_paths = load_rgb_entries(
        tum_dir,
        args.max_frames,
        args.start_timestamp,
        args.end_timestamp,
    )
    R, t, image_hw, intrinsics = load_calibration(calib_path, tum_dir)
    records_by_idx: Dict[int, Tuple[float, Dict[str, object]]] = {}

    print(
        "Exporting LiDAR depth labels: "
        f"sequence={sequence}, rgb_frames={len(image_times)}, "
        f"image_hw={image_hw}, depth_format={args.depth_format}, "
        f"bag={bag_path}",
        flush=True,
    )

    start_time = Time.from_sec(max(0.0, float(image_times[0]) - args.max_time_diff))
    end_time = Time.from_sec(float(image_times[-1]) + args.max_time_diff)
    with rosbag.Bag(str(bag_path), "r") as bag:
        topic_info = bag.get_type_and_topic_info().topics
        if args.lidar_topic not in topic_info:
            available = ", ".join(sorted(topic_info.keys()))
            raise KeyError(f"Topic {args.lidar_topic} not found in {bag_path}. Available topics: {available}")
        print(
            f"Reading {args.lidar_topic} from {start_time.to_sec():.6f} to {end_time.to_sec():.6f}",
            flush=True,
        )
        lidar_messages = 0
        last_progress = time.monotonic()
        for _, msg, stamp in bag.read_messages(
            topics=[args.lidar_topic],
            start_time=start_time,
            end_time=end_time,
        ):
            lidar_messages += 1
            lidar_time = float(stamp.to_sec())
            if lidar_time > float(image_times[-1]) + args.max_time_diff:
                break
            idx = int(np.searchsorted(image_times, lidar_time))
            candidates = []
            if idx < len(image_times):
                candidates.append(idx)
            if idx > 0:
                candidates.append(idx - 1)
            if idx + 1 < len(image_times):
                candidates.append(idx + 1)
            if not candidates:
                continue

            best_idx = min(candidates, key=lambda i: abs(image_times[i] - lidar_time))
            dt = abs(float(image_times[best_idx]) - lidar_time)
            if dt > args.max_time_diff:
                continue
            if best_idx in records_by_idx and dt >= records_by_idx[best_idx][0]:
                continue

            points = livox_points(msg)
            sparse_depth = project_sparse_depth(points, R, t, image_hw, intrinsics, args.min_depth, args.max_depth)
            if sparse_depth[2].size == 0:
                continue
            record = write_record(
                output_dir,
                tum_dir,
                rel_paths[best_idx],
                float(image_times[best_idx]),
                lidar_time,
                sparse_depth,
                image_hw,
                intrinsics,
                args.depth_format,
            )
            records_by_idx[best_idx] = (dt, record)
            if (
                args.progress_interval > 0
                and (
                    len(records_by_idx) == 1
                    or len(records_by_idx) % args.progress_interval == 0
                    or time.monotonic() - last_progress > 30.0
                )
            ):
                last_progress = time.monotonic()
                print(
                    "Progress: "
                    f"lidar_messages={lidar_messages}, "
                    f"frames_with_depth={len(records_by_idx)}/{len(image_times)}, "
                    f"last_rgb={image_times[best_idx]:.6f}, "
                    f"last_lidar={lidar_time:.6f}, "
                    f"valid_pixels={record['valid_pixels']}",
                    flush=True,
                )
            if len(records_by_idx) == len(image_times):
                break

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx in sorted(records_by_idx):
            handle.write(json.dumps(records_by_idx[idx][1]) + "\n")

    summary = {
        "sequence": sequence,
        "bag_path": str(bag_path),
        "tum_dir": str(tum_dir),
        "calib_path": str(calib_path),
        "lidar_topic": args.lidar_topic,
        "start_timestamp": args.start_timestamp,
        "end_timestamp": args.end_timestamp,
        "depth_format": args.depth_format,
        "first_image_timestamp": float(image_times[0]),
        "last_image_timestamp": float(image_times[-1]),
        "frames_in_rgb_txt": int(len(image_times)),
        "frames_with_depth": int(len(records_by_idx)),
        "manifest": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
