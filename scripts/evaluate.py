import argparse
import csv
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from gaussian.general_utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DPT-LSG tracking and mapping outputs.")
    parser.add_argument("config", help="Path to the dataset/config yaml used for the run.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="A concrete VINGS output directory containing rgbdnua/ and droid_c2w/. "
             "If omitted, the newest matching run directory under config['output']['save_dir'] is used.",
    )
    parser.add_argument(
        "--pose-dir-name",
        default=None,
        help="Override pose directory name. Default prefers droid_c2w_new over droid_c2w when available.",
    )
    parser.add_argument(
        "--alignment",
        default="sim3",
        choices=["none", "se3", "sim3"],
        help="Alignment used for tracking ATE. sim3 is the default for monocular evaluation.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for LPIPS. Defaults to cuda if available, else cpu.",
    )
    parser.add_argument("--skip-tracking", action="store_true", help="Skip tracking evaluation.")
    parser.add_argument("--skip-mapping", action="store_true", help="Skip mapping evaluation.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_output_dir(cfg: Dict, config_path: Path, explicit_output_dir: Optional[str]) -> Path:
    if explicit_output_dir is not None:
        output_dir = Path(explicit_output_dir).expanduser().resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
        return output_dir

    base_dir = Path(cfg["output"]["save_dir"]).expanduser().resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(
            f"Base output directory does not exist: {base_dir}. "
            "Pass --output-dir to point at a concrete VINGS run."
        )

    config_stem = config_path.stem
    candidates = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "rgbdnua").is_dir() or (child / "droid_c2w").is_dir() or (child / "droid_c2w_new").is_dir():
            if config_stem in child.name or config_stem.rstrip("yaml.") in child.name:
                candidates.append(child)

    if not candidates:
        for child in base_dir.iterdir():
            if child.is_dir() and ((child / "rgbdnua").is_dir() or (child / "droid_c2w").is_dir()):
                candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            f"Could not infer a concrete VINGS output directory under {base_dir}. "
            "Pass --output-dir explicitly."
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_numeric_stem(path: Path) -> float:
    stem = path.stem
    match = re.search(r"[-+]?\d*\.?\d+", stem)
    if match is None:
        raise ValueError(f"Could not parse numeric id from filename: {path.name}")
    return float(match.group(0))


def load_predicted_poses(pose_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    pose_files = sorted(pose_dir.glob("*.txt"), key=parse_numeric_stem)
    if not pose_files:
        raise FileNotFoundError(f"No pose txt files found in: {pose_dir}")

    timestamps = np.array([parse_numeric_stem(path) for path in pose_files], dtype=np.float64)
    poses = np.stack([np.loadtxt(path).reshape(4, 4) for path in pose_files], axis=0)
    return timestamps, poses


def load_ground_truth(cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    dataset_module = importlib.import_module(cfg["dataset"]["module"])
    dataset = dataset_module.get_dataset(cfg)
    if not hasattr(dataset, "load_gt_dict"):
        raise AttributeError(
            f"Dataset module {cfg['dataset']['module']} does not provide load_gt_dict(), "
            "so standalone tracking evaluation is unavailable."
        )
    gt_dict = dataset.load_gt_dict()
    return np.asarray(gt_dict["timestamps"], dtype=np.float64), np.asarray(gt_dict["c2ws"], dtype=np.float64)


def match_pose_sequences(
    pred_timestamps: np.ndarray,
    pred_poses: np.ndarray,
    gt_timestamps: np.ndarray,
    gt_poses: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pred_timestamps.ndim != 1 or gt_timestamps.ndim != 1:
        raise ValueError("Timestamps must be 1D arrays.")

    if len(gt_timestamps) < 2:
        max_dt = 1e-6
    else:
        diffs = np.diff(np.sort(gt_timestamps))
        positive_diffs = diffs[diffs > 0]
        max_dt = 0.5 * float(np.median(positive_diffs)) if len(positive_diffs) > 0 else 1e-3
        max_dt = max(max_dt, 1e-6)

    matched_pred = []
    matched_gt = []
    matched_ids = []
    for pred_idx, pred_ts in enumerate(pred_timestamps):
        insert_idx = np.searchsorted(gt_timestamps, pred_ts)
        candidates = []
        if insert_idx < len(gt_timestamps):
            candidates.append(insert_idx)
        if insert_idx > 0:
            candidates.append(insert_idx - 1)
        if not candidates:
            continue

        best_idx = min(candidates, key=lambda idx: abs(gt_timestamps[idx] - pred_ts))
        if abs(gt_timestamps[best_idx] - pred_ts) <= max_dt:
            matched_pred.append(pred_poses[pred_idx])
            matched_gt.append(gt_poses[best_idx])
            matched_ids.append(pred_ts)

    if len(matched_pred) < 2:
        raise RuntimeError(
            "Could not match enough predicted poses to ground-truth poses. "
            f"Matched {len(matched_pred)} frame(s)."
        )

    return (
        np.asarray(matched_ids, dtype=np.float64),
        np.stack(matched_pred, axis=0),
        np.stack(matched_gt, axis=0),
    )


def umeyama_alignment(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape:
        raise ValueError("Source and destination points must have the same shape.")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1

    rotation = u @ correction @ vt
    if with_scale:
        src_var = (src_centered ** 2).sum() / src.shape[0]
        scale = float(np.trace(np.diag(singular_values) @ correction) / src_var)
    else:
        scale = 1.0

    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def align_positions(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    alignment: str,
) -> np.ndarray:
    if alignment == "none":
        return pred_positions.copy()

    scale, rotation, translation = umeyama_alignment(
        pred_positions,
        gt_positions,
        with_scale=(alignment == "sim3"),
    )
    return (scale * (rotation @ pred_positions.T)).T + translation


def summarize(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "num_frames": int(arr.shape[0]),
    }


def evaluate_tracking(
    output_dir: Path,
    cfg: Dict,
    alignment: str,
    pose_dir_name: Optional[str],
    eval_dir: Path,
) -> Dict:
    if pose_dir_name is not None:
        pose_dir = output_dir / pose_dir_name
    elif (output_dir / "droid_c2w_new").is_dir():
        pose_dir = output_dir / "droid_c2w_new"
    else:
        pose_dir = output_dir / "droid_c2w"

    pred_timestamps, pred_poses = load_predicted_poses(pose_dir)
    gt_timestamps, gt_poses = load_ground_truth(cfg)
    matched_ids, matched_pred_poses, matched_gt_poses = match_pose_sequences(
        pred_timestamps, pred_poses, gt_timestamps, gt_poses
    )

    pred_positions = matched_pred_poses[:, :3, 3]
    gt_positions = matched_gt_poses[:, :3, 3]
    aligned_pred_positions = align_positions(pred_positions, gt_positions, alignment)
    translation_errors = np.linalg.norm(aligned_pred_positions - gt_positions, axis=1)
    summary = summarize(translation_errors)

    csv_path = eval_dir / "tracking_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "ate_translation"])
        for frame_id, error in zip(matched_ids, translation_errors):
            writer.writerow([frame_id, float(error)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(gt_positions[:, 0], gt_positions[:, 2], label="GT", linewidth=2)
    axes[0].plot(aligned_pred_positions[:, 0], aligned_pred_positions[:, 2], label=f"Pred ({alignment})", linewidth=2)
    axes[0].set_title("Trajectory (X-Z)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Z")
    axes[0].axis("equal")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    frame_axis = np.arange(len(translation_errors))
    axes[1].plot(frame_axis, translation_errors, label="ATE", linewidth=1.8)
    axes[1].axhline(summary["mean"], color="tab:orange", linestyle="--", label=f"mean={summary['mean']:.4f}")
    axes[1].axhline(summary["rmse"], color="tab:red", linestyle="--", label=f"rmse={summary['rmse']:.4f}")
    axes[1].axhline(summary["median"], color="tab:green", linestyle="--", label=f"median={summary['median']:.4f}")
    axes[1].axhline(summary["std"], color="tab:purple", linestyle="--", label=f"std={summary['std']:.4f}")
    axes[1].set_title("ATE Per Frame")
    axes[1].set_xlabel("Matched Frame Index")
    axes[1].set_ylabel("Translation Error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(eval_dir / "tracking_plots.png", dpi=180)
    plt.close(fig)

    return {
        "pose_dir": str(pose_dir),
        "alignment": alignment,
        "summary": summary,
        "matched_frame_ids": matched_ids.tolist(),
    }


def crop_mapping_tiles(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("rgbdnua image must be HxWx3.")

    height, width = image_bgr.shape[:2]
    tile_h = height // 2
    tile_w = width // 4
    gt_rgb = image_bgr[:tile_h, :tile_w]
    pred_rgb = image_bgr[tile_h: tile_h * 2, :tile_w]
    return gt_rgb, pred_rgb


def compute_masked_psnr(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> float:
    mse = ((pred - gt) ** 2)[..., valid_mask].mean()
    mse = torch.clamp(mse, min=1e-12)
    return float(20 * torch.log10(1.0 / torch.sqrt(mse)))


def compute_masked_ssim(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> float:
    pred_np = pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    gt_np = gt.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    valid_np = valid_mask.detach().cpu().numpy().astype(bool)
    _, ssim_map = structural_similarity(
        gt_np,
        pred_np,
        channel_axis=2,
        data_range=1.0,
        full=True,
    )
    return float(ssim_map[valid_np].mean())


def compute_masked_lpips(
    lpips_model: torch.nn.Module,
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    valid = valid_mask.unsqueeze(0).unsqueeze(0).to(dtype=pred.dtype, device=pred.device)
    pred_eval = pred * valid + gt * (1.0 - valid)
    pred_eval = pred_eval * 2.0 - 1.0
    gt_eval = gt * 2.0 - 1.0
    return float(lpips_model(pred_eval, gt_eval).item())


def evaluate_mapping(
    output_dir: Path,
    device: torch.device,
    eval_dir: Path,
) -> Dict:
    rgbdnua_dir = output_dir / "rgbdnua"
    if not rgbdnua_dir.is_dir():
        raise FileNotFoundError(f"rgbdnua directory does not exist: {rgbdnua_dir}")

    image_paths = sorted(rgbdnua_dir.glob("*.png"), key=parse_numeric_stem)
    if not image_paths:
        raise FileNotFoundError(f"No rgbdnua PNG files found in: {rgbdnua_dir}")

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    psnr_values = []
    ssim_values = []
    lpips_values = []
    frame_ids = []

    for image_path in image_paths:
        composite_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if composite_bgr is None:
            continue

        gt_bgr, pred_bgr = crop_mapping_tiles(composite_bgr)
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)

        gt_tensor = torch.from_numpy(gt_rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        pred_tensor = torch.from_numpy(pred_rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        valid_mask = (gt_tensor.sum(dim=1) > 0).squeeze(0)

        if not torch.any(valid_mask):
            continue

        frame_id = parse_numeric_stem(image_path)
        psnr = compute_masked_psnr(pred_tensor, gt_tensor, valid_mask)
        ssim = compute_masked_ssim(pred_tensor, gt_tensor, valid_mask)
        lpips_value = compute_masked_lpips(lpips_model, pred_tensor, gt_tensor, valid_mask)

        frame_ids.append(frame_id)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips_value)
        rows.append([frame_id, psnr, ssim, lpips_value])

    if not rows:
        raise RuntimeError(f"No valid rgbdnua frames could be evaluated in {rgbdnua_dir}.")

    csv_path = eval_dir / "mapping_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "psnr", "ssim", "lpips"])
        writer.writerows(rows)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(frame_ids, psnr_values, color="tab:blue", linewidth=1.8)
    axes[0].set_ylabel("PSNR")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frame_ids, ssim_values, color="tab:green", linewidth=1.8)
    axes[1].set_ylabel("SSIM")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(frame_ids, lpips_values, color="tab:red", linewidth=1.8)
    axes[2].set_ylabel("LPIPS")
    axes[2].set_xlabel("Frame Id")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(eval_dir / "mapping_plots.png", dpi=180)
    plt.close(fig)

    return {
        "rgbdnua_dir": str(rgbdnua_dir),
        "summary": {
            "psnr": summarize(psnr_values),
            "ssim": summarize(ssim_values),
            "lpips": summarize(lpips_values),
        },
        "num_frames": len(rows),
    }


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(str(config_path))
    output_dir = infer_output_dir(cfg, config_path, args.output_dir)
    eval_dir = output_dir / "evaluation"
    ensure_dir(eval_dir)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    results = {
        "config": str(config_path),
        "output_dir": str(output_dir),
        "device": str(device),
    }

    if not args.skip_tracking:
        results["tracking"] = evaluate_tracking(
            output_dir=output_dir,
            cfg=cfg,
            alignment=args.alignment,
            pose_dir_name=args.pose_dir_name,
            eval_dir=eval_dir,
        )

    if not args.skip_mapping:
        results["mapping"] = evaluate_mapping(
            output_dir=output_dir,
            device=device,
            eval_dir=eval_dir,
        )

    summary_path = eval_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved evaluation artifacts to: {eval_dir}")


if __name__ == "__main__":
    main()
