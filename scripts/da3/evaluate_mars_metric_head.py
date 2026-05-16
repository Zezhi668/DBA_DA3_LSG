#!/usr/bin/env python3
"""Evaluate DA3 metric-head checkpoints on MARS-LIVG sparse LiDAR labels."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from train_mars_metric_head import MarsSparseDepthDataset, da3_depth_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate DA3 checkpoint depth precision against sparse MARS-LIVG "
            "LiDAR labels from a lidar_depth_training/manifest.jsonl file."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        action="append",
        default=[],
        help=(
            "Checkpoint path to evaluate. Pass multiple times to compare. "
            "Use 'pretrained' to evaluate the base DA3Metric-Large model."
        ),
    )
    parser.add_argument("--source-dir", type=Path, default=Path("/home/server/VINGS_work/Depth-Anything-3/src"))
    parser.add_argument("--pretrained", type=str, default="/home/server/VINGS_work/DPT-LSG/ckpts/DA3Metric-Large")
    parser.add_argument("--model-name", type=str, default="da3metric-large")
    parser.add_argument("--image-size", type=int, nargs=2, default=(448, 560), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=300.0)
    parser.add_argument("--valid-min-pixels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-mod", type=int, default=1)
    parser.add_argument("--sample-remainder", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--median-align", action="store_true", help="Also scale-align each prediction by sparse GT median.")
    parser.add_argument("--no-canonical-focal-scale", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def checkpoint_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    return checkpoint


def load_model(args: argparse.Namespace, checkpoint_arg: str):
    if str(args.source_dir) not in sys.path:
        sys.path.insert(0, str(args.source_dir))
    from depth_anything_3.api import DepthAnything3

    if checkpoint_arg == "pretrained":
        source = args.pretrained
        if source.startswith("/") and not Path(source).exists():
            source = "depth-anything/DA3METRIC-LARGE"
        model = DepthAnything3.from_pretrained(source)
        label = "pretrained"
        checkpoint_meta = {"checkpoint": "pretrained", "source": source}
    else:
        checkpoint_path = Path(checkpoint_arg)
        model = DepthAnything3(model_name=args.model_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint_state_dict(checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        label = checkpoint_path.stem
        checkpoint_meta = {
            "checkpoint": str(checkpoint_path),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        }
        if isinstance(checkpoint, dict):
            checkpoint_meta["step"] = checkpoint.get("step")
            checkpoint_meta["init_checkpoint"] = checkpoint.get("init_checkpoint")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return label, model, device, checkpoint_meta


def build_loader(args: argparse.Namespace) -> tuple[DataLoader, int, int]:
    dataset = MarsSparseDepthDataset(args.manifest, tuple(args.image_size), args.valid_min_pixels)
    sample_mod = max(1, int(args.sample_mod))
    sample_remainder = int(args.sample_remainder) % sample_mod
    indices = [idx for idx in range(len(dataset)) if idx % sample_mod == sample_remainder]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(indices)
    if args.max_samples is not None:
        indices = indices[: int(args.max_samples)]
    if not indices:
        raise ValueError("No samples selected for evaluation.")
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, len(dataset), len(indices)


def autocast_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def maybe_median_align(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    aligned = pred.clone()
    for batch_idx in range(pred.shape[0]):
        mask = valid[batch_idx]
        if int(mask.sum().item()) == 0:
            continue
        pred_med = torch.median(pred[batch_idx][mask])
        gt_med = torch.median(gt[batch_idx][mask])
        if torch.isfinite(pred_med) and float(pred_med.item()) > 0.0:
            aligned[batch_idx] = pred[batch_idx] * (gt_med / pred_med)
    return aligned


def update_metrics(acc: Dict[str, float], pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, args: argparse.Namespace) -> None:
    label_mask = mask & torch.isfinite(gt) & (gt > args.min_depth) & (gt < args.max_depth)
    pred_positive = torch.isfinite(pred) & (pred > 0.0)
    valid = label_mask & pred_positive
    if args.median_align:
        pred = maybe_median_align(pred, gt, valid)
        pred_positive = torch.isfinite(pred) & (pred > 0.0)
        valid = label_mask & pred_positive

    acc["label_pixels"] += int(label_mask.sum().item())
    acc["invalid_pred_pixels"] += int((label_mask & ~pred_positive).sum().item())
    acc["pred_above_max_pixels"] += int((label_mask & pred_positive & (pred > args.max_depth)).sum().item())
    acc["pred_below_min_pixels"] += int((label_mask & pred_positive & (pred < args.min_depth)).sum().item())

    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return

    pred_v = pred[valid].float()
    gt_v = gt[valid].float()
    err = pred_v - gt_v
    abs_err = torch.abs(err)
    sq_err = err * err
    rel = abs_err / torch.clamp(gt_v, min=args.min_depth)
    sq_rel = sq_err / torch.clamp(gt_v, min=args.min_depth)
    log_err = torch.log(pred_v) - torch.log(gt_v)
    log_abs = torch.abs(log_err)
    ratio = torch.maximum(pred_v / gt_v, gt_v / pred_v)

    acc["valid_pixels"] += valid_count
    acc["abs_rel_sum"] += float(rel.sum().item())
    acc["sq_rel_sum"] += float(sq_rel.sum().item())
    acc["abs_sum"] += float(abs_err.sum().item())
    acc["sq_sum"] += float(sq_err.sum().item())
    acc["log_abs_sum"] += float(log_abs.sum().item())
    acc["log_sq_sum"] += float((log_err * log_err).sum().item())
    acc["log10_abs_sum"] += float(torch.abs(torch.log10(pred_v) - torch.log10(gt_v)).sum().item())
    acc["silog_d_sum"] += float(log_err.sum().item())
    acc["silog_d2_sum"] += float((log_err * log_err).sum().item())
    acc["delta1"] += int((ratio < 1.25).sum().item())
    acc["delta2"] += int((ratio < 1.25**2).sum().item())
    acc["delta3"] += int((ratio < 1.25**3).sum().item())


def finalize_metrics(acc: Dict[str, float]) -> Dict[str, float]:
    n = max(1.0, float(acc["valid_pixels"]))
    label_n = max(1.0, float(acc["label_pixels"]))
    log_mean = acc["silog_d_sum"] / n
    log_sq_mean = acc["silog_d2_sum"] / n
    silog = float(np.sqrt(max(log_sq_mean - log_mean * log_mean, 0.0)) * 100.0)
    return {
        "frames": int(acc["frames"]),
        "label_pixels": int(acc["label_pixels"]),
        "valid_pixels": int(acc["valid_pixels"]),
        "valid_pixel_fraction": float(acc["valid_pixels"] / label_n),
        "invalid_pred_fraction": float(acc["invalid_pred_pixels"] / label_n),
        "pred_below_min_fraction": float(acc["pred_below_min_pixels"] / label_n),
        "pred_above_max_fraction": float(acc["pred_above_max_pixels"] / label_n),
        "abs_rel": float(acc["abs_rel_sum"] / n),
        "sq_rel": float(acc["sq_rel_sum"] / n),
        "mae": float(acc["abs_sum"] / n),
        "rmse": float(np.sqrt(acc["sq_sum"] / n)),
        "rmse_log": float(np.sqrt(acc["log_sq_sum"] / n)),
        "log_l1": float(acc["log_abs_sum"] / n),
        "log10": float(acc["log10_abs_sum"] / n),
        "silog": silog,
        "delta1": float(acc["delta1"] / n),
        "delta2": float(acc["delta2"] / n),
        "delta3": float(acc["delta3"] / n),
        "training_loss_proxy": float((acc["log_abs_sum"] / n) + 0.2 * (acc["abs_rel_sum"] / n)),
    }


def evaluate_checkpoint(args: argparse.Namespace, loader: DataLoader, checkpoint_arg: str) -> Dict[str, object]:
    label, model, device, checkpoint_meta = load_model(args, checkpoint_arg)
    dtype = autocast_dtype(args.precision)
    acc: Dict[str, float] = {
        "frames": 0,
        "label_pixels": 0,
        "valid_pixels": 0,
        "invalid_pred_pixels": 0,
        "pred_above_max_pixels": 0,
        "pred_below_min_pixels": 0,
        "abs_rel_sum": 0.0,
        "sq_rel_sum": 0.0,
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "log_abs_sum": 0.0,
        "log_sq_sum": 0.0,
        "log10_abs_sum": 0.0,
        "silog_d_sum": 0.0,
        "silog_d2_sum": 0.0,
        "delta1": 0.0,
        "delta2": 0.0,
        "delta3": 0.0,
    }

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            gt = batch["depth"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            intr = batch["intrinsics"].to(device, non_blocking=True)
            enabled = dtype is not None and device.type == "cuda"
            autocast_ctx = torch.autocast(device.type, dtype=dtype) if enabled else nullcontext()
            with autocast_ctx:
                output = model.model(images[:, None])
                pred = da3_depth_tensor(output)
                if not args.no_canonical_focal_scale:
                    focal = 0.5 * (intr[:, 0] + intr[:, 1])
                    pred = pred * (focal[:, None, None] / 300.0)
                if pred.shape[-2:] != gt.shape[-2:]:
                    pred = F.interpolate(pred[:, None], size=gt.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            update_metrics(acc, pred.float(), gt.float(), mask.bool(), args)
            acc["frames"] += int(images.shape[0])

    metrics = finalize_metrics(acc)
    result = {
        "name": label,
        **checkpoint_meta,
        "metrics": metrics,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.sample_mod < 1:
        raise ValueError("--sample-mod must be >= 1")
    if not args.checkpoint:
        args.checkpoint = ["pretrained"]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    loader, total_records, selected_records = build_loader(args)
    results = []
    for checkpoint_arg in args.checkpoint:
        print(f"Evaluating {checkpoint_arg}", flush=True)
        result = evaluate_checkpoint(args, loader, checkpoint_arg)
        results.append(result)
        metrics = result["metrics"]
        print(
            "{name}: frames={frames} valid_pixels={valid_pixels} "
            "abs_rel={abs_rel:.5f} rmse={rmse:.3f} silog={silog:.3f} "
            "delta1={delta1:.4f}".format(name=result["name"], **metrics),
            flush=True,
        )

    payload = {
        "manifest": str(args.manifest),
        "total_records": total_records,
        "selected_records": selected_records,
        "image_size": list(args.image_size),
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "canonical_focal_scale": not args.no_canonical_focal_scale,
        "median_align": args.median_align,
        "results": results,
    }

    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
