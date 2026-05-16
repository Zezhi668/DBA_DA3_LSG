#!/usr/bin/env python3
"""Fine-tune the DA3 metric depth head on MARS-LIVG sparse LiDAR labels."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DA3Metric-Large on MARS-LIVG.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("/home/server/VINGS_work/Depth-Anything-3/src"))
    parser.add_argument("--pretrained", type=str, default="/home/server/VINGS_work/DPT-LSG/ckpts/DA3Metric-Large")
    parser.add_argument("--output", type=Path, default=Path("/home/server/VINGS_work/DPT-LSG/ckpts/da3_mars_metric_head.pth"))
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional DA3 training checkpoint to initialize from before continuing on this manifest.",
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Also restore optimizer/scaler state from --init-checkpoint when present.",
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=(448, 560), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Optional integer number of full dataloader epochs. "
            "When set, this overrides --steps with epochs * len(dataloader)."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=300.0)
    parser.add_argument("--valid-min-pixels", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--canonical-focal-scale", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--save-every", type=int, default=2000)
    return parser.parse_args()


class MarsSparseDepthDataset(Dataset):
    def __init__(self, manifest: Path, image_size: Tuple[int, int], min_valid_pixels: int):
        self.image_size = tuple(int(v) for v in image_size)
        self.records = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if int(record.get("valid_pixels", 0)) >= min_valid_pixels:
                        self.records.append(record)
        if not self.records:
            raise ValueError(f"No usable records found in {manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def _load_sparse_depth(
        self,
        record: Dict[str, object],
        out_h: int,
        out_w: int,
        orig_h: int,
        orig_w: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        depth_path = Path(str(record["depth"]))
        depth_format = str(record.get("depth_format", "dense_npy"))
        if depth_format != "sparse_npz" and depth_path.suffix != ".npz":
            depth = np.load(depth_path).astype(np.float32)
            mask = cv2.imread(str(record["mask"]), cv2.IMREAD_GRAYSCALE) > 0
            depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
            return depth, mask

        sparse = np.load(depth_path)
        if "height" in sparse and "width" in sparse:
            orig_h = int(sparse["height"])
            orig_w = int(sparse["width"])
        u = sparse["u"].astype(np.float32)
        v = sparse["v"].astype(np.float32)
        z = sparse["depth"].astype(np.float32)

        depth_flat = np.zeros(out_h * out_w, dtype=np.float32)
        if z.size == 0:
            return depth_flat.reshape(out_h, out_w), depth_flat.reshape(out_h, out_w).astype(bool)

        uu = np.rint(u * (out_w / float(orig_w))).astype(np.int64)
        vv = np.rint(v * (out_h / float(orig_h))).astype(np.int64)
        valid = (uu >= 0) & (uu < out_w) & (vv >= 0) & (vv < out_h) & np.isfinite(z) & (z > 0)
        uu, vv, z = uu[valid], vv[valid], z[valid]
        if z.size == 0:
            return depth_flat.reshape(out_h, out_w), depth_flat.reshape(out_h, out_w).astype(bool)

        flat = vv * out_w + uu
        order = np.argsort(z)
        flat_sorted = flat[order]
        z_sorted = z[order]
        unique_flat, first = np.unique(flat_sorted, return_index=True)
        depth_flat[unique_flat] = z_sorted[first]
        depth = depth_flat.reshape(out_h, out_w)
        return depth, depth > 0

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(record["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]
        out_h, out_w = self.image_size
        image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
        depth, mask = self._load_sparse_depth(record, out_h, out_w, orig_h, orig_w)

        fx, fy, cx, cy = [float(v) for v in record["intrinsics_fx_fy_cx_cy"]]
        scale_x = out_w / float(orig_w)
        scale_y = out_h / float(orig_h)
        intrinsics = torch.tensor(
            [fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y],
            dtype=torch.float32,
        )

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_t = (image_t - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": image_t,
            "depth": torch.from_numpy(depth).float(),
            "mask": torch.from_numpy(mask.astype(np.bool_)),
            "intrinsics": intrinsics,
        }


def da3_depth_tensor(output) -> torch.Tensor:
    depth = output["depth"] if isinstance(output, dict) else output.depth
    if depth.ndim == 5:
        depth = depth[:, 0]
    if depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    return depth.float()


def masked_metric_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    min_depth: float,
    max_depth: float,
) -> torch.Tensor:
    valid = (
        mask
        & torch.isfinite(pred)
        & torch.isfinite(gt)
        & (gt > min_depth)
        & (gt < max_depth)
        & (pred > min_depth)
        & (pred < max_depth)
    )
    if int(valid.sum().item()) == 0:
        return pred.sum() * 0.0
    pred_v = pred[valid]
    gt_v = gt[valid]
    log_l1 = (torch.log(pred_v) - torch.log(gt_v)).abs().mean()
    rel_l1 = ((pred_v - gt_v).abs() / torch.clamp(gt_v, min=min_depth)).mean()
    return log_l1 + 0.2 * rel_l1


def trainable_parameters(model, unfreeze_backbone: bool):
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.model.head.parameters():
        param.requires_grad_(True)
    if unfreeze_backbone:
        for param in model.model.backbone.parameters():
            param.requires_grad_(True)
    return [param for param in model.parameters() if param.requires_grad]


def checkpoint_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint)
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    if not isinstance(state, dict):
        raise TypeError("Checkpoint state_dict must be a mapping.")
    return state


def load_checkpoint(
    model,
    optimizer,
    scaler,
    args: argparse.Namespace,
) -> int:
    if args.init_checkpoint is None:
        return 0
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
    state = checkpoint_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"loaded init checkpoint {args.init_checkpoint} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})",
        flush=True,
    )
    if args.resume_optimizer and isinstance(checkpoint, dict):
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("restored optimizer state", flush=True)
        if "scaler" in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
            print("restored AMP scaler state", flush=True)
        return int(checkpoint.get("step", 0))
    return 0


def save_checkpoint(model, optimizer, scaler, args: argparse.Namespace, step: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "steps": args.steps,
            "epochs": args.epochs,
            "model_name": "da3metric-large",
            "image_size": list(args.image_size),
            "source_pretrained": args.pretrained,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.source_dir) not in sys.path:
        sys.path.insert(0, str(args.source_dir))

    from depth_anything_3.api import DepthAnything3

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = MarsSparseDepthDataset(args.manifest, tuple(args.image_size), args.valid_min_pixels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    if len(loader) <= 0:
        raise ValueError(
            "Dataloader has no batches. Reduce --batch-size or check the training manifest."
        )
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be a positive integer when provided.")
        args.steps = int(args.epochs) * len(loader)
    print(
        f"training records={len(dataset)} batch_size={args.batch_size} "
        f"steps_per_epoch={len(loader)} epochs={args.epochs} steps={args.steps}",
        flush=True,
    )
    iterator = iter(loader)

    pretrained = args.pretrained
    if pretrained.startswith("/") and not Path(pretrained).exists():
        print(f"pretrained path does not exist: {pretrained}; falling back to depth-anything/DA3METRIC-LARGE")
        pretrained = "depth-anything/DA3METRIC-LARGE"
    model = DepthAnything3.from_pretrained(pretrained).to(device)
    model.train()
    params = trainable_parameters(model, args.unfreeze_backbone)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    if args.precision == "bf16":
        autocast_dtype = torch.bfloat16
    elif args.precision == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.precision == "fp16")
    start_step = load_checkpoint(model, optimizer, scaler, args)

    for local_step in range(1, args.steps + 1):
        step = start_step + local_step
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        images = batch["image"].to(device, non_blocking=True)
        gt_depth = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        intr = batch["intrinsics"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=device.type == "cuda" and args.precision != "fp32"):
            output = model.model(images[:, None])
            pred = da3_depth_tensor(output)
            if args.canonical_focal_scale:
                focal = 0.5 * (intr[:, 0] + intr[:, 1])
                pred = pred * (focal[:, None, None] / 300.0)
            if pred.shape[-2:] != gt_depth.shape[-2:]:
                pred = F.interpolate(pred[:, None], size=gt_depth.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            loss = masked_metric_loss(pred, gt_depth, mask, args.min_depth, args.max_depth)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 50 == 0:
            print(f"step={step} loss={float(loss.detach().cpu()):.6f}")
        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, scaler, args, step, args.output.with_name(f"{args.output.stem}_step{step}{args.output.suffix}"))

    save_checkpoint(model, optimizer, scaler, args, start_step + args.steps, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
