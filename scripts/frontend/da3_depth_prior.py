"""Depth Anything 3 metric depth priors for the DROID frontend.

Splat-SLAM's DSPO alternates photometric DBA with a monocular depth prior

    r_mono(u) = d_i(u) - 1 / (theta_i D_mono_i(u) + gamma_i)

because its monocular depth predictor is relative.  DA3 Metric already
predicts canonical metric depth, so this frontend uses the simpler prior

    r_da3(u) = d_i(u) - 1 / (s_f D_DA3_i(u) + eps)

where ``s_f`` is the optional DA3 canonical-focal conversion.  For
DA3METRIC-LARGE this conversion is ``focal / 300``.  The residual is
spliced into DROID by writing ``disps_sens``; the CUDA BA kernel already adds
the depth-prior Schur terms when ``disps_sens > 0``.
"""

from __future__ import annotations

import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _cfg_get(cfg: Dict[str, object], key: str, default):
    return cfg[key] if key in cfg else default


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@dataclass
class DA3DepthPriorConfig:
    enabled: bool = False
    model_id: str = "depth-anything/DA3METRIC-LARGE"
    model_name: str = "da3metric-large"
    source_dir: Optional[str] = None
    model_dir: Optional[str] = None
    checkpoint_path: Optional[str] = None
    device: str = "cuda:0"
    precision: str = "float16"
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    min_depth: float = 0.2
    max_depth: float = 300.0
    depth_scale: float = 1.0
    canonical_focal_scale: bool = True
    replace_existing_depth: bool = True
    async_mode: bool = True
    frontend_refine_updates: int = 1
    seed_initialized_window: bool = True
    max_seed_keyframes: int = 8
    max_pending: int = 2
    anchor_mode: str = "affine"
    align_min_pixels: int = 512
    align_trim: float = 0.1
    align_min_scale: float = 0.25
    align_max_scale: float = 4.0
    align_min_shift: float = -80.0
    align_max_shift: float = 80.0
    align_blend: float = 1.0
    require_sane_fit_for_replace: bool = False

    @classmethod
    def from_cfg(cls, cfg: Dict[str, object]) -> "DA3DepthPriorConfig":
        da3_cfg = dict(cfg.get("da3", {}))
        tracker_device = cfg.get("device", {}).get("tracker", "cuda:0")
        return cls(
            enabled=_as_bool(_cfg_get(da3_cfg, "enabled", False)),
            model_id=str(_cfg_get(da3_cfg, "model_id", "depth-anything/DA3METRIC-LARGE")),
            model_name=str(_cfg_get(da3_cfg, "model_name", "da3metric-large")),
            source_dir=_cfg_get(da3_cfg, "source_dir", None),
            model_dir=_cfg_get(da3_cfg, "model_dir", None),
            checkpoint_path=_cfg_get(da3_cfg, "checkpoint_path", None),
            device=str(_cfg_get(da3_cfg, "device", tracker_device)),
            precision=str(_cfg_get(da3_cfg, "precision", "float16")).lower(),
            process_res=int(_cfg_get(da3_cfg, "process_res", 504)),
            process_res_method=str(_cfg_get(da3_cfg, "process_res_method", "upper_bound_resize")),
            min_depth=float(_cfg_get(da3_cfg, "min_depth", 0.2)),
            max_depth=float(_cfg_get(da3_cfg, "max_depth", 300.0)),
            depth_scale=float(_cfg_get(da3_cfg, "depth_scale", 1.0)),
            canonical_focal_scale=_as_bool(_cfg_get(da3_cfg, "canonical_focal_scale", True)),
            replace_existing_depth=_as_bool(_cfg_get(da3_cfg, "replace_existing_depth", True)),
            async_mode=_as_bool(_cfg_get(da3_cfg, "async", _cfg_get(da3_cfg, "async_mode", True))),
            frontend_refine_updates=int(_cfg_get(da3_cfg, "frontend_refine_updates", 1)),
            seed_initialized_window=_as_bool(_cfg_get(da3_cfg, "seed_initialized_window", True)),
            max_seed_keyframes=int(_cfg_get(da3_cfg, "max_seed_keyframes", 8)),
            max_pending=int(_cfg_get(da3_cfg, "max_pending", 2)),
            anchor_mode=str(_cfg_get(da3_cfg, "anchor_mode", "affine")).lower(),
            align_min_pixels=int(_cfg_get(da3_cfg, "align_min_pixels", 512)),
            align_trim=float(_cfg_get(da3_cfg, "align_trim", 0.1)),
            align_min_scale=float(_cfg_get(da3_cfg, "align_min_scale", 0.25)),
            align_max_scale=float(_cfg_get(da3_cfg, "align_max_scale", 4.0)),
            align_min_shift=float(_cfg_get(da3_cfg, "align_min_shift", -80.0)),
            align_max_shift=float(_cfg_get(da3_cfg, "align_max_shift", 80.0)),
            align_blend=float(_cfg_get(da3_cfg, "align_blend", 1.0)),
            require_sane_fit_for_replace=_as_bool(_cfg_get(da3_cfg, "require_sane_fit_for_replace", False)),
        )


class DA3MetricLargePredictor:
    """Lazy Depth Anything 3 Metric Large predictor.

    The official API uses ``DepthAnything3.from_pretrained`` and handles
    autocast internally. Keep parameters in fp32 by default because DA3's input
    processor feeds fp32 images into the network before its own autocast scope.
    """

    def __init__(self, config: DA3DepthPriorConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = self._resolve_dtype(config.precision)
        self.model = None

    @staticmethod
    def _resolve_dtype(precision: str) -> torch.dtype:
        precision = precision.lower()
        if precision in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if precision in {"fp16", "float16", "half"}:
            return torch.float16
        if precision in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported DA3 precision: {precision}")

    def _model_source(self) -> str:
        if self.config.model_dir:
            if not Path(self.config.model_dir).exists():
                print(
                    f"DA3 model_dir does not exist: {self.config.model_dir}. "
                    f"Falling back to Hugging Face model_id {self.config.model_id}."
                )
                return self.config.model_id
            return self.config.model_dir
        return self.config.model_id

    def _load(self) -> None:
        if self.model is not None:
            return

        if self.config.source_dir and self.config.source_dir not in sys.path:
            sys.path.insert(0, self.config.source_dir)

        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise ImportError(
                "Depth Anything 3 is not installed in this environment. "
                "Clone the official repository and set da3.source_dir to its "
                "src directory, or install it with --no-deps so pip does not "
                "replace the DPT-LSG torch stack."
            ) from exc

        checkpoint_path = self.config.checkpoint_path
        if checkpoint_path and checkpoint_path.endswith((".pt", ".pth")):
            model = DepthAnything3(model_name=self.config.model_name)
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"DA3 checkpoint load warning: {len(missing)} missing keys")
            if unexpected:
                print(f"DA3 checkpoint load warning: {len(unexpected)} unexpected keys")
        else:
            model = DepthAnything3.from_pretrained(self._model_source())

        model = model.to(device=self.device)
        model.eval()
        self.model = model

    @staticmethod
    def _intrinsics_to_matrix(intrinsics: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if intrinsics is None:
            return None
        intr = intrinsics.detach().float().cpu().reshape(-1)
        if intr.numel() < 4:
            return None
        # Dataset packets pass resized intrinsics as [fx, fy, cx, cy].
        fx, fy, cx, cy = [float(v.item()) for v in intr[:4]]
        K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        return K[None]

    @staticmethod
    def _focal_scale(intrinsics: Optional[torch.Tensor]) -> float:
        if intrinsics is None:
            return 1.0
        intr = intrinsics.detach().float().reshape(-1)
        if intr.numel() < 2:
            return 1.0
        # DA3METRIC-LARGE converts canonical output to metric depth as
        # depth_m = raw_output * focal / 300.
        focal = 0.5 * (intr[0].item() + intr[1].item())
        return float(focal / 300.0)

    def predict(
        self,
        image_chw_uint8: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a ``(H, W)`` metric proxy depth tensor on ``self.device``."""
        self._load()
        assert self.model is not None

        if image_chw_uint8.ndim != 3 or image_chw_uint8.shape[0] != 3:
            raise ValueError("DA3 expects a CHW RGB uint8 image")

        height, width = int(image_chw_uint8.shape[-2]), int(image_chw_uint8.shape[-1])
        image_np = image_chw_uint8.detach().permute(1, 2, 0).cpu().numpy()
        intrinsics_np = self._intrinsics_to_matrix(intrinsics)

        with torch.inference_mode():
            prediction = self.model.inference(
                [image_np],
                intrinsics=intrinsics_np,
                process_res=self.config.process_res,
                process_res_method=self.config.process_res_method,
                export_dir=None,
            )

        depth = getattr(prediction, "depth", None)
        if depth is None:
            raise RuntimeError("DA3 prediction did not expose a depth field")
        if isinstance(depth, np.ndarray):
            depth_tensor = torch.from_numpy(depth)
        else:
            depth_tensor = torch.as_tensor(depth)
        if depth_tensor.ndim == 3:
            depth_tensor = depth_tensor[0]
        depth_tensor = depth_tensor.to(device=self.device, dtype=torch.float32)

        if depth_tensor.shape[-2:] != (height, width):
            depth_tensor = F.interpolate(
                depth_tensor[None, None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        if self.config.canonical_focal_scale:
            depth_tensor = depth_tensor * self._focal_scale(intrinsics)
        if self.config.depth_scale != 1.0:
            depth_tensor = depth_tensor * float(self.config.depth_scale)

        return depth_tensor


class DroidDA3DepthPrior:
    """Keyframe-only DA3 depth hook for ``DepthVideo.disps_sens``."""

    def __init__(self, config: DA3DepthPriorConfig):
        self.config = config
        self.enabled = bool(config.enabled)
        self.predictor: Optional[DA3MetricLargePredictor] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.pending: Dict[float, Future] = {}
        self.processed_tstamps = set()
        self.anchor_params: Dict[float, Tuple[float, float, str]] = {}
        self._seeded_initialized_window = False
        if self.enabled:
            self.predictor = DA3MetricLargePredictor(config)
            if config.async_mode:
                self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="da3-depth")

    @classmethod
    def from_cfg(cls, cfg: Dict[str, object]) -> "DroidDA3DepthPrior":
        return cls(DA3DepthPriorConfig.from_cfg(cfg))

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None

    @staticmethod
    def _full_res_intrinsics(video, index: int) -> torch.Tensor:
        return video.intrinsics[index].detach() * 8.0

    @staticmethod
    def _timestamp(video, index: int) -> float:
        return float(video.tstamp[index].detach().cpu().item())

    def _predict_local(self, video, index: int) -> Tuple[float, torch.Tensor]:
        assert self.predictor is not None
        image = video.images[index].detach()
        intrinsics = self._full_res_intrinsics(video, index)
        tstamp = self._timestamp(video, index)
        depth = self.predictor.predict(image, intrinsics)
        return tstamp, depth

    def _local_index_from_timestamp(self, video, tstamp: float) -> Optional[int]:
        count = int(video.counter.value)
        if count <= 0:
            return None
        stamps = video.tstamp[:count].detach().cpu()
        matches = torch.where(torch.isclose(stamps, torch.tensor(tstamp, dtype=stamps.dtype), atol=1e-6))[0]
        if matches.numel() == 0:
            return None
        return int(matches[-1].item())

    def _attach_depth(self, video, index: int, depth: torch.Tensor) -> None:
        depth = depth.to(device=video.disps_sens.device, dtype=torch.float32)
        depth, scale, shift, mode = self._align_depth_anchor(video, index, depth)
        valid = torch.isfinite(depth) & (depth > self.config.min_depth) & (depth < self.config.max_depth)
        disp_full = torch.zeros_like(depth, dtype=torch.float32)
        disp_full[valid] = 1.0 / torch.clamp(depth[valid], min=1e-6)
        disp_ds = disp_full[3::8, 3::8].contiguous()
        can_replace = self.config.replace_existing_depth
        if self.config.require_sane_fit_for_replace:
            can_replace = can_replace and self._alignment_is_sane(scale, shift, mode)

        with video.get_lock():
            if can_replace:
                video.disps_sens[index] = disp_ds
            else:
                keep_existing = video.disps_sens[index] > 0
                video.disps_sens[index] = torch.where(keep_existing, video.disps_sens[index], disp_ds)
            video.disps[index] = torch.where(video.disps_sens[index] > 0, video.disps_sens[index], video.disps[index])
        self.anchor_params[self._timestamp(video, index)] = (scale, shift, mode)

    def _alignment_is_sane(self, scale: float, shift: float, mode: str) -> bool:
        if "no-fit" in mode or "unknown" in mode or "clamped" in mode:
            return False
        if mode == "fixed" and self.config.anchor_mode not in {"none", "fixed", "metric"}:
            return False
        return (
            np.isfinite(scale)
            and np.isfinite(shift)
            and self.config.align_min_scale <= scale <= self.config.align_max_scale
            and self.config.align_min_shift <= shift <= self.config.align_max_shift
        )

    def _align_depth_anchor(
        self,
        video,
        index: int,
        da3_depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float, str]:
        """Fit a DSPO-style depth affine to the current DROID depth estimate.

        Splat-SLAM optimizes a per-frame linear depth transform for relative
        monocular depth. The existing DROID BA kernel only accepts fixed inverse
        depth measurements, so we estimate the affine outside BA and refresh the
        anchor asynchronously when DA3 finishes.
        """
        mode = self.config.anchor_mode
        if mode in {"none", "fixed", "metric"}:
            return da3_depth, 1.0, 0.0, "fixed"

        da3_ds = da3_depth[3::8, 3::8].detach().float()
        droid_disp = video.disps[index].detach().float()
        droid_valid = torch.isfinite(droid_disp) & (droid_disp > 1.0 / self.config.max_depth)
        droid_depth = torch.zeros_like(droid_disp)
        droid_depth[droid_valid] = 1.0 / torch.clamp(droid_disp[droid_valid], min=1e-6)

        valid = (
            torch.isfinite(da3_ds)
            & torch.isfinite(droid_depth)
            & (da3_ds > self.config.min_depth)
            & (da3_ds < self.config.max_depth)
            & (droid_depth > self.config.min_depth)
            & (droid_depth < self.config.max_depth)
        )
        if int(valid.sum().item()) < self.config.align_min_pixels:
            return da3_depth, 1.0, 0.0, "fixed-no-fit"

        x = da3_ds[valid]
        y = droid_depth[valid]
        if x.numel() > 50000:
            step = max(1, x.numel() // 50000)
            x = x[::step]
            y = y[::step]

        ratio = y / torch.clamp(x, min=1e-6)
        ratio_valid = torch.isfinite(ratio) & (ratio > self.config.align_min_scale) & (ratio < self.config.align_max_scale)
        x = x[ratio_valid]
        y = y[ratio_valid]
        ratio = ratio[ratio_valid]
        if x.numel() < self.config.align_min_pixels:
            return da3_depth, 1.0, 0.0, "fixed-no-fit"

        trim = min(max(float(self.config.align_trim), 0.0), 0.45)
        if trim > 0.0 and x.numel() >= 32:
            lo = torch.quantile(ratio, trim)
            hi = torch.quantile(ratio, 1.0 - trim)
            keep = (ratio >= lo) & (ratio <= hi)
            x = x[keep]
            y = y[keep]
            ratio = ratio[keep]

        if x.numel() < self.config.align_min_pixels:
            return da3_depth, 1.0, 0.0, "fixed-no-fit"

        if mode in {"scale", "linear-scale"}:
            scale = torch.median(ratio)
            shift = torch.zeros_like(scale)
            fit_mode = "scale"
        elif mode in {"affine", "linear", "dspo"}:
            x_mean = x.mean()
            y_mean = y.mean()
            var = torch.mean((x - x_mean) ** 2)
            if not torch.isfinite(var) or float(var.item()) < 1e-6:
                scale = torch.median(ratio)
                shift = torch.zeros_like(scale)
                fit_mode = "scale-fallback"
            else:
                scale = torch.mean((x - x_mean) * (y - y_mean)) / var
                shift = y_mean - scale * x_mean
                fit_mode = "affine"
        else:
            return da3_depth, 1.0, 0.0, "fixed-unknown-mode"

        raw_scale = scale
        raw_shift = shift
        scale = torch.clamp(scale, self.config.align_min_scale, self.config.align_max_scale)
        shift = torch.clamp(shift, self.config.align_min_shift, self.config.align_max_shift)
        if (
            not torch.isclose(raw_scale, scale, rtol=1e-4, atol=1e-4)
            or not torch.isclose(raw_shift, shift, rtol=1e-4, atol=1e-4)
        ):
            fit_mode = f"{fit_mode}-clamped"
        blend = min(max(float(self.config.align_blend), 0.0), 1.0)
        if blend < 1.0:
            scale = (1.0 - blend) + blend * scale
            shift = blend * shift

        aligned = scale * da3_depth + shift
        aligned = torch.where(aligned > self.config.min_depth, aligned, da3_depth)
        return aligned, float(scale.item()), float(shift.item()), fit_mode

    def poll_ready(self, video) -> int:
        if not self.enabled or not self.pending:
            return 0
        attached = 0
        for tstamp, future in list(self.pending.items()):
            if not future.done():
                continue
            del self.pending[tstamp]
            try:
                result_tstamp, depth = future.result()
            except Exception as exc:
                print(f"DA3 depth prediction failed for timestamp {tstamp}: {exc}")
                continue
            index = self._local_index_from_timestamp(video, result_tstamp)
            if index is None:
                continue
            self._attach_depth(video, index, depth)
            self.processed_tstamps.add(result_tstamp)
            attached += 1
        return attached

    def submit_or_attach(self, video, index: int) -> bool:
        if not self.enabled:
            return False
        tstamp = self._timestamp(video, index)
        if tstamp in self.processed_tstamps or tstamp in self.pending:
            return False

        if self.executor is not None:
            if len(self.pending) >= max(1, int(self.config.max_pending)):
                return False
            self.pending[tstamp] = self.executor.submit(self._predict_local, video, index)
            return False

        result_tstamp, depth = self._predict_local(video, index)
        self._attach_depth(video, index, depth)
        self.processed_tstamps.add(result_tstamp)
        return True

    def refine_frontend(self, graph) -> None:
        if not self.enabled or graph is None:
            return
        if self.config.async_mode:
            return
        for _ in range(max(0, int(self.config.frontend_refine_updates))):
            if graph.ii.numel() > 0:
                graph.update(None, None, use_inactive=True)

    def on_keyframe(self, video, local_index: int, graph=None) -> None:
        if not self.enabled or local_index < 0 or local_index >= int(video.counter.value):
            return
        attached = self.submit_or_attach(video, local_index)
        if attached:
            self.refine_frontend(graph)

    def seed_initialized_window(self, video, t1: int, graph=None) -> None:
        if not self.enabled or self._seeded_initialized_window:
            return
        if not self.config.seed_initialized_window:
            self._seeded_initialized_window = True
            return

        end = max(0, int(t1) - 1)
        start = max(0, end - int(self.config.max_seed_keyframes))
        attached_any = False
        for index in range(start, end):
            attached_any = self.submit_or_attach(video, index) or attached_any
        self._seeded_initialized_window = True
        if attached_any:
            self.refine_frontend(graph)
