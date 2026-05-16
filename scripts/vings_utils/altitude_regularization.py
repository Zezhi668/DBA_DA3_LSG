"""Terrain-agnostic constant-altitude pose regularization.

This loss anchors camera translation, not scene depth:

    L_altitude = lambda_alt / N * sum_i rho(t_i,z - z_anchor)

where ``t_i,z`` is the camera center Z coordinate in the world or current
submap frame.  No per-pixel depth variance term is used, so terrain, buildings,
trees, and hills remain free to vary in the depth maps and Gaussian geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg: Dict[str, object], key: str, default):
    return cfg[key] if key in cfg else default


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@dataclass
class ConstantAltitudeConfig:
    enabled: bool = False
    lambda_alt: float = 0.0
    axis: int = 2
    huber_delta: float = 1.0
    anchor_mode: str = "submap_first"
    fixed_z: Optional[float] = None

    @classmethod
    def from_cfg(cls, cfg: Dict[str, object], nested_key: str = "constant_altitude") -> "ConstantAltitudeConfig":
        alt_cfg = dict(cfg.get(nested_key, {}))
        if not alt_cfg and nested_key != "altitude":
            alt_cfg = dict(cfg.get("altitude", {}))
        return cls(
            enabled=_as_bool(_cfg_get(alt_cfg, "enabled", False)),
            lambda_alt=float(_cfg_get(alt_cfg, "lambda_alt", 0.0)),
            axis=int(_cfg_get(alt_cfg, "axis", 2)),
            huber_delta=float(_cfg_get(alt_cfg, "huber_delta", 1.0)),
            anchor_mode=str(_cfg_get(alt_cfg, "anchor_mode", "submap_first")),
            fixed_z=_cfg_get(alt_cfg, "fixed_z", None),
        )


class ConstantAltitudeLoss(nn.Module):
    """Soft constant-altitude loss over camera extrinsics."""

    def __init__(self, config: ConstantAltitudeConfig):
        super().__init__()
        self.config = config

    @classmethod
    def from_cfg(cls, cfg: Dict[str, object], nested_key: str = "constant_altitude") -> "ConstantAltitudeLoss":
        return cls(ConstantAltitudeConfig.from_cfg(cfg, nested_key=nested_key))

    @property
    def active(self) -> bool:
        return bool(self.config.enabled and self.config.lambda_alt > 0.0)

    def anchor_from_poses(self, c2w_poses: torch.Tensor) -> torch.Tensor:
        axis = int(self.config.axis)
        if self.config.fixed_z is not None:
            return torch.as_tensor(self.config.fixed_z, dtype=c2w_poses.dtype, device=c2w_poses.device)
        tz = c2w_poses[..., axis, 3].detach()
        if self.config.anchor_mode in {"mean", "submap_mean", "batch_mean"}:
            return tz.mean()
        return tz.reshape(-1)[0]

    def forward(
        self,
        c2w_poses: torch.Tensor,
        anchor_z: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.active:
            return c2w_poses.new_zeros(())
        if c2w_poses.numel() == 0:
            return c2w_poses.new_zeros(())

        axis = int(self.config.axis)
        if anchor_z is None:
            anchor_z = self.anchor_from_poses(c2w_poses)
        else:
            anchor_z = torch.as_tensor(anchor_z, dtype=c2w_poses.dtype, device=c2w_poses.device)

        residual = c2w_poses[..., axis, 3] - anchor_z
        if weights is not None:
            weights = weights.to(device=c2w_poses.device, dtype=c2w_poses.dtype)
            residual = residual * weights

        delta = float(self.config.huber_delta)
        if delta > 0.0:
            loss = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=delta, reduction="mean")
        else:
            loss = (residual * residual).mean()
        return float(self.config.lambda_alt) * loss


def numpy_constant_altitude_error(sim3_vec: np.ndarray, anchor_z: float, axis: int, sim3_exp_fn) -> np.ndarray:
    """Return the 1D altitude residual for a Sim3 pose-graph node."""
    sim3 = sim3_exp_fn(sim3_vec)
    return np.asarray([float(sim3.t[int(axis)] - anchor_z)], dtype=np.float64)

