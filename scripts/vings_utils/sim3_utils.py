import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class Sim3Transform:
    # Standard similarity transform on points: x' = s * R * x + t
    R: np.ndarray
    t: np.ndarray
    s: float


def _as_float64(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def skew(w: np.ndarray) -> np.ndarray:
    w = _as_float64(w).reshape(3)
    wx, wy, wz = w
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ],
        dtype=np.float64,
    )


def so3_exp(w: np.ndarray) -> np.ndarray:
    w = _as_float64(w).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + skew(w)
    wx = skew(w / theta)
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(theta) * wx
        + (1.0 - math.cos(theta)) * (wx @ wx)
    )


def so3_log(R: np.ndarray) -> np.ndarray:
    R = _as_float64(R).reshape(3, 3)
    trace = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(trace)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    vee = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    )
    return 0.5 * theta / math.sin(theta) * vee


def sim3_get_V(w: np.ndarray, lam: float) -> np.ndarray:
    w = _as_float64(w).reshape(3)
    theta2 = float(w @ w)
    if theta2 > 1e-9:
        theta = math.sqrt(theta2)
        X = math.sin(theta) / theta
        Y = (1.0 - math.cos(theta)) / theta2
        Z = (1.0 - X) / theta2
        W = (0.5 - Y) / theta2
    else:
        Y = 0.5 - theta2 / 24.0
        Z = 1.0 / 6.0 - theta2 / 120.0
        W = 1.0 / 24.0 - theta2 / 720.0

    lam2 = lam * lam
    lam3 = lam2 * lam
    exp_min_lam = math.exp(-lam)
    if lam2 > 1e-9:
        A = (1.0 - exp_min_lam) / lam
        alpha = 1.0 / (1.0 + theta2 / lam2)
        beta = (exp_min_lam - 1.0 + lam) / lam2
        mu = (1.0 - lam + 0.5 * lam2 - exp_min_lam) / lam3
    else:
        A = 1.0 - lam / 2.0 + lam2 / 6.0
        alpha = 0.0
        beta = 0.5 - lam / 6.0 + lam2 / 24.0 - lam3 / 120.0
        mu = 1.0 / 6.0 - lam / 24.0 + lam2 / 120.0 - lam3 / 720.0

    gamma = Y - lam * Z
    upsilon = Z - lam * W
    B = alpha * (beta - gamma) + gamma
    C = alpha * (mu - upsilon) + upsilon
    wx = skew(w)
    ident = np.eye(3, dtype=np.float64)
    return A * ident + B * wx + C * (wx @ wx)


def sim3_identity() -> Sim3Transform:
    return Sim3Transform(
        R=np.eye(3, dtype=np.float64),
        t=np.zeros(3, dtype=np.float64),
        s=1.0,
    )


def sim3_exp(xi: np.ndarray) -> Sim3Transform:
    xi = _as_float64(xi).reshape(7)
    w = xi[:3]
    u = xi[3:6]
    lam = float(xi[6])
    R = so3_exp(w)
    V = sim3_get_V(w, lam)
    s = math.exp(lam)
    t = s * (V @ u)
    return Sim3Transform(R=R, t=t, s=s)


def sim3_log(S: Sim3Transform) -> np.ndarray:
    w = so3_log(S.R)
    lam = math.log(max(float(S.s), 1e-12))
    V = sim3_get_V(w, lam)
    u = np.linalg.solve(V, _as_float64(S.t).reshape(3) / max(float(S.s), 1e-12))
    return np.concatenate([w, u, np.array([lam], dtype=np.float64)], axis=0)


def sim3_compose(A: Sim3Transform, B: Sim3Transform) -> Sim3Transform:
    R = A.R @ B.R
    s = float(A.s) * float(B.s)
    t = float(A.s) * (A.R @ B.t) + A.t
    return Sim3Transform(R=R, t=t, s=s)


def sim3_inverse(S: Sim3Transform) -> Sim3Transform:
    R_inv = S.R.T
    s_inv = 1.0 / max(float(S.s), 1e-12)
    t_inv = -s_inv * (R_inv @ S.t)
    return Sim3Transform(R=R_inv, t=t_inv, s=s_inv)


def sim3_transform_points(S: Sim3Transform, pts: np.ndarray) -> np.ndarray:
    pts = _as_float64(pts).reshape(-1, 3)
    return float(S.s) * (pts @ S.R.T) + S.t.reshape(1, 3)


def pose3_matrix_to_sim3(T: np.ndarray) -> Sim3Transform:
    T = _as_float64(T).reshape(4, 4)
    return Sim3Transform(
        R=T[:3, :3].copy(),
        t=T[:3, 3].copy(),
        s=1.0,
    )


def sim3_to_pose3_matrix(S: Sim3Transform) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = S.R
    matrix[:3, 3] = S.t
    return matrix


def sim3_to_seven_vector(S: Sim3Transform) -> np.ndarray:
    return sim3_log(S).astype(np.float64)


def umeyama_similarity(src_pts: np.ndarray, dst_pts: np.ndarray) -> Sim3Transform:
    # Solve dst ~= s * R * src + t.
    src = _as_float64(src_pts).reshape(-1, 3)
    dst = _as_float64(dst_pts).reshape(-1, 3)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Need at least 3 paired 3D points for Umeyama alignment.")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    covariance = (dst_centered.T @ src_centered) / float(src.shape[0])
    U, D, Vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0.0:
        correction[-1, -1] = -1.0

    R = U @ correction @ Vt
    src_var = np.mean(np.sum(src_centered * src_centered, axis=1))
    s = float(np.trace(np.diag(D) @ correction) / max(src_var, 1e-12))
    t = dst_mean - s * (R @ src_mean)
    return Sim3Transform(R=R, t=t, s=s)


def ransac_umeyama(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    iters: int = 256,
    sample_size: int = 4,
    min_inliers: int = 20,
    inlier_threshold: float = 0.25,
) -> Tuple[Optional[Sim3Transform], Optional[np.ndarray]]:
    src = _as_float64(src_pts).reshape(-1, 3)
    dst = _as_float64(dst_pts).reshape(-1, 3)
    if src.shape != dst.shape or src.shape[0] < max(sample_size, 3):
        return None, None

    best_model = None
    best_inliers = None
    best_count = -1
    best_error = np.inf
    rng = np.random.default_rng()

    adaptive_threshold = max(
        float(inlier_threshold),
        0.01 * max(float(np.median(np.linalg.norm(dst, axis=1))), 1e-6),
    )

    for _ in range(int(iters)):
        sample_indices = rng.choice(src.shape[0], size=int(sample_size), replace=False)
        sample_src = src[sample_indices]
        if np.linalg.matrix_rank(sample_src - sample_src.mean(axis=0, keepdims=True)) < 2:
            continue

        try:
            model = umeyama_similarity(src[sample_indices], dst[sample_indices])
        except Exception:
            continue

        residual = np.linalg.norm(sim3_transform_points(model, src) - dst, axis=1)
        inliers = residual < adaptive_threshold
        inlier_count = int(inliers.sum())
        if inlier_count == 0:
            continue
        inlier_error = float(residual[inliers].mean())
        if inlier_count > best_count or (inlier_count == best_count and inlier_error < best_error):
            best_model = model
            best_inliers = inliers
            best_count = inlier_count
            best_error = inlier_error

    if best_model is None or best_count < int(min_inliers):
        return None, None

    refined = umeyama_similarity(src[best_inliers], dst[best_inliers])
    return refined, best_inliers


def numerical_jacobian(fun: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = _as_float64(x).reshape(-1)
    y0 = _as_float64(fun(x)).reshape(-1)
    jacobian = np.zeros((y0.size, x.size), dtype=np.float64)
    for idx in range(x.size):
        dx = np.zeros_like(x)
        dx[idx] = eps
        y_plus = _as_float64(fun(x + dx)).reshape(-1)
        y_minus = _as_float64(fun(x - dx)).reshape(-1)
        jacobian[:, idx] = (y_plus - y_minus) / (2.0 * eps)
    return jacobian
