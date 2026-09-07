"""
pose_utils.py — Shared SE(3)/SO(3) helpers and an on-manifold base-frame pose
filter, used by feature_pose.py, build_feature_map.py and pencil_pose_tracker.py.

These mirror the math already validated in your test_se3_math.py / approach6dof.
Only numpy + OpenCV are required (both pure-CPU on Apple Silicon).
"""
from __future__ import annotations
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# SE(3) / SO(3) helpers
# ---------------------------------------------------------------------------
def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from R (3x3) and t (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).flatten()
    return T


def R_from_rvec(rv: np.ndarray) -> np.ndarray:
    """Rodrigues rotation vector -> 3x3 rotation matrix."""
    R, _ = cv2.Rodrigues(np.asarray(rv, dtype=np.float64).reshape(3, 1))
    return R


def inv_T(T: np.ndarray) -> np.ndarray:
    """Analytical inverse of a rigid-body transform (uses R^T)."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation onto SO(3) via SVD, forcing det = +1."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def so3_log(R: np.ndarray) -> np.ndarray:
    """Matrix log of SO(3) -> rotation vector (axis * angle)."""
    rv, _ = cv2.Rodrigues(orthonormalize(R))
    return rv.flatten()


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Matrix exp of so(3): rotation vector -> rotation matrix."""
    R, _ = cv2.Rodrigues(np.asarray(w, dtype=np.float64).reshape(3, 1))
    return R


def geodesic_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle (degrees) between two rotations."""
    return float(np.degrees(np.linalg.norm(so3_log(Ra.T @ Rb))))


# ---------------------------------------------------------------------------
# On-manifold pose filter in the BASE frame
# ---------------------------------------------------------------------------
class PoseFilter:
    """Exponential low-pass for a (quasi-)static rigid body in the BASE frame.

    Rotation is blended along the SO(3) geodesic, translation linearly. The
    estimate is held for up to `max_stale` frames when detection drops out,
    so the output stays smooth and valid through brief occlusion. This is the
    same filter philosophy used in your approach6dof pipeline.
    """

    def __init__(self, alpha_R: float = 0.5, alpha_t: float = 0.5, max_stale: int = 15):
        self.alpha_R = float(alpha_R)
        self.alpha_t = float(alpha_t)
        self.max_stale = int(max_stale)
        self.R: np.ndarray | None = None
        self.t: np.ndarray | None = None
        self.stale = 0

    def update(self, R: np.ndarray, t: np.ndarray) -> None:
        if self.R is None:
            self.R = orthonormalize(R)
            self.t = np.asarray(t, dtype=np.float64).copy()
        else:
            dr = so3_log(self.R.T @ R)                       # geodesic increment
            self.R = orthonormalize(self.R @ so3_exp(self.alpha_R * dr))
            self.t = (1.0 - self.alpha_t) * self.t + self.alpha_t * np.asarray(t)
        self.stale = 0

    def hold(self) -> None:
        if self.R is not None:
            self.stale += 1

    def valid(self) -> bool:
        return self.R is not None and self.stale <= self.max_stale

    def pose(self) -> np.ndarray | None:
        if self.R is None:
            return None
        return make_T(self.R, self.t)

    def reset(self) -> None:
        self.R = None
        self.t = None
        self.stale = 0
