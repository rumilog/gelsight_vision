"""Pixel <-> plate-frame geometry for the D555 color camera.

Chain used everywhere in the bead pipeline:

    pixel (u, v) -> undistort (plumb_bob) -> ray in the optical frame
      -> rotate/translate into the plate frame -> intersect plane z = z_surface

Units: the plate frame is in **millimetres** here (the ROS side converts to
metres only when it builds a PoseGoal).

The camera pose (optical frame in the plate frame) comes from either
  * TF: `plate -> <color optical frame>` at capture time (`from_tf`), or
  * PnP on known plate points, e.g. the plate corners (`pose_from_plane_points`).
Measured on 2026-11-04: with camera_focus commanded to z = 0.30 m the color
lens sits ~440 mm above the plate, so `camera_focus` is not the optical origin.
Don't assume one from the other; compare both (`compare_poses`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


def quat_to_rot(q_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q_xyzw)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class CameraModel:
    """Pinhole + plumb_bob camera with a pose in the plate frame (mm)."""

    K: np.ndarray                      # 3x3 intrinsics
    D: np.ndarray                      # k1, k2, p1, p2, k3
    R_plate_opt: np.ndarray = field(default_factory=lambda: np.eye(3))  # optical -> plate
    C_plate_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))  # optical centre
    width: int = 0
    height: int = 0
    pose_source: str = 'unset'

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_intrinsics(cls, fx, fy, cx, cy, d, width=0, height=0) -> 'CameraModel':
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        d = np.zeros(5) if d is None or len(d) == 0 else np.asarray(list(d)[:5], float)
        if len(d) < 5:
            d = np.r_[d, np.zeros(5 - len(d))]
        return cls(K=K, D=d, width=int(width), height=int(height))

    @classmethod
    def from_camera_info_json(cls, path, stream: str = 'color') -> 'CameraModel':
        """`camera_info.json` as written by motion_collect_overlap / the bead nodes."""
        ci = json.loads(Path(path).read_text())[stream]
        return cls.from_intrinsics(ci['fx'], ci['fy'], ci['cx'], ci['cy'], ci.get('d'),
                                   ci.get('width', 0), ci.get('height', 0))

    def with_pose(self, R_plate_opt, C_plate_mm, source: str) -> 'CameraModel':
        return CameraModel(self.K.copy(), self.D.copy(), np.asarray(R_plate_opt, float),
                           np.asarray(C_plate_mm, float), self.width, self.height, source)

    def with_tf(self, translation_m: Sequence[float], quat_xyzw: Sequence[float],
                source: str = 'tf') -> 'CameraModel':
        """Pose from a `plate -> optical` TF (parent = plate, translation in metres)."""
        return self.with_pose(quat_to_rot(quat_xyzw), np.asarray(translation_m, float) * 1000.0,
                              source)

    # ---------------------------------------------------------------- pixels
    def normalize(self, uv) -> np.ndarray:
        """Undistorted normalized image coordinates for pixel(s) `uv` (N, 2)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        x = (uv[:, 0] - self.K[0, 2]) / self.K[0, 0]
        y = (uv[:, 1] - self.K[1, 2]) / self.K[1, 1]
        x0, y0 = x.copy(), y.copy()
        k1, k2, p1, p2, k3 = self.D
        for _ in range(20):
            r2 = x * x + y * y
            rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            x, y = (x0 - dx) / rad, (y0 - dy) / rad
        return np.c_[x, y]

    def distort_project(self, xn, yn) -> np.ndarray:
        k1, k2, p1, p2, k3 = self.D
        r2 = xn * xn + yn * yn
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        return np.c_[self.K[0, 0] * xd + self.K[0, 2], self.K[1, 1] * yd + self.K[1, 2]]

    # ---------------------------------------------------------------- plate
    def pixel_to_plate(self, uv, z_surface_mm: float = 0.0) -> np.ndarray:
        """Back-project pixel(s) onto the plane z = z_surface_mm. Returns (N, 3) mm."""
        n = self.normalize(uv)
        rays = (self.R_plate_opt @ np.c_[n, np.ones(len(n))].T).T
        s = (z_surface_mm - self.C_plate_mm[2]) / rays[:, 2]
        return self.C_plate_mm[None] + s[:, None] * rays

    def plate_to_pixel(self, xyz_mm) -> np.ndarray:
        xyz = np.atleast_2d(np.asarray(xyz_mm, float))
        if xyz.shape[1] == 2:
            xyz = np.c_[xyz, np.zeros(len(xyz))]
        p = (self.R_plate_opt.T @ (xyz - self.C_plate_mm[None]).T).T
        return self.distort_project(p[:, 0] / p[:, 2], p[:, 1] / p[:, 2])

    def mm_per_px_at(self, z_surface_mm: float = 0.0) -> float:
        return float(abs(self.C_plate_mm[2] - z_surface_mm) / self.K[0, 0])

    def summary(self) -> dict:
        tilt = float(np.degrees(np.arccos(min(1.0, abs(self.R_plate_opt[2, 2])))))
        return {
            'pose_source': self.pose_source,
            'optical_centre_plate_mm': [round(float(v), 2) for v in self.C_plate_mm],
            'optical_axis_tilt_deg': round(tilt, 3),
            'mm_per_px_at_plate': round(self.mm_per_px_at(0.0), 4),
            'K': self.K.tolist(), 'D': self.D.tolist(),
            'R_plate_opt': self.R_plate_opt.tolist(),
        }


def pose_from_plane_points(cam: CameraModel, uv, xy_plate_mm) -> tuple[np.ndarray, np.ndarray]:
    """Planar PnP (>= 4 points on z = 0, e.g. the plate corners).

    Returns (R_plate_opt, C_plate_mm). Homography + orthonormalization; good
    to a fraction of a pixel on clean corners, which is all we need here.
    """
    n = cam.normalize(uv)
    P = np.asarray(xy_plate_mm, float)
    if len(P) < 4:
        raise ValueError('need >= 4 plane points for PnP')
    A = []
    for (X, Y), (u, v) in zip(P, n):
        A += [[-X, -Y, -1, 0, 0, 0, u * X, u * Y, u],
              [0, 0, 0, -X, -Y, -1, v * X, v * Y, v]]
    H = np.linalg.svd(np.asarray(A))[2][-1].reshape(3, 3)
    lam = 2.0 / (np.linalg.norm(H[:, 0]) + np.linalg.norm(H[:, 1]))
    r1, r2, t = lam * H[:, 0], lam * H[:, 1], lam * H[:, 2]
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    R = np.c_[r1, r2, np.cross(r1, r2)]
    U, _, Vt = np.linalg.svd(R)
    R_opt_plate = U @ Vt
    if np.linalg.det(R_opt_plate) < 0:
        R_opt_plate = U @ np.diag([1, 1, -1]) @ Vt
    return R_opt_plate.T, -R_opt_plate.T @ t


def compare_poses(a: CameraModel, b: CameraModel, probe_xy_mm=None) -> dict:
    """How far apart two pose estimates put the same pixels on the plate."""
    if probe_xy_mm is None:
        probe_xy_mm = [[x, y] for x in (25, 75, 125) for y in (25, 75, 125)]
    uv = a.plate_to_pixel(np.asarray(probe_xy_mm, float))
    pb = b.pixel_to_plate(uv)[:, :2]
    d = np.linalg.norm(pb - np.asarray(probe_xy_mm, float), axis=1)
    return {
        'centre_delta_mm': [round(float(v), 2) for v in (b.C_plate_mm - a.C_plate_mm)],
        'plate_point_disagreement_mm': {'mean': round(float(d.mean()), 2),
                                        'max': round(float(d.max()), 2)},
    }
