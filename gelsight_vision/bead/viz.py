"""Overlay drawing for the bead pipeline (cv2, BGR images)."""

from __future__ import annotations

import cv2
import numpy as np

from gelsight_vision.bead.camera_model import CameraModel
from gelsight_vision.bead.planner import GEL_LONG_MM, GEL_SHORT_MM, Station, rot2


def footprint_xy(centre_mm, yaw_rad: float, long_mm: float = GEL_LONG_MM,
                 short_mm: float = GEL_SHORT_MM) -> np.ndarray:
    """Gel footprint corners (5, 2), long axis along R(yaw) @ x."""
    R = rot2(yaw_rad)
    L, S = R[:, 0], R[:, 1]
    c = np.asarray(centre_mm, float)
    return np.array([c + a * long_mm / 2 * L + b * short_mm / 2 * S
                     for a, b in [(1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1)]])


def _poly(img, cam, xy, color, thick):
    uv = cam.plate_to_pixel(np.asarray(xy, float))
    cv2.polylines(img, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], False, color, thick,
                  cv2.LINE_AA)
    return uv


def draw_plan(image_bgr: np.ndarray, cam: CameraModel, centerline_xy_mm=None,
              stations: list[Station] | None = None, extra_lines=None,
              crop_to_mm=None) -> np.ndarray:
    """Centerline (red), bead stations (green footprints), plate refs (yellow).

    `extra_lines`: list of (xy_mm array, BGR color) e.g. a GelSight-felt bead line.
    `crop_to_mm`: (x0, x1, y0, y1) plate box to crop the output to (+ margin).
    """
    img = image_bgr.copy()
    if centerline_xy_mm is not None and len(centerline_xy_mm):
        _poly(img, cam, centerline_xy_mm, (0, 0, 255), 1)
    for xy, col in (extra_lines or []):
        _poly(img, cam, xy, col, 1)
    for st in stations or []:
        col = (0, 255, 0) if st.kind == 'bead' else (0, 220, 255)
        uv = _poly(img, cam, footprint_xy(st.target_mm, st.yaw_rad), col, 2)
        cv2.putText(img, f"{'T' if st.kind == 'bead' else 'P'}{st.index}",
                    tuple(int(v) for v in uv[0] + [3, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                    cv2.LINE_AA)
        if st.image_target_mm is not None and st.image_target_mm != st.target_mm:
            a = cam.plate_to_pixel(np.asarray(st.image_target_mm))[0]
            b = cam.plate_to_pixel(np.asarray(st.target_mm))[0]
            cv2.arrowedLine(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 0, 255),
                            1, cv2.LINE_AA, tipLength=0.3)
    if crop_to_mm is not None:
        x0, x1, y0, y1 = crop_to_mm
        uv = cam.plate_to_pixel(np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]], float))
        m = 20
        u0, v0 = np.maximum(uv.min(0).astype(int) - m, 0)
        u1, v1 = np.minimum(uv.max(0).astype(int) + m, [img.shape[1], img.shape[0]])
        img = img[v0:v1, u0:u1]
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    return img
