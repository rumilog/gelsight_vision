"""ROS-free planning entry point shared by touch_bead_stations and plan_bead_offline."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from gelsight_vision.bead.camera_model import CameraModel
from gelsight_vision.bead.centerline import BeadLine, bead_centerline, render_plate_ortho
from gelsight_vision.bead.planner import (Calibration, Station, bead_stations, pick_stations,
                                          plate_reference_stations)
from gelsight_vision.bead.viz import draw_plan


@dataclass
class BeadPlan:
    line: BeadLine
    stations: list[Station]
    refs: list[Station]
    plan: dict
    images: dict = field(default_factory=dict)   # filename -> BGR image


def plan_bead(img_bgr: np.ndarray, cam: CameraModel, calib: Calibration | None = None,
              n_stations: int = 3, n_plate_refs: int = 3, plate_mm=(150.0, 150.0),
              ortho_res_mm: float = 0.5, max_bead_width_mm: float = 25.0,
              plate_ref_clearance_mm: float = 22.0, plate_ref_points_mm=None,
              yaw_override_rad: float | None = None, extra_lines=None,
              plate_ref_mode: str = 'beside') -> BeadPlan:
    """plate_ref_mode: 'beside' (one plate touch next to each station) or 'spread'
    (n_plate_refs points spread over the plate, >= 25 mm from the edges)."""
    calib = calib or Calibration()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ortho = render_plate_ortho(rgb, cam, (0.0, plate_mm[0]), (0.0, plate_mm[1]), ortho_res_mm)
    line = bead_centerline(ortho, max_bead_width_mm=max_bead_width_mm)
    idx = pick_stations(line, n_stations)
    stations = bead_stations(line, idx, calib, yaw_override_rad)
    refs = plate_reference_stations(
        line, calib, n_plate_refs, plate_mm[0], plate_mm[1],
        clearance_mm=plate_ref_clearance_mm, manual_xy_mm=plate_ref_points_mm,
        beside=stations if plate_ref_mode == 'beside' else None) if n_plate_refs > 0 else []
    o = (255 * ortho.image[..., ::-1]).astype(np.uint8)
    mask_img = np.where(line.mask[..., None], o, (o * 0.3).astype(np.uint8))
    lo, hi = line.xy_mm.min(0) - 15, line.xy_mm.max(0) + 15
    images = {
        'ortho_mask.png': mask_img,
        'plan_overlay.png': draw_plan(img_bgr, cam, line.xy_mm, refs + stations, extra_lines),
        'plan_overlay_zoom.png': draw_plan(img_bgr, cam, line.xy_mm, refs + stations, extra_lines,
                                           crop_to_mm=(lo[0], hi[0], lo[1], hi[1])),
    }
    plan = {
        'camera': cam.summary(), 'calibration': calib.to_dict(),
        'bead': {'length_mm': round(line.length_mm, 1),
                 'visual_width_median_mm': round(float(np.nanmedian(line.width_mm)), 2),
                 'centerline_xy_mm': np.round(line.xy_mm, 2).tolist(),
                 'visual_width_mm': [None if not np.isfinite(w) else round(float(w), 2)
                                     for w in line.width_mm]},
        'stations': [s.to_dict() for s in stations],
        'plate_refs': [s.to_dict() for s in refs],
    }
    return BeadPlan(line, stations, refs, plan, images)
