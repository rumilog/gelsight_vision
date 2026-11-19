"""Touch planning: bead stations, tooltip yaw, plate reference points, calibration.

Conventions
-----------
* Plate frame in mm (x, y); tooltip yaw psi is a rotation about plate +z and
  the PoseGoal orientation is quaternion_from_euler(0, 0, psi).
* At psi = 0 the GelSight long axis (19.9 mm, raw-image width) lies along
  plate x (that's how every overlap scan so far ran). So to lay the long axis
  ACROSS the bead, psi = angle of the bead normal. With a 7-10 mm bead that
  leaves ~5 mm margin per side, vs ~1 mm with the short (11.2 mm) axis across.
* Yaw is folded into (-80 deg, 100 deg] so the wrist turns at most ~100 deg from
  the scan orientation and a bead along x always gets +90 deg (no 180 deg flips
  of the gel image between neighbouring stations).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from gelsight_vision.bead.centerline import BeadLine

GEL_LONG_MM = 19.9
GEL_SHORT_MM = 11.2


def rot2(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def yaw_across(normal_xy, fold_low_deg: float = -80.0) -> float:
    """Tooltip yaw (rad) putting the gel long axis along `normal_xy`."""
    psi = float(np.arctan2(normal_xy[1], normal_xy[0]))
    lo = np.radians(fold_low_deg)
    return float((psi - lo) % np.pi + lo)


@dataclass
class Calibration:
    """Output of calib_spot_touch (see bead.calib.solve_spot_calibration).

    camera correction:  true = p + b0 + B @ (p - p_ref)        (plate mm)
    gel centre offset:  gel_centre = tooltip + R(psi) @ c_tool  (mm)
    gel image axes:     plate_dir = R(psi) @ R(phi) @ diag(1, flip) @ gel_dir
    """

    b0_mm: list = field(default_factory=lambda: [0.0, 0.0])
    B: list = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    p_ref_mm: list = field(default_factory=lambda: [75.0, 75.0])
    c_tool_mm: list = field(default_factory=lambda: [0.0, 0.0])
    phi_deg: float = 0.0
    flip: int = 1
    residual_rms_mm: float | None = None
    source: str = 'identity'

    @classmethod
    def load(cls, path) -> 'Calibration':
        if not path:
            return cls()
        d = json.loads(Path(path).read_text())
        d = d.get('calibration', d)
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys}, source=str(path))

    def correct_point(self, p_mm) -> np.ndarray:
        p = np.asarray(p_mm, float)
        return p + np.asarray(self.b0_mm) + (np.asarray(self.B) @ (p - np.asarray(self.p_ref_mm)))

    def tooltip_for(self, target_mm, psi: float) -> np.ndarray:
        """Tooltip xy that puts the gel centre on `target_mm` at yaw psi."""
        return np.asarray(target_mm, float) - rot2(psi) @ np.asarray(self.c_tool_mm, float)

    def gel_to_plate_dir(self, psi: float) -> np.ndarray:
        """2x2 mapping gel-image mm (u right along long axis, v down) -> plate mm."""
        return rot2(psi) @ rot2(np.radians(self.phi_deg)) @ np.diag([1.0, float(self.flip)])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Station:
    kind: str                 # 'bead' or 'plate_ref'
    index: int
    target_mm: list           # where the gel centre should land (plate mm)
    tooltip_mm: list          # commanded gelsight_tooltip xy (plate mm)
    yaw_rad: float
    s_mm: float | None = None
    tangent: list | None = None
    normal: list | None = None
    visual_width_mm: float | None = None
    image_target_mm: list | None = None   # before calibration correction

    def to_dict(self) -> dict:
        d = asdict(self)
        d['yaw_deg'] = round(float(np.degrees(self.yaw_rad)), 2)
        return d


def pick_stations(line: BeadLine, k: int = 3, end_margin_mm: float = 8.0,
                  min_spacing_mm: float = GEL_SHORT_MM, tab_width_factor: float = 1.2,
                  chase_width_mm: float = 1.0) -> list[int]:
    """Indices into `line` for k bead stations.

    Even coverage (centres of k equal segments) over the usable stretch: ends,
    run-on/run-off tabs and craters (visual width > tab_width_factor x median)
    are excluded with `end_margin_mm` clearance. A station is nudged toward a
    visual-width anomaly within its segment only if the anomaly > chase_width_mm.
    """
    s, w = line.s_mm, line.width_mm
    wf = ndi.median_filter(np.nan_to_num(w, nan=np.nanmedian(w)), 5)
    med = float(np.median(wf))
    ok = np.isfinite(w) & (wf < tab_width_factor * med)
    ok[0] = ok[-1] = False
    ds = float(np.median(np.diff(s))) if len(s) > 1 else 1.0
    ok = ndi.binary_erosion(ok, iterations=max(1, int(end_margin_mm / ds)))
    if not ok.any():
        raise RuntimeError('pick_stations: no usable bead stretch (bead too short?)')
    lo, hi = s[ok][0], s[ok][-1]
    seg = (hi - lo) / k
    anomaly = np.abs(wf - med)
    chosen: list[float] = []
    for c in lo + (np.arange(k) + 0.5) * seg:
        win = ok & (np.abs(s - c) < 0.35 * seg)
        if chosen:
            win &= np.min(np.abs(s[:, None] - np.array(chosen)[None]), axis=1) > min_spacing_mm
        i = int(np.argmin(np.abs(s - c) + 1e9 * ~ok))
        if win.any() and anomaly[win].max() > chase_width_mm:
            i = int(np.flatnonzero(win)[np.argmax(anomaly[win])])
        chosen.append(float(s[i]))
    return [int(np.argmin(np.abs(s - c))) for c in chosen]


def bead_stations(line: BeadLine, idx: list[int], calib: Calibration,
                  yaw_override_rad: float | None = None) -> list[Station]:
    out = []
    for j, i in enumerate(idx):
        p_img = line.xy_mm[i]
        target = calib.correct_point(p_img)
        psi = yaw_across(line.normal[i]) if yaw_override_rad is None else yaw_override_rad
        out.append(Station(
            kind='bead', index=j + 1,
            target_mm=[round(float(v), 3) for v in target],
            tooltip_mm=[round(float(v), 3) for v in calib.tooltip_for(target, psi)],
            yaw_rad=psi, s_mm=round(float(line.s_mm[i]), 2),
            tangent=[round(float(v), 4) for v in line.tangent[i]],
            normal=[round(float(v), 4) for v in line.normal[i]],
            visual_width_mm=(None if np.isnan(line.width_mm[i])
                             else round(float(line.width_mm[i]), 2)),
            image_target_mm=[round(float(v), 3) for v in p_img]))
    return out


def plate_reference_stations(line: BeadLine, calib: Calibration, n: int = 3,
                             plate_x_mm: float = 150.0, plate_y_mm: float = 150.0,
                             edge_margin_mm: float = 25.0, clearance_mm: float = 22.0,
                             manual_xy_mm=None, beside: list[Station] | None = None,
                             ) -> list[Station]:
    """Plate-only touches for the plate plane (z reference for bead height).

    Points are >= clearance_mm from the bead centerline (gel half-length 10 mm
    + bead half-width ~5 mm + margin) and >= edge_margin_mm inside the plate
    edge (clamps sit on the edges). Yaw 0.

    * `beside` given (default in plan_bead): one point next to each bead
      station along its normal, alternating sides -- local plate reference, far
      from the clamps. `n` is ignored.
    * otherwise: n points spread over the plate by farthest-point sampling.
    """
    def inside(p):
        return (edge_margin_mm <= p[0] <= plate_x_mm - edge_margin_mm
                and edge_margin_mm <= p[1] <= plate_y_mm - edge_margin_mm)

    def clear(p):
        return np.min(np.linalg.norm(line.xy_mm - p, axis=1)) >= clearance_mm - 1e-6

    if manual_xy_mm:
        pts = np.asarray(manual_xy_mm, float).reshape(-1, 2)
    elif beside:
        pts = []
        for j, st in enumerate(beside):
            c, nrm = np.asarray(st.image_target_mm, float), np.asarray(st.normal, float)
            # Alternate sides so the plate plane is pinned across the bead too. Try the
            # wanted side at every distance before giving up on it: with the distance
            # loop outside, a centerline wiggle of a few tenths of a mm made the 22 mm
            # point fail and flipped it to the other side (all 5 refs on one side,
            # captures_bead/2026-11-19_11-04-31).
            options = [c + side * dist * nrm
                       for side in ((1, -1) if j % 2 == 0 else (-1, 1))
                       for dist in (clearance_mm, clearance_mm + 3, clearance_mm + 6)]
            ok = [p for p in options if inside(p) and clear(p)]
            if ok:
                pts.append(ok[0])
        pts = np.asarray(pts).reshape(-1, 2)
    else:
        g = np.arange(edge_margin_mm, max(plate_x_mm - edge_margin_mm, edge_margin_mm) + 1e-6, 2.0)
        h = np.arange(edge_margin_mm, max(plate_y_mm - edge_margin_mm, edge_margin_mm) + 1e-6, 2.0)
        cand = np.array([[x, y] for x in g for y in h])
        d = np.min(np.linalg.norm(cand[:, None] - line.xy_mm[None], axis=2), axis=1)
        cand = cand[d >= clearance_mm]
        if len(cand) == 0:
            return []
        mid = line.xy_mm[len(line.xy_mm) // 2]
        sel = [int(np.argmax(np.linalg.norm(cand - mid, axis=1)))]
        while len(sel) < min(n, len(cand)):
            dd = np.min(np.linalg.norm(cand[:, None] - cand[sel][None], axis=2), axis=1)
            sel.append(int(np.argmax(dd)))
        pts = cand[sel]
    out = []
    for j, p in enumerate(pts):
        target = calib.correct_point(p)
        out.append(Station(kind='plate_ref', index=j + 1,
                           target_mm=[round(float(v), 3) for v in target],
                           tooltip_mm=[round(float(v), 3) for v in calib.tooltip_for(target, 0.0)],
                           yaw_rad=0.0, image_target_mm=[round(float(v), 3) for v in p]))
    return out


def fit_plate_plane(xy_mm, z_mm, min_spread_mm: float = 10.0) -> tuple[np.ndarray, float]:
    """z = a x + b y + c through the plate-reference contacts. Returns (coef, rms).

    If the points lie nearly on a line (spread across it < min_spread_mm, e.g. all
    refs on one side of the bead), the tilt across that line is not measurable:
    a few hundredths of a mm of contact noise over a few mm of spread, extrapolated
    22 mm to the bead, would come out as tenths of a mm. Then only the tilt along
    the line is fitted and the plate is taken as level across it.
    """
    xy = np.asarray(xy_mm, float)
    z = np.asarray(z_mm, float)
    if len(z) == 0:
        return np.array([0.0, 0.0, np.nan]), float('nan')
    if len(z) < 3:  # level plate assumed
        return np.array([0.0, 0.0, float(np.mean(z))]), float(np.std(z))
    m = xy.mean(0)
    _, sv, vt = np.linalg.svd(xy - m, full_matrices=False)
    spread = sv / np.sqrt(len(z))           # std of the points along each principal axis
    if spread[1] * 4 < min_spread_mm:       # ~ full width across the line < min_spread_mm
        t = (xy - m) @ vt[0]
        k, c0 = np.polyfit(t, z, 1)
        a, b = k * vt[0]
        coef = np.array([a, b, c0 - k * float(vt[0] @ m)])
    else:
        A = np.c_[xy, np.ones(len(z))]
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    A = np.c_[xy, np.ones(len(z))]
    return coef, float(np.sqrt(np.mean((A @ coef - z) ** 2)))
