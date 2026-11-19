"""ROS-free tests for gelsight_vision.bead (camera model, centerline, planner, calib).

Run: python3 -m pytest test/test_bead_offline.py -q   (needs numpy, scipy, opencv)
"""

import math
import os

import numpy as np
import pytest

from gelsight_vision.bead.calib import analyze_bead_sweep, solve_spot_calibration
from gelsight_vision.bead.camera_model import CameraModel, pose_from_plane_points
from gelsight_vision.bead.centerline import bead_centerline, render_plate_ortho
from gelsight_vision.bead.gel_features import locate_compact_feature, read_crown
from gelsight_vision.bead.planner import (Calibration, bead_stations, pick_stations,
                                          plate_reference_stations, rot2, yaw_across)

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, '..', 'captures_overlap', '2026-11-04_17-34-24', 'realsense')


def rig_camera(tilt_deg=1.5, yaw_deg=90.0, centre=(73.0, 74.0, 440.0)):
    """Down-looking camera like the rig: image up = +x, image left = +y."""
    cam = CameraModel.from_intrinsics(645.0, 644.0, 624.0, 357.0,
                                      [-0.055, 0.06, 0.0004, -0.0006, -0.019], 1280, 720)
    # optical z -> plate -z; build from a yaw about z and a small tilt about x
    down = np.diag([1.0, -1.0, -1.0])
    a = math.radians(yaw_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    t = math.radians(tilt_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    return cam.with_pose(Rz @ Rx @ down, np.asarray(centre, float), 'synthetic')


def synth_image(cam, bead_fn, width_mm=9.0, plate=150.0):
    """Render a gray plate on a dark table with a bright bead y = bead_fn(x)."""
    h, w = 720, 1280
    vv, uu = np.mgrid[0:h, 0:w]
    xyz = cam.pixel_to_plate(np.c_[uu.ravel(), vv.ravel()])
    x, y = xyz[:, 0], xyz[:, 1]
    img = np.full(x.shape, 0.15)
    on = (x >= 0) & (x <= plate) & (y >= 0) & (y <= plate)
    img[on] = 0.45 + 0.1 * (x[on] / plate)          # illumination gradient
    d = np.abs(y - bead_fn(x))
    bead = on & (d < width_mm / 2) & (x > 10) & (x < 140)
    img[bead] = 0.85 - 0.1 * (d[bead] / (width_mm / 2))
    img = img.reshape(h, w)
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1)
    return np.repeat(img[..., None], 3, axis=2)


def test_camera_roundtrip_and_pnp():
    cam = rig_camera()
    pts = np.array([[10, 20, 0], [75, 75, 0], [140, 130, 3.0]], float)
    uv = cam.plate_to_pixel(pts)
    back = np.vstack([cam.pixel_to_plate(uv[i:i + 1], z)[0] for i, z in enumerate(pts[:, 2])])
    assert np.allclose(back, pts, atol=1e-4)
    corners = np.array([[150, 150], [150, 0], [0, 0], [0, 150]], float)
    R, C = pose_from_plane_points(cam, cam.plate_to_pixel(corners), corners)
    assert np.allclose(C, cam.C_plate_mm, atol=0.05)
    assert np.allclose(R, cam.R_plate_opt, atol=1e-4)


@pytest.mark.parametrize('curve', ['straight', 'curved'])
def test_centerline_and_stations(curve):
    cam = rig_camera()
    fn = (lambda x: 78.0 + 0.03 * (x - 75)) if curve == 'straight' else \
        (lambda x: 75.0 + 12.0 * np.sin((x - 10) / 130 * np.pi))
    img = synth_image(cam, fn)
    ortho = render_plate_ortho(img, cam)
    line = bead_centerline(ortho)
    err = np.abs(line.xy_mm[:, 1] - fn(line.xy_mm[:, 0]))
    inner = (line.xy_mm[:, 0] > 20) & (line.xy_mm[:, 0] < 130)
    assert np.median(err[inner]) < 0.6 and err[inner].max() < 1.5
    assert 115 < line.length_mm < 150
    idx = pick_stations(line, 3)
    st = bead_stations(line, idx, Calibration())
    assert len(st) == 3
    for s in st:
        long_axis = rot2(s.yaw_rad)[:, 0]
        assert abs(long_axis @ np.asarray(s.tangent)) < 0.05   # long axis across the bead
        assert -math.radians(80) - 1e-9 <= s.yaw_rad <= math.radians(100) + 1e-9
    refs = plate_reference_stations(line, Calibration(), beside=st)
    assert len(refs) == 3
    for r in refs:
        assert np.min(np.linalg.norm(line.xy_mm - r.target_mm, axis=1)) > 21.5


def test_yaw_fold_is_consistent():
    assert math.isclose(yaw_across([0.0, 1.0]), math.pi / 2)
    assert math.isclose(yaw_across([0.0, -1.0]), math.pi / 2)
    assert math.isclose(yaw_across([1.0, 0.0]), 0.0, abs_tol=1e-12)


def test_spot_calibration_recovers_truth():
    rng = np.random.default_rng(1)
    b0, c = np.array([2.5, -3.4]), np.array([1.2, -0.7])
    phi, flip = math.radians(178.0), -1
    A = rot2(phi) @ np.diag([1.0, flip])
    recs, no_jog = [], []
    for k, p_img in enumerate([[30, 40], [120, 35], [110, 120], [40, 115], [75, 75]]):
        p_img = np.asarray(p_img, float)
        s_true = p_img + b0
        for yaw in (0.0, math.pi / 2):
            jogs = [(0, 0), (2, 0), (0, 2)] if yaw == 0.0 else [(0, 0)]
            for jog in jogs:
                R = rot2(yaw)
                p_cmd = p_img + np.asarray(jog, float)
                g = p_cmd + R @ c
                f = np.linalg.solve(R @ A, s_true - g) + rng.normal(0, 0.1, 2)
                rec = {'spot': k + 1, 'yaw_rad': yaw, 'p_img_mm': p_img.tolist(),
                       'p_cmd_mm': p_cmd.tolist(), 'f_mm': f.tolist(), 'found': True}
                recs.append(rec)
                if jog == (0, 0):
                    no_jog.append(rec)
    assert any('jog' in w for w in
               solve_spot_calibration(no_jog, model='translation')['diagnostics']['warnings'])
    out = solve_spot_calibration(recs, model='translation')
    cal = out['calibration']
    assert np.allclose(cal['b0_mm'], b0, atol=0.2)
    assert np.allclose(cal['c_tool_mm'], c, atol=0.2)
    assert cal['flip'] == flip
    assert abs(((cal['phi_deg'] - 178.0) + 180) % 360 - 180) < 2.0
    assert cal['residual_rms_mm'] < 0.3
    # the result loads as a planner Calibration and inverts correctly
    calib = Calibration(**{k: v for k, v in cal.items()})
    assert np.allclose(calib.correct_point([30, 40]), np.array([30, 40]) + b0, atol=0.2)
    tool = calib.tooltip_for([30, 40], 0.0)
    assert np.allclose(tool + c, [30, 40], atol=0.2)


def test_bead_sweep_analysis():
    rng = np.random.default_rng(2)
    xc, plate_z, h0 = 1.3, 6.8, 2.9
    touches = []
    for d in np.arange(-8, 8.5, 1.0):
        rel = d - xc
        h = h0 if abs(rel) < 6 else max(0.0, h0 - (abs(rel) - 6) * 1.5)
        u = -rel + rng.normal(0, 0.3)        # crown sits at xc - d along gel +u
        touches.append({'offset_mm': float(d), 'tooltip_mm': [75 + d, 75.0],
                        'contact_z_mm': plate_z + h + rng.normal(0, 0.1),
                        'crown_u_mm': u, 'crown_near_edge': abs(u) > 7.9, 'valid': True})
    refs = [{'tooltip_mm': [75 + 22, 75.0], 'contact_z_mm': plate_z},
            {'tooltip_mm': [75 - 22, 75.0], 'contact_z_mm': plate_z}]
    res = analyze_bead_sweep(touches, refs)
    assert abs(res['bead_height_plateau_mm'] - h0) < 0.25
    f = res['crown_fit']
    assert abs(f['alpha'] + 1) < 0.15
    assert f['gel_u_points_along_normal'] is True
    assert abs(f['crown_offset_from_station_mm'] - xc) < 0.4
    assert f['single_frame_precision_mm'] < 0.6


def _gel_pair(band_u_mm=None, dent_uv_mm=None):
    h, w = 720, 1280
    mmpx = 19.9 / w
    vv, uu = np.mgrid[0:h, 0:w]
    u = (uu - (w - 1) / 2) * mmpx
    v = (vv - (h - 1) / 2) * mmpx
    rng = np.random.default_rng(3)
    base = 120 + 10 * np.sin(uu / 90.0)[..., None] + np.zeros((h, w, 3))
    frame = base.copy()
    if band_u_mm is not None:      # crown band ~4 mm wide, flanks out of contact
        frame += (60 * np.exp(-0.5 * ((u - band_u_mm) / 1.2) ** 2))[..., None]
    if dent_uv_mm is not None:     # plate contact everywhere + a 1.5 mm dent
        frame += 25
        frame += (70 * np.exp(-0.5 * (((u - dent_uv_mm[0]) ** 2 + (v - dent_uv_mm[1]) ** 2)
                                      / 0.5 ** 2)))[..., None]
    frame += rng.normal(0, 2, frame.shape)
    return frame.clip(0, 255).astype(np.uint8), base.clip(0, 255).astype(np.uint8)


def test_read_crown_and_feature():
    raw, base = _gel_pair(band_u_mm=2.7)
    cr = read_crown(raw, base)
    assert abs(cr.crown_u_mm - 2.7) < 0.2
    assert not cr.plate_both_sides and not cr.crown_near_edge
    raw, base = _gel_pair(dent_uv_mm=(-3.1, 1.8))
    ft = locate_compact_feature(raw, base)
    assert ft.found and abs(ft.u_mm + 3.1) < 0.2 and abs(ft.v_mm - 1.8) < 0.2


@pytest.mark.skipif(not os.path.exists(os.path.join(SNAPSHOT, 'color.png')),
                    reason='2026-11-04 snapshot not on this machine')
def test_real_snapshot_plan():
    import cv2
    from gelsight_vision.bead.pipeline import plan_bead
    img = cv2.imread(os.path.join(SNAPSHOT, 'color.png'))
    cam = CameraModel.from_camera_info_json(os.path.join(SNAPSHOT, 'camera_info.json'))
    px = np.array([[498, 236], [714, 240], [712, 462], [496, 460]], float)
    mm = np.array([[150, 150], [150, 0], [0, 0], [0, 150]], float)
    cam = cam.with_pose(*pose_from_plane_points(cam, px, mm), 'corners')
    bp = plan_bead(img, cam)
    assert 140 < bp.line.length_mm < 165
    assert len(bp.stations) == 3 and len(bp.refs) == 3
    for s in bp.stations:          # bead runs along plate x at y ~ 73-76 mm
        assert 70 < s.target_mm[1] < 80
        assert abs(math.degrees(s.yaw_rad) - 90) < 6
