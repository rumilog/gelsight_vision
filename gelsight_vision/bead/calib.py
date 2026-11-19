"""Solvers for the two rig calibrations.

Experiment 1 -- spot touch (calib_spot_touch)
---------------------------------------------
Touch K marked spots (centre-punch dents work: visible to the D555 and felt
by the gel) at >= 2 tool yaws. For touch i at spot k:

    spot_k (true)   = p_img_k + b0 + B (p_img_k - p_ref)          camera correction
    gel centre      = p_cmd_i + R(psi_i) c                         TCP -> gel offset
    spot_k (gel)    = gel centre + R(psi_i) R(phi) diag(1, flip) f_i

f_i = feature position in the gel image (mm). For fixed (phi, flip) this is
linear in (b0, B, c); phi is grid-searched. Two yaws separate b0 from c.
Known jogs (touching a spot at p_img + (jog, 0) and p_img + (0, jog)) are
what pin down (phi, flip): without them every spot carries the same
information and phi trades off against b0 / c.

Experiment 2 -- bead sweep (calib_bead_sweep)
---------------------------------------------
Touch one bead station at offsets d along the bead normal, long axis across
the bead. Robot contact Z vs d gives the bead height plateau and how far
off-centre a touch can be; the crown position in each frame vs d gives the
gel-axis sign and a single-frame crown localization precision.
"""

from __future__ import annotations

import numpy as np

from gelsight_vision.bead.planner import fit_plate_plane, rot2


# ---------------------------------------------------------------- experiment 1
def _spot_system(recs, phi, flip, model, p_ref):
    A = rot2(phi) @ np.diag([1.0, float(flip)])
    rows, rhs = [], []
    for r in recs:
        R = rot2(r['yaw_rad'])
        pimg = np.asarray(r['p_img_mm'], float)
        pcmd = np.asarray(r['p_cmd_mm'], float)
        f = np.asarray(r['f_mm'], float)
        y = pcmd - pimg + R @ A @ f
        dp = pimg - p_ref
        for a in range(2):
            row = [1.0 if a == 0 else 0.0, 1.0 if a == 1 else 0.0]
            if model == 'affine':
                row += [dp[0] if a == 0 else 0.0, dp[1] if a == 0 else 0.0,
                        dp[0] if a == 1 else 0.0, dp[1] if a == 1 else 0.0]
            row += list(-R[a])
            rows.append(row)
            rhs.append(y[a])
    return np.asarray(rows), np.asarray(rhs)


def _solve(recs, phi, flip, model, p_ref):
    M, y = _spot_system(recs, phi, flip, model, p_ref)
    x, *_ = np.linalg.lstsq(M, y, rcond=None)
    res = (M @ x - y).reshape(-1, 2)
    return x, res


def solve_spot_calibration(records: list[dict], model: str = 'auto',
                           phi_step_deg: float = 0.5) -> dict:
    """records: dicts with p_img_mm, p_cmd_mm, yaw_rad, f_mm, found, spot.

    Returns {'calibration': {...planner.Calibration fields...}, 'diagnostics': {...}}.
    """
    recs = [r for r in records if r.get('found', True)]
    spots = sorted({r['spot'] for r in recs})
    yaws = sorted({round(float(np.degrees(r['yaw_rad'])), 1) for r in recs})
    if len(recs) < 3:
        raise ValueError(f'need >= 3 usable touches, got {len(recs)}')
    if model == 'auto':
        model = 'affine' if len(spots) >= 4 else 'translation'
    p_ref = np.mean([r['p_img_mm'] for r in recs], axis=0)
    best = None
    for flip in (1, -1):
        for phi in np.radians(np.arange(-180, 180, phi_step_deg)):
            _, res = _solve(recs, phi, flip, model, p_ref)
            cost = float(np.sum(res ** 2))
            if best is None or cost < best[0]:
                best = (cost, phi, flip)
    _, phi, flip = best
    for phi_f in phi + np.radians(np.arange(-phi_step_deg, phi_step_deg, 0.02)):
        _, res = _solve(recs, phi_f, flip, model, p_ref)
        if float(np.sum(res ** 2)) < best[0]:
            best = (float(np.sum(res ** 2)), phi_f, flip)
    _, phi, flip = best
    x, res = _solve(recs, phi, flip, model, p_ref)
    b0 = x[:2]
    B = x[2:6].reshape(2, 2) if model == 'affine' else np.zeros((2, 2))
    c = x[-2:]
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    per_spot = {}
    for s in spots:
        p = np.asarray(next(r['p_img_mm'] for r in recs if r['spot'] == s), float)
        corr = b0 + B @ (p - p_ref)
        per_spot[str(s)] = {'image_xy_mm': [round(float(v), 2) for v in p],
                            'correction_mm': [round(float(v), 2) for v in corr],
                            'image_error_mm': round(float(np.linalg.norm(corr)), 2)}
    warnings = []
    jogged = {tuple(np.round(np.subtract(r['p_cmd_mm'], r['p_img_mm']), 3)) for r in recs}
    if len(jogged) < 3:
        warnings.append('fewer than 2 distinct jogs: the gel image axes (phi, flip) are not '
                        'identifiable; touch each spot at known small offsets (jog_mm)')
    if len(yaws) < 2:
        warnings.append('only one yaw: camera bias and gel-centre offset are not separable '
                        '(c is folded into b0)')
    if model == 'affine' and len(spots) < 4:
        warnings.append('affine model with < 4 spots is under-determined')
    return {
        'calibration': {
            'b0_mm': [round(float(v), 3) for v in b0],
            'B': [[round(float(v), 5) for v in row] for row in B],
            'p_ref_mm': [round(float(v), 3) for v in p_ref],
            'c_tool_mm': [round(float(v), 3) for v in c],
            'phi_deg': round(float(np.degrees(phi)), 2),
            'flip': int(flip),
            'residual_rms_mm': round(rms, 3),
        },
        'diagnostics': {
            'model': model, 'n_touches_used': len(recs),
            'n_touches_dropped': len(records) - len(recs),
            'spots': per_spot, 'yaws_deg': yaws,
            'residuals_mm': [[round(float(v), 3) for v in r] for r in res],
            'warnings': warnings,
        },
    }


# ---------------------------------------------------------------- experiment 2
def analyze_bead_sweep(touches: list[dict], plate_refs: list[dict],
                       height_window_mm: float = 3.0) -> dict:
    """touches: dicts with offset_mm, tooltip_mm (xy), contact_z_mm, crown_u_mm,
    crown_near_edge, valid.  plate_refs: dicts with tooltip_mm, contact_z_mm."""
    t = [r for r in touches if r.get('valid', True) and r.get('contact_z_mm') is not None]
    if len(t) < 3:
        raise ValueError(f'need >= 3 valid sweep touches, got {len(t)}')
    d = np.array([r['offset_mm'] for r in t], float)
    order = np.argsort(d)
    d = d[order]
    t = [t[i] for i in order]
    z = np.array([r['contact_z_mm'] for r in t], float)
    xy = np.array([r['tooltip_mm'] for r in t], float)

    pr = [r for r in plate_refs if r.get('contact_z_mm') is not None]
    if pr:
        coef, plane_rms = fit_plate_plane([r['tooltip_mm'] for r in pr],
                                          [r['contact_z_mm'] for r in pr])
        h = z - (np.c_[xy, np.ones(len(z))] @ coef)
        h_ref = 'plate plane from reference touches'
    else:
        coef, plane_rms = None, None
        h = z - np.min(z)
        h_ref = 'lowest sweep touch (no plate refs!)'

    # crown position in gel frame vs offset: u = alpha * d + beta
    u = np.array([np.nan if r.get('crown_u_mm') is None else r['crown_u_mm'] for r in t], float)
    good = np.isfinite(u) & ~np.array([bool(r.get('crown_near_edge', False)) for r in t])
    fit = None
    if good.sum() >= 3:
        keep = good.copy()
        for _ in range(3):
            alpha, beta = np.polyfit(d[keep], u[keep], 1)
            r = u - (alpha * d + beta)
            s = 1.4826 * np.median(np.abs(r[keep])) + 1e-6
            keep = good & (np.abs(r) < max(3 * s, 0.3))
        alpha, beta = np.polyfit(d[keep], u[keep], 1)
        sgn = -float(np.sign(alpha)) if alpha != 0 else 1.0
        xc = d + sgn * u           # crown position along the normal, per frame
        fit = {
            'alpha': round(float(alpha), 3), 'beta_mm': round(float(beta), 3),
            'gel_u_points_along_normal': bool(sgn > 0),
            'crown_offset_from_station_mm': round(float(np.median(xc[keep])), 3),
            'single_frame_precision_mm': round(float(np.std(xc[keep])), 3),
            'n_used': int(keep.sum()),
            'usable_offset_range_mm': [round(float(d[good].min()), 1),
                                       round(float(d[good].max()), 1)],
            'warning': None if 0.8 < abs(alpha) < 1.2 else
            f'|alpha| = {abs(alpha):.2f}, expected ~1: crown estimate unreliable on this bead',
        }
    xc0 = fit['crown_offset_from_station_mm'] if fit else 0.0
    near = np.abs(d - xc0) <= height_window_mm
    plateau = float(np.median(h[near])) if near.any() else float('nan')
    drop = d[h < 0.5 * plateau] if np.isfinite(plateau) and plateau > 0 else np.array([])
    return {
        'height_reference': h_ref,
        'plate_plane': None if coef is None else {'coef': [float(v) for v in coef],
                                                  'rms_mm': round(float(plane_rms), 3)},
        'offsets_mm': d.tolist(),
        'contact_height_mm': [round(float(v), 3) for v in h],
        'bead_height_plateau_mm': round(plateau, 3),
        'height_halves_at_offsets_mm': [round(float(v), 1) for v in drop],
        'crown_fit': fit,
        'notes': ('contact_height = tooltip contact Z above the plate plane, not corrected '
                  'for gel compression; with the long axis across the bead it is ~ the '
                  'highest surface under the 19.9 mm pad.'),
    }
