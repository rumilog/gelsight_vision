"""Analyze a calib_spot_touch session (experiment 1) from its files.

Reads <session>/spot_*_pose.json + spot_*_raw.png + gelsight_baseline_raw.png,
locates the marked spot in every GelSight frame, and solves for

    * the camera correction (image-predicted plate point -> true plate point),
    * the GelSight centre offset from the gelsight_tooltip TCP (tool frame),
    * the GelSight image axes (rotation phi + flip) relative to the tool.

Writes <session>/calib_result.json (pass it to touch_bead_stations / calib_bead_sweep
as calib_path) and <session>/calib_features.png (detections for a visual check).

    python3 -m gelsight_vision.tools.analyze_spot_calib <session_dir> [--feature-mm 1.5]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np

from gelsight_vision.bead.calib import solve_spot_calibration
from gelsight_vision.bead.gel_features import contact_map, locate_compact_feature


def analyze_spot_session(session_dir: str, feature_mm: float = 1.5, min_snr: float = 6.0,
                         model: str = 'auto') -> dict:
    base_path = os.path.join(session_dir, 'gelsight_baseline_raw.png')
    baseline = cv2.imread(base_path)
    if baseline is None:
        raise FileNotFoundError(base_path)
    records, tiles = [], []
    for pose_path in sorted(glob.glob(os.path.join(session_dir, 'spot_*_pose.json'))):
        pose = json.load(open(pose_path))
        meta = pose.get('meta', {})
        raw = cv2.imread(pose_path.replace('_pose.json', '_raw.png'))
        if raw is None or not pose.get('valid_contact', False):
            continue
        feat = locate_compact_feature(raw, baseline, feature_mm=feature_mm, min_snr=min_snr)
        records.append({'stem': pose['stem'], 'spot': meta['spot'], 'yaw_rad': pose['yaw_rad'],
                        'p_img_mm': meta['p_img_mm'], 'p_cmd_mm': pose['target_plate_mm'],
                        'f_mm': [feat.u_mm, feat.v_mm], 'snr': feat.snr, 'found': feat.found})
        cmap, mmpx = contact_map(raw, baseline)
        norm = cv2.normalize(cmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
        h, w = cmap.shape
        u = int(round(feat.u_mm / mmpx + (w - 1) / 2))
        v = int(round(feat.v_mm / mmpx + (h - 1) / 2))
        cv2.drawMarker(vis, (u, v), (0, 255, 0) if feat.found else (0, 0, 255),
                       cv2.MARKER_CROSS, 16, 2)
        cv2.putText(vis, f"{pose['stem']} snr {feat.snr:.0f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)
        tiles.append(vis)
    if not records:
        raise RuntimeError(f'no usable spot touches in {session_dir}')
    if tiles:
        cols = 4
        blank = np.zeros_like(tiles[0])
        rows = [np.hstack(tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols])))
                for i in range(0, len(tiles), cols)]
        cv2.imwrite(os.path.join(session_dir, 'calib_features.png'), np.vstack(rows))
    result = solve_spot_calibration(records, model=model)
    result['records'] = records
    result['settings'] = {'feature_mm': feature_mm, 'min_snr': min_snr}
    with open(os.path.join(session_dir, 'calib_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result


def print_summary(result: dict) -> None:
    c, d = result['calibration'], result['diagnostics']
    print(f"model {d['model']}, {d['n_touches_used']} touches used "
          f"({d['n_touches_dropped']} dropped: feature not found)")
    print(f"camera correction b0 = {c['b0_mm']} mm (at {c['p_ref_mm']}), B = {c['B']}")
    print(f"gel centre offset from TCP (tool frame) c = {c['c_tool_mm']} mm")
    print(f"gel image axes: phi = {c['phi_deg']} deg, flip = {c['flip']}")
    print(f"residual rms = {c['residual_rms_mm']} mm")
    for s, v in d['spots'].items():
        print(f"  spot {s}: image said {v['image_xy_mm']}, off by {v['image_error_mm']} mm")
    for w in d['warnings']:
        print('WARNING:', w)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('session_dir')
    ap.add_argument('--feature-mm', type=float, default=1.5)
    ap.add_argument('--min-snr', type=float, default=6.0)
    ap.add_argument('--model', choices=['auto', 'translation', 'affine'], default='auto')
    a = ap.parse_args()
    print_summary(analyze_spot_session(a.session_dir, a.feature_mm, a.min_snr, a.model))


if __name__ == '__main__':
    main()
