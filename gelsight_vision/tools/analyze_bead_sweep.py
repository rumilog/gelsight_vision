"""Analyze a calib_bead_sweep session (experiment 2) from its files.

Reads <session>/sweep_*_pose.json (+ _raw.png, _depth.npy), plate_ref_*_pose.json,
gelsight_baseline_raw.png and sweep.json. Reports:

    * bead height plateau from robot contact Z above the plate plane,
    * how far off-centre a touch can be before the crown leaves the pad,
    * gel-axis sign along the bead normal + single-frame crown localization precision,
    * the bead cross-section from overlaid GelSight profiles (approximate anchoring).

Writes <session>/sweep_result.json and <session>/sweep_summary.png (needs matplotlib).

    python3 -m gelsight_vision.tools.analyze_bead_sweep <session_dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np

from gelsight_vision.bead.calib import analyze_bead_sweep
from gelsight_vision.bead.gel_features import GEL_FULL_WIDTH_MM, read_crown


def _load(session_dir: str, prefix: str) -> list[dict]:
    return [json.load(open(p))
            for p in sorted(glob.glob(os.path.join(session_dir, f'{prefix}_*_pose.json')))]


def analyze_sweep_session(session_dir: str, gel_depth_sign: float = -1.0) -> dict:
    sweep_meta = json.load(open(os.path.join(session_dir, 'sweep.json')))
    keep = float(sweep_meta.get('params', {}).get('gel_keep', 0.75))
    baseline = cv2.imread(os.path.join(session_dir, 'gelsight_baseline_raw.png'))
    touches, profiles = [], []
    for pose in _load(session_dir, 'sweep'):
        meta = pose.get('meta', {})
        stem = os.path.join(session_dir, pose['stem'])
        crown = None
        raw = cv2.imread(stem + '_raw.png')
        if raw is not None and baseline is not None and pose.get('valid_contact'):
            crown = read_crown(raw, baseline).to_dict()
        touches.append({'offset_mm': meta['offset_mm'], 'tooltip_mm': pose['target_plate_mm'],
                        'contact_z_mm': pose.get('contact_z_mm'),
                        'valid': pose.get('valid_contact', False),
                        'crown_u_mm': None if crown is None else crown['crown_u_mm'],
                        'crown_near_edge': None if crown is None else crown['crown_near_edge'],
                        'crown': crown})
        if os.path.exists(stem + '_depth.npy'):
            dep = np.load(stem + '_depth.npy').astype(float)
            prof = gel_depth_sign * np.median(dep, axis=0)
            mmpx = GEL_FULL_WIDTH_MM * keep / dep.shape[1]
            u = (np.arange(dep.shape[1]) - (dep.shape[1] - 1) / 2) * mmpx
            profiles.append((meta['offset_mm'], u, prof))
    refs = [{'tooltip_mm': r['target_plate_mm'], 'contact_z_mm': r.get('contact_z_mm')}
            for r in _load(session_dir, 'plate_ref') if r.get('valid_contact')]
    result = analyze_bead_sweep(touches, refs)
    result['touches'] = touches
    result['station'] = sweep_meta.get('station')
    with open(os.path.join(session_dir, 'sweep_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    _plot(session_dir, result, profiles)
    return result


def _plot(session_dir, res, profiles):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed; skipping sweep_summary.png')
        return
    d = np.asarray(res['offsets_mm'])
    h = np.asarray(res['contact_height_mm'])
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].plot(d, h, 'ko-')
    ax[0].axhline(res['bead_height_plateau_mm'], color='g', ls='--',
                  label=f"plateau {res['bead_height_plateau_mm']:.2f} mm")
    ax[0].set(xlabel='tool offset along bead normal (mm)', ylabel='contact Z above plate (mm)',
              title='robot contact height vs offset')
    ax[0].legend()
    t = res['touches']
    u = np.array([np.nan if x['crown_u_mm'] is None else x['crown_u_mm'] for x in t])
    dd = np.array([x['offset_mm'] for x in t])
    ax[1].plot(dd, u, 'bo', label='crown in gel frame')
    fit = res.get('crown_fit')
    if fit:
        xx = np.linspace(dd.min(), dd.max(), 2)
        ax[1].plot(xx, fit['alpha'] * xx + fit['beta_mm'], 'r-',
                   label=f"u = {fit['alpha']:.2f} d + {fit['beta_mm']:.2f}; "
                         f"precision {fit['single_frame_precision_mm']:.2f} mm")
    ax[1].axhspan(-GEL_FULL_WIDTH_MM / 2, GEL_FULL_WIDTH_MM / 2, color='0.92')
    ax[1].set(xlabel='tool offset (mm)', ylabel='crown u (mm)',
              title='crown position inside the GelSight frame')
    ax[1].legend(fontsize=8)
    sgn = 1.0 if (fit is None or fit['gel_u_points_along_normal']) else -1.0
    hmap = dict(zip(res['offsets_mm'], res['contact_height_mm']))
    cm = plt.get_cmap('viridis')
    span = max(1e-6, float(np.ptp(dd)))
    for off, uu, prof in profiles:
        x = off + sgn * uu
        top = hmap.get(off)
        y = prof - prof.max() + (top if top is not None else 0.0)
        ax[2].plot(x, y, color=cm((off - dd.min()) / span), lw=0.8)
    ax[2].set(xlabel='position across bead (mm, 0 = planned station)',
              ylabel='height (mm, gel profile anchored at contact Z)',
              title='cross-section from overlaid GelSight profiles (approx.)')
    fig.tight_layout()
    fig.savefig(os.path.join(session_dir, 'sweep_summary.png'), dpi=90)
    plt.close(fig)


def print_summary(res: dict) -> None:
    print(f"height reference: {res['height_reference']}")
    print(f"bead height plateau (contact Z): {res['bead_height_plateau_mm']} mm")
    print(f"contact height halves at offsets: {res['height_halves_at_offsets_mm']} mm")
    f = res.get('crown_fit')
    if f:
        print(f"crown fit: alpha {f['alpha']}, crown sits {f['crown_offset_from_station_mm']} mm "
              f"from the planned station along the normal; single-frame precision "
              f"{f['single_frame_precision_mm']} mm over offsets {f['usable_offset_range_mm']}")
        if f['warning']:
            print('WARNING:', f['warning'])
    else:
        print('crown fit: not enough frames with a crown reading')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('session_dir')
    ap.add_argument('--gel-depth-sign', type=float, default=-1.0)
    a = ap.parse_args()
    print_summary(analyze_sweep_session(a.session_dir, a.gel_depth_sign))


if __name__ == '__main__':
    main()
