"""Compare the runs of an overlap sweep (motion_collect_overlap) side by side.

A sweep directory holds one `overlap_X.XX/` run per overlap, each scanning the
same plate area and already stitched by `stitch_session_solved`. For every run
this crops the solved depth to the requested area (all runs share its centre)
and reports:

  * cost: touches, scan wall-time, seconds per touch, valid-contact fraction
  * internal consistency: per-pixel RMS disagreement between overlapping
    tiles after the solve (seam lines for overlap 0)
  * agreement with a reference surface, after removing the best-fit plane
    (offset + tilt are only pinned by contact Z, so they're excluded):
    total RMS, split into large-scale shape (low-pass) and fine detail
    (high-pass) RMS, plus the detail-map correlation.

There is no ground truth, so the default reference is the *consensus*: the
pixelwise median of every run's detrended surface. Each run's large-scale
shape carries its own random contact-Z error; the median cancels much of it,
whereas comparing against any single run (even the densest) folds that run's
errors into everyone's score. Pass `ref_overlap` to compare against one run.

Read the columns accordingly: `detail_*` (texture < ~1 mm) and the pair
residual are what overlap actually controls. `shape_*` (large-scale form) is
mostly set by contact-Z noise in every run -- overlap vs none helps it, but
differences between overlapping runs there are largely luck of the touches.

Writes `compare_overlaps.png` (colour relief per run over its difference
from the reference), `compare_overlaps.csv` and `compare_overlaps.json` in the
sweep directory.

    python -m gelsight_vision.gelsight.compare_overlaps <sweep_dir> [--ref 0.5]
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from gelsight_vision.gelsight.stitch import (
    _colorize,
    _diverging_lut,
    _fill_small_holes,
    _hillshade,
    _label,
    _resize_nan,
)


def _crop_to_area(depth: np.ndarray, area_mm: tuple[float, float], mm_per_px: float) -> np.ndarray:
    """Centre-crop a mosaic (rows ~ plate x, cols ~ plate y) to the area."""
    h, w = depth.shape
    ah = min(h, int(round(area_mm[0] / mm_per_px)))
    aw = min(w, int(round(area_mm[1] / mm_per_px)))
    r0, c0 = (h - ah) // 2, (w - aw) // 2
    return depth[r0:r0 + ah, c0:c0 + aw]


def _detrend(z: np.ndarray) -> np.ndarray:
    """Subtract the least-squares plane through the finite pixels."""
    m = np.isfinite(z)
    if m.sum() < 3:
        return z
    v, u = np.nonzero(m)
    A = np.column_stack([v, u, np.ones(len(v))]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, z[m].astype(np.float64), rcond=None)
    vv, uu = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    return (z - (coef[0] * vv + coef[1] * uu + coef[2])).astype(np.float32)


def _highpass(z: np.ndarray, sigma_px: float) -> np.ndarray:
    filled = _fill_small_holes(z, 3 * sigma_px)
    base = np.nan_to_num(filled, nan=float(np.nanmedian(z)))
    return np.where(np.isfinite(z), z - cv2.GaussianBlur(base, (0, 0), sigma_px), np.nan)


def compare_overlap_runs(
    sweep_dir: Union[str, Path],
    ref_overlap: Optional[float] = None,
    detail_sigma_mm: float = 1.0,
    panel_px: int = 420,
) -> Optional[dict]:
    sweep_dir = Path(sweep_dir)
    runs = []
    for meta_path in sorted(sweep_dir.glob("overlap_*/session.json")):
        run_dir = meta_path.parent
        solve_path = run_dir / "gelsight_solved_solve.json"
        depth_path = run_dir / "gelsight_solved_depth.npy"
        if not (solve_path.exists() and depth_path.exists()):
            print(f"compare: {run_dir.name} not stitched yet; skipping")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        with open(solve_path) as f:
            solve = json.load(f)
        runs.append(dict(dir=run_dir, meta=meta, solve=solve, depth=np.load(depth_path)))
    if len(runs) < 1:
        print(f"compare: no stitched runs in {sweep_dir}")
        return None

    runs.sort(key=lambda r: -r['meta']['overlap'])
    if ref_overlap is None:
        ref = runs[0]
    else:
        ref = min(runs, key=lambda r: abs(r['meta']['overlap'] - ref_overlap))
    area_mm = tuple(a * 1000.0 for a in ref['meta']['area_m'])
    mm_per_px = ref['solve']['mm_per_px']

    for r in runs:
        r['crop'] = _crop_to_area(r['depth'], area_mm, r['solve']['mm_per_px'])
    h = min(r['crop'].shape[0] for r in runs)
    w = min(r['crop'].shape[1] for r in runs)
    for r in runs:
        r['crop'] = r['crop'][:h, :w]

    sig = detail_sigma_mm / mm_per_px
    if ref_overlap is None and len(runs) >= 3:
        ref = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref_flat = np.nanmedian(np.stack([_detrend(r['crop']) for r in runs]), axis=0)
        ref_label = "consensus (median of all runs)"
    else:
        ref_flat = _detrend(ref['crop'])
        ref_label = f"overlap {ref['meta']['overlap']:.2f} run"
    ref_detail = _highpass(ref_flat, sig)

    rows = []
    for r in runs:
        meta, solve, crop = r['meta'], r['solve'], r['crop']
        timing = meta.get('timing', {})
        row = {
            'overlap': meta['overlap'],
            'grid': f"{meta['touch_x']}x{meta['touch_y']}",
            'touches': meta['touch_x'] * meta['touch_y'],
            'scan_time_s': timing.get('total_s'),
            's_per_touch': timing.get('mean_touch_s'),
            'valid_contact_frac': timing.get('valid_contact_frac'),
            'coverage_frac': float(np.isfinite(crop).mean()),
            'solve_mode': solve.get('solve_mode'),
            'pair_residual_rms_mm': (solve.get('pair_residual_rms_mm') or {}).get('solved'),
            'is_reference': ref is not None and r is ref,
        }
        diff = _detrend(crop - ref_flat)
        dfin = diff[np.isfinite(diff)]
        detail_diff = _highpass(diff, sig)
        dd = detail_diff[np.isfinite(detail_diff)]
        shape_diff = diff - np.nan_to_num(detail_diff)
        sd = shape_diff[np.isfinite(shape_diff)]
        rms = lambda a: float(np.sqrt(np.mean(a ** 2))) if a.size else None  # noqa: E731
        row['rms_vs_ref_mm'] = rms(dfin)
        row['shape_rms_vs_ref_mm'] = rms(sd)
        row['detail_rms_vs_ref_mm'] = rms(dd)
        row['p95_abs_vs_ref_mm'] = float(np.percentile(np.abs(dfin), 95)) if dfin.size else None
        det = _highpass(crop, sig)
        m = np.isfinite(det) & np.isfinite(ref_detail)
        row['detail_corr_vs_ref'] = float(np.corrcoef(det[m], ref_detail[m])[0, 1]) if m.sum() > 10 else None
        r['diff'] = diff
        rows.append(row)

    # ---- montage: colour relief (top) over difference from reference (bottom)
    scale = panel_px / max(h, w)
    ph, pw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    px_mm = mm_per_px / scale
    ref_small = _fill_small_holes(_resize_nan(ref_flat, (ph, pw)), 2.0)
    lo, hi = (float(v) for v in np.nanpercentile(ref_small, [1, 99]))
    dlim = max(
        (float(np.nanpercentile(np.abs(r['diff']), 99)) for r in runs if r is not ref),
        default=0.1,
    ) or 0.1
    lut = _diverging_lut()
    cols = []
    for r, row in zip(runs, rows):
        z = _fill_small_holes(_resize_nan(_detrend(r['crop']), (ph, pw)), 2.0)
        shade = np.nan_to_num(_hillshade(z, px_mm, 2.0), nan=1.0)[..., None]
        top = (_colorize(z, lo, hi) * (0.35 + 0.65 * shade)).astype(np.uint8)
        bot = _colorize(_resize_nan(r['diff'], (ph, pw)), -dlim, dlim, lut)
        t = row['scan_time_s']
        title = (f"ov {row['overlap']:.2f} | {row['touches']} touches"
                 + (f" | {t / 60:.1f} min" if t else ""))
        resid = row['pair_residual_rms_mm']
        sub = ("REFERENCE" if row['is_reference']
               else f"vs ref {row['rms_vs_ref_mm']:.3f} (shape {row['shape_rms_vs_ref_mm']:.3f}, "
                    f"detail {row['detail_rms_vs_ref_mm']:.3f}) mm")
        sub += f" | resid {resid:.3f}" if resid is not None else ""
        col = np.vstack([_label(top, title), _label(bot, sub)])
        cols.append(col)
    gap = np.full((cols[0].shape[0], 10, 3), 255, np.uint8)
    montage = cols[0]
    for c in cols[1:]:
        montage = np.hstack([montage, gap, c])
    header = _label(
        np.full((4, montage.shape[1], 3), 255, np.uint8),
        f"area {area_mm[0]:.1f} x {area_mm[1]:.1f} mm | top: detrended height {lo:.2f}..{hi:.2f} mm | "
        f"bottom: difference from {ref_label}, +-{dlim:.3f} mm",
    )
    montage = np.vstack([header, montage])
    cv2.imwrite(str(sweep_dir / "compare_overlaps.png"), montage)

    with open(sweep_dir / "compare_overlaps.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result = {'area_mm': list(area_mm), 'reference': ref_label, 'runs': rows}
    with open(sweep_dir / "compare_overlaps.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"compare: area {area_mm[0]:.1f} x {area_mm[1]:.1f} mm, reference = {ref_label}")
    print(f"  {'overlap':>7} {'touches':>7} {'time':>8} {'resid':>7} {'vs_ref':>7} "
          f"{'shape':>7} {'detail':>7} {'detail_r':>8}")
    for row in rows:
        t = row['scan_time_s']
        print(f"  {row['overlap']:>7.2f} {row['touches']:>7d} "
              f"{(f'{t / 60:.1f}m' if t else '-'):>8} "
              f"{(row['pair_residual_rms_mm'] or float('nan')):>7.3f} "
              f"{row['rms_vs_ref_mm']:>7.3f} {row['shape_rms_vs_ref_mm']:>7.3f} "
              f"{row['detail_rms_vs_ref_mm']:>7.3f} {row['detail_corr_vs_ref']:>8.2f}")
    print(f"  -> {sweep_dir / 'compare_overlaps.png'} (+ .csv, .json)")
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compare the runs of an overlap sweep")
    ap.add_argument("sweep_dir")
    ap.add_argument("--ref", type=float, default=None, help="overlap of the reference run")
    args = ap.parse_args()
    compare_overlap_runs(args.sweep_dir, ref_overlap=args.ref)
