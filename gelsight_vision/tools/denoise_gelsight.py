"""Denoise a single gelsight touch (or a whole session's tiles) and dump
before/after PLYs so we can eyeball the effect before rolling it into the
stitched pipeline.

Recipe per tile:
  1. Load raw depth (mm) + BGR image.
  2. Mask outliers: |z| > outlier_mm -> NaN (matches stitch.py).
  3. Plane-detrend (matches stitch_session_anchored: removes DC + tilt).
  4. Denoise: cv2.bilateralFilter on the detrended float32 depth.
     - preserves weld-bead edges (edge-aware)
     - kills per-pixel Poisson ringing (spatial smoothing)
  5. Backproject to a metric point cloud in the tile's own local frame
     (x,y from pixel scale; z = denoised mm -> m). Same scale convention as
     stitch.write_anchored_ply so the numbers are apples-to-apples.

Usage:
  python -m gelsight_vision.tools.denoise_gelsight <session_dir> <touch_index>
  python -m gelsight_vision.tools.denoise_gelsight <session_dir> all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


_VERT_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
])

# Gelsight FOV constants (match stitch.write_anchored_ply defaults).
GEL_FULL_W_M = 0.020
GEL_FULL_H_M = 0.015
BORDER_FRACTION = 0.25  # 25% trimmed on each side -> keep = 0.5


def _fit_plane(depth_mm: np.ndarray, outlier_mm: float) -> np.ndarray:
    """Same plane fit as stitch._fit_plane, kept local so this tool has no
    circular deps and can be iterated on independently."""
    h, w = depth_mm.shape
    v_grid, u_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    mask = np.isfinite(depth_mm) & (np.abs(depth_mm) < outlier_mm)
    if int(mask.sum()) < 100:
        return np.zeros_like(depth_mm, dtype=np.float32)
    A = np.stack([u_grid[mask], v_grid[mask], np.ones_like(u_grid[mask])], axis=1).astype(np.float64)
    b = depth_mm[mask].astype(np.float64)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    return (coeffs[0] * u_grid + coeffs[1] * v_grid + coeffs[2]).astype(np.float32)


def _bilateral_denoise(
    depth_mm: np.ndarray,
    d: int = 9,
    sigma_color_mm: float = 0.5,
    sigma_space_px: float = 7.0,
) -> np.ndarray:
    """cv2.bilateralFilter on float32 depth. NaNs are temporarily filled with
    the tile median so they don't leak zeros into neighbors, then restored
    to NaN in the output."""
    finite = np.isfinite(depth_mm)
    fill = float(np.nanmedian(depth_mm)) if finite.any() else 0.0
    src = np.where(finite, depth_mm, fill).astype(np.float32)
    out = cv2.bilateralFilter(src, d=d, sigmaColor=sigma_color_mm, sigmaSpace=sigma_space_px)
    return np.where(finite, out, np.nan).astype(np.float32)


def _backproject_ply(
    depth_mm: np.ndarray,
    img_bgr: np.ndarray | None,
    ply_path: Path,
) -> int:
    """Same pixel-scale convention as stitch.write_anchored_ply for a single
    tile (touch_x=touch_y=1)."""
    h, w = depth_mm.shape
    keep = 1.0 - 2.0 * BORDER_FRACTION
    m_per_px_y = (GEL_FULL_W_M * keep) / h
    m_per_px_x = (GEL_FULL_H_M * keep) / w

    v_grid, u_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x_m = (u_grid - w / 2.0) * m_per_px_x
    y_m = (v_grid - h / 2.0) * m_per_px_y
    z_m = depth_mm * 1e-3

    valid = np.isfinite(z_m)
    n = int(valid.sum())
    if n == 0:
        return 0

    verts = np.empty(n, dtype=_VERT_DTYPE)
    verts['x'] = x_m[valid].astype(np.float32)
    verts['y'] = y_m[valid].astype(np.float32)
    verts['z'] = z_m[valid].astype(np.float32)
    if img_bgr is not None and img_bgr.shape[:2] == (h, w):
        verts['r'] = img_bgr[valid, 2]
        verts['g'] = img_bgr[valid, 1]
        verts['b'] = img_bgr[valid, 0]
    else:
        verts['r'] = 200
        verts['g'] = 200
        verts['b'] = 200

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(ply_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(verts.tobytes())
    return n


def _side_by_side_preview(
    detrended: np.ndarray,
    denoised: np.ndarray,
    out_path: Path,
    clip_mm: float = 1.0,
) -> None:
    """Two-panel viridis preview clipped tight around 0 (post-detrend), so
    the noise reduction is visible pixel-level."""
    def _to_color(arr: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(arr, nan=0.0)
        a = np.clip(a, -clip_mm, clip_mm)
        norm = ((a + clip_mm) / (2 * clip_mm) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)

    left = _to_color(detrended)
    right = _to_color(denoised)
    gap = np.full((left.shape[0], 6, 3), 255, dtype=np.uint8)
    combined = np.concatenate([left, gap, right], axis=1)
    cv2.imwrite(str(out_path), combined)


def denoise_touch(
    session_dir: Path,
    index: int,
    outlier_mm: float = 5.0,
    bilateral_d: int = 9,
    sigma_color_mm: float = 0.5,
    sigma_space_px: float = 7.0,
) -> None:
    stem = f"gelsight_{index:04d}"
    depth_path = session_dir / f"{stem}_depth.npy"
    img_path = session_dir / f"{stem}.png"
    if not depth_path.exists():
        print(f"denoise: {depth_path} missing")
        return

    raw = np.load(depth_path).astype(np.float32)
    img = cv2.imread(str(img_path)) if img_path.exists() else None

    masked = np.where(np.abs(raw) > outlier_mm, np.nan, raw)
    plane = _fit_plane(masked, outlier_mm=outlier_mm)
    detrended = masked - plane
    denoised = _bilateral_denoise(
        detrended, d=bilateral_d,
        sigma_color_mm=sigma_color_mm, sigma_space_px=sigma_space_px,
    )

    raw_ply = session_dir / f"{stem}_denoise_before.ply"
    denoised_ply = session_dir / f"{stem}_denoise_after.ply"
    preview_png = session_dir / f"{stem}_denoise_preview.png"

    n_before = _backproject_ply(detrended, img, raw_ply)
    n_after = _backproject_ply(denoised, img, denoised_ply)
    _side_by_side_preview(detrended, denoised, preview_png)

    def _stats(arr: np.ndarray) -> str:
        f = arr[np.isfinite(arr)]
        return (
            f"std {f.std():.3f} mm  |z|<=0.3mm coverage "
            f"{100*(np.abs(f) <= 0.3).mean():.1f}%"
        )

    print(f"touch {index}: {stem}_depth.npy ({raw.shape[0]}x{raw.shape[1]})")
    print(f"  detrended  -> {raw_ply.name} ({n_before} pts) {_stats(detrended)}")
    print(f"  denoised   -> {denoised_ply.name} ({n_after} pts) {_stats(denoised)}")
    print(f"  preview    -> {preview_png.name} (left=detrended, right=denoised, clipped ±1 mm)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    parser.add_argument(
        "index",
        help="1-based touch index, or 'all' to run every touch in the session",
    )
    parser.add_argument("--outlier-mm", type=float, default=5.0)
    parser.add_argument("--bilateral-d", type=int, default=9)
    parser.add_argument("--sigma-color-mm", type=float, default=0.5)
    parser.add_argument("--sigma-space-px", type=float, default=7.0)
    args = parser.parse_args()

    session_dir: Path = args.session_dir
    if not session_dir.exists():
        raise SystemExit(f"session dir not found: {session_dir}")

    if args.index == "all":
        indices = sorted(
            int(p.stem.split("_")[1])
            for p in session_dir.glob("gelsight_????_depth.npy")
        )
    else:
        indices = [int(args.index)]

    for idx in indices:
        denoise_touch(
            session_dir, idx,
            outlier_mm=args.outlier_mm,
            bilateral_d=args.bilateral_d,
            sigma_color_mm=args.sigma_color_mm,
            sigma_space_px=args.sigma_space_px,
        )


if __name__ == "__main__":
    main()
