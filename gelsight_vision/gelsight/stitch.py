"""Stitch the per-touch GelSight captures of a session into one grid view."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from gelsight_vision.gelsight.height_map import save_height_preview

_NAME_RE = re.compile(r"^gelsight_(\d{4})\.png$")


def _fit_plane(depth: np.ndarray, outlier_mm: float = 5.0) -> np.ndarray:
    """Least-squares fit of z = a*u + b*v + c to the depth map, then evaluate.
    Pixels with |z| > outlier_mm are excluded from the fit but the returned
    plane is dense at every pixel."""
    h, w = depth.shape
    v_grid, u_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    mask = np.isfinite(depth) & (np.abs(depth) < outlier_mm)
    if int(mask.sum()) < 100:
        return np.zeros_like(depth, dtype=np.float32)
    A = np.stack(
        [u_grid[mask], v_grid[mask], np.ones_like(u_grid[mask])],
        axis=1,
    ).astype(np.float64)
    b = depth[mask].astype(np.float64)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    a_c, b_c, c_c = coeffs
    plane = a_c * u_grid + b_c * v_grid + c_c
    return plane.astype(np.float32)


def _grid_position(capture_index: int, touch_x: int, touch_y: int) -> tuple[int, int]:
    """Map 1-indexed capture number to (row, col) in the stitched grid.

    Grid is `touch_x` rows tall by `touch_y` columns wide. Captures fill the
    current row left-to-right before moving one row down. With touch_x=3 and
    touch_y=3 this gives [[1,2,3],[4,5,6],[7,8,9]] -- capture 1 at top-left,
    final capture at bottom-right.
    """
    k = capture_index - 1
    row = k // touch_y
    col = k % touch_y
    return row, col


def stitch_session(
    session_dir: Union[str, Path],
    touch_x: int,
    touch_y: int,
    outlier_mm: float = 5.0,
) -> None:
    """Read all `gelsight_NNNN.png` + `_depth.npy` pairs from `session_dir` and
    write:
      - `gelsight_stitched.png` (RGB panorama)
      - `gelsight_stitched_depth.npy` (raw stitched depth, mm)
      - `gelsight_stitched_depth_preview.png` (raw preview)
      - `gelsight_stitched_depth_debiased.npy` (session-bias subtracted, mm)
      - `gelsight_stitched_depth_debiased_preview.png` (clipped tight around 0)
      - `gelsight_session_bias.npy` (the per-touch systematic bias map that
        was subtracted, in pre-rotation orientation, mm)

    Bias correction: gsrobotics' Poisson-integration reconstruction has a
    per-touch bias that appears identically on every touch (typically a
    strong top/bottom band, ~5 mm range). Median-averaging across all tiles
    exposes it (real sample geometry averages out; the artifact survives).
    Subtracting it drops per-pixel std by ~30% and reveals real weld texture.
    Outlier pixels (|z| > outlier_mm) are NaN'd before both the bias
    calculation and the stitch.
    """
    session_dir = Path(session_dir)
    img_paths = sorted(
        p for p in session_dir.glob("gelsight_*.png") if _NAME_RE.match(p.name)
    )
    if not img_paths:
        print(f"stitch: no captures found in {session_dir}")
        return

    expected = touch_x * touch_y
    if len(img_paths) != expected:
        print(
            f"stitch: expected {expected} captures, found {len(img_paths)}; "
            "stitching available frames into their grid slots"
        )

    first_img = cv2.rotate(cv2.imread(str(img_paths[0])), cv2.ROTATE_90_COUNTERCLOCKWISE)
    first_depth = np.rot90(
        np.load(img_paths[0].with_name(img_paths[0].stem + "_depth.npy")), k=1,
    )
    h, w = first_img.shape[:2]
    dh, dw = first_depth.shape

    # First pass: load every tile's raw depth (pre-rotation) with outliers
    # masked out, so we can compute a robust session-bias map.
    raw_depths: dict[int, np.ndarray] = {}
    for path in img_paths:
        match = _NAME_RE.match(path.name)
        if not match:
            continue
        idx = int(match.group(1))
        d = np.load(path.with_name(path.stem + "_depth.npy")).astype(np.float32)
        d = np.where(np.abs(d) > outlier_mm, np.nan, d)
        raw_depths[idx] = d

    # Median across tiles keeps the systematic per-touch artifact and cancels
    # out per-tile sample content (different tiles see different geometry).
    bias = np.nanmedian(np.stack(list(raw_depths.values())), axis=0)

    stitched_img = np.zeros((touch_x * h, touch_y * w, 3), dtype=first_img.dtype)
    stitched_depth = np.full(
        (touch_x * dh, touch_y * dw), np.nan, dtype=np.float32,
    )
    stitched_debiased = np.full(
        (touch_x * dh, touch_y * dw), np.nan, dtype=np.float32,
    )

    for path in img_paths:
        match = _NAME_RE.match(path.name)
        if not match:
            continue
        idx = int(match.group(1))
        row, col = _grid_position(idx, touch_x, touch_y)
        if not (0 <= row < touch_x and 0 <= col < touch_y):
            print(f"stitch: capture {idx} falls outside the {touch_x}x{touch_y} grid; skipping")
            continue
        img = cv2.rotate(cv2.imread(str(path)), cv2.ROTATE_90_COUNTERCLOCKWISE)
        raw = raw_depths[idx]
        debiased = raw - bias
        depth_rot = np.rot90(raw, k=1)
        debiased_rot = np.rot90(debiased, k=1)
        stitched_img[row * h:(row + 1) * h, col * w:(col + 1) * w] = img
        stitched_depth[row * dh:(row + 1) * dh, col * dw:(col + 1) * dw] = depth_rot
        stitched_debiased[row * dh:(row + 1) * dh, col * dw:(col + 1) * dw] = debiased_rot

    img_out = session_dir / "gelsight_stitched.png"
    depth_out = session_dir / "gelsight_stitched_depth.npy"
    preview_out = session_dir / "gelsight_stitched_depth_preview.png"
    debiased_out = session_dir / "gelsight_stitched_depth_debiased.npy"
    debiased_preview_out = session_dir / "gelsight_stitched_depth_debiased_preview.png"
    bias_out = session_dir / "gelsight_session_bias.npy"

    cv2.imwrite(str(img_out), stitched_img)
    np.save(depth_out, stitched_depth)
    np.save(debiased_out, stitched_debiased)
    np.save(bias_out, bias)

    # Raw preview: keep the existing behavior (NaN-fill with session median).
    preview_depth = np.where(
        np.isnan(stitched_depth), np.nanmedian(stitched_depth), stitched_depth
    )
    save_height_preview(preview_depth, preview_out)

    # Debiased preview: clip tight around 0 so real sub-mm signal is visible.
    debiased_for_viz = np.nan_to_num(stitched_debiased, nan=0.0)
    clip_mm = 1.0
    clipped = np.clip(debiased_for_viz, -clip_mm, clip_mm)
    norm = ((clipped + clip_mm) / (2 * clip_mm) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(debiased_preview_out), color)

    finite = stitched_debiased[np.isfinite(stitched_debiased)]
    print(f"stitched image -> {img_out}")
    print(f"stitched depth (raw)      -> {depth_out}")
    print(f"stitched depth (debiased) -> {debiased_out} "
          f"(std {finite.std():.3f} mm, "
          f"|z|<=1mm coverage {100*(np.abs(finite)<=1).mean():.1f}%)")
    print(f"session bias map          -> {bias_out}")
    print(f"raw preview      -> {preview_out}")
    print(f"debiased preview -> {debiased_preview_out}")


def write_stitched_ply(
    session_dir: Union[str, Path],
    touch_x: int,
    touch_y: int,
    gelsight_full_width_m: float = 0.020,
    gelsight_full_height_m: float = 0.015,
    border_fraction: float = 0.25,
) -> None:
    """Backproject the stitched gelsight height map to a metric point cloud
    saved as `gelsight_stitched_cloud.ply` next to the stitch outputs.

    Coordinates are in meters, centered on the panorama midpoint. Lateral
    scale is derived from the touch-grid geometry: each tile covers
    `gelsight_full_*_m * (1 - 2*border_fraction)` meters and is 320x240 px
    after resize (per HeightMapper defaults), so per-pixel scale works out
    to (cropped_fov_m / 320) = (cropped_fov_m / 240) = uniform.

    Z (height) is assumed to be in millimeters (gsrobotics Poisson output)
    and converted to meters for consistency with the D555 stereo PLYs.
    """
    session_dir = Path(session_dir)
    # Prefer the debiased stitch if present (it cancels the per-touch Poisson
    # boundary artifact); fall back to raw for older sessions.
    debiased_path = session_dir / "gelsight_stitched_depth_debiased.npy"
    raw_path = session_dir / "gelsight_stitched_depth.npy"
    depth_path = debiased_path if debiased_path.exists() else raw_path
    img_path = session_dir / "gelsight_stitched.png"
    if not depth_path.exists():
        print(f"stitch ply: {depth_path} missing; skipping")
        return

    print(f"stitch ply: using {depth_path.name}")
    depth = np.load(depth_path)
    img = cv2.imread(str(img_path)) if img_path.exists() else None

    keep = 1.0 - 2.0 * border_fraction  # =0.5 by default
    h, w = depth.shape
    # Per-pixel scale from the panorama geometry. touch_x tiles run down the
    # panorama (rows); touch_y run across (cols). Uniform because tiles are
    # resized to the same aspect ratio as their FOV.
    m_per_px_y = (touch_x * gelsight_full_width_m * keep) / h
    m_per_px_x = (touch_y * gelsight_full_height_m * keep) / w

    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x_m = (u - w / 2.0) * m_per_px_x
    y_m = (v - h / 2.0) * m_per_px_y
    z_m = depth * 1e-3  # mm -> m

    valid = np.isfinite(z_m)
    n = int(valid.sum())
    if n == 0:
        print("stitch ply: no finite depth pixels; skipping")
        return

    verts = np.empty(n, dtype=np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
    ]))
    verts['x'] = x_m[valid].astype(np.float32)
    verts['y'] = y_m[valid].astype(np.float32)
    verts['z'] = z_m[valid].astype(np.float32)
    if img is not None and img.shape[:2] == (h, w):
        # OpenCV loads BGR; PLY expects RGB.
        verts['r'] = img[valid, 2]
        verts['g'] = img[valid, 1]
        verts['b'] = img[valid, 0]
    else:
        verts['r'] = 200
        verts['g'] = 200
        verts['b'] = 200

    ply_path = session_dir / "gelsight_stitched_cloud.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(ply_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(verts.tobytes())
    print(
        f"gelsight stitched point cloud -> {ply_path} ({n} pts, "
        f"scale={m_per_px_x*1000:.4f} mm/px)"
    )


def stitch_session_anchored(
    session_dir: Union[str, Path],
    touch_x: int,
    touch_y: int,
    outlier_mm: float = 5.0,
    gel_depth_sign: int = 1,
) -> None:
    """Build a plate-frame stitched depth map using per-touch pose JSONs.

    Per tile:
        1. Fit a plane to the raw gelsight depth (outliers masked) and subtract
           it. Removes DC + per-tile tilt. Remaining signal is sub-mm local
           relief of the sample surface.
        2. Add the tile's `contact_Z` (from `gelsight_NNNN_pose.json`,
           translation.z in the plate frame, in meters -> mm) to anchor the
           detrended relief in the plate frame.

    Result: a single depth map where every pixel is metric plate-frame Z
    (mm), so weld-bead tiles sit higher and flat-plate tiles sit lower —
    matching what the robot actually saw at contact.

    `gel_depth_sign` picks the sign convention: +1 means "positive gelsight
    depth = raised sample feature (higher plate Z)". Flip to -1 if the first
    stitched image looks inverted.
    """
    session_dir = Path(session_dir)
    img_paths = sorted(
        p for p in session_dir.glob("gelsight_*.png") if _NAME_RE.match(p.name)
    )
    if not img_paths:
        print(f"stitch anchored: no captures in {session_dir}")
        return

    first_depth = np.rot90(
        np.load(img_paths[0].with_name(img_paths[0].stem + "_depth.npy")), k=1,
    )
    dh, dw = first_depth.shape  # after rotation

    stitched_anchored = np.full(
        (touch_x * dh, touch_y * dw), np.nan, dtype=np.float32,
    )
    contact_z_per_tile: dict[int, float] = {}
    missing_poses: list[int] = []

    for path in img_paths:
        match = _NAME_RE.match(path.name)
        if not match:
            continue
        idx = int(match.group(1))
        row, col = _grid_position(idx, touch_x, touch_y)
        if not (0 <= row < touch_x and 0 <= col < touch_y):
            continue

        pose_path = path.with_name(path.stem + "_pose.json")
        if not pose_path.exists():
            missing_poses.append(idx)
            continue
        with open(pose_path) as f:
            pose = json.load(f)
        contact_z_m = float(pose['translation']['z'])
        contact_z_per_tile[idx] = contact_z_m

        d = np.load(path.with_name(path.stem + "_depth.npy")).astype(np.float32)
        d = np.where(np.abs(d) > outlier_mm, np.nan, d)
        plane = _fit_plane(d, outlier_mm=outlier_mm)
        detrended = d - plane  # sub-mm local relief around zero
        anchored_mm = contact_z_m * 1000.0 + gel_depth_sign * detrended
        anchored_rot = np.rot90(anchored_mm, k=1)
        stitched_anchored[row * dh:(row + 1) * dh, col * dw:(col + 1) * dw] = anchored_rot

    if not contact_z_per_tile:
        print(
            "stitch anchored: no pose JSONs found — anchored stitch not "
            "written. Ensure set_plate is publishing plate on /tf_static and "
            "re-run the session."
        )
        return

    if missing_poses:
        print(
            f"stitch anchored: {len(missing_poses)} tiles missing pose JSON: "
            f"{missing_poses[:10]}{'...' if len(missing_poses) > 10 else ''}"
        )

    depth_out = session_dir / "gelsight_stitched_depth_anchored.npy"
    preview_out = session_dir / "gelsight_stitched_depth_anchored_preview.png"
    np.save(depth_out, stitched_anchored)

    finite = stitched_anchored[np.isfinite(stitched_anchored)]
    # Preview stretches over the actual anchored range so the tile-to-tile
    # jumps (weld bead vs flat plate) become the dominant visual signal.
    lo, hi = np.percentile(finite, [1, 99])
    clip = np.clip(np.nan_to_num(stitched_anchored, nan=lo), lo, hi)
    if hi > lo:
        norm = ((clip - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(clip, dtype=np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(preview_out), color)

    zs = np.array(list(contact_z_per_tile.values()))
    print(f"stitch anchored: {len(contact_z_per_tile)} tiles anchored")
    print(f"  contact-Z range across tiles: {zs.min()*1000:.2f} to "
          f"{zs.max()*1000:.2f} mm  (spread {zs.ptp()*1000:.2f} mm)")
    print(f"  anchored depth  -> {depth_out} "
          f"(range [{finite.min():.2f}, {finite.max():.2f}] mm)")
    print(f"  anchored preview -> {preview_out}")


def write_anchored_ply(
    session_dir: Union[str, Path],
    touch_x: int,
    touch_y: int,
    gelsight_full_width_m: float = 0.020,
    gelsight_full_height_m: float = 0.015,
    border_fraction: float = 0.25,
) -> None:
    """Backproject the anchored (plate-frame Z) stitched depth into a metric
    PLY that's directly comparable to the D555 stereo PLY."""
    session_dir = Path(session_dir)
    depth_path = session_dir / "gelsight_stitched_depth_anchored.npy"
    img_path = session_dir / "gelsight_stitched.png"
    if not depth_path.exists():
        print(f"anchored ply: {depth_path} missing; skipping")
        return

    depth_mm = np.load(depth_path)
    img = cv2.imread(str(img_path)) if img_path.exists() else None

    keep = 1.0 - 2.0 * border_fraction
    h, w = depth_mm.shape
    m_per_px_y = (touch_x * gelsight_full_width_m * keep) / h
    m_per_px_x = (touch_y * gelsight_full_height_m * keep) / w

    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    # Panorama is centered on the tile grid origin — matches the touch-loop
    # convention where captures are laid out around the plate frame origin.
    x_m = (u - w / 2.0) * m_per_px_x
    y_m = (v - h / 2.0) * m_per_px_y
    z_m = depth_mm * 1e-3  # anchored map is mm -> convert to m

    valid = np.isfinite(z_m)
    n = int(valid.sum())
    if n == 0:
        print("anchored ply: no finite pixels")
        return

    verts = np.empty(n, dtype=np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
    ]))
    verts['x'] = x_m[valid].astype(np.float32)
    verts['y'] = y_m[valid].astype(np.float32)
    verts['z'] = z_m[valid].astype(np.float32)
    if img is not None and img.shape[:2] == (h, w):
        verts['r'] = img[valid, 2]
        verts['g'] = img[valid, 1]
        verts['b'] = img[valid, 0]
    else:
        verts['r'] = 200
        verts['g'] = 200
        verts['b'] = 200

    ply_path = session_dir / "gelsight_stitched_cloud_anchored.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(ply_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(verts.tobytes())
    print(
        f"gelsight anchored point cloud -> {ply_path} ({n} pts, "
        f"z_range [{z_m[valid].min()*1000:.2f}, {z_m[valid].max()*1000:.2f}] mm)"
    )


# ---------------------------------------------------------------------------
# Overlap-based per-tile Z offset solve
# ---------------------------------------------------------------------------

def _diverging_lut() -> np.ndarray:
    """Blue-white-red 256-entry BGR LUT for signed maps (cv2.applyColorMap)."""
    pos = [0.0, 0.25, 0.5, 0.75, 1.0]
    rgb = [(33, 102, 172), (103, 169, 207), (247, 247, 247),
           (239, 138, 98), (178, 24, 43)]
    t = np.linspace(0.0, 1.0, 256)
    lut = np.stack(
        [np.interp(t, pos, [c[ch] for c in rgb]) for ch in (2, 1, 0)], axis=1,
    )
    return lut.astype(np.uint8).reshape(256, 1, 3)


def _colorize(
    z: np.ndarray, lo: float, hi: float, cmap=cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """Map a float array to BGR over [lo, hi]; NaN -> dark gray."""
    nan_mask = ~np.isfinite(z)
    span = max(hi - lo, 1e-9)
    norm = np.clip((np.where(nan_mask, lo, z) - lo) / span, 0.0, 1.0)
    out = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    out[nan_mask] = (40, 40, 40)
    return out


def _fill_small_holes(z: np.ndarray, sigma_px: float) -> np.ndarray:
    """Normalized-convolution fill of NaN holes up to a few sigma wide.
    Larger holes (e.g. a missing tile) stay NaN. Viz only."""
    m = np.isfinite(z).astype(np.float32)
    num = cv2.GaussianBlur(np.nan_to_num(z).astype(np.float32) * m, (0, 0), sigma_px)
    den = cv2.GaussianBlur(m, (0, 0), sigma_px)
    with np.errstate(invalid='ignore', divide='ignore'):
        fill = np.where(den > 1e-3, num / den, np.nan)
    return np.where(m > 0, z, fill).astype(np.float32)


def _resize_nan(z: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Area-downsample a NaN-holed float map to `shape` (rows, cols)."""
    m = np.isfinite(z).astype(np.float32)
    dsize = (shape[1], shape[0])
    num = cv2.resize(np.nan_to_num(z).astype(np.float32) * m, dsize, interpolation=cv2.INTER_AREA)
    den = cv2.resize(m, dsize, interpolation=cv2.INTER_AREA)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0.25, num / den, np.nan).astype(np.float32)


def _hillshade(z_mm: np.ndarray, px_mm: float, exaggeration: float) -> np.ndarray:
    """Lambertian shading in [0, 1], light from the upper-left at 45 deg."""
    zf = cv2.GaussianBlur(np.nan_to_num(z_mm, nan=float(np.nanmedian(z_mm))), (0, 0), 1.0)
    gy, gx = np.gradient(zf * exaggeration, px_mm)
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    light = np.array([-1.0, -1.0, np.sqrt(2.0)])
    light /= np.linalg.norm(light)
    shade = np.clip(n @ light, 0.0, 1.0)
    shade[~np.isfinite(z_mm)] = np.nan
    return shade.astype(np.float32)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Prepend a white title strip to a BGR panel."""
    strip = np.full((28, img.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(strip, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def _write_ply(path: Path, x_m, y_m, z_m, bgr: Optional[np.ndarray]) -> int:
    """Binary little-endian XYZRGB PLY from flat arrays. Returns vertex count."""
    n = int(x_m.size)
    verts = np.empty(n, dtype=np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
    ]))
    verts['x'], verts['y'], verts['z'] = x_m, y_m, z_m
    if bgr is not None:
        verts['r'], verts['g'], verts['b'] = bgr[:, 2], bgr[:, 1], bgr[:, 0]
    else:
        verts['r'] = verts['g'] = verts['b'] = 200
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(verts.tobytes())
    return n


_ORIENTATIONS = {
    'none': lambda a: a,
    'fliplr': np.fliplr,
    'flipud': np.flipud,
    'rot180': lambda a: a[::-1, ::-1],
}


def _detect_tile_orientation(
    rgb: dict, grid_rc: dict, touch_y: int, row_step: int, col_step: int,
    depth_rows: int, min_overlap_px: int, max_pairs: int = 60,
) -> tuple[str, dict]:
    """Score each candidate in-tile flip by the median NCC of high-passed
    gray RGB over right/down neighbour overlaps. Returns (best, scores)."""
    def hp(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return g - cv2.GaussianBlur(g, (0, 0), 4)

    def ncc(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))

    gray = {k: hp(v) for k, v in rgb.items()}
    rh, rw = next(iter(gray.values())).shape
    s = rh / depth_rows
    by_rc = {rc: k for k, rc in grid_rc.items()}
    pairs = []
    for k, (r, c) in grid_rc.items():
        for dr, dc in ((0, 1), (1, 0)):
            nb = by_rc.get((r + dr, c + dc))
            dR, dC = int(round(dr * row_step * s)), int(round(dc * col_step * s))
            if nb is None or k not in gray or nb not in gray:
                continue
            if rh - dR < min_overlap_px * s or rw - dC < min_overlap_px * s:
                continue
            pairs.append((k, nb, dR, dC))
    by_dir = {d: [p for p in pairs if (p[3] > 0) == d] for d in (True, False)}
    scores = {}
    for name, f in _ORIENTATIONS.items():
        # A flip along the seam leaves that direction's NCC unchanged, so
        # score right and down neighbours separately and keep the weaker.
        per_dir = []
        for ps in by_dir.values():
            ps = ps[::max(1, len(ps) // max_pairs)]
            if ps:
                per_dir.append(np.median([
                    ncc(f(gray[a])[dR:, dC:], f(gray[b])[:rh - dR, :rw - dC]) for a, b, dR, dC in ps
                ]))
        scores[name] = float(min(per_dir)) if per_dir else float('nan')
    best = max(scores, key=lambda n: -np.inf if np.isnan(scores[n]) else scores[n])
    return best, scores


def _band_slope(band: np.ndarray, px_mm: float) -> np.ndarray:
    """Per-line least-squares slope (mm/mm) across a NaN-holed edge band."""
    m = np.isfinite(band)
    t = np.broadcast_to(np.arange(band.shape[1]) * px_mm, band.shape)
    n = m.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        tb = np.where(m, t, 0).sum(axis=1) / n
        vb = np.where(m, band, 0).sum(axis=1) / n
        dt = np.where(m, t - tb[:, None], 0)
        dv = np.where(m, band - vb[:, None], 0)
        slope = (dt * dv).sum(axis=1) / (dt * dt).sum(axis=1)
    return np.where(n >= band.shape[1] // 2, slope, np.nan)


def _robust_plane_inliers(
    rc: np.ndarray, z: np.ndarray, thresh: float, iters: int = 5,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Fit z = a*row + b*col + c with iterative trimming.
    Returns (inlier mask, [a, b, c] or None if too few points)."""
    inl = np.isfinite(z)
    coef = None
    for _ in range(iters):
        if inl.sum() < 3:
            break
        A = np.column_stack([rc[inl], np.ones(int(inl.sum()))])
        coef, *_ = np.linalg.lstsq(A, z[inl], rcond=None)
        res = np.column_stack([rc, np.ones(len(z))]) @ coef - z
        new = np.isfinite(z) & (np.abs(res) <= thresh)
        if np.array_equal(new, inl):
            break
        inl = new
    return inl, coef


def stitch_session_solved(
    session_dir: Union[str, Path],
    touch_x: int,
    touch_y: int,
    *,
    touch_overlap: float = 0.0,
    gelsight_full_width_m: float = 0.020,
    gelsight_full_height_m: float = 0.015,
    border_fraction: float = 0.25,
    height_mm_per_unit: Optional[float] = None,
    gel_depth_sign: int = 1,
    outlier_mm: float = 1.0,
    debias: bool = True,
    min_overlap_px: int = 8,
    overlap_stride: int = 4,
    orientation: str = 'auto',
    seam_band_px: int = 24,
    seam_skip_px: int = 4,
    seam_sigma_mm: float = 0.05,
    seam_slope_sigma: float = 0.02,
    prior_sigma_mm: float = 1.0,
    tilt_prior_sigma: float = 0.05,
    contact_outlier_mm: float = 1.5,
    safe_z_m: float = 0.02,
    irls_iters: int = 4,
    shade_exaggeration: float = 2.0,
    detail_sigma_mm: float = 1.0,
    ply_stride: int = 2,
) -> Optional[dict]:
    """Stitch the touch grid with per-tile Z offsets solved from tile overlaps.

    Every tile k gets three unknowns -- offset c_k plus tilts (a_k, b_k) --
    so its plate-frame surface is  relief_k(y, x) + c_k + a_k*y + b_k*x,
    where relief_k is the session-debiased, plane-detrended gelsight height
    (mm) and (y, x) are tile-local mm coordinates. They are solved jointly by
    weighted least squares (Huber IRLS) over:

      * Overlap constraints, for neighbours whose measured step is smaller
        than the tile FOV by >= `min_overlap_px`: both tiles must agree at
        every shared pixel (sampled every `overlap_stride` px). All 8
        neighbours are used, diagonals included.
      * Seam constraints otherwise (abutting tiles, the default grid): each
        tile's edge band (`seam_band_px` wide, `seam_skip_px` in from the
        noisy Poisson boundary) is median-reduced per line and both tile
        planes must agree at the shared seam. Height agreement on a single
        line can't see the tilt *across* the seam (a ramp that meets its
        neighbours at both edges satisfies it), so each seam also requires
        the across-seam slope (tile tilt + edge-band relief slope) to match
        on both sides (`seam_slope_sigma`, mm/mm).
      * A weak prior pulling c_k to the tile's robot contact Z
        (`prior_sigma_mm`). Touches that stopped at `safe_z_m` (force tripped
        before contact) or sit > `contact_outlier_mm` off a robust plane
        through all contacts get no prior and are placed purely by their
        neighbours.
      * A weak prior pulling every tile's tilt to the slope of that robust
        contact plane (`tilt_prior_sigma`, mm/mm), so slope drift along a
        chain of seams stays bounded.

    Seams dominate locally (`seam_sigma_mm` << `prior_sigma_mm`); contact Z
    only pins the large-scale shape that chained seams can't observe.

    `orientation` flips every rotated tile before stitching ('none', 'fliplr',
    'flipud', 'rot180'); 'auto' picks the flip under which overlapping RGB
    neighbours agree best, and falls back to 'none' without overlap.

    gsrobotics' Poisson heights are in lateral-pixel units, so the default
    `height_mm_per_unit` is the tile's mm/px (10 mm / 320 px = 0.03125).

    Writes `gelsight_solved_*` next to the captures (depth npy + previews,
    RGB mosaic, shaded relief, overlays, before/after comparison, PLY, and a
    JSON with the per-tile solution). Returns that JSON as a dict.
    """
    session_dir = Path(session_dir)
    img_paths = sorted(
        p for p in session_dir.glob("gelsight_*.png") if _NAME_RE.match(p.name)
    )
    if not img_paths:
        print(f"stitch solved: no captures in {session_dir}")
        return None

    # ---- load tiles (rotated to panorama orientation) ----------------------
    raw: dict[int, np.ndarray] = {}
    rgb: dict[int, np.ndarray] = {}
    contact_z: dict[int, float] = {}
    pose_xy: dict[int, np.ndarray] = {}
    for path in img_paths:
        idx = int(_NAME_RE.match(path.name).group(1))
        row, col = _grid_position(idx, touch_x, touch_y)
        if not (0 <= row < touch_x and 0 <= col < touch_y):
            continue
        raw[idx] = np.load(path.with_name(path.stem + "_depth.npy")).astype(np.float32)
        rgb[idx] = cv2.rotate(cv2.imread(str(path)), cv2.ROTATE_90_COUNTERCLOCKWISE)
        pose_path = path.with_name(path.stem + "_pose.json")
        if pose_path.exists():
            with open(pose_path) as f:
                pose = json.load(f)
            if 'translation' in pose:
                t = pose['translation']
                pose_xy[idx] = np.array([t['x'], t['y']]) * 1000.0
                # motion_collect_overlap flags touches whose force tripped
                # before real contact; their Z is not a surface height.
                if pose.get('valid_contact', True):
                    contact_z[idx] = float(t['z']) * 1000.0
    ids = sorted(raw)
    pos = {k: i for i, k in enumerate(ids)}
    grid_rc = {k: _grid_position(k, touch_x, touch_y) for k in ids}

    keep = 1.0 - 2.0 * border_fraction
    th, tw = np.rot90(raw[ids[0]], k=1).shape  # (320, 240)
    mm_per_px = gelsight_full_width_m * 1000.0 * keep / th
    unit_mm = mm_per_px if height_mm_per_unit is None else height_mm_per_unit

    # ---- lattice step from poses (falls back to nominal) -------------------
    row_step = th * (1.0 - touch_overlap)
    col_step = tw * (1.0 - touch_overlap)
    step_source = "nominal"
    if len(pose_xy) >= 3 and touch_x >= 2 and touch_y >= 2:
        ks = sorted(pose_xy)
        A = np.array([[1.0, *grid_rc[k]] for k in ks])
        coef, *_ = np.linalg.lstsq(A, np.array([pose_xy[k] for k in ks]), rcond=None)
        row_step = float(np.linalg.norm(coef[1])) / mm_per_px
        col_step = float(np.linalg.norm(coef[2])) / mm_per_px
        step_source = "poses"
    # Within 2 px of butting up -> treat as exactly abutting (no hairline gaps).
    row_step = th if abs(row_step - th) < 2.0 else row_step
    col_step = tw if abs(col_step - tw) < 2.0 else col_step
    row_step_i, col_step_i = int(round(row_step)), int(round(col_step))
    # Round each tile's origin from the float step, so a fractional step (e.g.
    # 214.4 px at overlap 0.33) doesn't accumulate rounding drift down the grid.
    origin = {
        k: (int(round(grid_rc[k][0] * row_step)), int(round(grid_rc[k][1] * col_step)))
        for k in ids
    }
    ov_rows, ov_cols = th - row_step_i, tw - col_step_i
    solve_mode = "overlap" if min(ov_rows, ov_cols) >= min_overlap_px else (
        "seam" if max(ov_rows, ov_cols) < min_overlap_px else "overlap+seam")
    print(
        f"stitch solved: {len(ids)} tiles, step {row_step_i}x{col_step_i} px "
        f"({step_source}), overlap {ov_rows}/{ov_cols} px "
        f"({'overlap' if ov_rows >= min_overlap_px else 'seam'}/"
        f"{'overlap' if ov_cols >= min_overlap_px else 'seam'} constraints)"
    )

    # In-tile orientation: with real overlap, pick the flip under which
    # overlapping neighbours actually agree (seam-only data can't tell).
    orient_scores = None
    if orientation == 'auto':
        orientation = 'none'
        if max(ov_rows, ov_cols) >= min_overlap_px:
            best, orient_scores = _detect_tile_orientation(
                rgb, grid_rc, touch_y, row_step, col_step, th, min_overlap_px,
            )
            score_str = ", ".join(f"{n} {v:.2f}" for n, v in orient_scores.items())
            print(f"  tile orientation NCC: {score_str}")
            if best != 'none' and orient_scores[best] - orient_scores['none'] > 0.1:
                print(f"  -> applying '{best}' to every tile (overlaps agree far better)")
                orientation = best
    flip = _ORIENTATIONS[orientation]
    rgb = {k: np.ascontiguousarray(flip(v)) for k, v in rgb.items()}

    # Session median kills the per-touch Poisson artifact (see stitch_session).
    bias = np.median(np.stack([raw[k] for k in ids]), axis=0) if debias else 0.0
    relief: dict[int, np.ndarray] = {}
    for k in ids:
        d = flip(np.rot90(raw[k] - bias, k=1)) * unit_mm * gel_depth_sign
        d = d - np.nanmedian(d)  # offset is solved for; centre so _fit_plane's mask holds
        d = np.where(np.abs(d) > outlier_mm, np.nan, d)
        relief[k] = np.ascontiguousarray(d - _fit_plane(d, outlier_mm=outlier_mm)).astype(np.float32)

    yy = (np.arange(th) - (th - 1) / 2.0) * mm_per_px  # tile-local mm, rows
    xx = (np.arange(tw) - (tw - 1) / 2.0) * mm_per_px  # tile-local mm, cols

    # ---- pairwise constraints ----------------------------------------------
    # Each pair yields samples (eA, yA, xA, eB, yB, xB): relief values and the
    # tile-local coords at which the two tile planes must agree. With real
    # overlap every tile is paired with all 8 neighbours (right, down and both
    # diagonals cover each pair once); otherwise only edge neighbours via seams.
    has_overlap = ov_rows >= min_overlap_px and ov_cols >= min_overlap_px
    neighbours = ((0, 1), (1, 0), (1, 1), (1, -1)) if has_overlap else ((0, 1), (1, 0))
    ctr_r, ctr_c = (th - 1) / 2.0, (tw - 1) / 2.0
    pairs = []
    for k in ids:
        r, c = grid_rc[k]
        for dr, dc in neighbours:
            if not (r + dr < touch_x and 0 <= c + dc < touch_y):
                continue
            nb = (r + dr) * touch_y + (c + dc) + 1
            if nb not in relief:
                continue
            RA, RB = relief[k], relief[nb]
            dR = origin[nb][0] - origin[k][0]  # B's origin in A's pixels
            dC = origin[nb][1] - origin[k][1]
            if th - abs(dR) >= min_overlap_px and tw - abs(dC) >= min_overlap_px:
                rA = np.arange(max(0, dR), min(th, th + dR), overlap_stride)
                cA = np.arange(max(0, dC), min(tw, tw + dC), overlap_stride)
                gr, gc = np.meshgrid(rA, cA, indexing='ij')
                gr, gc = gr.ravel(), gc.ravel()
                eA, eB = RA[gr, gc], RB[gr - dR, gc - dC]
                ok = np.isfinite(eA) & np.isfinite(eB)
                if ok.sum() < 10:
                    continue
                gr, gc = gr[ok], gc[ok]
                pairs.append(dict(
                    a=pos[k], b=pos[nb], horiz=dr == 0, slope_t=None,
                    eA=eA[ok], eB=eB[ok],
                    yA=(gr - ctr_r) * mm_per_px, xA=(gc - ctr_c) * mm_per_px,
                    yB=(gr - dR - ctr_r) * mm_per_px, xB=(gc - dC - ctr_c) * mm_per_px,
                ))
                continue
            if dr and dc:  # diagonal without overlap: corners only touch
                continue
            horiz = dc == 1
            step = dC if horiz else dR
            if not horiz:  # transpose so the seam is always across columns
                RA, RB = RA.T, RB.T
            L, W = RA.shape  # L = along seam, W = across
            lines = np.arange(0, L, 2)
            bA = RA[lines, W - seam_skip_px - seam_band_px:W - seam_skip_px]
            bB = RB[lines, seam_skip_px:seam_skip_px + seam_band_px]
            with warnings.catch_warnings():  # all-NaN lines -> NaN, dropped below
                warnings.simplefilter("ignore", RuntimeWarning)
                eA, eB = np.nanmedian(bA, axis=1), np.nanmedian(bB, axis=1)
            eA_slope = _band_slope(bA, mm_per_px)
            eB_slope = _band_slope(bB, mm_per_px)
            ok = np.isfinite(eA) & np.isfinite(eB)
            if ok.sum() < 10:
                continue
            slope_t = None
            sl = np.isfinite(eA_slope) & np.isfinite(eB_slope)
            if sl.sum() >= 10:
                # bA + sA = bB + sB  ->  bA - bB = sB - sA  (across-seam tilts)
                slope_t = float(np.median(eB_slope[sl] - eA_slope[sl]))
            seam = ((W - 1) + step) / 2.0  # seam position in A coords
            along = (lines[ok] - (L - 1) / 2.0) * mm_per_px
            acrossA = np.full(int(ok.sum()), (seam - (W - 1) / 2.0) * mm_per_px)
            acrossB = acrossA - step * mm_per_px
            if horiz:  # (y, x) = (along, across)
                yA, xA, yB, xB = along, acrossA, along, acrossB
            else:      # transposed: (y, x) = (across, along)
                yA, xA, yB, xB = acrossA, along, acrossB, along
            pairs.append(dict(a=pos[k], b=pos[nb], horiz=horiz, slope_t=slope_t,
                              eA=eA[ok], eB=eB[ok], yA=yA, xA=xA, yB=yB, xB=xB))

    # ---- contact-Z prior ---------------------------------------------------
    n_t = len(ids)
    zc = np.array([contact_z.get(k, np.nan) for k in ids])
    no_contact = np.isfinite(zc) & (np.abs(zc - safe_z_m * 1000.0) < 0.05)
    rc_arr = np.array([grid_rc[k] for k in ids], dtype=float)
    prior_ok, plane = _robust_plane_inliers(
        rc_arr, np.where(no_contact, np.nan, zc), contact_outlier_mm,
    )
    if prior_ok.sum() == 0:  # no usable poses: gauge-fix around zero
        prior_target, prior_w = np.zeros(n_t), np.full(n_t, 1e-6)
    else:
        prior_target = np.where(prior_ok, zc, 0.0)
        prior_w = np.where(prior_ok, 1.0 / prior_sigma_mm ** 2, 0.0)
    if plane is not None and prior_ok.sum() >= 3:
        # Contact plane is per grid step; convert to mm/mm in tile-local (y, x).
        tilt_target = (plane[0] / (row_step * mm_per_px), plane[1] / (col_step * mm_per_px))
        tilt_w = 1.0 / tilt_prior_sigma ** 2
    else:
        tilt_target, tilt_w = (0.0, 0.0), 1e-6

    # ---- weighted least squares + Huber IRLS -------------------------------
    def jac(p):
        return np.column_stack([
            np.ones_like(p['yA']), p['yA'], p['xA'],
            -np.ones_like(p['yB']), -p['yB'], -p['xB'],
        ])

    for p in pairs:
        p['J'] = jac(p)
        p['t'] = p['eB'] - p['eA']
        p['w0'] = 1.0 / (seam_sigma_mm ** 2 * len(p['t']))
        p['w'] = np.full(len(p['t']), p['w0'])

    n_u = 3 * n_t
    u = np.zeros(n_u)
    for it in range(irls_iters):
        AtA = np.zeros((n_u, n_u))
        Atb = np.zeros(n_u)
        for p in pairs:
            cols = np.r_[3 * p['a']:3 * p['a'] + 3, 3 * p['b']:3 * p['b'] + 3]
            Jw = p['J'] * p['w'][:, None]
            AtA[np.ix_(cols, cols)] += Jw.T @ p['J']
            Atb[cols] += Jw.T @ p['t']
            if p['slope_t'] is not None:
                ia = 3 * p['a'] + (2 if p['horiz'] else 1)
                ib = 3 * p['b'] + (2 if p['horiz'] else 1)
                ws = 1.0 / seam_slope_sigma ** 2
                AtA[ia, ia] += ws
                AtA[ib, ib] += ws
                AtA[ia, ib] -= ws
                AtA[ib, ia] -= ws
                Atb[ia] += ws * p['slope_t']
                Atb[ib] -= ws * p['slope_t']
        ci = np.arange(n_t) * 3
        AtA[ci, ci] += prior_w
        Atb[ci] += prior_w * prior_target
        AtA[ci + 1, ci + 1] += tilt_w
        AtA[ci + 2, ci + 2] += tilt_w
        Atb[ci + 1] += tilt_w * tilt_target[0]
        Atb[ci + 2] += tilt_w * tilt_target[1]
        u = np.linalg.solve(AtA, Atb)
        if not pairs:
            break
        res = [p['J'] @ np.r_[u[3 * p['a']:3 * p['a'] + 3], u[3 * p['b']:3 * p['b'] + 3]] - p['t']
               for p in pairs]
        allres = np.concatenate(res)
        scale = 1.4826 * np.median(np.abs(allres - np.median(allres))) + 1e-9
        for p, r in zip(pairs, res):
            p['w'] = p['w0'] * np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(r), 1e-12))

    sol = u.reshape(n_t, 3)  # columns: c, a (d/dy), b (d/dx)

    # Seam step = mean disagreement across a shared edge; compare against the
    # naive placement (contact Z offset, no tilt) that stitch_session_anchored
    # effectively uses.
    naive = np.zeros((n_t, 3))
    naive[:, 0] = np.where(np.isfinite(zc), zc, np.nanmedian(zc) if np.isfinite(zc).any() else 0.0)

    def seam_steps(v):
        out = []
        for p in pairs:
            r = p['J'] @ np.r_[v[p['a']], v[p['b']]] - p['t']
            out.append(float(np.median(r)))
        return np.array(out)

    steps_naive, steps_solved = seam_steps(naive), seam_steps(sol)

    def pair_residual_rms(v):
        # Per-pixel disagreement between overlapping tiles (or seam lines).
        if not pairs:
            return float('nan')
        r = np.concatenate([p['J'] @ np.r_[v[p['a']], v[p['b']]] - p['t'] for p in pairs])
        return float(np.sqrt(np.mean(r ** 2)))

    # ---- composite depth + RGB (feathered in overlaps) ---------------------
    H = int(round(row_step * (touch_x - 1))) + th
    W = int(round(col_step * (touch_y - 1))) + tw
    ramp_r = np.minimum(np.arange(th) + 1, th - np.arange(th)).astype(np.float32)
    ramp_c = np.minimum(np.arange(tw) + 1, tw - np.arange(tw)).astype(np.float32)
    feather = np.minimum(ramp_r[:, None], ramp_c[None, :])
    plane_tile = lambda c, a, b: c + a * yy[:, None] + b * xx[None, :]  # noqa: E731

    def composite(v):
        acc = np.zeros((H, W), np.float64)
        wsum = np.zeros((H, W), np.float64)
        for k in ids:
            r0, c0 = origin[k]
            z = relief[k] + plane_tile(*v[pos[k]])
            m = np.isfinite(z)
            acc[r0:r0 + th, c0:c0 + tw] += np.where(m, z, 0.0) * feather * m
            wsum[r0:r0 + th, c0:c0 + tw] += feather * m
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(wsum > 0, acc / wsum, np.nan).astype(np.float32)

    depth_solved = composite(sol)
    depth_naive = composite(naive)

    rh, rw = rgb[ids[0]].shape[:2]
    s_rgb = rh / th
    RH, RW = int(round(H * s_rgb)), int(round(W * s_rgb))
    feather_rgb = cv2.resize(feather, (rw, rh), interpolation=cv2.INTER_AREA)[..., None]
    racc = np.zeros((RH, RW, 3), np.float64)
    rws = np.zeros((RH, RW, 1), np.float64)
    for k in ids:
        r0, c0 = (int(round(o * s_rgb)) for o in origin[k])
        hh, ww = min(rh, RH - r0), min(rw, RW - c0)
        racc[r0:r0 + hh, c0:c0 + ww] += rgb[k][:hh, :ww] * feather_rgb[:hh, :ww]
        rws[r0:r0 + hh, c0:c0 + ww] += feather_rgb[:hh, :ww]
    rgb_mosaic = np.where(rws > 0, racc / np.maximum(rws, 1e-9), 40).astype(np.uint8)

    # ---- outputs -----------------------------------------------------------
    out = lambda name: session_dir / f"gelsight_solved_{name}"  # noqa: E731
    np.save(out("depth.npy"), depth_solved)
    cv2.imwrite(str(out("rgb.png")), rgb_mosaic)

    # Visualizations at the RGB mosaic's resolution so every PNG aligns 1:1.
    viz_px_mm = mm_per_px / s_rgb
    dz = _fill_small_holes(_resize_nan(depth_solved, (RH, RW)), 3.0)
    dz_naive = _fill_small_holes(_resize_nan(depth_naive, (RH, RW)), 3.0)
    lo, hi = (float(v) for v in np.nanpercentile(dz, [1, 99]))
    save_height_preview(dz, out("depth_preview.png"), scale_factor=1.0, value_range=(lo, hi))

    # Detail: high-pass relief so sub-mm texture isn't swamped by plate tilt.
    sig = detail_sigma_mm / viz_px_mm
    low = _fill_small_holes(dz, 3 * sig)
    low = cv2.GaussianBlur(np.nan_to_num(low, nan=float(np.nanmedian(dz))), (0, 0), sig)
    detail = dz - low
    dlim = float(np.nanpercentile(np.abs(detail), 99))
    lut = _diverging_lut()
    save_height_preview(detail, out("detail_preview.png"), colormap=lut,
                        scale_factor=1.0, value_range=(-dlim, dlim))

    shade = _hillshade(dz, viz_px_mm, shade_exaggeration)
    shade_u8 = (np.nan_to_num(shade, nan=0.16) * 255).astype(np.uint8)
    cv2.imwrite(str(out("shaded.png")), shade_u8)

    color = _colorize(dz, lo, hi)
    shade3 = np.nan_to_num(shade, nan=1.0)[..., None]
    relief_shaded = (color * (0.35 + 0.65 * shade3)).astype(np.uint8)
    cv2.imwrite(str(out("color_relief.png")), relief_shaded)
    overlay = cv2.addWeighted(rgb_mosaic, 0.55, color, 0.45, 0.0)
    cv2.imwrite(str(out("rgb_depth_overlay.png")), overlay)
    rgb_shaded = (rgb_mosaic * (0.4 + 0.6 * shade3)).astype(np.uint8)
    cv2.imwrite(str(out("rgb_shaded.png")), rgb_shaded)

    # Before/after: naive contact-Z placement vs solved, same color scale,
    # plus the per-tile offset correction (solved c - contact Z).
    corr = np.full((RH, RW), np.nan, np.float32)
    for k in ids:
        if np.isfinite(zc[pos[k]]):
            r0, c0 = (int(round(o * s_rgb)) for o in origin[k])
            corr[r0:r0 + rh, c0:c0 + rw] = sol[pos[k], 0] - zc[pos[k]]
    clim = float(np.nanmax(np.abs(corr))) if np.isfinite(corr).any() else 1.0
    shade_naive = np.nan_to_num(_hillshade(dz_naive, viz_px_mm, shade_exaggeration), nan=1.0)[..., None]
    panels = [
        _label((_colorize(dz_naive, lo, hi) * (0.35 + 0.65 * shade_naive)).astype(np.uint8),
               f"naive contact-Z  |  {lo:.2f}..{hi:.2f} mm"),
        _label(relief_shaded, f"{solve_mode}-solved  |  {lo:.2f}..{hi:.2f} mm"),
        _label(_colorize(corr, -clim, clim, lut), f"offset correction  |  +-{clim:.2f} mm"),
    ]
    gap = np.full((panels[0].shape[0], 12, 3), 255, np.uint8)
    cv2.imwrite(str(out("comparison.png")),
                np.hstack([panels[0], gap, panels[1], gap, panels[2]]))

    # Point cloud (meters, centred on the mosaic), coloured from the RGB mosaic.
    v, uu = np.mgrid[0:H:ply_stride, 0:W:ply_stride]
    zs = depth_solved[v, uu]
    ok = np.isfinite(zs)
    rv = np.clip((v[ok] * s_rgb).astype(int), 0, RH - 1)
    ru = np.clip((uu[ok] * s_rgb).astype(int), 0, RW - 1)
    n_pts = _write_ply(
        out("cloud.ply"),
        ((uu[ok] - W / 2.0) * mm_per_px * 1e-3).astype(np.float32),
        ((v[ok] - H / 2.0) * mm_per_px * 1e-3).astype(np.float32),
        (zs[ok] * 1e-3).astype(np.float32),
        rgb_mosaic[rv, ru],
    )

    rms = lambda a: float(np.sqrt(np.mean(a ** 2))) if a.size else float('nan')  # noqa: E731
    summary = {
        'touch_x': touch_x, 'touch_y': touch_y,
        'mm_per_px': mm_per_px, 'height_mm_per_unit': unit_mm,
        'step_px': [row_step_i, col_step_i], 'step_source': step_source,
        'overlap_px': [ov_rows, ov_cols], 'solve_mode': solve_mode,
        'n_pairs': len(pairs),
        'orientation': orientation, 'orientation_scores': orient_scores,
        'pair_residual_rms_mm': {'naive': pair_residual_rms(naive), 'solved': pair_residual_rms(sol)},
        'seam_step_rms_mm': {'naive': rms(steps_naive), 'solved': rms(steps_solved)},
        'seam_step_max_mm': {
            'naive': float(np.abs(steps_naive).max()) if pairs else None,
            'solved': float(np.abs(steps_solved).max()) if pairs else None,
        },
        'no_contact_tiles': [k for k in ids if no_contact[pos[k]]],
        'flagged_invalid_tiles': [k for k in ids if k in pose_xy and k not in contact_z],
        'prior_outlier_tiles': [
            k for k in ids
            if np.isfinite(zc[pos[k]]) and not no_contact[pos[k]] and not prior_ok[pos[k]]
        ],
        'tiles': {
            str(k): {
                'grid_rc': list(grid_rc[k]),
                'contact_z_mm': None if not np.isfinite(zc[pos[k]]) else float(zc[pos[k]]),
                'offset_mm': float(sol[pos[k], 0]),
                'tilt_y_mm_per_mm': float(sol[pos[k], 1]),
                'tilt_x_mm_per_mm': float(sol[pos[k], 2]),
                'prior_used': bool(prior_ok[pos[k]]),
            } for k in ids
        },
    }
    with open(out("solve.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  pair residual RMS (per pixel): {summary['pair_residual_rms_mm']['solved']:.3f} mm")
    print(f"  seam step RMS: naive {summary['seam_step_rms_mm']['naive']:.3f} mm -> "
          f"solved {summary['seam_step_rms_mm']['solved']:.3f} mm over {len(pairs)} seams")
    print(f"  no-contact tiles (z at safe_z): {summary['no_contact_tiles']}")
    print(f"  flagged invalid by collector:   {summary['flagged_invalid_tiles']}")
    print(f"  contact-Z outliers (no prior):  {summary['prior_outlier_tiles']}")
    print(f"  depth  -> {out('depth.npy')} ({lo:.2f}..{hi:.2f} mm, p1..p99)")
    print(f"  rgb    -> {out('rgb.png')}")
    print(f"  views  -> depth_preview, detail_preview, shaded, color_relief, "
          f"rgb_depth_overlay, rgb_shaded, comparison (.png)")
    print(f"  cloud  -> {out('cloud.ply')} ({n_pts} pts)")
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=stitch_session_solved.__doc__.split("\n")[0])
    ap.add_argument("session_dir")
    ap.add_argument("--touch-x", type=int, required=True)
    ap.add_argument("--touch-y", type=int, required=True)
    ap.add_argument("--touch-overlap", type=float, default=0.0)
    ap.add_argument("--sign", type=int, default=1, choices=(-1, 1))
    ap.add_argument("--seam-sigma-mm", type=float, default=0.05)
    ap.add_argument("--prior-sigma-mm", type=float, default=1.0)
    args = ap.parse_args()
    stitch_session_solved(
        args.session_dir, args.touch_x, args.touch_y,
        touch_overlap=args.touch_overlap, gel_depth_sign=args.sign,
        seam_sigma_mm=args.seam_sigma_mm, prior_sigma_mm=args.prior_sigma_mm,
    )
