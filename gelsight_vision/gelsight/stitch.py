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


def _inpaint_untrusted(z: np.ndarray, good: np.ndarray, work_px: int = 1024) -> np.ndarray:
    """Replace `z` where `good` is False by a smooth fill from the good
    surface around it. Solved at reduced resolution (cv2 Navier-Stokes
    inpainting on a uint16-scaled copy keeps it fast), then the fill is
    upsampled and only written into the bad pixels."""
    bad = ~good
    if not bad.any() or not good.any():
        return z
    H, W = z.shape
    s = max(1, int(np.ceil(max(H, W) / work_px)))
    zs = cv2.resize(np.where(good, z, np.nan).astype(np.float32), (W // s, H // s),
                    interpolation=cv2.INTER_AREA)
    gs = np.isfinite(zs)
    lo = float(np.nanmin(zs))
    q = np.where(gs, zs - lo, 0).astype(np.float32)
    fill = cv2.inpaint(q, (~gs).astype(np.uint8), max(3, 8 // s), cv2.INPAINT_NS) + lo
    fill = cv2.resize(fill, (W, H), interpolation=cv2.INTER_LINEAR)
    keep_nan = ~np.isfinite(z) & ~cv2.dilate(np.isfinite(z).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    out = np.where(bad, fill, z).astype(np.float32)
    out[keep_nan] = np.nan  # don't invent data outside the scanned area
    return out


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


def _detect_beads(
    rgb_mosaic: np.ndarray,
    valid: np.ndarray,
    px_mm: float,
    work_mm: float = 0.125,
    bg_mm: float = 20.0,
    thresh_mad: float = 6.0,
    close_mm: float = 0.75,
    open_mm: float = 0.75,
    min_area_mm2: float = 10.0,
    border_mm: float = 1.0,
) -> dict:
    """Find raised (or sunk) features -- weld beads, spot welds -- in the
    GelSight RGB mosaic, without using any height data.

    GelSight shading departs from the flat-gel colour wherever the surface
    slopes, so a bead's flanks light up and its crown shows as a thin
    flat-coloured line between them. Per pixel we take the RGB distance from
    the local plate colour (a `bg_mm` median), threshold it at the plate
    median + `thresh_mad` robust sigmas, close (`close_mm`) to bridge the
    crown line, open (`open_mm`) to drop scratches and spatter, fill holes,
    and keep blobs of at least `min_area_mm2`. Any number of blobs (lines,
    dots, none) is fine: each is one feature.

    Returns a dict with the work-resolution grid (`work_mm`, `labels`, 0 =
    plate) and per-feature `beads` (area, half-width, centroid, bbox), plus
    `dev` (the deviation image) for debugging.
    """
    f = max(1, int(round(work_mm / px_mm)))
    work = px_mm * f
    h, w = rgb_mosaic.shape[0] // f, rgb_mosaic.shape[1] // f
    x = cv2.resize(rgb_mosaic[:h * f, :w * f], (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    ok = cv2.resize(valid[:h * f, :w * f].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.999
    b = int(round(border_mm / work))
    if b:
        ok = cv2.erode(ok.astype(np.uint8), np.ones((2 * b + 1, 2 * b + 1), np.uint8)).astype(bool)

    # Plate colour: large median on a 4x coarser grid (medianBlur needs uint8
    # for big apertures), back up to the work grid.
    d = 4
    small = cv2.resize(x, (max(1, w // d), max(1, h // d)), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(bg_mm / (work * d))) | 1)
    bg = cv2.medianBlur(np.clip(small, 0, 255).astype(np.uint8), k)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    dev = cv2.GaussianBlur(np.linalg.norm(x - bg, axis=-1), (0, 0), max(0.25 / work, 0.5))

    beads: list[dict] = []
    labels = np.zeros((h, w), np.int32)
    if ok.sum() < 100:
        return dict(work_mm=work, labels=labels, beads=beads, dev=dev, thresh=float('nan'))
    med = float(np.median(dev[ok]))
    mad = 1.4826 * float(np.median(np.abs(dev[ok] - med))) + 1e-6
    thresh = med + thresh_mad * mad
    m = ((dev > thresh) & ok).astype(np.uint8)

    def disk(r_mm):
        r = max(1, int(round(r_mm / work)))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk(close_mm))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(open_mm))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)  # fills holes
    filled &= ok.astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(filled, connectivity=8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA] * work ** 2
        if area < min_area_mm2:
            continue
        j = len(beads) + 1
        labels[lab == i] = j
        x0, y0, bw, bh = (int(stats[i, s]) for s in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                     cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        beads.append(dict(id=j, area_mm2=float(area), bbox=[y0, x0, bh, bw],
                          centroid_mm=[float((cent[i, 1] + 0.5) * work), float((cent[i, 0] + 0.5) * work)]))
    return dict(work_mm=work, labels=labels, beads=beads, dev=dev, thresh=thresh)


def _contact_map(
    img: np.ndarray, baseline: Optional[np.ndarray], px_mm: float,
    base_tex: float, tex_ratio: float = 2.0, col_thresh: float = 40.0, close_mm: float = 0.3,
) -> np.ndarray:
    """Where the gel actually touches the part, from one GelSight frame.

    Gel pressed on metal picks up its micro-texture (machining marks, pits,
    spatter); gel held off the part (e.g. beside a tall bead the tip is
    resting on) stays smooth like the no-contact baseline, and the depth
    model reads it as flat surface. Contact = local high-pass texture above
    `tex_ratio` x the baseline's texture level, or colour departing from the
    baseline by more than `col_thresh` (sloped contact without texture),
    closed by `close_mm`. Returns a bool map at the frame's resolution."""
    f = img.astype(np.float32)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 0.15 / px_mm)
    tex = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), 0.5 / px_mm))
    contact = tex > tex_ratio * base_tex
    if baseline is not None and baseline.shape == img.shape:
        col = cv2.GaussianBlur(np.linalg.norm(f - baseline, axis=-1), (0, 0), 0.5 / px_mm)
        contact |= col > col_thresh
    r = max(1, int(round(close_mm / px_mm)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.morphologyEx(contact.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)


def _bead_profile(labels: np.ndarray, j: int, work: float) -> tuple[np.ndarray, float]:
    """Unit dome over feature `j`: 0 at its outline, 1 along its medial axis
    (a line bead's centreline, a dot's centre). Uses a circular-arc cross
    section on the inward distance, normalised by the feature's half-width
    (99th percentile of that distance, robust to blobby ends)."""
    inside = (labels == j).astype(np.uint8)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5) * work
    half = float(np.percentile(dist[inside > 0], 99)) if inside.any() else 0.0
    if half <= 0:
        return np.zeros(labels.shape, np.float32), 0.0
    u = np.clip(dist / half, 0.0, 1.0)
    return np.sqrt(1.0 - (1.0 - u) ** 2).astype(np.float32), half


def stitch_session_solved(
    session_dir: Union[str, Path],
    touch_x: int,
    touch_y: int,
    *,
    touch_overlap: float = 0.0,
    gelsight_full_width_m: float = 0.0199,
    gelsight_full_height_m: float = 0.015,
    border_fraction: float = 0.25,
    height_mm_per_unit: Optional[float] = None,
    gel_depth_sign: int = -1,
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
    steep_slope: float = 0.0,
    steep_dilate_mm: float = 0.5,
    steep_smooth_mm: float = 0.15,
    bead_prior: bool = False,
    bead_prior_sigma_mm: float = 0.5,
    bead_thresh_mad: float = 6.0,
    bead_min_area_mm2: float = 10.0,
    bead_open_mm: float = 0.75,
    output_tag: Optional[str] = None,
    edge_trim: float = 0.0,
    contact_mask: bool = False,
    contact_tex_ratio: float = 2.0,
    contact_col: float = 40.0,
    contact_z_is_peak: bool = False,
    bead_height_from_z: bool = False,
    huber: str = 'all',
    out_dir: Optional[Union[str, Path]] = None,
    write_images: bool = True,
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
    `height_mm_per_unit` is the tile's mm/px (about 0.031 at 320 px across a
    50% crop). Tiles must have square pixels: the scale comes from the long
    side only, and the lattice step from the poses, so a frame that was
    squashed to 4:3 gets stretched and "overlaps" strips that are different
    places on the part (HeightMapper(preserve_aspect=True) avoids that).

    `gel_depth_sign` is -1 because the gsrobotics depth is negative where the
    surface pushes into the gel: on the 2026-11-04 weld plate +1 turned the
    raised bead into a valley, and -1 agrees with the robot contact heights.
    `gelsight_full_width_m` is the long side of the streamed frame, measured
    as 19.9 mm by registering neighbouring raw frames against the robot step
    (the 16:9 frame is 19.9 x 11.2 mm, not 20 x 15).

    `steep_slope` (mm/mm, 0 = off) distrusts tile pixels whose relief slope
    exceeds it, plus `steep_dilate_mm` around them. GelSight infers slope
    from shading, which fails at steep edges such as weld toes: the slope
    saturates and the gel just past the edge reads the opposite way, so the
    toe comes out as a spike. Distrusted pixels are left out of the overlap
    constraints; in the composite any trusted tile covering the same spot
    wins, and spots no tile reads with trust are inpainted from the trusted
    surface around them (steep edges become smooth ramps between the
    trusted heights on either side).

    `bead_prior` (off by default; off = the solve above, unchanged) is a
    Z-blind alternative to the two robot priors. Raised features (weld beads,
    spot welds -- any number, any shape, or none) are found in the GelSight
    RGB mosaic (`_detect_beads`), and each gets a unit dome template that is
    0 at its outline and 1 along its medial axis (`_bead_profile`). Each tile's
    offset and tilt are then pulled (`bead_prior_sigma_mm`, `tilt_prior_sigma`)
    toward  sum_j h_j * (template j over that tile),  i.e. toward a flat
    plate at height 0 plus one dome per feature, where the dome amplitudes h_j
    are extra unknowns solved with everything else (either sign, so a groove
    comes out as a dip). The overlaps still set the local shape; the prior
    only replaces what contact Z used to pin: the large-scale shape, which is
    now "flat plate, features crest along their centre". Contact Z is not
    used in the solve; if present it only sets a constant display datum (so
    heights stay in plate-frame mm) and is reported for comparison under
    `bead_prior.z_check`. Outputs go to `gelsight_beadprior_*` unless
    `output_tag` says otherwise, so the default outputs are left alone.

    `edge_trim` (fraction of each tile axis, per edge; 0 = off) drops that
    band at every tile border from the overlap constraints and nearly zeroes
    its weight in the composite. The gsrobotics Poisson integration leaves
    large ramps at tile borders that differ from tile to tile (the session
    median only removes the shared part); with real overlap there is enough
    tile interior to give the border up.

    `contact_mask` (off by default) finds, per tile, where the gel actually
    touched the part (`_contact_map`: micro-texture or colour departing from
    `gelsight_baseline.png`). Gel held off the part -- beside a bead taller
    than the gel can wrap, the tip rests on the crown and the plate around it
    is never touched -- reads as flat and, through the overlaps, pins the bead
    crown to plate level. No-contact pixels are left out of each tile's
    centring/detrend and of the overlap constraints, and only fill the
    composite where no tile touched.

    `contact_z_is_peak` (contact-Z mode): the force stop happens when the
    highest point under the gel touches, so contact Z is the tile's *peak*
    height, not its mean; the prior then targets c_k = Z_k - p99(relief_k).
    Use with a large `contact_outlier_mm` so bead tiles keep their prior.

    `bead_height_from_z` (with `bead_prior`): the solve stays Z-blind, then
    each feature's dome is rescaled so its crown sits at the median contact-Z
    excess (over the robust contact plane) of the tiles on that crown -- one
    robot number per feature, averaged over its tiles. The gel's photometric
    slopes read steep weld flanks several times too shallow, so this is the
    only height scale for a tall bead short of a gel calibration.

    `huber` picks which overlap pairs the IRLS may down-weight: 'all' (the
    default), 'off' (plain least squares), or 'nonbead' (pairs touching a
    `bead_prior` feature keep full weight -- they disagree more than plate
    pairs by nature, and down-weighting them flattens the bead).

    `out_dir` writes outputs there instead of next to the captures, and
    `write_images=False` writes only the depth npy and solve JSON (plus the
    bead map), for quick parameter sweeps.

    Writes `gelsight_<output_tag>_*` (default `gelsight_solved_*`) next to the captures (depth npy + previews,
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

    # Per-tile contact maps at relief resolution (see `contact_mask`).
    contact: dict[int, np.ndarray] = {}
    if contact_mask:
        rgb_px_mm = mm_per_px * th / rgb[ids[0]].shape[0]
        bpath = session_dir / "gelsight_baseline.png"
        base_img = cv2.imread(str(bpath)) if bpath.exists() else None
        if base_img is not None:
            base_img = np.ascontiguousarray(flip(cv2.rotate(base_img, cv2.ROTATE_90_COUNTERCLOCKWISE)))
            bg = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            bhp = bg - cv2.GaussianBlur(bg, (0, 0), 0.15 / rgb_px_mm)
            base_tex = float(np.median(np.sqrt(cv2.GaussianBlur(bhp * bhp, (0, 0), 0.5 / rgb_px_mm))))
            base_f = base_img.astype(np.float32)
        else:  # no baseline: texture floor from the smoothest 5% of all tiles
            base_f = None
            samp = []
            for k in ids[::max(1, len(ids) // 50)]:
                g = cv2.cvtColor(rgb[k], cv2.COLOR_BGR2GRAY).astype(np.float32)
                hp = g - cv2.GaussianBlur(g, (0, 0), 0.15 / rgb_px_mm)
                samp.append(np.percentile(np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), 0.5 / rgb_px_mm)), 5))
            base_tex = float(np.median(samp))
            print(f"  contact mask: no {bpath.name}, texture floor from tiles only")
        for k in ids:
            cm = _contact_map(rgb[k], base_f, rgb_px_mm, base_tex, contact_tex_ratio, contact_col)
            contact[k] = cv2.resize(cm.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA) > 0.5
        frac = np.array([contact[k].mean() for k in ids])
        print(f"  contact mask: median {np.median(frac):.0%} of tile in contact, "
              f"{int((frac < 0.5).sum())} tiles under 50%")

    relief: dict[int, np.ndarray] = {}
    for k in ids:
        d = flip(np.rot90(raw[k] - bias, k=1)) * unit_mm * gel_depth_sign
        dc = np.where(contact[k], d, np.nan) if contact else d  # centre/detrend on contact only
        if contact and np.isfinite(dc).sum() < 100:
            dc = d
        d = d - np.nanmedian(dc)  # offset is solved for; centre so _fit_plane's mask holds
        dc = dc - np.nanmedian(dc)
        d = np.where(np.abs(d) > outlier_mm, np.nan, d)
        dc = np.where(np.abs(dc) > outlier_mm, np.nan, dc)
        relief[k] = np.ascontiguousarray(d - _fit_plane(dc, outlier_mm=outlier_mm)).astype(np.float32)

    # Per-tile trust: False near steep relief (see `steep_slope`).
    trust: dict[int, np.ndarray] = {}
    if steep_slope > 0:
        sig = max(steep_smooth_mm / mm_per_px, 0.5)
        dil = int(round(steep_dilate_mm / mm_per_px))
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1)) if dil else None
        for k in ids:
            d = relief[k]
            fin = np.isfinite(d)
            sm = cv2.GaussianBlur(np.where(fin, d, np.nanmedian(d)).astype(np.float32), (0, 0), sig)
            gy_, gx_ = np.gradient(sm, mm_per_px)
            steep = (np.hypot(gx_, gy_) > steep_slope).astype(np.uint8)
            if kern is not None:
                steep = cv2.dilate(steep, kern)
            trust[k] = fin & ~steep.astype(bool)
        frac = 1.0 - np.mean([t.mean() for t in trust.values()])
        print(f"  steep-slope mask: {frac:.1%} of tile pixels distrusted (> {steep_slope} mm/mm)")
    # Relief used for the overlap constraints: distrusted pixels -> NaN.
    relief_fit = {k: np.where(trust[k], relief[k], np.nan) if trust else relief[k] for k in ids}
    if contact:
        relief_fit = {k: np.where(contact[k], v, np.nan) for k, v in relief_fit.items()}
    edge_keep = None
    if edge_trim > 0:
        er, ec = int(round(edge_trim * th)), int(round(edge_trim * tw))
        edge_keep = np.zeros((th, tw), bool)
        edge_keep[er:th - er, ec:tw - ec] = True
        relief_fit = {k: np.where(edge_keep, v, np.nan) for k, v in relief_fit.items()}
        print(f"  edge trim: {er} / {ec} px per tile edge left out of overlaps")

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
            RA, RB = relief_fit[k], relief_fit[nb]
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

    # ---- RGB mosaic (independent of the Z solve) ---------------------------
    H = int(round(row_step * (touch_x - 1))) + th
    W = int(round(col_step * (touch_y - 1))) + tw
    ramp_r = np.minimum(np.arange(th) + 1, th - np.arange(th)).astype(np.float32)
    ramp_c = np.minimum(np.arange(tw) + 1, tw - np.arange(tw)).astype(np.float32)
    feather = np.minimum(ramp_r[:, None], ramp_c[None, :])

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
        if contact_z_is_peak:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                peak = np.array([np.nanpercentile(relief_fit[k], 99) for k in ids])
            prior_target = np.where(prior_ok & np.isfinite(peak), zc - np.nan_to_num(peak), prior_target)
        prior_w = np.where(prior_ok, 1.0 / prior_sigma_mm ** 2, 0.0)
    if plane is not None and prior_ok.sum() >= 3:
        # Contact plane is per grid step; convert to mm/mm in tile-local (y, x).
        tilt_target = (plane[0] / (row_step * mm_per_px), plane[1] / (col_step * mm_per_px))
        tilt_w = 1.0 / tilt_prior_sigma ** 2
    else:
        tilt_target, tilt_w = (0.0, 0.0), 1e-6

    # ---- bead-shape prior (Z-blind; replaces the two priors above) ---------
    # G[t, p, j]: plane fit (p = offset, d/dy, d/dx) of feature j's unit dome
    # over tile t's footprint, so the prior is  tile_params ~= G @ h.
    G = np.zeros((n_t, 3, 0))
    beads_found: Optional[dict] = None
    bead_list: list[dict] = []
    if bead_prior:
        beads_found = _detect_beads(
            rgb_mosaic, rws[..., 0] > 0, mm_per_px / s_rgb,
            thresh_mad=bead_thresh_mad, min_area_mm2=bead_min_area_mm2, open_mm=bead_open_mm,
        )
        bead_list = beads_found['beads']
        work = beads_found['work_mm']
        lab_w = beads_found['labels']
        G = np.zeros((n_t, 3, len(bead_list)))
        # Tile footprints on the work grid; global mm = (pixel index + 0.5) * pitch.
        fh, fw = th * mm_per_px, tw * mm_per_px
        foot = []
        for k in ids:
            y0, x0 = origin[k][0] * mm_per_px, origin[k][1] * mm_per_px
            i0, i1 = max(0, int(np.ceil(y0 / work - 0.5))), min(lab_w.shape[0], int(np.floor((y0 + fh) / work - 0.5)) + 1)
            j0, j1 = max(0, int(np.ceil(x0 / work - 0.5))), min(lab_w.shape[1], int(np.floor((x0 + fw) / work - 0.5)) + 1)
            yl = (np.arange(i0, i1) + 0.5) * work - (y0 + fh / 2.0)
            xl = (np.arange(j0, j1) + 0.5) * work - (x0 + fw / 2.0)
            Yl, Xl = np.meshgrid(yl, xl, indexing='ij')
            A = np.column_stack([np.ones(Yl.size), Yl.ravel(), Xl.ravel()])
            foot.append((i0, i1, j0, j1, np.linalg.pinv(A) if A.shape[0] >= 3 else None))
        for jb, bd in enumerate(bead_list):
            prof, half = _bead_profile(lab_w, bd['id'], work)
            bd['half_width_mm'] = half
            by0, bx0, bh, bw = bd['bbox']
            for t, (i0, i1, j0, j1, P) in enumerate(foot):
                if P is None or i1 <= by0 or i0 >= by0 + bh or j1 <= bx0 or j0 >= bx0 + bw:
                    continue
                G[t, :, jb] = P @ prof[i0:i1, j0:j1].ravel()
        prior_target, prior_w = np.zeros(n_t), np.full(n_t, 1.0 / bead_prior_sigma_mm ** 2)
        tilt_target, tilt_w = (0.0, 0.0), 1.0 / tilt_prior_sigma ** 2
        print(f"  bead prior: {len(bead_list)} feature(s) "
              + ", ".join(f"#{b['id']} {b['area_mm2']:.0f} mm^2 (half-width {b['half_width_mm']:.2f} mm)"
                          for b in bead_list)
              + "; contact Z not used in the solve")
    n_b = G.shape[2]

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

    if huber not in ('all', 'off', 'nonbead'):
        raise ValueError(f"huber must be 'all', 'off' or 'nonbead', not {huber!r}")
    if huber == 'nonbead' and not bead_prior:
        raise ValueError("huber='nonbead' needs bead_prior=True (it uses the detected features)")
    bead_tile = np.abs(G[:, 0, :]).sum(axis=1) > 0.02 if n_b else np.zeros(n_t, bool)
    n_u = 3 * n_t + n_b  # tile (c, a, b) triples, then one dome amplitude per feature
    if n_b:
        # Prior residual  tile_params - G h  (target 0): its tile-tile diagonal
        # is the prior_w / tilt_w added below; these are the cross terms.
        wv = np.empty(3 * n_t)
        wv[0::3], wv[1::3], wv[2::3] = prior_w, tilt_w, tilt_w
        Gm = G.reshape(3 * n_t, n_b)
        WG = wv[:, None] * Gm
        bead_block = (WG, Gm.T @ WG + 1e-9 * np.eye(n_b))
    u = np.zeros(n_u)
    free_tiles: set[int] = set()
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
        if n_b:
            T = slice(0, 3 * n_t)
            Hb = slice(3 * n_t, n_u)
            AtA[T, Hb] -= bead_block[0]
            AtA[Hb, T] -= bead_block[0].T
            AtA[Hb, Hb] += bead_block[1]
        try:
            u = np.linalg.solve(AtA, Atb)
        except np.linalg.LinAlgError:
            # A tile with no usable overlap and no prior (e.g. masked out by
            # contact_mask + edge_trim with its contact Z rejected) is
            # unconstrained; a tiny ridge parks it at 0 instead of failing.
            u = np.linalg.solve(AtA + 1e-6 * np.eye(n_u), Atb)
            if it == 0:
                free_tiles.update(ids[i] for i in range(n_t) if AtA[3 * i, 3 * i] < 1e-9)
                print(f"  warning: {len(free_tiles)} unconstrained tile(s) "
                      f"{sorted(free_tiles)[:10]} -> ridge, left out of the composite")
        if not pairs:
            break
        res = [p['J'] @ np.r_[u[3 * p['a']:3 * p['a'] + 3], u[3 * p['b']:3 * p['b'] + 3]] - p['t']
               for p in pairs]
        if huber == 'off':
            break
        allres = np.concatenate(res)
        scale = 1.4826 * np.median(np.abs(allres - np.median(allres))) + 1e-9
        for p, r in zip(pairs, res):
            if huber == 'nonbead' and (bead_tile[p['a']] or bead_tile[p['b']]):
                continue
            p['w'] = p['w0'] * np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(r), 1e-12))

    sol = u[:3 * n_t].reshape(n_t, 3).copy()  # columns: c, a (d/dy), b (d/dx)
    h_bead = u[3 * n_t:]

    bead_summary = None
    if bead_prior:
        # Display datum only: the Z-blind solve puts the plate at 0, so shift it
        # to the plate's median contact Z (if any) to keep plate-frame mm.
        on_plate = np.abs(G[:, 0, :]).sum(axis=1) < 1e-3 if n_b else np.ones(n_t, bool)
        zp = np.isfinite(zc) & ~no_contact & on_plate
        datum = float(np.median(zc[zp])) if zp.any() else 0.0
        sol[:, 0] += datum
        # Comparison against contact Z (not used above): per feature, the
        # robot-measured excess over the plate's robust contact plane on tiles
        # that sit mostly on that feature's crown, vs the solved excess there.
        z_check = []
        zres = zc - (np.column_stack([rc_arr, np.ones(n_t)]) @ plane) if plane is not None else None
        for jb, bd in enumerate(bead_list):
            crown = (G[:, 0, jb] > 0.5) & np.isfinite(zc) & ~no_contact
            row = dict(id=bd['id'], h_mm=float(h_bead[jb]), n_crown_tiles=int(crown.sum()))
            if crown.any():
                row['solved_crown_tile_excess_mm'] = float(np.median(sol[crown, 0] - datum))
                if zres is not None:
                    row['contact_z_crown_tile_excess_mm'] = float(np.median(zres[crown]))
            z_check.append(row)
            print(f"  bead #{bd['id']}: dome h = {h_bead[jb]:+.3f} mm" + (
                f"; crown tiles solved {row['solved_crown_tile_excess_mm']:+.3f} mm vs contact-Z "
                f"{row.get('contact_z_crown_tile_excess_mm', float('nan')):+.3f} mm above plate"
                if crown.any() else ""))
        bead_summary = dict(
            sigma_mm=bead_prior_sigma_mm, tilt_sigma=tilt_prior_sigma,
            thresh_mad=bead_thresh_mad, min_area_mm2=bead_min_area_mm2, open_mm=bead_open_mm,
            work_mm=beads_found['work_mm'], display_datum_mm=datum,
            features=[dict(b, h_mm=float(h_bead[i])) for i, b in enumerate(bead_list)],
            z_check=z_check,
        )

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

    # ---- composite depth (feathered in overlaps; RGB mosaic built above) ---
    plane_tile = lambda c, a, b: c + a * yy[:, None] + b * xx[None, :]  # noqa: E731

    def composite(v):
        acc = np.zeros((H, W), np.float64)
        wsum = np.zeros((H, W), np.float64)
        tsum = np.zeros((H, W), np.float64) if trust else None
        for k in ids:
            if k in free_tiles:
                continue
            r0, c0 = origin[k]
            z = relief[k] + plane_tile(*v[pos[k]])
            m = np.isfinite(z)
            w = feather * m
            if edge_keep is not None:  # trimmed border only fills where nothing else does
                w = w * np.where(edge_keep, 1.0, 1e-3)
            if contact:  # likewise gel that never touched the part
                w = w * np.where(contact[k], 1.0, 1e-3)
            if trust:  # trusted readings outweigh distrusted ones 1000:1
                w = w * np.where(trust[k], 1.0, 1e-3)
                tsum[r0:r0 + th, c0:c0 + tw] += feather * trust[k]
            acc[r0:r0 + th, c0:c0 + tw] += np.where(m, z, 0.0) * w
            wsum[r0:r0 + th, c0:c0 + tw] += w
        with np.errstate(invalid='ignore', divide='ignore'):
            out = np.where(wsum > 0, acc / wsum, np.nan).astype(np.float32)
        if trust:
            out = _inpaint_untrusted(out, (tsum > 0) & np.isfinite(out))
        return out

    depth_solved = composite(sol)
    if bead_prior and bead_height_from_z and bead_summary is not None:
        lab_w, work = beads_found['labels'], beads_found['work_mm']
        for jb, (bd, row) in enumerate(zip(bead_list, bead_summary['z_check'])):
            target = row.get('contact_z_crown_tile_excess_mm')
            if target is None or row['n_crown_tiles'] < 3:
                continue
            delta = target - float(h_bead[jb])
            prof = _bead_profile(lab_w, bd['id'], work)[0]
            # work grid -> depth grid (both start at the mosaic origin)
            up = cv2.resize(prof, (int(round(prof.shape[1] * work / mm_per_px)),
                                   int(round(prof.shape[0] * work / mm_per_px))), interpolation=cv2.INTER_LINEAR)
            hh, ww = min(H, up.shape[0]), min(W, up.shape[1])
            depth_solved[:hh, :ww] += delta * up[:hh, :ww]
            row['dome_rescaled_to_mm'] = target
            print(f"  bead #{bd['id']}: dome rescaled {h_bead[jb]:+.3f} -> {target:+.3f} mm "
                  f"(contact-Z median over {row['n_crown_tiles']} crown tiles)")
    depth_naive = composite(naive) if write_images else None

    # ---- outputs -----------------------------------------------------------
    tag = output_tag or ("beadprior" if bead_prior else "solved")
    dest = Path(out_dir) if out_dir is not None else session_dir
    dest.mkdir(parents=True, exist_ok=True)
    out = lambda name: dest / f"gelsight_{tag}_{name}"  # noqa: E731
    np.save(out("depth.npy"), depth_solved)
    if write_images:
        cv2.imwrite(str(out("rgb.png")), rgb_mosaic)
    if bead_prior:
        # Detected features: outline + unit dome over the work-res RGB mosaic.
        lab_w = beads_found['labels']
        small = cv2.resize(rgb_mosaic, (lab_w.shape[1], lab_w.shape[0]), interpolation=cv2.INTER_AREA)
        dome = np.zeros(lab_w.shape, np.float32)
        for bd in bead_list:
            dome = np.maximum(dome, _bead_profile(lab_w, bd['id'], beads_found['work_mm'])[0])
        heat = cv2.applyColorMap((dome * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        vis = np.where(lab_w[..., None] > 0, cv2.addWeighted(small, 0.4, heat, 0.6, 0.0), small)
        cnts, _ = cv2.findContours((lab_w > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 1)
        cv2.imwrite(str(out("beads.png")), vis)

    lo = hi = float('nan')
    n_pts = 0
    if write_images:
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
        'steep_slope': steep_slope, 'steep_dilate_mm': steep_dilate_mm,
        'edge_trim': edge_trim, 'huber': huber, 'gel_depth_sign': gel_depth_sign,
        'contact_mask': contact_mask, 'contact_z_is_peak': contact_z_is_peak,
        'bead_height_from_z': bead_height_from_z,
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
                'prior_used': bool(prior_ok[pos[k]]) and not bead_prior,
            } for k in ids
        },
    }
    if bead_summary is not None:
        summary['bead_prior'] = bead_summary
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
          f"rgb_depth_overlay, rgb_shaded, comparison{', beads' if bead_prior else ''} (.png)")
    print(f"  cloud  -> {out('cloud.ply')} ({n_pts} pts)")
    return summary


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=stitch_session_solved.__doc__.split("\n")[0])
    ap.add_argument("session_dir")
    # Grid size / overlap default to the run's session.json
    # (motion_collect_overlap), so partial runs stitch with just the path.
    ap.add_argument("--touch-x", type=int)
    ap.add_argument("--touch-y", type=int)
    ap.add_argument("--touch-overlap", type=float)
    ap.add_argument("--sign", type=int, default=1, choices=(-1, 1))
    ap.add_argument("--seam-sigma-mm", type=float, default=0.05)
    ap.add_argument("--prior-sigma-mm", type=float, default=1.0)
    ap.add_argument("--bead-prior", action="store_true",
                    help="Z-blind solve: flat plate + one dome per feature found in the RGB "
                         "mosaic, instead of the contact-Z priors (writes gelsight_beadprior_*)")
    ap.add_argument("--bead-prior-sigma-mm", type=float, default=0.5)
    ap.add_argument("--bead-thresh-mad", type=float, default=6.0)
    ap.add_argument("--bead-min-area-mm2", type=float, default=10.0)
    ap.add_argument("--bead-open-mm", type=float, default=0.75)
    ap.add_argument("--tag", default=None, help="output prefix gelsight_<tag>_* (default solved / beadprior)")
    args = ap.parse_args()
    extra = {}
    meta_path = Path(args.session_dir) / "session.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        args.touch_x = args.touch_x or meta.get("touch_x")
        args.touch_y = args.touch_y or meta.get("touch_y")
        rep = meta.get("reprocessed")
        if rep:  # reprocess_overlap_scan output: use its tile geometry and sign
            ov = [o for o in rep.get("true_overlap", []) if o is not None]
            if args.touch_overlap is None and ov:
                args.touch_overlap = min(ov)
            extra = dict(gelsight_full_width_m=rep["frame_width_m"],
                         border_fraction=(1.0 - rep["keep_fraction"]) / 2.0, outlier_mm=3.0)
            if not any(a.startswith("--sign") for a in sys.argv[1:]):
                args.sign = rep.get("gel_depth_sign", args.sign)
        if args.touch_overlap is None:
            args.touch_overlap = meta.get("overlap", 0.0)
    if not (args.touch_x and args.touch_y):
        ap.error("--touch-x/--touch-y required (no session.json with a grid)")
    stitch_session_solved(
        args.session_dir, args.touch_x, args.touch_y,
        touch_overlap=args.touch_overlap or 0.0, gel_depth_sign=args.sign,
        seam_sigma_mm=args.seam_sigma_mm, prior_sigma_mm=args.prior_sigma_mm,
        bead_prior=args.bead_prior, bead_prior_sigma_mm=args.bead_prior_sigma_mm,
        bead_thresh_mad=args.bead_thresh_mad, bead_min_area_mm2=args.bead_min_area_mm2,
        bead_open_mm=args.bead_open_mm, output_tag=args.tag, **extra,
    )
