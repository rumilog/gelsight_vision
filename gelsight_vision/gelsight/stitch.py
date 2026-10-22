"""Stitch the per-touch GelSight captures of a session into one grid view."""

from __future__ import annotations

import json
import re
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
