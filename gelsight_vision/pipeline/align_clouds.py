"""Align the D555 stereo PLY (camera optical frame) with the anchored
gelsight PLY (world frame) and write a combined PLY.

Approach: use the stereo capture's `camera_focus` world pose as the proxy for
where the D555 optical origin sits in world. This works cleanly when the URDF
defines `camera_focus` at (or very near) the D555 depth camera's optical
origin — the common convention. If the two frames don't coincide there'll be
a fixed rigid offset visible in the combined PLY that a static
tool0->camera_link transform would remove.

Optical->ROS rotation is the standard camera convention:
    optical: X right, Y down, Z forward
    ROS   : X forward, Y left, Z up
so R_opt_to_ros = [[0,0,1],[-1,0,0],[0,-1,0]].
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

import numpy as np


_VERT_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
])

# Fixed rotation from optical frame to camera_focus frame. Since camera_focus's
# world quaternion is just a yaw (~90°) but the camera is physically looking
# straight down at the plate, there must be a URDF-baked rotation between
# camera_focus and the optical frame that encodes "looking down."
#
# For a down-looking camera:
#   optical Z (forward, out of lens) -> world -Z (down)
#   optical X (right in image)       -> world  X  (with yaw applied)
#   optical Y (down in image)        -> world -Y  (with yaw applied)
# giving a 180° rotation about the X axis: flip Y and Z.
#
# If the alignment is X/Y-swapped or mirrored after the first run, adjust
# this matrix (there are 24 orthogonal orientations to try). Long-term fix:
# publish a `tool0 -> gelsight_vision_d555_link` static transform so
# `world -> infra1_optical` resolves through TF directly.
R_OPT_TO_CAMFOCUS = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=np.float64)


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - w * z),     2 * (x * z + w * y)],
        [    2 * (x * y + w * z), 1 - 2 * (x * x + z * z),     2 * (y * z - w * x)],
        [    2 * (x * z - w * y),     2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _read_binary_ply(path: Path) -> np.ndarray:
    """Parse a little-endian PLY written by our writers (x/y/z float + r/g/b uchar)."""
    with open(path, 'rb') as f:
        header_bytes = b''
        while True:
            line = f.readline()
            header_bytes += line
            if line.strip() == b'end_header':
                break
        header = header_bytes.decode('ascii')
        n = None
        for line in header.splitlines():
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
                break
        if n is None:
            raise ValueError(f"couldn't parse vertex count in {path}")
        raw = f.read(n * _VERT_DTYPE.itemsize)
    return np.frombuffer(raw, dtype=_VERT_DTYPE, count=n)


def _write_binary_ply(path: Path, verts: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(verts.tobytes())


def transform_stereo_to_world(
    session_dir: Union[str, Path],
    subsample_stride: int = 1,
    color_tint: tuple[int, int, int] | None = None,
) -> Path | None:
    """Read `stereo_0001_cloud.ply` (camera optical frame) + `stereo_0001_pose.json`
    (camera_focus in world) and write `stereo_0001_cloud_world.ply` transformed
    into world coords.

    `subsample_stride`: keep 1 of every N points (memory guard — the stereo
    cloud is ~3M points at 2x upsample). Default 1 keeps all points.
    `color_tint`: if set, replace point colors with this RGB (useful for
    distinguishing stereo vs gelsight in the combined viz).
    Returns the output path, or None if inputs are missing.
    """
    session_dir = Path(session_dir)
    ply_path = session_dir / "stereo_0001_cloud.ply"
    pose_path = session_dir / "stereo_0001_pose.json"
    if not ply_path.exists() or not pose_path.exists():
        print(f"stereo-to-world: missing {ply_path.name} or {pose_path.name}")
        return None

    verts = _read_binary_ply(ply_path)
    if subsample_stride > 1:
        verts = verts[::subsample_stride]
    pts_opt = np.stack([verts['x'], verts['y'], verts['z']], axis=1).astype(np.float64)

    # Convert optical -> camera_focus body convention.
    pts_ros = pts_opt @ R_OPT_TO_CAMFOCUS.T

    with open(pose_path) as f:
        pose = json.load(f)
    t = np.array([
        pose['translation']['x'],
        pose['translation']['y'],
        pose['translation']['z'],
    ], dtype=np.float64)
    q = pose['rotation']
    R_world_cam = _quat_to_matrix(q['x'], q['y'], q['z'], q['w'])

    pts_world = pts_ros @ R_world_cam.T + t

    out = np.empty(len(verts), dtype=_VERT_DTYPE)
    out['x'] = pts_world[:, 0].astype(np.float32)
    out['y'] = pts_world[:, 1].astype(np.float32)
    out['z'] = pts_world[:, 2].astype(np.float32)
    if color_tint is None:
        out['r'] = verts['r']
        out['g'] = verts['g']
        out['b'] = verts['b']
    else:
        out['r'], out['g'], out['b'] = color_tint

    out_path = session_dir / "stereo_0001_cloud_world.ply"
    _write_binary_ply(out_path, out)
    print(
        f"stereo -> world: {out_path.name} ({len(out)} pts, "
        f"z range [{pts_world[:, 2].min():.3f}, {pts_world[:, 2].max():.3f}] m)"
    )
    return out_path


def write_combined_ply(
    session_dir: Union[str, Path],
    stereo_stride: int = 8,
    tint_stereo: bool = True,
) -> Path | None:
    """Concatenate the anchored gelsight PLY and the world-frame stereo PLY
    into `combined_cloud.ply` (both in world coords).

    `stereo_stride`: subsample the stereo cloud so it doesn't drown out
    gelsight detail in viz (default 8x = keep ~460k of 3.7M stereo points).
    `tint_stereo`: color stereo points blue so they visually separate from
    the gelsight (colored by IR/gel imagery) in Foxglove.
    """
    session_dir = Path(session_dir)
    stereo_world = transform_stereo_to_world(
        session_dir,
        subsample_stride=stereo_stride,
        color_tint=(80, 120, 220) if tint_stereo else None,
    )
    if stereo_world is None:
        return None

    gel_path = session_dir / "gelsight_stitched_cloud_anchored.ply"
    if not gel_path.exists():
        print(f"combined: missing {gel_path.name} — skipping")
        return None

    stereo_verts = _read_binary_ply(stereo_world)
    gel_verts = _read_binary_ply(gel_path)

    combined = np.concatenate([stereo_verts, gel_verts])
    out_path = session_dir / "combined_cloud.ply"
    _write_binary_ply(out_path, combined)

    # Report the Z overlap between the two clouds. If they align, stereo Z
    # (world) at the plate area should match anchored gelsight Z (world).
    sz = np.array([stereo_verts['x'], stereo_verts['y'], stereo_verts['z']]).T
    gz = np.array([gel_verts['x'], gel_verts['y'], gel_verts['z']]).T
    print(
        f"combined -> {out_path} "
        f"({len(stereo_verts)} stereo + {len(gel_verts)} gel = {len(combined)} pts)"
    )
    print(f"  stereo world Z: [{sz[:, 2].min():.4f}, {sz[:, 2].max():.4f}] m, "
          f"median {np.median(sz[:, 2]):.4f}")
    print(f"  gel    world Z: [{gz[:, 2].min():.4f}, {gz[:, 2].max():.4f}] m, "
          f"median {np.median(gz[:, 2]):.4f}")
    print(f"  Z median offset (stereo - gel): "
          f"{(np.median(sz[:, 2]) - np.median(gz[:, 2])) * 1000:+.2f} mm "
          f"(that's the fixed camera_focus <-> optical origin gap to fix)")
    # XY overlap check: does the gelsight rectangle sit inside the stereo FOV?
    print(
        f"  stereo XY: x [{sz[:, 0].min():.3f}, {sz[:, 0].max():.3f}] "
        f"y [{sz[:, 1].min():.3f}, {sz[:, 1].max():.3f}]"
    )
    print(
        f"  gel    XY: x [{gz[:, 0].min():.3f}, {gz[:, 0].max():.3f}] "
        f"y [{gz[:, 1].min():.3f}, {gz[:, 1].max():.3f}]"
    )
    return out_path


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("usage: python -m gelsight_vision.pipeline.align_clouds <session_dir>")
        sys.exit(1)
    write_combined_ply(sys.argv[1])
