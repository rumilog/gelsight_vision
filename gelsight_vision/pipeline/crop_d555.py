"""Crop the full D555 overhead frame to the `rgb_stitched` portrait region.

The Gelsight-Welds dataset's `rgb_stitched` panorama is a fixed-pixel crop of
the D555 overview taken when the robot is at the `camera_focus` pose. Default
constants reproduce scripts/v2/01_crop_d555.py in the Gelsight-Welds repo;
re-tune CENTER_*_PX / CELL_*_PX (or pass them as kwargs) if the camera or
plate drift between collection batches.

Output for the default 10x10 grid at 33x44 px/cell is 440x330 (HxW) BGR.
"""

from __future__ import annotations

import numpy as np


GRID: tuple[int, int] = (10, 10)
CENTER_X_PX: int = 565
CENTER_Y_PX: int = 360
CELL_W_PX: int = 33
CELL_H_PX: int = 44


def grid_top_left(
    *,
    center_x: int = CENTER_X_PX,
    center_y: int = CENTER_Y_PX,
    cell_w: int = CELL_W_PX,
    cell_h: int = CELL_H_PX,
    grid: tuple[int, int] = GRID,
) -> tuple[int, int]:
    """Pixel of the grid's top-left corner.

    The cell whose center sits at robot (0, 0) is at
    (row = grid_rows - grid_rows // 2, col = grid_cols - grid_cols // 2) -- (5, 5)
    for a 10x10 grid. CENTER_*_PX names that cell's pixel center; the grid's
    top-left is half a cell up-left of it.
    """
    nr, nc = grid
    col_center = nc - nc // 2
    row_center = nr - nr // 2
    left = center_x - col_center * cell_w - cell_w / 2
    top = center_y - row_center * cell_h - cell_h / 2
    return int(round(left)), int(round(top))


def crop_rgb_stitched(
    d555_frame: np.ndarray,
    *,
    center_x: int = CENTER_X_PX,
    center_y: int = CENTER_Y_PX,
    cell_w: int = CELL_W_PX,
    cell_h: int = CELL_H_PX,
    grid: tuple[int, int] = GRID,
) -> np.ndarray:
    """Crop a full D555 frame down to the rgb_stitched portrait panorama.

    Args:
        d555_frame: Full BGR (or RGB) frame from the D555, e.g. 720x1280x3.
        center_x, center_y, cell_w, cell_h, grid: Crop parameters; defaults
            match the Gelsight-Welds v2 dataset build.

    Returns:
        Cropped slice with shape (grid_rows * cell_h, grid_cols * cell_w, C).
        Same dtype and channel order as the input.
    """
    nr, nc = grid
    left, top = grid_top_left(
        center_x=center_x, center_y=center_y,
        cell_w=cell_w, cell_h=cell_h, grid=grid,
    )
    grid_h = nr * cell_h
    grid_w = nc * cell_w
    return d555_frame[top:top + grid_h, left:left + grid_w]


def tile_bounds(
    capture_index: int,
    *,
    center_x: int = CENTER_X_PX,
    center_y: int = CENTER_Y_PX,
    cell_w: int = CELL_W_PX,
    cell_h: int = CELL_H_PX,
    grid: tuple[int, int] = GRID,
) -> tuple[int, int, int, int]:
    """Pixel bounds (x0, y0, x1, y1) of one tile in the full D555 frame."""
    nr, nc = grid
    if not 1 <= capture_index <= nr * nc:
        raise ValueError(f"capture_index must be 1..{nr * nc}, got {capture_index}")
    k = capture_index - 1
    row, col = divmod(k, nc)
    left, top = grid_top_left(
        center_x=center_x, center_y=center_y,
        cell_w=cell_w, cell_h=cell_h, grid=grid,
    )
    x0 = left + col * cell_w
    y0 = top + row * cell_h
    return x0, y0, x0 + cell_w, y0 + cell_h


def crop_tile(
    d555_frame: np.ndarray,
    capture_index: int,
    *,
    rotate: bool = True,
    **kwargs,
) -> np.ndarray:
    """Slice one tile from the D555 frame, matching the dataset's `rgb_image`.

    The dataset rotates each tile 90 degrees clockwise after slicing so the
    long axis is horizontal (44 wide x 33 tall in landscape). Pass
    `rotate=False` to keep the raw portrait slice.
    """
    x0, y0, x1, y1 = tile_bounds(capture_index, **kwargs)
    tile = d555_frame[y0:y1, x0:x1]
    if rotate:
        import cv2
        tile = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
    return tile
