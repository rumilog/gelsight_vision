"""Mappings between capture_index, (tile_row, tile_col), and plate-frame (x, y).

The 10x10 touch grid spans ~100mm x 75mm of the plate. Origin (0, 0) in the
plate frame is the center of cell (row=5, col=5); the grid is asymmetric --
5 cells on the +x / +y side, 4 cells on the -x / -y side, because
`touch_x // 2 = 5` for a touch_x = 10 grid.

Per-tile step is 10mm x 7.5mm, the gelsight sensor's lateral FOV after the
border crop (`keep_fraction = 0.5` against a 20mm x 15mm full-sensor FOV).

These conventions are the ones encoded in motion_collect_sample.py and stored
as `robot_x_m` / `robot_y_m` columns in the Gelsight-Welds HF dataset, so they
must stay in sync with both.
"""

from __future__ import annotations

GRID: tuple[int, int] = (10, 10)
STEP_X_M: float = 0.010
STEP_Y_M: float = 0.0075


def capture_index_to_tile(capture_index: int) -> tuple[int, int]:
    """1-indexed capture_index -> (tile_row, tile_col), both 0-indexed."""
    n = GRID[0] * GRID[1]
    if not 1 <= capture_index <= n:
        raise ValueError(f"capture_index must be 1..{n}, got {capture_index}")
    return divmod(capture_index - 1, GRID[1])


def tile_to_capture_index(tile_row: int, tile_col: int) -> int:
    nr, nc = GRID
    if not (0 <= tile_row < nr and 0 <= tile_col < nc):
        raise ValueError(f"tile ({tile_row}, {tile_col}) out of bounds for grid {GRID}")
    return tile_row * nc + tile_col + 1


def capture_index_to_xy(capture_index: int) -> tuple[float, float]:
    """1-indexed capture_index -> (robot_x, robot_y) in plate-frame meters."""
    tr, tc = capture_index_to_tile(capture_index)
    nr, nc = GRID
    x = (nr // 2 - tr) * STEP_X_M
    y = (nc // 2 - tc) * STEP_Y_M
    return x, y
