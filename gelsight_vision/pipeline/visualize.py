"""Visualization helpers for the agentic pipeline."""

from __future__ import annotations

import cv2
import numpy as np


GRID: tuple[int, int] = (10, 10)


def _score_color_bgr(score: float) -> tuple[int, int, int]:
    """Green (low) -> yellow -> red (high). Returns a BGR tuple for cv2."""
    s = float(np.clip(score, 0.0, 1.0))
    if s < 0.5:
        t = s / 0.5
        return (0, 255, int(255 * t))
    t = (s - 0.5) / 0.5
    return (0, int(255 * (1 - t)), 255)


def draw_weld_patch_overlay(
    rgb_stitched_bgr: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    *,
    grid: tuple[int, int] = GRID,
    grid_color: tuple[int, int, int] = (180, 180, 180),
    label_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Return a copy of `rgb_stitched_bgr` with a `grid` overlay drawn on top,
    every tile whose `scores[r, c] >= threshold` outlined in a score-colored
    box (green -> yellow -> red), and the capture_index labeled in the corner.
    """
    canvas = rgb_stitched_bgr.copy()
    H, W = canvas.shape[:2]
    nr, nc = grid
    if scores.shape != grid:
        raise ValueError(f"scores shape {scores.shape} doesn't match grid {grid}")
    tile_h, tile_w = H // nr, W // nc

    for c in range(nc + 1):
        x = min(c * tile_w, W - 1)
        cv2.line(canvas, (x, 0), (x, nr * tile_h - 1), grid_color, 1)
    for r in range(nr + 1):
        y = min(r * tile_h, H - 1)
        cv2.line(canvas, (0, y), (nc * tile_w - 1, y), grid_color, 1)

    for r in range(nr):
        for c in range(nc):
            score = float(scores[r, c])
            if score < threshold:
                continue
            color = _score_color_bgr(score)
            x0, y0 = c * tile_w, r * tile_h
            x1, y1 = x0 + tile_w - 1, y0 + tile_h - 1
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            label = str(r * nc + c + 1)
            org = (x0 + 3, y0 + 13)
            cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, label_color, 1, cv2.LINE_AA)

    return canvas


def show(image_bgr: np.ndarray, title: str = "preview", scale: int = 2) -> None:
    """Display `image_bgr` in a window; block until any key is pressed.

    `scale` upsamples with nearest-neighbor purely for visibility -- the saved
    overlay PNG stays at native resolution.
    """
    img = image_bgr
    if scale != 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)
