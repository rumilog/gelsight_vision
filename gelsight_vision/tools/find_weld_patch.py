"""Tool 1: find weld-bearing patches in the D555 overhead.

CNN-based scorer + threshold filter. The CNN is `WeldPatchCNN` (small 3-block
conv), trained against the Gelsight-Welds dataset's per-tile `rgb_image`
column (44x33 landscape RGB tiles). Sigmoid output -> probability of weld
relief in [0, 1].

`score_patches` does the heavy work (model load + per-tile inference);
`find_weld_patch` is a pure threshold over those scores. Keep them separate
so the visualization layer can color tiles by score without re-running the
CNN.

The model bundle is cached on first call so repeated invocations are cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from gelsight_vision.pipeline.crop_d555 import crop_tile
from gelsight_vision.tools.weld_patch_cnn import (
    DEFAULT_MODEL_PATH,
    ModelBundle,
    load_model,
    preprocess,
)


GRID: tuple[int, int] = (10, 10)


_BUNDLE_CACHE: dict[str, ModelBundle] = {}


def _get_bundle(model_path: Path, device: Optional[torch.device]) -> ModelBundle:
    key = f"{model_path}|{device}"
    if key not in _BUNDLE_CACHE:
        _BUNDLE_CACHE[key] = load_model(model_path, device=device)
    return _BUNDLE_CACHE[key]


def score_patches(
    d555_frame_bgr: np.ndarray,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    device: Optional[torch.device] = None,
    grid: tuple[int, int] = GRID,
    **crop_kwargs,
) -> np.ndarray:
    """Score every patch in `grid` with the CNN. Returns an (nr, nc) float32
    array in [0, 1] -- the predicted probability of weld relief per tile.

    Tiles are sliced and rotated to match the dataset's `rgb_image` convention
    (44x33 landscape) before being passed to the CNN.
    """
    bundle = _get_bundle(model_path, device)
    nr, nc = grid

    tensors: list[torch.Tensor] = []
    for ci in range(1, nr * nc + 1):
        tile_bgr = crop_tile(d555_frame_bgr, ci, rotate=True, grid=grid, **crop_kwargs)
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        tensors.append(preprocess(Image.fromarray(tile_rgb), bundle.input_size))

    batch = torch.stack(tensors).to(bundle.device)
    with torch.no_grad():
        logits = bundle.model(batch)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    return probs.reshape(nr, nc)


def find_weld_patch(
    scores: np.ndarray,
    threshold: float = 0.5,
) -> list[int]:
    """Return sorted 1-indexed capture_indexes whose CNN score >= threshold."""
    nr, nc = scores.shape
    selected: list[int] = []
    for r in range(nr):
        for c in range(nc):
            if float(scores[r, c]) >= threshold:
                selected.append(r * nc + c + 1)
    return selected
