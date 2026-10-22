"""Convert a GelSight Mini image into a surface-height tensor.

Wraps gsrobotics' Reconstruction3D so each pixel of the output corresponds to
gel-surface displacement in z (in the raw units produced by Poisson integration
of the predicted normals; per the gsrobotics docs the lateral scale is
~0.0634 mm/pixel at 320x240).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch

from ..gsrobotics.utilities.image_processing import crop_and_resize
from ..gsrobotics.utilities.reconstruction import Reconstruction3D

DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_BORDER_FRACTION = 0.25
DEFAULT_MARKER_THRESHOLD = (0, 70)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "gsrobotics" / "models" / "nnmini.pt"

ImageLike = Union[np.ndarray, str, Path]


class HeightMapper:
    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        border_fraction: float = DEFAULT_BORDER_FRACTION,
        marker_threshold: tuple[int, int] = DEFAULT_MARKER_THRESHOLD,
        use_gpu: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.border_fraction = border_fraction
        self.marker_threshold = marker_threshold
        self.reconstruction = Reconstruction3D(
            image_width=width, image_height=height, use_gpu=use_gpu
        )
        if self.reconstruction.load_nn(str(model_path)) is None:
            raise RuntimeError(f"Failed to load NN model from {model_path}")
        # Bypass the 50-frame streaming auto-baseline; callers manage baseline explicitly.
        self.reconstruction.depth_map_zero_counter = 100

    def calibrate(self, baseline_image: ImageLike) -> None:
        """Set the zero baseline from an untouched-gel image. Subsequent
        height maps are returned relative to this baseline."""
        rgb = self._prepare(baseline_image)
        self.reconstruction.depth_map_zero = np.zeros((self.height, self.width))
        depth, *_ = self.reconstruction.get_depthmap(
            image=rgb, markers_threshold=self.marker_threshold
        )
        self.reconstruction.depth_map_zero = depth

    def image_to_height_tensor(self, image: ImageLike) -> torch.Tensor:
        """Return a (H, W) float tensor of gel-surface height for one image."""
        rgb = self._prepare(image)
        depth, *_ = self.reconstruction.get_depthmap(
            image=rgb, markers_threshold=self.marker_threshold
        )
        return torch.from_numpy(depth).float()

    def crop_native(self, image: np.ndarray) -> np.ndarray:
        """Apply the same border crop the depth pipeline uses, but at the
        image's native resolution (no resize). Use this when saving the
        captured BGR frame so it covers the same physical area as the
        computed depth map."""
        return crop_and_resize(
            image=image,
            target_size=None,
            border_fraction=self.border_fraction,
        )

    @property
    def keep_fraction(self) -> float:
        """Fraction of each axis that survives the border crop. Multiply
        a full-sensor FOV (in mm) by this to get the cropped FOV."""
        return 1.0 - 2.0 * self.border_fraction

    def _prepare(self, image: ImageLike) -> np.ndarray:
        if isinstance(image, (str, Path)):
            loaded = cv2.imread(str(image))
            if loaded is None:
                raise FileNotFoundError(f"Could not read image: {image}")
            image = loaded
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return crop_and_resize(
            image=rgb,
            target_size=(self.width, self.height),
            border_fraction=self.border_fraction,
        )


_default_mapper: Optional[HeightMapper] = None


def image_to_height_tensor(
    image: ImageLike,
    mapper: Optional[HeightMapper] = None,
) -> torch.Tensor:
    """One-shot convenience: load a BGR image (path or array) and return a
    (H, W) height tensor. Reuses a module-level HeightMapper across calls so
    the NN is loaded only once."""
    global _default_mapper
    if mapper is None:
        if _default_mapper is None:
            _default_mapper = HeightMapper()
        mapper = _default_mapper
    return mapper.image_to_height_tensor(image)


PX_TO_MM = 0.0634  # gsrobotics-documented lateral scale for the 320x240 image
# the depth pipeline operates on (after the 15% border crop + resize).


def save_height_preview(
    height: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
    colormap: int = cv2.COLORMAP_VIRIDIS,
    percentile_clip: tuple[float, float] = (1.0, 99.0),
    num_ticks: int = 5,
    unit: str = "mm",
    scale_factor: float = PX_TO_MM,
    value_range: Optional[tuple[float, float]] = None,
) -> None:
    """Save a colormapped PNG preview of a (H, W) height array with a vertical
    scale bar on the right. Tick values are displayed in `unit`, computed as
    raw_height * scale_factor. The default scale_factor (PX_TO_MM = 0.0634)
    matches the gsrobotics lateral pixel size at 320x240; the depth output of
    poisson_dct_neumann is in the same pixel units, so this gives an
    approximate mm reading. The saved .npy keeps the raw values; only the
    preview labels are scaled.

    `colormap` may be an OpenCV colormap id or a (256, 1, 3) uint8 LUT.
    `value_range` (lo, hi) overrides the percentile clip. NaN pixels are
    drawn dark gray."""
    if isinstance(height, torch.Tensor):
        height = height.detach().cpu().numpy()
    nan_mask = ~np.isfinite(height)
    if value_range is not None:
        lo, hi = value_range
    else:
        lo, hi = np.nanpercentile(height, percentile_clip)
    clipped = np.clip(np.where(nan_mask, lo, height), lo, hi)
    if hi - lo < 1e-9:
        normalized = np.zeros_like(clipped, dtype=np.uint8)
    else:
        normalized = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, colormap)
    preview[nan_mask] = (40, 40, 40)

    h, w = preview.shape[:2]
    gap = 10
    bar_w = 20
    label_w = 90
    tick_len = 4
    header_h = 22 if unit else 0

    gradient = np.linspace(255, 0, h, dtype=np.uint8).reshape(-1, 1)
    colorbar = cv2.applyColorMap(np.repeat(gradient, bar_w, axis=1), colormap)

    canvas_h = h + header_h
    canvas_w = w + gap + bar_w + tick_len + label_w
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    canvas[header_h:header_h + h, :w] = preview
    bar_x = w + gap
    canvas[header_h:header_h + h, bar_x:bar_x + bar_w] = colorbar
    cv2.rectangle(
        canvas, (bar_x, header_h), (bar_x + bar_w - 1, header_h + h - 1),
        (0, 0, 0), 1,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0, 0, 0)

    if unit:
        header = f"[{unit}]"
        (text_w, text_h), _ = cv2.getTextSize(header, font, font_scale, 1)
        text_x = bar_x + (bar_w - text_w) // 2
        cv2.putText(
            canvas, header, (text_x, header_h - 6),
            font, font_scale, color, 1, cv2.LINE_AA,
        )

    for i in range(num_ticks):
        frac = i / (num_ticks - 1) if num_ticks > 1 else 0.0
        y = header_h + int(round(frac * (h - 1)))
        value = (hi - frac * (hi - lo)) * scale_factor
        cv2.line(
            canvas, (bar_x + bar_w, y), (bar_x + bar_w + tick_len, y), color, 1,
        )
        text = f"{value:.3f}"
        (_, text_h), _ = cv2.getTextSize(text, font, font_scale, 1)
        text_y = max(header_h + text_h, min(header_h + h - 1, y + text_h // 2))
        cv2.putText(
            canvas, text, (bar_x + bar_w + tick_len + 2, text_y),
            font, font_scale, color, 1, cv2.LINE_AA,
        )

    cv2.imwrite(str(path), canvas)
