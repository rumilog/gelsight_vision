"""Small CNN that scores RGB tiles for weld presence.

Ported verbatim from agentic_sr's tools/welds/models/weld_patch_cnn.py so the
checkpoint trained over there loads here without surgery. Input expected by
`preprocess` is a 44x33 landscape RGB tile (PIL Image), matching the
Gelsight-Welds dataset's `rgb_image` column.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


DEFAULT_MODEL_PATH: Path = Path(__file__).parent / "models" / "weld_patch_classifier.pt"
INPUT_SIZE: int = 64
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class WeldPatchCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


@dataclass(frozen=True)
class ModelBundle:
    model: WeldPatchCNN
    device: torch.device
    input_size: int = INPUT_SIZE


def preprocess(rgb_image: Image.Image, size: int = INPUT_SIZE) -> torch.Tensor:
    """Resize + ImageNet-normalize a PIL RGB image to a (3, size, size) tensor.

    Matches torchvision's Resize(size) + ToTensor() + Normalize(ImageNet) but
    avoids the torchvision dependency. BILINEAR matches torchvision's default
    interpolation for PIL inputs, which is what the checkpoint was trained on.
    """
    img = rgb_image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_model(
    path: Path | None = None,
    device: torch.device | None = None,
) -> ModelBundle:
    model_path = path or DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = WeldPatchCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return ModelBundle(
        model=model,
        device=device,
        input_size=int(checkpoint.get("input_size", INPUT_SIZE)),
    )
