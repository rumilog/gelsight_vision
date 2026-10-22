"""Tool 3: Phase-2 cross-modal RGB->depth inpainting.

Given the stitched D555 RGB and per-patch gelsight depth from touched tiles,
returns a dense depth panorama with the rest of the patches inpainted by the
trained Palette diffusion model.

Lazy + module-cached: first call loads weights, beta schedule, and (if
`use_compile=True`) torch.compile()s the denoise net. Subsequent calls reuse
the same Network instance.

Normalization caveat (carried from agentic_sr):
- Training used per-sample min-max -> [-1, 1] over the full assembled depth.
- At inference we only have observed patches, so we compute the range from
  the *downsampled* observed pixels (which match what training saw).
- Model output is clamped to that observed range when denormalized; if the
  un-observed patches contain depths outside that range, the prediction is
  biased toward the observed bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from gelsight_vision.tools.palette.network import Network


_PKG_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = _PKG_ROOT / "palette_models" / "checkpoint" / "2000_Network.pth"

IMAGE_SIZE: tuple[int, int] = (320, 240)   # (H, W) portrait
GRID: tuple[int, int] = (10, 10)
ROTK: int = 1                              # per-tile rotation to match rgb_stitched orientation

_UNET_ARGS = dict(
    in_channel=4, out_channel=1, inner_channel=64,
    channel_mults=[1, 2, 4, 8], attn_res=[8], num_head_channels=32,
    res_blocks=2, dropout=0.2, image_size=320,
)
_BETA_SCHEDULE = dict(
    train=dict(schedule="linear", n_timestep=2000, linear_start=1e-6, linear_end=0.01),
    test=dict(schedule="linear", n_timestep=1000, linear_start=1e-4, linear_end=0.09),
)
# Use the train schedule at inference: training's val_step ran under it, so it
# is what the weights were tuned against. The test schedule was never validated
# and produced visibly worse samples in agentic_sr's experiments.
_INFERENCE_PHASE = "train"


@dataclass
class ModelHandle:
    net: Network
    device: torch.device
    image_size: tuple[int, int]
    grid: tuple[int, int]


_CACHE: dict[str, ModelHandle] = {}


def _rgb_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB image -> (3, H, W) tensor in [-1, 1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_model(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    *,
    device: Optional[torch.device] = None,
    use_compile: bool = False,
) -> ModelHandle:
    """Load the Phase-2 Network. Cached by (checkpoint, device, compile).

    `use_compile` defaults to False because torch 2.12's inductor backend has
    known codegen issues with this UNet config ("Too few arguments for ... CSE")
    and a successful compile only saves ~30% wall time on a 3060 (~30-90s of
    warmup, then a slightly faster sampling loop). Re-enable opportunistically
    when a newer torch ships the fix.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = f"{checkpoint}|{device}|{use_compile}"
    if key in _CACHE:
        return _CACHE[key]

    net = Network(
        unet=_UNET_ARGS, beta_schedule=_BETA_SCHEDULE,
        module_name="guided_diffusion",
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    # The 6 "unexpected" keys are the noise-schedule buffers
    # (gammas/sqrt_recip_gammas/...); set_new_noise_schedule below overwrites
    # them with the train-phase values, so strict=False is correct.
    net.load_state_dict(state, strict=False)
    net.eval()
    net.set_new_noise_schedule(device=device, phase=_INFERENCE_PHASE)

    if use_compile and device.type == "cuda":
        # Compiles the inner denoise UNet only -- this is the function called
        # `num_timesteps` times per restoration(), so it's where the speedup
        # would live. First call eats ~30-90s of compile time. Default mode
        # (not "reduce-overhead") because CUDA Graphs causes additional
        # incompatibilities on top of the inductor CSE issue.
        try:
            net.denoise_fn = torch.compile(net.denoise_fn)
        except Exception as e:  # noqa: BLE001
            print(f"[super_resolution] torch.compile disabled: {e}")

    handle = ModelHandle(net=net, device=device,
                         image_size=IMAGE_SIZE, grid=GRID)
    _CACHE[key] = handle
    return handle


def _per_tile_to_canvas_position(capture_index: int, grid: tuple[int, int]) -> tuple[int, int]:
    """1-indexed capture_index -> (tile_row, tile_col), 0-indexed."""
    return divmod(int(capture_index) - 1, grid[1])


def super_resolution(
    rgb_stitched: Image.Image | np.ndarray,
    observed_depths: dict[int, np.ndarray],
    *,
    model: Optional[ModelHandle] = None,
    seed: Optional[int] = None,
    autocast: bool = True,
) -> np.ndarray:
    """Phase-2 inpainting from RGB + observed patch depths to dense depth.

    Args:
        rgb_stitched: Full stitched RGB panorama (PIL Image or HxWx3 ndarray).
            Resized to IMAGE_SIZE if necessary.
        observed_depths: {capture_index: (240, 320) float32 depth} from
            touched patches. Patches NOT in the dict are inpainted.
        model: Optional pre-loaded ModelHandle; otherwise loads the default.
        seed: Optional torch RNG seed for reproducibility.
        autocast: Run sampling inside torch.cuda.amp.autocast(fp16). The
            GroupNorm32 layer Palette uses explicitly upcasts internally, so
            fp16 is safe for sampling here.

    Returns:
        (H, W) float32 ndarray -- dense depth panorama in the same absolute
        scale as the input observed_depths (denormalized using observed min/max).
    """
    handle = model or load_model()
    net = handle.net
    device = handle.device
    H, W = handle.image_size
    nr, nc = handle.grid
    tile_h, tile_w = H // nr, W // nc

    rgb = (
        rgb_stitched
        if isinstance(rgb_stitched, Image.Image)
        else Image.fromarray(np.asarray(rgb_stitched))
    )
    rgb = rgb.convert("RGB")
    if rgb.size != (W, H):
        rgb = rgb.resize((W, H), Image.BICUBIC)
    cond = _rgb_to_tensor(rgb).unsqueeze(0).to(device)  # (1, 3, H, W)

    if not observed_depths:
        raise ValueError(
            "observed_depths is empty -- Phase 2 needs at least one observed patch"
        )

    canvas_raw = np.zeros((H, W), dtype=np.float32)
    mask_np = np.ones((H, W), dtype=np.float32)
    for capture_index, depth in observed_depths.items():
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.shape != (240, 320):
            raise ValueError(
                f"expected per-tile depth shape (240, 320), got {depth_arr.shape} "
                f"for ci={capture_index}"
            )
        depth_rot = np.rot90(depth_arr, k=ROTK)
        pil = Image.fromarray(np.ascontiguousarray(depth_rot), mode="F")
        pil = pil.resize((tile_w, tile_h), Image.BILINEAR)
        tile_arr = np.asarray(pil, dtype=np.float32)
        tr, tc = _per_tile_to_canvas_position(capture_index, handle.grid)
        if not (0 <= tr < nr and 0 <= tc < nc):
            raise ValueError(
                f"capture_index {capture_index} out of bounds for grid {handle.grid}"
            )
        canvas_raw[tr * tile_h:(tr + 1) * tile_h,
                   tc * tile_w:(tc + 1) * tile_w] = tile_arr
        mask_np[tr * tile_h:(tr + 1) * tile_h,
                tc * tile_w:(tc + 1) * tile_w] = 0.0

    obs_bool = mask_np < 0.5
    obs_min = float(canvas_raw[obs_bool].min())
    obs_max = float(canvas_raw[obs_bool].max())
    obs_span = max(obs_max - obs_min, 1e-8)

    canvas_norm = np.zeros_like(canvas_raw)
    canvas_norm[obs_bool] = ((canvas_raw[obs_bool] - obs_min) / obs_span) * 2.0 - 1.0

    gt = torch.from_numpy(canvas_norm).unsqueeze(0).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

    if seed is not None:
        torch.manual_seed(int(seed))
    y_t = torch.randn_like(gt)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if autocast and device.type == "cuda"
        else _NullContext()
    )
    with torch.no_grad(), autocast_ctx:
        output, _ = net.restoration(cond, y_t=y_t, y_0=gt, mask=mask, sample_num=1)

    out_norm = output.squeeze().detach().cpu().float().numpy().clip(-1.0, 1.0)
    out_abs = ((out_norm + 1.0) / 2.0) * obs_span + obs_min
    return out_abs.astype(np.float32)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False
