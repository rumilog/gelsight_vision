# Agentic-SR Phase-2 checkpoint bundle

Contents:
  checkpoint/2000_Network.pth      — model weights (use this for inference)
  checkpoint/2000_Network_ema.pth  — EMA weights (DO NOT use; see notes)
  checkpoint/config.json           — Palette config that defines the architecture
  examples/                        — 6× (input, output) PNG pairs

To use:
  1. Clone Agentic-SR + its submodules.
  2. Copy checkpoint/ into:
     githubs/Palette-Image-to-Image-Diffusion-Models/experiments/
       train_palette_gelsight_phase2_v2_260616_192638/
  3. `uv sync` from the repo root.
  4. `from agentic_sr import generate_depth_map` — see CLAUDE.md for the full API.

Notes:
  - Use the non-EMA checkpoint at this training length. EMA is ~10× worse on
    masked-only MAE (see CLAUDE.md "EMA trap" section).
  - Inference: ~30-60 s per call (2000-step diffusion sampling).
  - Trained on the v2 config of `ppak10/Gelsight-Welds` HF dataset (~28 sessions).
