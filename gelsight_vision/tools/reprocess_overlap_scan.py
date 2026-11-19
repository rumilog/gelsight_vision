"""Rebuild an overlap scan from its raw GelSight frames and re-stitch it.

Scans collected before 2026-11-11 assumed a 20 x 15 mm frame, kept the central
50% and resized it to 4:3. The streamed frame is really 16:9, about
19.9 x 11.2 mm, so along the short side the tiles had no overlap at all and the
stitcher matched strips that are different places on the part. The raw frames
(`gelsight_NNNN_raw.png`) cover twice the area per axis, so they do overlap.

This recomputes every tile from the raw frame with a wider crop and square
pixels, into a new run folder that the stitcher and the 3D viewer read like any
other:

    gelsight_NNNN.png         cropped frame
    gelsight_NNNN_depth.npy   height map (rows = short side), raw model units
    gelsight_NNNN_pose.json   copied
    session.json              copied, with a `reprocessed` block added
    gelsight_solved_*         stitch outputs

Usage:
    python3 -m gelsight_vision.tools.reprocess_overlap_scan <session>/overlap_0.25
    # -> <session>/overlap_0.25_reprocessed/
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

from gelsight_vision.gelsight.height_map import HeightMapper
from gelsight_vision.gelsight.stitch import stitch_session_solved

_RAW_RE = re.compile(r"^gelsight_(\d+)_raw\.png$")


def reprocess(
    run_dir: Path,
    out_dir: Path,
    keep: float = 0.75,
    width_px: int = 480,
    frame_width_m: float = 0.0199,
    gel_depth_sign: int = -1,
    outlier_mm: float = 3.0,
    stitch: bool = True,
    bead_prior: bool = False,
) -> dict | None:
    """Recompute tiles from `run_dir`'s raw frames into `out_dir`, then stitch.

    `keep` is the fraction of each frame axis used per tile. The default 0.75
    gives 14.9 x 8.4 mm tiles; with the 7.5 x 5.625 mm step of the 2026-11-04
    scan that is 50% / 33% overlap. `width_px` sets the tile resolution (480 px
    across 14.9 mm keeps the original 0.031 mm/px).
    """
    run_dir, out_dir = Path(run_dir).expanduser().resolve(), Path(out_dir).expanduser().resolve()
    raws = sorted(p for p in run_dir.iterdir() if _RAW_RE.match(p.name))
    if not raws:
        raise FileNotFoundError(f"no gelsight_NNNN_raw.png frames in {run_dir}")
    session = json.loads((run_dir / "session.json").read_text())
    touch_x, touch_y = int(session["touch_x"]), int(session["touch_y"])
    out_dir.mkdir(parents=True, exist_ok=True)

    border = (1.0 - keep) / 2.0
    mapper = HeightMapper(width=width_px, border_fraction=border, preserve_aspect=True)
    baseline = run_dir / "gelsight_baseline_raw.png"
    if baseline.exists():
        frame = cv2.imread(str(baseline))
        mapper.calibrate(frame)
        cv2.imwrite(str(out_dir / "gelsight_baseline.png"), mapper.crop_native(frame))
        shutil.copy2(baseline, out_dir / baseline.name)

    frame_hw = None
    for i, raw in enumerate(raws, 1):
        stem = raw.name[: -len("_raw.png")]
        frame = cv2.imread(str(raw))
        frame_hw = frame.shape[:2]
        cv2.imwrite(str(out_dir / f"{stem}.png"), mapper.crop_native(frame))
        np.save(out_dir / f"{stem}_depth.npy", mapper.image_to_height_tensor(frame).numpy())
        pose = run_dir / f"{stem}_pose.json"
        if pose.exists():
            shutil.copy2(pose, out_dir / pose.name)
        if i % 100 == 0 or i == len(raws):
            print(f"  reprocessed {i}/{len(raws)} frames")
    if (run_dir / "timing.json").exists():
        shutil.copy2(run_dir / "timing.json", out_dir / "timing.json")

    # True tile footprint and overlap, for the record (frame pixels are square).
    fov = (frame_width_m * keep, frame_width_m * keep * frame_hw[0] / frame_hw[1])
    step = session.get("step_m", [None, None])
    overlap = [None if s is None else 1.0 - s / f for s, f in zip(step, fov)]
    session["reprocessed"] = {
        "source": str(run_dir), "keep_fraction": keep, "tile_px": [mapper.width, mapper.height],
        "frame_width_m": frame_width_m, "fov_m": list(fov), "true_overlap": overlap,
        "gel_depth_sign": gel_depth_sign,
    }
    session["fov_m"] = list(fov)
    (out_dir / "session.json").write_text(json.dumps(session, indent=2))
    print(f"tiles -> {out_dir}  ({fov[0] * 1e3:.1f} x {fov[1] * 1e3:.1f} mm, "
          f"overlap {overlap[0]:.0%} x {overlap[1]:.0%})" if None not in overlap else f"tiles -> {out_dir}")
    if not stitch:
        return None
    return stitch_session_solved(
        out_dir, touch_x, touch_y,
        touch_overlap=min(o for o in overlap if o is not None) if any(o is not None for o in overlap) else 0.0,
        gelsight_full_width_m=frame_width_m, border_fraction=border,
        gel_depth_sign=gel_depth_sign, outlier_mm=outlier_mm, bead_prior=bead_prior,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path, help="run folder with gelsight_NNNN_raw.png, e.g. <session>/overlap_0.25")
    ap.add_argument("--out", type=Path, default=None, help="output run folder (default <run_dir>_reprocessed)")
    ap.add_argument("--keep", type=float, default=0.75, help="fraction of each frame axis per tile")
    ap.add_argument("--width-px", type=int, default=480, help="tile width in px (long side)")
    ap.add_argument("--frame-width-mm", type=float, default=19.9, help="long side of the streamed frame")
    ap.add_argument("--sign", type=int, default=-1, choices=(-1, 1), help="gel depth sign")
    ap.add_argument("--outlier-mm", type=float, default=3.0,
                    help="relief further than this from the tile median is dropped")
    ap.add_argument("--no-stitch", action="store_true")
    ap.add_argument("--bead-prior", action="store_true",
                    help="stitch Z-blind with the bead-shape prior (writes gelsight_beadprior_*)")
    a = ap.parse_args()
    run = a.run_dir.expanduser().resolve()
    out = a.out or run.with_name(run.name + "_reprocessed")
    reprocess(run, out, keep=a.keep, width_px=a.width_px, frame_width_m=a.frame_width_mm / 1e3,
              gel_depth_sign=a.sign, outlier_mm=a.outlier_mm, stitch=not a.no_stitch,
              bead_prior=a.bead_prior)


if __name__ == "__main__":
    main()
