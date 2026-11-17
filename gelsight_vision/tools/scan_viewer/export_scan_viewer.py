"""Export a stitched overlap scan for the browser 3D viewer, and optionally serve it.

Reads the stitch outputs from one run folder of a `motion_collect_overlap`
session (e.g. `<session>/overlap_0.25/`):

    gelsight_solved_depth.npy   (H, W) float32 plate-frame height, mm
    gelsight_solved_rgb.png     RGB mosaic (integer multiple of H, W)
    gelsight_solved_solve.json  grid size + lattice step
    gelsight_NNNN_pose.json     tooltip poses, used to place the mosaic in the plate frame

and writes a self-contained viewer folder:

    index.html      the viewer (copied from this package)
    heights.bin     float32 (rows, cols) heights in mm, downsampled by --stride
    texture.jpg     RGB mosaic resized to (cols, rows) * --tex-scale
    meta.json       grid size, mm per sample, pixel -> plate affine

Usage:
    python -m gelsight_vision.tools.scan_viewer.export_scan_viewer \
        ~/HuggingFace/Datasets/gelsight-overlap-scans/2026-11-04_17-34-24/overlap_0.25 --serve
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import warnings
import webbrowser
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def _block_mean(a: np.ndarray, s: int) -> np.ndarray:
    """NaN-aware s x s block mean (crops the ragged edge)."""
    if s == 1:
        return a.astype(np.float32)
    h, w = (a.shape[0] // s) * s, (a.shape[1] // s) * s
    b = a[:h, :w].reshape(h // s, s, w // s, s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks -> NaN, filled later
        return np.nanmean(b, axis=(1, 3)).astype(np.float32)


def _fill_nans(z: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(z)
    if not bad.any():
        return z
    filled = cv2.inpaint(np.where(bad, 0, z).astype(np.float32), bad.astype(np.uint8), 3,
                         cv2.INPAINT_TELEA)
    return np.where(bad, filled, z).astype(np.float32)


def _pixel_to_plate(run_dir: Path, solve: dict, H: int, W: int, mm_per_px: float) -> dict:
    """Least-squares affine from full-res mosaic (row, col) to plate (x, y) mm,
    fitted on tile centres vs. the measured tooltip pose of each touch."""
    tx, ty = solve["touch_x"], solve["touch_y"]
    rs, cs = solve["step_px"]
    th, tw = H - (tx - 1) * rs, W - (ty - 1) * cs  # tile size in px
    A, B = [], []
    for p in sorted(run_dir.glob("gelsight_*_pose.json")):
        d = json.loads(p.read_text())
        if not d.get("valid_contact", True):
            continue
        r, c = d["grid"]["row"], d["grid"]["col"]
        t = d.get("plate_pose", d)["translation"]
        A.append([1.0, r * rs + th / 2.0, c * cs + tw / 2.0])
        B.append([t["x"] * 1e3, t["y"] * 1e3])
    if len(A) < 3:
        # Fall back to "mosaic centred on the origin, rows -> -x, cols -> -y".
        return {"x": [H / 2 * mm_per_px, -mm_per_px, 0.0],
                "y": [W / 2 * mm_per_px, 0.0, -mm_per_px], "fit": "fallback"}
    coef, *_ = np.linalg.lstsq(np.array(A), np.array(B), rcond=None)
    resid = np.array(B) - np.array(A) @ coef
    return {
        "x": coef[:, 0].tolist(),  # plate_x = x0 + xr*row + xc*col
        "y": coef[:, 1].tolist(),
        "fit": "poses",
        "n_tiles": len(A),
        "rms_mm": float(np.sqrt((resid ** 2).sum(1).mean())),
    }


def export(run_dir: Path, out_dir: Path, stride: int, tex_scale: float) -> Path:
    run_dir = run_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    depth = np.load(run_dir / "gelsight_solved_depth.npy")
    solve = json.loads((run_dir / "gelsight_solved_solve.json").read_text())
    mm_per_px = float(solve["mm_per_px"])
    H, W = depth.shape

    z = _fill_nans(_block_mean(depth, stride))
    rows, cols = z.shape
    z.astype("<f4").tofile(out_dir / "heights.bin")

    rgb = cv2.imread(str(run_dir / "gelsight_solved_rgb.png"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(run_dir / "gelsight_solved_rgb.png")
    # crop the RGB mosaic to the same footprint the downsampled grid covers
    sr = rgb.shape[0] / H
    rgb = rgb[: int(round(rows * stride * sr)), : int(round(cols * stride * sr))]
    tw, th = int(round(cols * stride * tex_scale)), int(round(rows * stride * tex_scale))
    tw, th = min(tw, 8192), min(th, 8192)
    rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / "texture.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])

    session = run_dir.parent.name
    meta = {
        "session": session,
        "run": run_dir.name,
        "rows": rows,
        "cols": cols,
        "stride": stride,
        "mm_per_sample": mm_per_px * stride,
        "mm_per_px_full": mm_per_px,
        "height_min_mm": float(np.min(z)),
        "height_max_mm": float(np.max(z)),
        "height_p1_mm": float(np.percentile(z, 1)),
        "height_p99_mm": float(np.percentile(z, 99)),
        # affine on *full-res* pixel (row, col); sample (i, j) -> pixel (i*stride + stride/2 - 0.5, ...)
        "pixel_to_plate_mm": _pixel_to_plate(run_dir, solve, H, W, mm_per_px),
        "texture": {"file": "texture.jpg", "width": tw, "height": th},
        "solve": {k: solve.get(k) for k in ("touch_x", "touch_y", "solve_mode",
                                            "pair_residual_rms_mm", "seam_step_rms_mm")},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.copy2(HERE / "index.html", out_dir / "index.html")
    print(f"viewer data -> {out_dir}  ({rows} x {cols} samples @ {meta['mm_per_sample']:.4f} mm, "
          f"texture {tw} x {th}, plate fit {meta['pixel_to_plate_mm']['fit']})")
    return out_dir


def serve(out_dir: Path, port: int, open_browser: bool) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"serving {out_dir} at {url}  (Ctrl+C to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path, help="run folder, e.g. <session>/overlap_0.25")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default <run_dir>/viewer)")
    ap.add_argument("--stride", type=int, default=4,
                    help="downsample factor for the mesh (4 -> 0.125 mm grid, ~1.1M vertices)")
    ap.add_argument("--tex-scale", type=float, default=1.0,
                    help="texture pixels per full-res depth pixel (1.0 = depth resolution; capped at 8192 px)")
    ap.add_argument("--serve", action="store_true", help="serve the viewer on localhost after export")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--skip-export", action="store_true", help="only serve an existing --out folder")
    a = ap.parse_args()

    run_dir = a.run_dir.expanduser().resolve()
    if a.out:
        out = a.out.expanduser().resolve()
    elif a.skip_export and (run_dir / "meta.json").exists():
        out = run_dir  # pointed straight at an exported viewer folder
    else:
        out = run_dir / "viewer"
    if a.skip_export and not (out / "meta.json").exists():
        ap.error(f"no exported viewer in {out} (run without --skip-export first)")
    if not a.skip_export:
        export(a.run_dir, out, a.stride, a.tex_scale)
    if a.serve:
        serve(out, a.port, not a.no_browser)


if __name__ == "__main__":
    main()
