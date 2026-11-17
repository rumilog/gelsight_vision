# Progress

## Datasets

### GelSight overlap scans: `rlogh/gelsight-overlap-scans` (Hugging Face, private)

- URL: https://huggingface.co/datasets/rlogh/gelsight-overlap-scans
- Owner: HF account `rlogh`. To share it, add collaborators or make it public under Settings on the dataset page.
- Contents: raw session folders from `motion_collect_overlap.py`, copied as-is. The dataset README covers the layout, units, tile numbering and known limitations.
- Local copies: `captures_overlap/<session>/`. These aren't in git (`captures**` is in `.gitignore`).

| Session | Area / overlap | Touches | Notes |
|---|---|---|---|
| `2026-11-04_17-34-24` | 130 × 135 mm at plate (75, 75) mm, 25% | 408 (17 × 24), all valid | 3.0 N, 20 mm/s; RealSense snapshot in `realsense/` |
| `2026-11-16_20-15-41` | 130 × 135 mm at plate (75, 75) mm, 50% | 875 (25 × 35), all valid | 3.0 N, 20 mm/s; GelSight only (`use_realsense:=False`, RealSense unplugged), 80 min. Use `overlap_0.50_reprocessed/` (fixed geometry and sign, includes a `viewer/` export), which is also on HF |

Download:

```bash
hf auth login   # once per machine, write token from huggingface.co -> Settings -> Access Tokens
hf download rlogh/gelsight-overlap-scans --repo-type dataset --local-dir ~/HuggingFace/Datasets/gelsight-overlap-scans
```

Add a new session (run only after `motion_collect_overlap` has exited, i.e. the arm has homed):

```bash
cd ~/ros2_ws/src/gelsight_vision/captures_overlap
hf upload rlogh/gelsight-overlap-scans <session> <session> --repo-type dataset
```

Then add a row to the Sessions table in the dataset README and in the table above.

3D viewer for the stitched scan (browser, three.js; needs internet for the CDN):

```bash
python3 -m gelsight_vision.tools.scan_viewer.export_scan_viewer \
    ~/HuggingFace/Datasets/gelsight-overlap-scans/<session>/overlap_0.25 --serve
```

This writes `<run>/viewer/` (heights, texture, meta, `index.html`) and opens http://127.0.0.1:8765/. `--stride 2` gives a finer mesh (0.0625 mm, ~4.5M vertices). Add `--skip-export` to re-serve an existing export. The viewer has orbit and fly (WASD) navigation, height exaggeration, RGB / height / high-pass detail / clay shading, a movable light, and a hover readout plus two-click measuring in plate-frame mm.

### Reconstruction fixes (2026-11-11)

The `2026-11-04_17-34-24` stitch showed the weld bead as a valley. Three separate causes, all fixed in code:

1. **Frame size.** The GelSight stream is 16:9 and covers 19.9 x 11.2 mm (measured by registering neighbouring raw frames against the robot step), not 20 x 15 mm. With the central 50% crop the tiles were 9.9 x 5.6 mm, so the 5.625 mm step across the weld left **no overlap in y**; the stitcher stretched each tile by 4/3 and matched strips that were different places on the plate. `GELSIGHT_FULL_HEIGHT_M` is now 0.0112, and the collector uses `HeightMapper(preserve_aspect=True)` (square pixels, 320 x 180 tiles), so a 25% overlap scan really overlaps 25% on both axes.
2. **Depth sign.** gsrobotics depth is negative where the surface pushes into the gel. `stitch_session_solved` now defaults to `gel_depth_sign=-1`.
3. **Contact Z.** The tooltip TF only updates at 10 Hz, and the pose was read the instant the descent stopped, so contact Z was wherever the arm was up to 0.1 s earlier (a 1.7 mm sawtooth in visit order). The collector now reads the pose after the arm has been at rest (`pose_at_stop` keeps the old reading). In the 2026-11-04 session contact Z is not usable per touch.

Rebuild an old scan from its raw frames (wider crop, true scale, corrected sign) and re-stitch:

```bash
python3 -m gelsight_vision.tools.reprocess_overlap_scan \
    ~/HuggingFace/Datasets/gelsight-overlap-scans/<session>/overlap_0.25
# -> <session>/overlap_0.25_reprocessed/  (14.9 x 8.4 mm tiles, 50% x 33% overlap)
```

Result for `2026-11-04_17-34-24`: plate flat to 0.085 mm RMS with both sides level, bead a continuous ridge 0.6-1.3 mm high (median 0.8) and 7-9 mm wide. The reprocessed run is local only; the Hugging Face copy still has the old `gelsight_solved_*` files.

Still open: the top of the bead reads flat with slightly raised edges rather than crowned. Where the pad rests on the bead the gel does not reach the plate beside it, and a region the gel is not touching reads up to ~0.45 mm off, which is the size of that edge/centre difference, so this scan cannot settle the bead-top shape or its absolute height. A rescan with the fixes above gives real overlap and a usable contact Z per touch to anchor it.

The older tactile + D555 dataset, `ppak10/Gelsight-Welds` (local clone at `~/HuggingFace/Datasets/Gelsight-Welds`, layout `source/v1`, `source/v2`), is separate. Its build scripts only understand the older `captures/` layout.

## Rig setup (as of 2026-11-04)

- Plate frame: `ros2 run gelsight_vision set_plate` puts `plate` at world (0, 0, 0) with no rotation, at the corner of the 150 × 150 mm plate, which extends along world +x / +y.
- Topics from the camera PC: GelSight `/tool_side_gelsight/image_raw`; RealSense `/tool_side_cam/*`.
- Overlap scan used for the session above:

  ```bash
  ros2 run gelsight_vision motion_collect_overlap --ros-args \
      -p mock:=False -p use_gelsight:=True \
      -p area_x_mm:=130.0 -p area_y_mm:=135.0 \
      -p center_x:=0.075 -p center_y:=0.075 -p overlaps:="[0.25]"
  ```

- `mock:=True` is for the mock arm only. It skips the force stop. To run the real arm without the GelSight, use `use_gelsight:=False`.
- If moves "succeed" but the arm doesn't move, check `ros2 topic echo --once /ur20/io_and_status_controller/safety_mode`. `mode: 3` is a protective stop; clear it on the teach pendant.
