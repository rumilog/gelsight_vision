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

The same dataset also has `bead_touch/<date>/`, which holds camera-guided bead touch sessions (see [Bead pipeline](#bead-pipeline-first-hardware-runs-2026-11-19)). Each date folder has its own README. To download only that part:

```bash
hf download rlogh/gelsight-overlap-scans --repo-type dataset --include "bead_touch/2026-11-19/*" \
    --local-dir ~/HuggingFace/Datasets/gelsight-overlap-scans
```

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

## Bead pipeline: first hardware runs (2026-11-19)

The D555 photo finds the weld bead, then the GelSight touches stations along it and records bead height (contact Z above a plate plane fitted to nearby plate touches). Nodes and parameters are in `gelsight_vision/bead/README.md`. The sessions below are on Hugging Face under `rlogh/gelsight-overlap-scans` → `bead_touch/2026-11-19/{calib_spot,calib_sweep,bead_stations}/<session>/`. That folder's README covers the file layout and units. The sweeps there are re-analyzed with the band-midpoint crown reading, and the originals are kept as `sweep_result_peak.json`. Local copies are in `captures_*/`, which is not in git.

| Session | What | Result |
|---|---|---|
| `captures_calib_spot/2026-11-19_10-30-28` | Spot calibration on electrical-tape squares | 12/12 touches landed where clicked, but `calib_result.json` is not usable (1.7 mm residual): the tape was larger than the gel view, so the feature finder hit a different tape edge each time. Use ~3 mm dots and click their centres. |
| `captures_calib_sweep/2026-11-19_10-35-29` | Bead sweep at plate x = 50.5 mm, ±8 mm in 1 mm steps | Crown tracking slope -0.97, single-frame precision 0.08 mm, bead height 1.71 mm. The contact-band centre is 3.27 mm along the bead normal from the image centerline. `calib_from_sweep_band.json` in this folder is hand-made from that. |
| `captures_calib_sweep/2026-11-19_11-00-29` | Same sweep, centred on the GelSight-felt bead at (50.5, 80.4) mm | 1.74 mm; band centre at y = 77.2 mm, matching `10-35-29` |
| `captures_calib_sweep/2026-11-19_11-19-32` | Re-measure at x = 25.9 mm, ±2 mm | 1.59 mm; band centre at y = 77.7 mm (77.1 at x = 50) |
| `captures_bead/2026-11-19_10-40-59` | 3 stations, old 6.5 mm (peak-based) correction | 1.58 / 1.74 / 1.88 mm at x = 22 / 49 / 76; the centerline stopped at x = 97 (fixed since) |
| `captures_bead/2026-11-19_11-13-42` | 5 stations, old correction | T5 read 4.54 mm: the GelSight frame shows a smooth cylinder across the bead, most likely the sensor cable under the gel at yaw ~90°. Excluded. |
| `captures_bead/2026-11-19_11-24-16` | 5 stations, `calib_from_sweep_band.json` | All crowns within ±0.8 mm of gel centre, no re-touch, refs on alternating sides (plane RMS 0.027 mm). Heights below. |

Bead height along the bead (`11-24-16`, mm): 1.53 at x = 26, 1.76 at 50, 1.83 at 75, 2.38 at 96, 2.51 at 123. The other runs agree to within 0.11 mm at the same x. Height rises along x, with a ~0.5 mm step between x = 75 and 96. x ≈ 95 is where the D555 image shows a dark notch in the bead, and only two stations sit in that stretch. Heights are not corrected for gel compression (~±0.3 mm). They are also well above the 0.6–1.3 mm that the reprocessed `2026-11-04` overlap scan reads from gel profiles, which that section already notes can't settle absolute height.

Command that produced `11-24-16`:

```bash
ros2 run gelsight_vision touch_bead_stations --ros-args -p mock:=False -p slow_z:=0.0 \
  -p calib_path:=$HOME/ros2_ws/src/gelsight_vision/captures_calib_sweep/2026-11-19_10-35-29/calib_from_sweep_band.json \
  -p n_stations:=5 \
  -p camera_pose_source:=corners \
  -p plate_corners_px:="[498.,236.,714.,240.,712.,462.,496.,460.]" \
  -p plate_corners_mm:="[150.,150.,150.,0.,0.,0.,0.,150.]"
```

Code fixed on 2026-11-19 (merged from `bead-pipeline-hardware-fixes`):

- **Centerline ends.** The centerline now covers the whole bead (139 mm). The old end-finding stopped at a narrow notch.
- **Crown reading.** It is now the midpoint of the gel contact band. The bead reads flat-topped with raised edges, so the old peak reading jumped between edges (a 3.78 mm re-touch moved it 6.4 mm). Calibrations from the old reading, such as `calib_from_sweep.json` (6.5 mm), don't match it.
- **Plate references.** They now alternate sides of the bead. The plane fit no longer extrapolates a cross-bead tilt from refs that all sit on one side; that was a 0.19 mm error in simulation.
- **Sweep cross-section panel.** Fixed: it had wrong units, a wrong anchor, and drew where the gel hung free.
- **Preview window.** It no longer blocks Ctrl+C.

Open:

- **Camera TF is wrong.** `rgt_manager` (not on the robot PC) publishes `gelsight_vision -> tool_side_cam_link` as 100 mm straight down with no rotation, so TF thinks the D555 looks sideways and pixel-to-plate rays miss the plate by metres. The safety check refused those targets. Solving the pose from the plate corners gives the mount as x 0.0417, y 0.0577, z -0.0966 m, quaternion (x -0.0148, y 0.7088, z -0.0062, w 0.7052), which is a 90° pitch, good to a few mm. Until that is put in `rgt_manager`'s config, use `camera_pose_source:=corners` with the pixels above. Those pixels are only valid for the `camera_focus` (75, 75, 300) mm capture pose and this plate placement.
- **`slow_z:=0.0` (single-speed descent) is in use.** The robot PC froze hard during the first real spot-calibration descent with the two-speed descent on, and the system journal lost the last minutes, so the cause is unknown. Contact force overshoots to 3.8–6 N against the 3 N setting.
- **Watch the GelSight cable at yaw ~90°** (see `11-13-42` T5).
- D555 image stamps are ~27 min behind the TF stamps (probably the camera PC clock). Harmless while nodes use the latest TF; worth syncing (chrony/NTP).

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
