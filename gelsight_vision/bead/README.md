# Bead pipeline: D555 image -> GelSight touches on the weld bead

Three nodes on a shared base (`bead/rig_node.py`). The geometry and analysis are
ROS-free (`camera_model`, `centerline`, `planner`, `gel_features`, `calib`, `pipeline`)
and covered by `test/test_bead_offline.py`.

| Node | What it does | Touches | Output |
|---|---|---|---|
| `calib_spot_touch` | Experiment 1: camera → tool calibration on marked spots | 4 per spot | `calib_result.json` |
| `calib_bead_sweep` | Experiment 2: step across the bead at one station | 2 + 17 | `sweep_result.json`, `sweep_summary.png` |
| `touch_bead_stations` | Production chain: image → bead line → k stations → heights | k + k | `bead_stations.json`, `plan_overlay*.png` |

Build: `colcon build --packages-select gelsight_vision && source install/setup.bash`

## Before the first real run
1. **Mock first** (mock arm, no sensor), with a saved snapshot so no camera is needed:
   ```bash
   ros2 run gelsight_vision touch_bead_stations --ros-args -p mock:=True -p use_gelsight:=False \
     -p image_path:=$CAP/captures_overlap/2026-11-04_17-34-24/realsense/color.png \
     -p camera_pose_source:=corners \
     -p plate_corners_px:="[498.,236.,714.,240.,712.,462.,496.,460.]" \
     -p plate_corners_mm:="[150.,150.,150.,0.,0.,0.,0.,150.]"
   ```
   Watch the wrist: bead stations use **yaw ≈ 90°** (gel long axis across the bead).
   Check the GelSight cable and housing clear everything at that yaw before going real.
2. Every node shows a preview and asks `[y/N]` before touching (`confirm:=False` to skip,
   `show_preview:=False` when headless). Targets outside the plate (±5 mm) or with
   |yaw| > 100° are refused.
3. **Check the camera pose once.** On a live capture the node logs where it thinks the
   lens is (`camera pose from tf: centre [...] mm`). The 2026-11-04 snapshot puts it ~440 mm
   above the plate. If you also pass `plate_corners_px/_mm`, `camera_pose_compare.json`
   shows how far TF and the corner fit disagree.
4. Descent is fast to `slow_z` (tooltip z = 14 mm in the plate frame), then 3 mm/s.
   Plate contact is at tooltip z ≈ 6–7.5 mm.

## Experiment 1: spot calibration (~5 min)
Centre-punch 4–5 dents spread over a plate, away from the bead and clamps. They should
be visible as dark dots in the D555 image.
```bash
ros2 run gelsight_vision calib_spot_touch --ros-args -p mock:=False
```
Click the dents in the image window (Enter = done, Backspace = undo), or pass
`-p spots_px:="[u1,v1,u2,v2,...]"`. Each spot gets touched at yaw 0 (centre, +2 mm x,
+2 mm y) and yaw 90 (centre). The jogs are what solve the gel image axes.
The node prints:
- the camera correction (how far off the image-predicted points were),
- the GelSight centre offset from the `gelsight_tooltip` TCP,
- the gel axes and the residual.

Check `calib_features.png`: every cross should sit on the dent. Re-run the solve offline
with `python3 -m gelsight_vision.tools.analyze_spot_calib <session> [--feature-mm 2]`.

## Experiment 2: bead sweep (~4 min)
```bash
ros2 run gelsight_vision calib_bead_sweep --ros-args -p mock:=False \
  -p calib_path:=<spot session>/calib_result.json
# or a manual station: -p station_plate_mm:="[75.0, 78.0]" -p bead_dir_deg:=0.0
```
Two plate refs at ±22 mm, then offsets 0, ±1 … ±8 mm across the bead. It reports:
- the bead height plateau from contact Z,
- the offsets where the height drops off (that's the usable tolerance),
- the gel-axis sign, and how precisely one frame finds the crown (fit slope should be ≈ ±1).

The crown-detection thresholds in `gel_features.read_crown` are first guesses. Tune them
on this output.

## Production run
```bash
ros2 run gelsight_vision touch_bead_stations --ros-args -p mock:=False \
  -p calib_path:=<spot session>/calib_result.json [-p n_stations:=3]
```
Plate references go next to each station (`plate_ref_mode:=beside`, 22 mm along the
normal). `plate_ref_mode:=spread` spreads them over the plate, ≥ 25 mm from the edges.
With a calibration loaded, a station whose crown reads > `retouch_mm` (3 mm) off the gel
centre is re-touched once, shifted onto the crown.

`bead_stations.json`, per station:
- arc length and plate position,
- yaw,
- contact Z and height above the plate plane (not yet corrected for gel compression),
- crown reading, and the raw GelSight frame filename.

Offline planning, no ROS: `python3 -m gelsight_vision.tools.plan_bead_offline <color.png> --corners-px ... --corners-mm ...`

## Known limits
- The crown reading (`read_crown().crown_u_mm`) is the midpoint of the gel contact band.
  The as-welded bead reads flat-topped with raised edges, so the strongest-contact point
  (kept as `crown_peak_u_mm`) sits ~3 mm to one side and can jump between edges.
  Calibrations made from peak readings (e.g. `calib_from_sweep.json`, 6.5 mm) don't
  match the midpoint reading; use `calib_from_sweep_band.json` (3.27 mm) from 2026-11-19.
- Bead segmentation was tuned on one snapshot. The bright region is ~2× wider than the raised bead,
  and the centerline can sit a few mm off the crown. Experiment 1 and the re-touch absorb that.
- Height is robot contact Z minus the plate plane, with no gel-compression correction.
  Plate-touch scatter was 0.33 mm std with the old single-speed descent.
- Smoke-tested in a simulated rig (stubbed ROS, synthetic GelSight frames). Not yet run on hardware.
