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
