"""Plan bead touch stations from a saved D555 color image -- no ROS, no robot.

Same planning code as the touch_bead_stations node (bead.pipeline.plan_bead).
Camera pose from plate corners (PnP) or a saved camera_pose.json (TF).

    python3 -m gelsight_vision.tools.plan_bead_offline \
        <captures_overlap>/2026-11-04_17-34-24/realsense/color.png \
        --corners-px 498 236 714 240 712 462 496 460 \
        --corners-mm 150 150 150 0 0 0 0 150 --out /tmp/bead_plan
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from gelsight_vision.bead.camera_model import CameraModel, compare_poses, pose_from_plane_points
from gelsight_vision.bead.pipeline import plan_bead
from gelsight_vision.bead.planner import Calibration


def camera_for(image_path: str, camera_info: str | None, corners_px, corners_mm,
               source: str = 'auto') -> CameraModel:
    folder = os.path.dirname(os.path.abspath(image_path))
    info_path = camera_info or os.path.join(folder, 'camera_info.json')
    cam = CameraModel.from_camera_info_json(info_path)
    tf_cam = corner_cam = None
    pose_path = os.path.join(folder, 'camera_pose.json')
    if os.path.exists(pose_path):
        tf = json.load(open(pose_path)).get('plate_to_optical')
        if tf:
            t, r = tf['translation'], tf['rotation']
            tf_cam = cam.with_tf([t['x'], t['y'], t['z']], [r['x'], r['y'], r['z'], r['w']])
    if corners_px and corners_mm:
        R, C = pose_from_plane_points(cam, np.reshape(corners_px, (-1, 2)),
                                      np.reshape(corners_mm, (-1, 2)))
        corner_cam = cam.with_pose(R, C, 'corners')
    if tf_cam is not None and corner_cam is not None:
        print('TF vs corners:', compare_poses(tf_cam, corner_cam))
    chosen = {'tf': tf_cam, 'corners': corner_cam}.get(source) if source != 'auto' else (
        tf_cam if tf_cam is not None else corner_cam)
    if chosen is None:
        raise SystemExit('no camera pose: pass --corners-px/--corners-mm '
                         'or provide camera_pose.json')
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('image')
    ap.add_argument('--camera-info', default=None)
    ap.add_argument('--corners-px', type=float, nargs='*', default=None)
    ap.add_argument('--corners-mm', type=float, nargs='*', default=None)
    ap.add_argument('--pose-source', choices=['auto', 'tf', 'corners'], default='auto')
    ap.add_argument('--plate-mm', type=float, nargs=2, default=[150.0, 150.0])
    ap.add_argument('--stations', type=int, default=3)
    ap.add_argument('--plate-refs', type=int, default=3)
    ap.add_argument('--plate-ref-mode', choices=['beside', 'spread'], default='beside')
    ap.add_argument('--calib', default='')
    ap.add_argument('--gel-line', default=None,
                    help='optional .npy of (x, y) plate mm, drawn in cyan for comparison')
    ap.add_argument('--out', default='bead_plan')
    a = ap.parse_args()

    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f'cannot read {a.image}')
    cam = camera_for(a.image, a.camera_info, a.corners_px, a.corners_mm, a.pose_source)
    extra = [(np.load(a.gel_line), (255, 255, 0))] if a.gel_line else None
    bp = plan_bead(img, cam, Calibration.load(a.calib), a.stations, a.plate_refs, a.plate_mm,
                   extra_lines=extra, plate_ref_mode=a.plate_ref_mode)
    os.makedirs(a.out, exist_ok=True)
    for name, im in bp.images.items():
        cv2.imwrite(os.path.join(a.out, name), im)
    with open(os.path.join(a.out, 'plan.json'), 'w') as f:
        json.dump(bp.plan, f, indent=2)
    c = bp.plan['camera']
    print(f"camera ({c['pose_source']}): centre {c['optical_centre_plate_mm']} mm, "
          f"{c['mm_per_px_at_plate']} mm/px; bead {bp.plan['bead']['length_mm']} mm long, "
          f"visual width {bp.plan['bead']['visual_width_median_mm']} mm")
    for s in bp.refs + bp.stations:
        print(f"  {s.kind:9s} {s.index}: gel centre {s.target_mm} mm, tooltip {s.tooltip_mm} mm, "
              f"yaw {np.degrees(s.yaw_rad):.1f} deg")
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
