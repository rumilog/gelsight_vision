"""D555 image -> bead centerline -> GelSight touches at k bead stations.

    capture (or load) color image  -> camera pose (TF or plate corners)
    -> plate-frame ortho image     -> bead mask + centerline (plate mm)
    -> k stations along the bead   -> tooltip yaw: gel long axis ACROSS the bead
    -> n plate-reference touches   -> plate plane (z reference)
    -> touch every station         -> contact Z, GelSight frame, crown reading
    -> bead_stations.json          -> per-station height above the plate

Outputs in captures_bead/<timestamp>/: color.png, camera_*.json, plan.json,
plan_overlay.png, ortho_mask.png, plate_ref_NN_* / bead_NN_* touch files,
bead_stations.json.

Examples
--------
Plan only from the saved 2026-11-04 snapshot (no camera, mock arm still needed
because the node waits for the robot action servers):

    ros2 run gelsight_vision touch_bead_stations --ros-args -p mock:=True \
        -p use_gelsight:=False -p plan_only:=True \
        -p image_path:=<captures_overlap>/2026-11-04_17-34-24/realsense/color.png \
        -p camera_pose_source:=corners \
        -p plate_corners_px:="[498.,236.,714.,240.,712.,462.,496.,460.]" \
        -p plate_corners_mm:="[150.,150.,150.,0.,0.,0.,0.,150.]"

(For no ROS at all use tools/plan_bead_offline.py.)

Full mock run (mock arm, no sensor):  -p mock:=True -p use_gelsight:=False
Real run:  -p mock:=False [-p calib_path:=.../calib_result.json]
"""

from __future__ import annotations

import math
import threading

import cv2
import numpy as np
import rclpy

from gelsight_vision.bead.gel_features import read_crown
from gelsight_vision.bead.pipeline import plan_bead
from gelsight_vision.bead.planner import Calibration, fit_plate_plane
from gelsight_vision.bead.rig_node import BeadRigNode


class TouchBeadStations(BeadRigNode):
    def __init__(self):
        super().__init__('touch_bead_stations', 'captures_bead')
        p = self.declare_parameter
        p('n_stations', 3)
        p('n_plate_refs', 3)                # 0 disables plate refs
        p('plate_ref_mode', 'beside')       # 'beside' each station, or 'spread' over the plate
        p('plate_ref_points_mm', [0.0])     # x1, y1, x2, y2, ... overrides auto choice
        p('plate_ref_clearance_mm', 22.0)
        p('calib_path', '')                 # calib_result.json from calib_spot_touch
        p('ortho_res_mm', 0.5)
        p('max_bead_width_mm', 25.0)
        p('yaw_override_deg', float('nan'))  # nan -> long axis across the bead
        p('plan_only', False)
        # One corrective re-touch when the crown reads > retouch_mm off the gel
        # centre. Needs calib_path (gel image axes); ignored otherwise.
        p('retouch_mm', 3.0)
        g = self.get_parameter
        self.n_stations = int(g('n_stations').value)
        self.n_plate_refs = int(g('n_plate_refs').value)
        self.plate_ref_mode = str(g('plate_ref_mode').value)
        refs = [float(v) for v in g('plate_ref_points_mm').value]
        self.plate_ref_points = refs if len(refs) >= 2 else None
        self.plate_ref_clearance = float(g('plate_ref_clearance_mm').value)
        self.calib = Calibration.load(str(g('calib_path').value))
        self.ortho_res = float(g('ortho_res_mm').value)
        self.max_bead_width = float(g('max_bead_width_mm').value)
        yo = float(g('yaw_override_deg').value)
        self.yaw_override = None if math.isnan(yo) else math.radians(yo)
        self.plan_only = bool(g('plan_only').value)
        self.retouch_mm = float(g('retouch_mm').value)

    # ------------------------------------------------------------------ plan
    def plan(self, img_bgr, cam):
        bp = plan_bead(img_bgr, cam, self.calib, self.n_stations, self.n_plate_refs,
                       self.plate_mm, self.ortho_res, self.max_bead_width,
                       self.plate_ref_clearance, self.plate_ref_points, self.yaw_override,
                       plate_ref_mode=self.plate_ref_mode)
        for name, im in bp.images.items():
            cv2.imwrite(self.out(name), im)
        self.write_json('plan.json', {**bp.plan, 'params': self.params_dict()})
        line, stations, refs = bp.line, bp.stations, bp.refs
        overlay = bp.images['plan_overlay.png']
        for s in refs + stations:
            self.get_logger().info(
                f"{s.kind} {s.index}: gel centre -> ({s.target_mm[0]:.1f}, {s.target_mm[1]:.1f}) "
                f"mm, tooltip ({s.tooltip_mm[0]:.1f}, {s.tooltip_mm[1]:.1f}) mm, "
                f"yaw {math.degrees(s.yaw_rad):.1f} deg")
        return line, stations, refs, overlay

    # ------------------------------------------------------------------ touch
    def touch_station(self, st, stem):
        rec = self.touch_at(stem, st.tooltip_mm[0], st.tooltip_mm[1], st.yaw_rad,
                            meta={'kind': st.kind, 'station': st.to_dict()})
        crown = None
        if self.use_gelsight and st.kind == 'bead' and self.last_raw is not None \
                and self.baseline_raw is not None and rec['valid_contact']:
            crown = read_crown(self.last_raw, self.baseline_raw).to_dict()
            if crown['crown_near_edge']:
                self.get_logger().warn(f'{stem}: crown near the gel edge '
                                       f"(u = {crown['crown_u_mm']:.1f} mm)")
        rec['crown'] = crown
        return rec

    def maybe_retouch(self, st, rec, stem):
        """Shift along the bead normal by the crown reading and touch once more."""
        crown = rec.get('crown')
        if (crown is None or self.calib.source == 'identity'
                or abs(crown['crown_u_mm']) < self.retouch_mm):
            return None
        shift = self.calib.gel_to_plate_dir(st.yaw_rad) @ np.array([crown['crown_u_mm'], 0.0])
        n = np.asarray(st.normal, float)
        delta = float(shift @ n) * n
        new_tool = np.asarray(st.tooltip_mm) + delta
        self.get_logger().info(f'{stem}: crown {crown["crown_u_mm"]:+.1f} mm off-centre; '
                               f're-touching shifted {np.round(delta, 2)} mm')
        st2 = type(st)(**{**st.__dict__, 'tooltip_mm': [float(v) for v in new_tool],
                          'target_mm': [float(v) for v in np.asarray(st.target_mm) + delta]})
        return self.touch_station(st2, stem + '_retouch')

    def summarize(self, line, results, ref_results):
        refs_ok = [r for r in ref_results if r['valid_contact'] and r['contact_z_mm'] is not None]
        coef, rms = fit_plate_plane([r['target_plate_mm'] for r in refs_ok],
                                    [r['contact_z_mm'] for r in refs_ok])
        out = []
        for st, rec in results:
            h = None
            if rec['contact_z_mm'] is not None and np.isfinite(coef[2]) and not self.mock:
                x, y = rec['target_plate_mm']
                h = rec['contact_z_mm'] - float(coef @ [x, y, 1.0])
            out.append({
                'station': st.index, 's_mm': st.s_mm, 'gel_centre_plate_mm': st.target_mm,
                'tooltip_plate_mm': rec['target_plate_mm'], 'tangent': st.tangent,
                'yaw_deg': round(math.degrees(st.yaw_rad), 2),
                'contact_z_mm': rec['contact_z_mm'],
                'height_above_plate_mm': None if h is None else round(h, 3),
                'valid_contact': rec['valid_contact'], 'crown': rec.get('crown'),
                'gel_frame': rec['stem'] + '_raw.png'})
        summary = {
            'frame_id': self.frame_id, 'units': 'mm',
            'plate_plane': {'coef_z_eq_ax_by_c': [float(v) for v in coef],
                            'n_refs': len(refs_ok), 'rms_mm': rms},
            'bead_length_mm': round(line.length_mm, 1),
            'stations': out,
            'mock': self.mock,
            'notes': (['MOCK RUN: no force stop, heights are not measured.'] if self.mock else [])
            + ['height_above_plate_mm = tooltip contact Z minus the plate plane; not '
                      'yet corrected for gel compression (expect ~0.3 mm noise today).',
                      'crown.plate_both_sides is a heuristic; validate with calib_bead_sweep.'],
        }
        self.write_json('bead_stations.json', summary)
        for s in out:
            self.get_logger().info(f"station {s['station']} (s={s['s_mm']} mm): height above "
                                   f"plate = {s['height_above_plate_mm']} mm")
        return summary


def main(args=None):
    rclpy.init(args=args)
    node = TouchBeadStations()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        touching = not node.plan_only
        if touching:
            node.calibrate_baseline()
            node.home()
        img, cam, _ = node.acquire_image()
        line, stations, refs, overlay = node.plan(img, cam)
        node.get_logger().info(f'plan -> {node.out("plan.json")}')
        node.preview(overlay, 'touch_bead_stations plan (any key)')
        if not touching:
            return
        if not node.ask(f'Touch {len(refs)} plate refs + {len(stations)} bead stations?'):
            node.get_logger().info('aborted by user')
            return
        ref_results = [node.touch_station(s, f'plate_ref_{s.index:02d}') for s in refs]
        results = []
        for st in stations:
            stem = f'bead_{st.index:02d}'
            rec = node.touch_station(st, stem)
            again = node.maybe_retouch(st, rec, stem)
            results.append((st, again if again is not None else rec))
        node.summarize(line, results, ref_results)
        node.home()
        node.get_logger().info(f'done -> {node.capture_dir}')
    except KeyboardInterrupt:
        node.get_logger().info('interrupted by user')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin.join(timeout=2.0)


if __name__ == '__main__':
    main()
