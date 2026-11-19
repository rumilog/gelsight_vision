"""Experiment 2: step the GelSight across the bead at one station.

    pick a station (auto: middle of the usable bead from the D555 image, or
    station_plate_mm + bead_dir_deg) -> 2 plate-reference touches beside it
    -> touches at offsets d in [offset_min_mm, offset_max_mm] along the bead
       normal, gel long axis across the bead
    -> analyze_bead_sweep: contact-Z cross-section, usable off-centre range,
       gel-axis sign, single-frame crown localization precision

Offsets run centre-out (0, +s, -s, +2s, ...) so an interrupted run still has
a centred, symmetric set. 17 touches at the defaults, ~3-4 min.

    ros2 run gelsight_vision calib_bead_sweep --ros-args -p mock:=False \
        [-p calib_path:=<calib_spot session>/calib_result.json]
    # manual station: -p station_plate_mm:="[75.0, 78.0]" -p bead_dir_deg:=0.0
    # dry run:        -p mock:=True -p use_gelsight:=False
Re-run the analysis: python3 -m gelsight_vision.tools.analyze_bead_sweep <session>
"""

from __future__ import annotations

import math
import threading

import cv2
import numpy as np
import rclpy

from gelsight_vision.bead.centerline import bead_centerline, render_plate_ortho
from gelsight_vision.bead.planner import (Calibration, Station, bead_stations, pick_stations,
                                          yaw_across)
from gelsight_vision.bead.rig_node import BeadRigNode
from gelsight_vision.bead.viz import draw_plan
from gelsight_vision.tools.analyze_bead_sweep import analyze_sweep_session, print_summary


def centre_out(lo: float, hi: float, step: float) -> list[float]:
    offs = [0.0]
    k = 1
    while k * step <= max(hi, -lo) + 1e-9:
        for v in (k * step, -k * step):
            if lo - 1e-9 <= v <= hi + 1e-9:
                offs.append(round(v, 4))
        k += 1
    return offs


class CalibBeadSweep(BeadRigNode):
    def __init__(self):
        super().__init__('calib_bead_sweep', 'captures_calib_sweep')
        p = self.declare_parameter
        p('station_plate_mm', [0.0])     # [x, y]; empty -> auto from the image
        p('bead_dir_deg', float('nan'))  # bead direction for a manual station
        p('station_fraction', 0.5)       # auto: where along the usable bead (0..1)
        p('offset_min_mm', -8.0)
        p('offset_max_mm', 8.0)
        p('offset_step_mm', 1.0)
        p('plate_ref_offset_mm', 22.0)   # plate refs at +/- this along the normal
        p('calib_path', '')
        p('max_bead_width_mm', 25.0)
        g = self.get_parameter
        sp = [float(v) for v in g('station_plate_mm').value]
        self.manual_station = sp if len(sp) == 2 else None
        self.bead_dir_deg = float(g('bead_dir_deg').value)
        self.station_fraction = float(g('station_fraction').value)
        self.offsets = centre_out(float(g('offset_min_mm').value),
                                  float(g('offset_max_mm').value),
                                  float(g('offset_step_mm').value))
        self.ref_offset = float(g('plate_ref_offset_mm').value)
        self.calib = Calibration.load(str(g('calib_path').value))
        self.max_bead_width = float(g('max_bead_width_mm').value)

    def station(self):
        """(target_mm, tangent, normal, img, cam)"""
        if self.manual_station is not None:
            if math.isnan(self.bead_dir_deg):
                raise ValueError('station_plate_mm needs bead_dir_deg')
            a = math.radians(self.bead_dir_deg)
            t = np.array([math.cos(a), math.sin(a)])
            target = self.calib.correct_point(self.manual_station)
            return target, t, np.array([-t[1], t[0]]), None, None
        img, cam, _ = self.acquire_image()
        ortho = render_plate_ortho(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cam,
                                   (0.0, self.plate_mm[0]), (0.0, self.plate_mm[1]))
        line = bead_centerline(ortho, max_bead_width_mm=self.max_bead_width)
        i_mid = pick_stations(line, 1)[0]
        if abs(self.station_fraction - 0.5) > 1e-6:
            idx = pick_stations(line, 21)
            i_mid = idx[min(len(idx) - 1, int(round(self.station_fraction * (len(idx) - 1))))]
        st = bead_stations(line, [i_mid], self.calib)[0]
        return (np.asarray(st.target_mm), np.asarray(st.tangent), np.asarray(st.normal),
                img, cam)

    def run(self):
        target, t, n, img, cam = self.station()
        psi = yaw_across(n)
        plan = []
        for side in (+1, -1):
            pt = target + side * self.ref_offset * n
            plan.append(('plate_ref', f'plate_ref_{1 if side > 0 else 2:02d}', pt, 0.0,
                         side * self.ref_offset))
        for d in self.offsets:
            stem = f'sweep_{"p" if d >= 0 else "m"}{abs(d):04.1f}'.replace('.', '_')
            plan.append(('sweep', stem, target + d * n, psi, d))
        stations = [Station(k if k == 'plate_ref' else 'bead', i + 1, list(pt),
                            list(self.calib.tooltip_for(pt, yaw)), yaw)
                    for i, (k, _, pt, yaw, _) in enumerate(plan)]
        self.write_json('sweep.json', {
            'station': {'target_mm': target.tolist(), 'tangent': t.tolist(),
                        'normal': n.tolist(), 'yaw_deg': math.degrees(psi)},
            'offsets_mm': self.offsets, 'calibration': self.calib.to_dict(),
            'params': self.params_dict()})
        if img is not None:
            overlay = draw_plan(img, cam, stations=[stations[0], stations[1], stations[2]])
            cv2.imwrite(self.out('sweep_overlay.png'), overlay)
            self.preview(overlay, 'calib_bead_sweep: station + plate refs (any key)')
        self.get_logger().info(f'station {np.round(target, 2)} mm, normal {np.round(n, 3)}, '
                               f'yaw {math.degrees(psi):.1f} deg, offsets {self.offsets}')
        if not self.ask(f'Touch 2 plate refs + {len(self.offsets)} sweep offsets?'):
            return
        for (kind, stem, pt, yaw, d), st in zip(plan, stations):
            self.touch_at(stem, st.tooltip_mm[0], st.tooltip_mm[1], yaw, meta={
                'experiment': 'sweep', 'kind': kind, 'offset_mm': d,
                'gel_centre_target_mm': [float(v) for v in pt]})
        if self.mock:
            self.get_logger().info('mock run: motion only, skipping the sweep analysis')
            return
        try:
            res = analyze_sweep_session(self.capture_dir)
            print_summary(res)
            self.get_logger().info(f'sweep result -> {self.out("sweep_result.json")}')
        except Exception as e:  # noqa: BLE001 -- keep the captured data either way
            self.get_logger().error(f'sweep analysis failed ({e!r}); data kept in '
                                    f'{self.capture_dir}, re-run tools/analyze_bead_sweep')


def main(args=None):
    rclpy.init(args=args)
    node = CalibBeadSweep()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        node.calibrate_baseline()
        node.home()
        node.run()
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
