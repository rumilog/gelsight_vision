"""Experiment 1: image spot -> GelSight touch calibration.

Mark 4-5 spots on a plate that both the D555 and the gel can see -- a
centre-punch dent is ideal (dark dot in the image, a sharp dent to the gel);
spread them over the plate, away from the bead. Then:

    capture image -> pick spots (click, or spots_px) -> map pixels to the plate
    -> touch each spot at every yaw in yaws_deg (default 0 and 90)
    -> locate the dent in each GelSight frame -> solve (analyze_spot_calib)

Result: calib_result.json with the camera correction, the gel-centre offset
from the gelsight_tooltip TCP, and the gel image axes. Feed it to
touch_bead_stations / calib_bead_sweep with calib_path:=.../calib_result.json.
Re-run the analysis offline (e.g. different --feature-mm) with
`python3 -m gelsight_vision.tools.analyze_spot_calib <session>`.

    ros2 run gelsight_vision calib_spot_touch --ros-args -p mock:=False
    # spots without a display:  -p spots_px:="[612.,300.,560.,410.,...]"
    # dry run on the mock arm:  -p mock:=True -p use_gelsight:=False -p spots_px:=...

About 4 spots x 2 yaws x ~10 s = under 2 minutes of touching.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import rclpy

from gelsight_vision.bead.rig_node import BeadRigNode
from gelsight_vision.bead.viz import draw_plan
from gelsight_vision.bead.planner import Station
from gelsight_vision.tools.analyze_spot_calib import analyze_spot_session, print_summary


class CalibSpotTouch(BeadRigNode):
    def __init__(self):
        super().__init__('calib_spot_touch', 'captures_calib_spot')
        p = self.declare_parameter
        p('spots_px', [0.0])          # u1, v1, u2, v2, ...; empty -> click on the image
        p('spots_plate_mm', [0.0])    # or give plate points directly (skips the image)
        p('yaws_deg', [0.0, 90.0])
        p('repeats', 1)
        # Known small offsets at these yaws: +jog along plate x and +jog along
        # plate y (needed to solve the gel image axes). Keep small so the dent
        # stays inside the 11.2 mm short side of the frame.
        p('jog_mm', 2.0)
        p('jog_yaws_deg', [0.0])
        p('feature_mm', 1.5)
        p('min_snr', 6.0)
        g = self.get_parameter
        self.spots_px = [float(v) for v in g('spots_px').value]
        self.spots_plate = [float(v) for v in g('spots_plate_mm').value]
        self.yaws = [float(v) for v in g('yaws_deg').value]
        self.repeats = int(g('repeats').value)
        self.jog = float(g('jog_mm').value)
        self.jog_yaws = [float(v) for v in g('jog_yaws_deg').value]
        self.feature_mm = float(g('feature_mm').value)
        self.min_snr = float(g('min_snr').value)

    def pick_spots(self) -> list[dict]:
        if len(self.spots_plate) >= 2:
            xy = np.asarray(self.spots_plate, float).reshape(-1, 2)
            return [{'spot': i + 1, 'pixel': None, 'p_img_mm': [float(a), float(b)]}
                    for i, (a, b) in enumerate(xy)]
        img, cam, _ = self.acquire_image()
        if len(self.spots_px) >= 2:
            uv = np.asarray(self.spots_px, float).reshape(-1, 2)
        else:
            self.get_logger().info('Click each marked spot, Enter when done, Backspace = undo')
            uv = np.asarray(self.click_points(img, 'calib_spot_touch: click spots'), float)
        if len(uv) == 0:
            raise RuntimeError('no spots selected')
        xy = cam.pixel_to_plate(uv)[:, :2]
        spots = [{'spot': i + 1, 'pixel': [float(u), float(v)], 'p_img_mm': [float(a), float(b)]}
                 for i, ((u, v), (a, b)) in enumerate(zip(uv, xy))]
        overlay = draw_plan(img, cam, stations=[
            Station('bead', s['spot'], s['p_img_mm'], s['p_img_mm'], np.radians(y))
            for s in spots for y in self.yaws])
        cv2.imwrite(self.out('spots_overlay.png'), overlay)
        self.preview(overlay, 'calib_spot_touch: planned touches (any key)')
        return spots

    def touch_plan(self, spots) -> list[tuple]:
        plan = []
        for s in spots:
            for yaw in self.yaws:
                jogs = [(0.0, 0.0)]
                if self.jog > 0 and any(abs(yaw - j) < 1e-6 for j in self.jog_yaws):
                    jogs += [(self.jog, 0.0), (0.0, self.jog)]
                for jx, jy in jogs:
                    for r in range(self.repeats):
                        tag = 'c' if (jx, jy) == (0.0, 0.0) else ('jx' if jx else 'jy')
                        stem = f"spot_{s['spot']:02d}_yaw{int(round(yaw)):+04d}_{tag}_r{r + 1}"
                        plan.append((stem, s, yaw, s['p_img_mm'][0] + jx, s['p_img_mm'][1] + jy))
        return plan

    def run(self) -> None:
        spots = self.pick_spots()
        plan = self.touch_plan(spots)
        self.write_json('spots.json', {'spots': spots, 'yaws_deg': self.yaws,
                                       'repeats': self.repeats, 'jog_mm': self.jog,
                                       'touches': [p[0] for p in plan],
                                       'params': self.params_dict()})
        if not self.ask(f'Touch {len(spots)} spots, {len(plan)} touches in total?'):
            return
        for stem, s, yaw, x, y in plan:
            self.touch_at(stem, x, y, np.radians(yaw), meta={
                'experiment': 'spot', 'spot': s['spot'], 'pixel': s['pixel'],
                'p_img_mm': s['p_img_mm']})
        if self.use_gelsight:
            try:
                result = analyze_spot_session(self.capture_dir, self.feature_mm, self.min_snr)
                print_summary(result)
                self.get_logger().info(f'calibration -> {self.out("calib_result.json")}')
            except Exception as e:  # noqa: BLE001 -- keep the captured data either way
                self.get_logger().error(
                    f'spot analysis failed ({e!r}); data kept in {self.capture_dir}, re-run '
                    'python3 -m gelsight_vision.tools.analyze_spot_calib <session>')


def main(args=None):
    rclpy.init(args=args)
    node = CalibSpotTouch()
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
