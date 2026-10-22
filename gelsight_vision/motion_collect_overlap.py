"""Overlap sweep: scan one plate area at several tile overlaps, time each
scan, stitch each, and compare the combined images.

For every overlap in `overlaps`, the area `area_x_mm` x `area_y_mm` (centred
on plate (`center_x`, `center_y`)) is covered by a grid whose step is
`(1 - overlap)` of the cropped gelsight FOV per axis -- overlap is a linear
fraction per axis, so 0.5 means each touch moves half a tile and every
interior tile overlaps all 8 neighbours. Grid size is chosen so the tiles
cover at least the requested area (overflow split evenly on both sides).

Session layout (per-touch files match motion_collect_sample.py):

    captures_overlap/<timestamp>/
        sweep.json                     requested area, overlaps, per-run summary
        compare_overlaps.png/.csv/.json   (gelsight/compare_overlaps.py)
        overlap_0.50/
            session.json               grid plan + timing summary
            timing.json                per-touch timestamps / durations / attempts
            gelsight_baseline.png
            gelsight_NNNN.png, _raw.png, _depth.npy, _depth_preview.png, _pose.json
            gelsight_solved_*          (stitch_session_solved)
        overlap_0.25/ ...

Each `_pose.json` holds the gelsight_tooltip pose at the force-stop instant in
world (the keys stitch.py reads) and plate frames, plus the commanded target,
start Z, descent distance, force, grid indices, attempts and `valid_contact`.
A touch whose force trips before `min_descent_m` of travel is retried.

Touches run in serpentine order to cut travel; files are still numbered
row-major (k -> row (k-1)//touch_y, col (k-1)%touch_y) so the stitcher's grid
mapping holds. Tactile-only: no D555 / IGEV.

    ros2 run gelsight_vision motion_collect_overlap --ros-args \\
        -p frame_id:=plate -p mock:=False \\
        -p area_x_mm:=30.0 -p area_y_mm:=30.0 -p overlaps:="[0.5, 0.33, 0.25, 0.1]"
"""

import json
import math
import os
import threading
import time
from datetime import datetime
from math import radians

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.action.client import ActionClient
from rclpy.node import Node
from robot_manager_interfaces.action import PoseGoal
from robot_manager_interfaces.srv import Home
from sensor_msgs.msg import Image
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import quaternion_from_euler

from gelsight_vision.gelsight.compare_overlaps import compare_overlap_runs
from gelsight_vision.gelsight.height_map import HeightMapper, save_height_preview
from gelsight_vision.gelsight.stitch import stitch_session_solved
from gelsight_vision.pipeline import touch_io

# Full-sensor lateral FOV of the gel; the depth pipeline keeps the central
# `keep_fraction` of each axis (see motion_collect_sample.py).
GELSIGHT_FULL_WIDTH_M = 0.020
GELSIGHT_FULL_HEIGHT_M = 0.015


def plan_grid(
    area_m: tuple[float, float], fov_m: tuple[float, float], overlap: float,
) -> dict:
    """Smallest grid with step (1 - overlap) * fov covering `area_m`.
    Axis 0 = plate x (grid rows), axis 1 = plate y (grid cols)."""
    n, step, cover = [], [], []
    for area, fov in zip(area_m, fov_m):
        st = fov * (1.0 - overlap)
        k = 1 if area <= fov else int(math.ceil((area - fov) / st - 1e-9)) + 1
        n.append(k)
        step.append(st)
        cover.append((k - 1) * st + fov)
    return {'touch_x': n[0], 'touch_y': n[1], 'step_m': step, 'coverage_m': cover}


class MotionCollectOverlap(Node):
    def __init__(self):
        super().__init__('motion_collect_overlap')

        _ = self.declare_parameter('ns', 'ur20')
        _ = self.declare_parameter('frame_id', 'plate')
        _ = self.declare_parameter('force_max', 4.0)
        _ = self.declare_parameter('mock', False)
        # Plate area to scan (mm) and its centre in the plate frame (m).
        _ = self.declare_parameter('area_x_mm', 30.0)
        _ = self.declare_parameter('area_y_mm', 30.0)
        _ = self.declare_parameter('center_x', 0.0)
        _ = self.declare_parameter('center_y', 0.0)
        # One full scan of the area per entry, in this order. Linear fraction
        # of the cropped FOV shared with each neighbour, per axis.
        _ = self.declare_parameter('overlaps', [0.5, 0.33, 0.25, 0.1])
        # Re-zero the gel height baseline before each scan of the sweep.
        _ = self.declare_parameter('rebaseline_each_run', True)
        # Plate-frame Z the tooltip approaches from before each descent.
        _ = self.declare_parameter('safe_z', 0.02)
        _ = self.declare_parameter('descent_speed', -0.025)
        # Contact must happen at least this far below the start Z, else the
        # force trip is treated as spurious and the touch retried.
        _ = self.declare_parameter('min_descent_m', 0.001)
        # Abort a descent that goes this far without reaching force_max.
        _ = self.declare_parameter('max_descent_m', 0.035)
        _ = self.declare_parameter('max_attempts', 3)
        # Readings >= force_max needed in a row to count as contact.
        _ = self.declare_parameter('trip_samples', 3)
        # Wait after each PTP move before zeroing the force sensor, so the
        # post-move transient isn't baked into the offset.
        _ = self.declare_parameter('settle_s', 0.5)
        _ = self.declare_parameter(
            'capture_root', '/home/ppak/ros2_ws/src/gelsight_vision/captures_overlap',
        )

        self.ns: str = str(self.get_parameter('ns').value)
        self.frame_id: str = str(self.get_parameter('frame_id').value)
        self.force_max: float = float(self.get_parameter('force_max').value)
        self.mock: bool = bool(self.get_parameter('mock').value)
        self.area_m = (
            float(self.get_parameter('area_x_mm').value) / 1000.0,
            float(self.get_parameter('area_y_mm').value) / 1000.0,
        )
        self.center = (
            float(self.get_parameter('center_x').value),
            float(self.get_parameter('center_y').value),
        )
        self.overlaps: list[float] = [float(o) for o in self.get_parameter('overlaps').value]
        self.rebaseline_each_run: bool = bool(self.get_parameter('rebaseline_each_run').value)
        self.safe_z: float = float(self.get_parameter('safe_z').value)
        self.descent_speed: float = float(self.get_parameter('descent_speed').value)
        self.min_descent_m: float = float(self.get_parameter('min_descent_m').value)
        self.max_descent_m: float = float(self.get_parameter('max_descent_m').value)
        self.max_attempts: int = int(self.get_parameter('max_attempts').value)
        self.trip_samples: int = int(self.get_parameter('trip_samples').value)
        self.settle_s: float = float(self.get_parameter('settle_s').value)
        bad = [o for o in self.overlaps if not 0.0 <= o < 1.0]
        if bad:
            raise ValueError(f"overlaps must be in [0, 1), got {bad}")

        capture_root = str(self.get_parameter('capture_root').value)
        session = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.sweep_dir = os.path.join(capture_root, session)
        os.makedirs(self.sweep_dir, exist_ok=True)
        # Per-run directory; the gelsight callback writes here.
        self.capture_dir = self.sweep_dir

        self.height_mapper = HeightMapper()
        self.cv_bridge = CvBridge()

        # Gelsight capture handshake: main thread sets `capture_stem` then the
        # flag; the image callback writes files and clears the flag.
        self.gelsight_calibrate = False
        self.gelsight_capture_image = False
        self.capture_stem: str | None = None

        self.pose_goal_client: ActionClient = ActionClient(
            self, PoseGoal, self.ns + "/pose_goal",
        )
        self.pose_goal_client.wait_for_server()
        self.home_client = self.create_client(Home, self.ns + "/home")
        while not self.home_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service: " + self.ns + "/home")

        touch_io.setup_force(self, self.ns)
        touch_io.setup_twist(self, self.ns)
        touch_io.setup_servo(self, self.ns)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.gelsight_subscriber = self.create_subscription(
            Image, 'gelsight_capture/image', self.gelsight_listener_callback, 10,
        )

    #### Gelsight
    def gelsight_listener_callback(self, msg):
        if self.gelsight_calibrate:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            baseline_path = os.path.join(self.capture_dir, "gelsight_baseline.png")
            cv2.imwrite(baseline_path, self.height_mapper.crop_native(frame))
            cv2.imwrite(os.path.join(self.capture_dir, "gelsight_baseline_raw.png"), frame)
            self.height_mapper.calibrate(frame)
            print(f"gelsight baseline -> {baseline_path}")
            self.gelsight_calibrate = False
            return
        if self.gelsight_capture_image:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            stem = os.path.join(self.capture_dir, self.capture_stem)
            cv2.imwrite(f"{stem}.png", self.height_mapper.crop_native(frame))
            cv2.imwrite(f"{stem}_raw.png", frame)
            height = self.height_mapper.image_to_height_tensor(frame).numpy()
            np.save(f"{stem}_depth.npy", height)
            save_height_preview(height, f"{stem}_depth_preview.png")
            print(f"gelsight captured frame -> {stem}.png (+ _raw, _depth.npy, _depth_preview)")
            self.gelsight_capture_image = False

    #### TF
    def lookup(self, parent: str, child: str = 'gelsight_tooltip') -> dict | None:
        """Latest parent -> child transform as a JSON-ready dict, or None."""
        try:
            tf = self.tf_buffer.lookup_transform(
                parent, child, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"TF lookup {parent} <- {child} failed: {e}")
            return None
        t, r = tf.transform.translation, tf.transform.rotation
        return {
            'frame_id': tf.header.frame_id,
            'child_frame_id': tf.child_frame_id,
            'stamp': {'sec': tf.header.stamp.sec, 'nanosec': tf.header.stamp.nanosec},
            'translation': {'x': t.x, 'y': t.y, 'z': t.z},
            'rotation': {'x': r.x, 'y': r.y, 'z': r.z, 'w': r.w},
        }

    #### Movement
    def move_z(self, speed):
        touch_io.move_z(self, speed, self.frame_id)

    def stop(self):
        touch_io.stop(self, self.frame_id)

    def calibrate_sensor(self, calibration_range=100):
        touch_io.calibrate_sensor(self, calibration_range)

    def run_action(self, action_client: ActionClient, goal_msg):
        result = action_client.send_goal(goal_msg)
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(result.result.message)
            exit(1)

    def tooltip_goal(self, x: float, y: float, orientation: Quaternion) -> PoseGoal.Goal:
        goal = PoseGoal.Goal()
        goal.target_pose = Pose(position=Point(x=x, y=y, z=self.safe_z), orientation=orientation)
        goal.velocity_scaling = 0.1
        goal.acceleration_scaling = 0.1
        goal.frame_id = self.frame_id
        goal.target_id = "gelsight_tooltip"
        goal.method = "PTP"
        return goal

    def descend_until_contact(self, start_z: float | None) -> str:
        """Servo down until force_max holds for `trip_samples` readings.
        Returns 'contact', or 'no_contact' if max_descent_m is exceeded."""
        if self.mock:
            for _ in range(1000):
                self.move_z(self.descent_speed)
                time.sleep(0.005)
            return 'contact'
        streak = 0
        ticks = 0
        while True:
            self.move_z(self.descent_speed)
            streak = streak + 1 if self.force >= self.force_max else 0
            if streak >= self.trip_samples:
                self.stop()
                return 'contact'
            ticks += 1
            if start_z is not None and streak == 0 and ticks % 50 == 0:  # ~10 Hz
                now = self.lookup('world')
                if now and start_z - now['translation']['z'] > self.max_descent_m:
                    self.stop()
                    return 'no_contact'
            time.sleep(0.002)

    def touch_and_capture(
        self, capture_index: int, row: int, col: int, x: float, y: float,
        orientation: Quaternion,
    ) -> dict:
        """Touch plate-frame (x, y), capture the gelsight, write the pose JSON.
        Retries when contact registers before `min_descent_m` of travel.
        Returns a timing/status record for timing.json."""
        t_start = time.time()
        goal = self.tooltip_goal(x, y, orientation)
        for attempt in range(1, self.max_attempts + 1):
            self.run_action(self.pose_goal_client, goal)
            time.sleep(self.settle_s)
            self.calibrate_sensor()
            start = self.lookup('world')
            start_z = start['translation']['z'] if start else None

            status = self.descend_until_contact(start_z)
            contact_world = self.lookup('world')
            contact_plate = self.lookup(self.frame_id)
            force_at_stop = float(self.force)
            if status == 'no_contact':
                self.get_logger().error(
                    f"touch {capture_index}: no contact within {self.max_descent_m * 1000:.0f} mm; skipping"
                )
                self.run_action(self.pose_goal_client, goal)
                return {'capture_index': capture_index, 'row': row, 'col': col,
                        'start_unix': t_start, 'duration_s': time.time() - t_start,
                        'attempts': attempt, 'captured': False, 'valid_contact': False}

            descent = None
            if start_z is not None and contact_world is not None:
                descent = start_z - contact_world['translation']['z']
            if self.mock or descent is None or descent >= self.min_descent_m:
                break
            self.get_logger().warn(
                f"touch {capture_index}: force tripped after only {descent * 1000:.2f} mm "
                f"(attempt {attempt}/{self.max_attempts}); retrying"
            )
            self.run_action(self.pose_goal_client, goal)
        else:
            self.get_logger().error(
                f"touch {capture_index}: no valid contact after {self.max_attempts} attempts; "
                "capturing anyway, flagged in pose JSON"
            )

        if contact_world is not None:
            cz = contact_world['translation']['z']
            self.get_logger().info(
                f"touch {capture_index} contact: world z={cz:.4f} m, "
                f"descent={descent * 1000 if descent is not None else float('nan'):.2f} mm, "
                f"F={force_at_stop:.2f} N"
            )

        time.sleep(2.0)
        self.capture_stem = f"gelsight_{capture_index:04d}"
        self.gelsight_capture_image = True
        while self.gelsight_capture_image:
            time.sleep(0.1)

        # World-frame pose keeps the keys stitch.py reads; the rest is extra.
        pose = dict(contact_world) if contact_world else {}
        valid = bool(descent is not None and descent >= self.min_descent_m) or self.mock
        pose.update({
            'plate_pose': contact_plate,
            'grid': {'capture_index': capture_index, 'row': row, 'col': col},
            'target_plate_m': {'x': x, 'y': y, 'z': self.safe_z},
            'start_z_world_m': start_z,
            'descent_m': descent,
            'valid_contact': valid,
            'attempts': attempt,
            'force_threshold_N': self.force_max,
            'force_at_stop_N': force_at_stop,
        })
        with open(os.path.join(self.capture_dir, f"{self.capture_stem}_pose.json"), 'w') as f:
            json.dump(pose, f, indent=2)

        self.run_action(self.pose_goal_client, goal)
        return {'capture_index': capture_index, 'row': row, 'col': col,
                'start_unix': t_start, 'duration_s': time.time() - t_start,
                'attempts': attempt, 'captured': True, 'valid_contact': valid,
                'descent_m': descent}

    def calibrate_baseline(self) -> None:
        self.get_logger().info("Capturing gelsight baseline frame for calibration...")
        self.gelsight_calibrate = True
        while self.gelsight_calibrate:
            time.sleep(0.1)

    def run_scan(self, overlap: float, orientation: Quaternion) -> dict:
        """Scan the area once at `overlap` into its own run directory, stitch
        it, and return the run summary."""
        run_dir = os.path.join(self.sweep_dir, f"overlap_{overlap:.2f}")
        os.makedirs(run_dir, exist_ok=True)
        self.capture_dir = run_dir

        keep = self.height_mapper.keep_fraction
        fov = (GELSIGHT_FULL_WIDTH_M * keep, GELSIGHT_FULL_HEIGHT_M * keep)
        plan = plan_grid(self.area_m, fov, overlap)
        nx, ny = plan['touch_x'], plan['touch_y']
        sx, sy = plan['step_m']
        meta = {
            'script': 'motion_collect_overlap',
            'overlap': overlap,
            'area_m': list(self.area_m), 'center_m': list(self.center),
            'fov_m': list(fov), **plan,
            'frame_id': self.frame_id, 'safe_z': self.safe_z,
            'force_max': self.force_max, 'min_descent_m': self.min_descent_m,
            'capture_order': 'files row-major: k -> row (k-1)//touch_y, col (k-1)%touch_y; '
                             'row 0 = max plate x, col 0 = max plate y; touched serpentine',
        }
        meta_path = os.path.join(run_dir, 'session.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        if self.rebaseline_each_run:
            self.calibrate_baseline()

        n = nx * ny
        self.get_logger().info(
            f"[overlap {overlap:.2f}] {nx}x{ny} = {n} touches, step "
            f"{sx * 1000:.2f} x {sy * 1000:.2f} mm, covers "
            f"{plan['coverage_m'][0] * 1000:.1f} x {plan['coverage_m'][1] * 1000:.1f} mm -> {run_dir}"
        )
        records = []
        t0 = time.time()
        done = 0
        for row in range(nx):
            x = self.center[0] + ((nx - 1) / 2.0 - row) * sx
            cols = range(ny) if row % 2 == 0 else range(ny - 1, -1, -1)
            for col in cols:
                y = self.center[1] + ((ny - 1) / 2.0 - col) * sy
                k = row * ny + col + 1
                done += 1
                elapsed = time.time() - t0
                eta = elapsed / (done - 1) * (n - done + 1) if done > 1 else float('nan')
                self.get_logger().info(
                    f"[overlap {overlap:.2f}] touch {done}/{n} (k={k}, row {row}, col {col}) "
                    f"-> plate ({x:.4f}, {y:.4f}); elapsed {elapsed / 60:.1f} min, eta {eta / 60:.1f} min"
                )
                records.append(self.touch_and_capture(k, row, col, x, y, orientation))
        total = time.time() - t0

        durations = [r['duration_s'] for r in records]
        meta['timing'] = {
            'total_s': total,
            'mean_touch_s': float(np.mean(durations)) if durations else None,
            'captured': sum(r['captured'] for r in records),
            'valid_contact_frac': float(np.mean([r['valid_contact'] for r in records])) if records else None,
            'retried_touches': sum(r['attempts'] > 1 for r in records),
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(run_dir, 'timing.json'), 'w') as f:
            json.dump(records, f, indent=2)
        self.get_logger().info(
            f"[overlap {overlap:.2f}] scan took {total / 60:.1f} min "
            f"({meta['timing']['mean_touch_s']:.1f} s/touch); stitching..."
        )

        stitch_session_solved(
            session_dir=run_dir, touch_x=nx, touch_y=ny, touch_overlap=overlap,
        )
        return {'overlap': overlap, 'dir': os.path.basename(run_dir),
                'touches': n, 'grid': [nx, ny], **meta['timing']}


def main(args=None):
    rclpy.init(args=args)
    node = MotionCollectOverlap()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.calibrate_baseline()

        request = Home.Request()
        request.speed = 0.1
        node.home_client.call(request)

        q = quaternion_from_euler(radians(0.0), radians(0.0), radians(0.0))
        orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        keep = node.height_mapper.keep_fraction
        fov = (GELSIGHT_FULL_WIDTH_M * keep, GELSIGHT_FULL_HEIGHT_M * keep)
        sweep = {
            'area_m': list(node.area_m), 'center_m': list(node.center),
            'overlaps': node.overlaps, 'runs': [],
        }
        for o in node.overlaps:
            p = plan_grid(node.area_m, fov, o)
            node.get_logger().info(
                f"plan: overlap {o:.2f} -> {p['touch_x']}x{p['touch_y']} = "
                f"{p['touch_x'] * p['touch_y']} touches"
            )
        sweep_path = os.path.join(node.sweep_dir, 'sweep.json')

        for o in node.overlaps:
            sweep['runs'].append(node.run_scan(o, orientation))
            with open(sweep_path, 'w') as f:
                json.dump(sweep, f, indent=2)

        compare_overlap_runs(node.sweep_dir)

        request = Home.Request()
        request.speed = 0.1
        node.home_client.call(request)

    except KeyboardInterrupt:
        node.get_logger().info("Script interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join()


if __name__ == '__main__':
    main()
