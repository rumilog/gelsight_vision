"""Shared ROS 2 base node for the bead nodes (touch_bead_stations, calib_*).

Touch mechanics are carried over from motion_collect_overlap.touch_and_capture
(retry on early force trips, contact pose read with the arm at rest, GelSight
stall handling) with three additions:

* tooltip yaw per touch (quaternion_from_euler(0, 0, yaw)),
* two-speed descent: `descent_speed` down to plate-frame tooltip z = `slow_z`,
  then `slow_descent_speed` until contact (cuts the 3 N -> ~4.4 N overshoot),
* D555 color capture that also records TF plate -> color optical frame, so
  pixels can be mapped to the plate without the camera_focus assumption.

Mock: `mock:=True` is the mock arm (no force stop), exactly like the other
nodes; pair with `use_gelsight:=False` when no sensor is attached, and with
`image_path:=.../color.png` to plan from a saved snapshot instead of the camera.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from math import degrees, radians

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.action.client import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from robot_manager_interfaces.action import PoseGoal
from robot_manager_interfaces.srv import Home
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import quaternion_from_euler

from gelsight_vision.bead.camera_model import CameraModel, compare_poses, pose_from_plane_points
from gelsight_vision.gelsight.height_map import HeightMapper, save_height_preview
from gelsight_vision.pipeline import touch_io

CONTACT_POSE_SETTLE_S = 0.5


def yaw_quaternion(yaw_rad: float) -> Quaternion:
    q = quaternion_from_euler(0.0, 0.0, yaw_rad)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


class BeadRigNode(Node):
    """Camera capture + yawed GelSight touches in the plate frame."""

    def __init__(self, node_name: str, capture_subdir: str):
        super().__init__(node_name)
        p = self.declare_parameter
        p('ns', 'ur20')
        p('frame_id', 'plate')
        p('mock', False)
        p('use_gelsight', True)
        p('gelsight_topic', '/tool_side_gelsight/image_raw')
        # Fraction of each GelSight axis kept for the depth map (0.75 -> 14.9 mm,
        # 480 px, 0.031 mm/px, matching tools/reprocess_overlap_scan.py).
        p('gel_keep', 0.75)
        p('force_max', 3.0)
        p('safe_z', 0.02)
        p('descent_speed', -0.020)
        # Plate-frame tooltip Z below which the descent slows down; 0 disables.
        # Plate contact is at tooltip z ~ 6-7.5 mm, as-welded beads add ~2-4 mm.
        p('slow_z', 0.014)
        p('slow_descent_speed', -0.003)
        p('min_descent_m', 0.001)
        p('max_descent_m', 0.035)
        p('max_attempts', 3)
        p('stall_s', 5.0)
        p('trip_samples', 3)
        p('settle_s', 0.5)
        # Camera: camera_focus target (plate frame, m) and stream namespace.
        p('realsense_ns', '/tool_side_cam')
        p('camera_x', 0.075)
        p('camera_y', 0.075)
        p('standoff_z', 0.30)
        p('realsense_timeout_s', 5.0)
        # Offline: plan from a saved color.png (camera_info.json next to it).
        p('image_path', '')
        # Camera pose: 'tf' (plate -> color optical TF), 'corners' (PnP on
        # plate_corners_px <-> plate_corners_mm), or 'auto' (tf, else corners).
        p('camera_pose_source', 'auto')
        p('plate_corners_px', [0.0])      # u1, v1, u2, v2, ... (>= 4 points)
        p('plate_corners_mm', [0.0])      # x1, y1, x2, y2, ... same order
        p('plate_x_mm', 150.0)
        p('plate_y_mm', 150.0)
        # Refuse targets further than this outside the plate, and yaws beyond this.
        p('plate_bounds_margin_mm', 5.0)
        p('max_abs_yaw_deg', 100.0)
        p('show_preview', True)
        p('confirm', True)               # ask on the terminal before touching
        p('capture_root', f'/home/ppak/ros2_ws/src/gelsight_vision/{capture_subdir}')

        g = self.get_parameter
        self.ns = str(g('ns').value)
        self.frame_id = str(g('frame_id').value)
        self.mock = bool(g('mock').value)
        self.use_gelsight = bool(g('use_gelsight').value)
        self.gelsight_topic = str(g('gelsight_topic').value)
        self.gel_keep = float(g('gel_keep').value)
        self.force_max = float(g('force_max').value)
        self.safe_z = float(g('safe_z').value)
        self.descent_speed = float(g('descent_speed').value)
        self.slow_z = float(g('slow_z').value)
        self.slow_descent_speed = float(g('slow_descent_speed').value)
        self.min_descent_m = float(g('min_descent_m').value)
        self.max_descent_m = float(g('max_descent_m').value)
        self.max_attempts = int(g('max_attempts').value)
        self.stall_s = float(g('stall_s').value)
        self.trip_samples = int(g('trip_samples').value)
        self.settle_s = float(g('settle_s').value)
        self.realsense_ns = str(g('realsense_ns').value).rstrip('/')
        self.camera_xy = (float(g('camera_x').value), float(g('camera_y').value))
        self.standoff_z = float(g('standoff_z').value)
        self.realsense_timeout_s = float(g('realsense_timeout_s').value)
        self.image_path = str(g('image_path').value)
        self.camera_pose_source = str(g('camera_pose_source').value)
        self.corners_px = [float(v) for v in g('plate_corners_px').value]
        self.corners_mm = [float(v) for v in g('plate_corners_mm').value]
        self.plate_mm = (float(g('plate_x_mm').value), float(g('plate_y_mm').value))
        self.bounds_margin_mm = float(g('plate_bounds_margin_mm').value)
        self.max_abs_yaw = radians(float(g('max_abs_yaw_deg').value))
        self.show_preview = bool(g('show_preview').value)
        self.confirm = bool(g('confirm').value)

        session = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.capture_dir = os.path.join(str(g('capture_root').value), session)
        os.makedirs(self.capture_dir, exist_ok=True)

        self.cv_bridge = CvBridge()
        border = (1.0 - self.gel_keep) / 2.0
        self.height_mapper = HeightMapper(width=int(round(640 * self.gel_keep)),
                                          border_fraction=border, preserve_aspect=True)
        self.gelsight_calibrate = False
        self.gelsight_capture_image = False
        self.capture_stem: str | None = None
        self.gelsight_last_frame_t = 0.0
        self.baseline_raw: np.ndarray | None = None
        self.last_raw: np.ndarray | None = None
        self.last_depth: np.ndarray | None = None

        self.pose_goal_client = ActionClient(self, PoseGoal, self.ns + '/pose_goal')
        self.pose_goal_client.wait_for_server()
        self.home_client = self.create_client(Home, self.ns + '/home')
        while not self.home_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service: ' + self.ns + '/home')
        touch_io.setup_force(self, self.ns)
        touch_io.setup_twist(self, self.ns)
        touch_io.setup_servo(self, self.ns)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        if self.use_gelsight:
            self.gelsight_subscriber = self.create_subscription(
                Image, self.gelsight_topic, self.gelsight_listener_callback, 10)
        self.rs_latest: dict = {}

    # ------------------------------------------------------------------ io
    def out(self, name: str) -> str:
        return os.path.join(self.capture_dir, name)

    def write_json(self, name: str, obj) -> str:
        path = self.out(name)
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)
        return path

    def params_dict(self) -> dict:
        return {name: prm.value for name, prm in self._parameters.items()}

    # ------------------------------------------------------------------ gelsight
    def gelsight_listener_callback(self, msg):
        self.gelsight_last_frame_t = time.time()
        if not (self.gelsight_calibrate or self.gelsight_capture_image):
            return
        frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.gelsight_calibrate:
            cv2.imwrite(self.out('gelsight_baseline_raw.png'), frame)
            cv2.imwrite(self.out('gelsight_baseline.png'), self.height_mapper.crop_native(frame))
            self.height_mapper.calibrate(frame)
            self.baseline_raw = frame.copy()
            self.gelsight_calibrate = False
            return
        stem = self.out(self.capture_stem)
        cv2.imwrite(f'{stem}_raw.png', frame)
        cv2.imwrite(f'{stem}.png', self.height_mapper.crop_native(frame))
        height = self.height_mapper.image_to_height_tensor(frame).numpy()
        np.save(f'{stem}_depth.npy', height)
        save_height_preview(height, f'{stem}_depth_preview.png')
        self.last_raw, self.last_depth = frame.copy(), height
        self.gelsight_capture_image = False

    def _wait_flag(self, attr: str) -> None:
        t0, next_warn = time.time(), 10.0
        while getattr(self, attr):
            if time.time() - t0 >= next_warn:
                self.get_logger().warn(
                    f'No GelSight frame for {time.time() - self.gelsight_last_frame_t:.0f} s on '
                    f'{self.gelsight_topic}; check the USB link / driver on the camera PC.')
                next_warn += 10.0
            time.sleep(0.1)

    def wait_for_gelsight_stream(self) -> None:
        if not self.use_gelsight:
            return
        t0, next_warn = time.time(), 1.0
        while time.time() - self.gelsight_last_frame_t > 1.0:
            if time.time() - t0 >= next_warn:
                self.get_logger().warn(f'Waiting for GelSight frames on {self.gelsight_topic}...')
                next_warn += 10.0
            time.sleep(0.1)

    def calibrate_baseline(self) -> None:
        if not self.use_gelsight:
            return
        self.get_logger().info('Capturing GelSight baseline (gel must not be touching anything)')
        self.gelsight_calibrate = True
        self._wait_flag('gelsight_calibrate')

    # ------------------------------------------------------------------ tf / motion
    def lookup(self, parent: str, child: str = 'gelsight_tooltip') -> dict | None:
        try:
            tf = self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time(),
                                                 timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'TF lookup {parent} <- {child} failed: {e}')
            return None
        t, r = tf.transform.translation, tf.transform.rotation
        return {'frame_id': tf.header.frame_id, 'child_frame_id': tf.child_frame_id,
                'stamp': {'sec': tf.header.stamp.sec, 'nanosec': tf.header.stamp.nanosec},
                'translation': {'x': t.x, 'y': t.y, 'z': t.z},
                'rotation': {'x': r.x, 'y': r.y, 'z': r.z, 'w': r.w}}

    def move_z(self, speed):
        touch_io.move_z(self, speed, self.frame_id)

    def stop(self):
        touch_io.stop(self, self.frame_id)

    def calibrate_sensor(self, calibration_range=100):
        touch_io.calibrate_sensor(self, calibration_range)

    def run_action(self, client: ActionClient, goal_msg):
        result = client.send_goal(goal_msg)
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f'pose goal failed: {result.result.message}')

    def home(self) -> None:
        req = Home.Request()
        req.speed = 0.1
        self.home_client.call(req)

    def goal(self, target_id: str, x: float, y: float, z: float, yaw_rad: float) -> PoseGoal.Goal:
        goal = PoseGoal.Goal()
        goal.target_pose = Pose(position=Point(x=x, y=y, z=z), orientation=yaw_quaternion(yaw_rad))
        goal.velocity_scaling = 0.1
        goal.acceleration_scaling = 0.1
        goal.frame_id = self.frame_id
        goal.target_id = target_id
        goal.method = 'PTP'
        return goal

    def check_target(self, x_mm: float, y_mm: float, yaw_rad: float) -> None:
        m = self.bounds_margin_mm
        if not (-m <= x_mm <= self.plate_mm[0] + m and -m <= y_mm <= self.plate_mm[1] + m):
            raise ValueError(f'target ({x_mm:.1f}, {y_mm:.1f}) mm is outside the plate '
                             f'{self.plate_mm} (+{m} mm); refusing to touch')
        if abs(yaw_rad) > self.max_abs_yaw + 1e-9:
            raise ValueError(f'yaw {degrees(yaw_rad):.1f} deg exceeds max_abs_yaw_deg')

    def _plate_z(self) -> float | None:
        now = self.lookup(self.frame_id)
        return None if now is None else now['translation']['z']

    def descend_until_contact(self, start_z: float | None) -> str:
        """Servo down (fast, then slow below slow_z) until force_max holds for
        trip_samples readings. Returns 'contact' or 'no_contact'."""
        if self.mock:
            # mock arm: no force sensor; descend to roughly where the plate would be
            # (tooltip z ~ 7 mm) instead of servoing 100 mm into the table.
            n = int(max(0.0, self.safe_z - 0.007) / abs(self.descent_speed) / 0.005)
            for _ in range(n):
                self.move_z(self.descent_speed)
                time.sleep(0.005)
            self.stop()
            return 'contact'
        streak = ticks = 0
        speed = self.descent_speed
        last_z, last_move_t = start_z, time.time()
        while True:
            self.move_z(speed)
            streak = streak + 1 if self.force >= self.force_max else 0
            if streak >= self.trip_samples:
                self.stop()
                return 'contact'
            ticks += 1
            if start_z is not None and streak == 0 and ticks % 50 == 0:  # ~10 Hz
                z = self._plate_z()
                if z is not None:
                    if self.slow_z > 0 and z < self.slow_z and speed != self.slow_descent_speed:
                        speed = self.slow_descent_speed
                    if start_z - z > self.max_descent_m:
                        self.stop()
                        return 'no_contact'
                    if abs(z - last_z) > 0.0002:
                        last_z, last_move_t = z, time.time()
                    elif time.time() - last_move_t > self.stall_s + (
                            0 if speed == self.descent_speed else 3 * self.stall_s):
                        self.stop()
                        raise RuntimeError(
                            f'tooltip has not moved in {self.stall_s:.0f} s (z={z:.4f} m, '
                            f'F={self.force:.2f} N); check /{self.ns}/servo_node/status')
            time.sleep(0.002)

    def touch_at(self, stem: str, x_mm: float, y_mm: float, yaw_rad: float,
                 meta: dict | None = None) -> dict:
        """Touch tooltip at plate (x_mm, y_mm) with yaw, capture, write <stem>_pose.json.

        Returns the pose record (also saved), incl. contact_z_mm in the plate frame.
        """
        self.check_target(x_mm, y_mm, yaw_rad)
        x, y = x_mm / 1000.0, y_mm / 1000.0
        goal = self.goal('gelsight_tooltip', x, y, self.safe_z, yaw_rad)
        t0 = time.time()
        contact_world = contact_plate = None
        descent = None
        status = 'contact'
        force_at_stop = float('nan')
        for attempt in range(1, self.max_attempts + 1):
            self.run_action(self.pose_goal_client, goal)
            self.wait_for_gelsight_stream()
            time.sleep(self.settle_s)
            self.calibrate_sensor()
            start = self.lookup(self.frame_id)
            start_z = start['translation']['z'] if start else None
            status = self.descend_until_contact(start_z)
            contact_world, contact_plate = self.lookup('world'), self.lookup(self.frame_id)
            force_at_stop = float(self.force)
            if status == 'no_contact':
                self.get_logger().error(f'{stem}: no contact within max_descent_m; skipping')
                break
            if start_z is not None and contact_plate is not None:
                descent = start_z - contact_plate['translation']['z']
            if self.mock or descent is None or descent >= self.min_descent_m:
                break
            self.get_logger().warn(f'{stem}: force tripped after {descent * 1000:.2f} mm '
                                   f'(attempt {attempt}/{self.max_attempts}); retrying')
            self.run_action(self.pose_goal_client, goal)

        at_stop = {'world': contact_world, 'plate': contact_plate}
        self.capture_stem = stem
        if status == 'contact':
            time.sleep(2.0 if self.use_gelsight else CONTACT_POSE_SETTLE_S)
            sw, sp = self.lookup('world'), self.lookup(self.frame_id)
            if sp is not None:
                contact_world, contact_plate = sw, sp
                if start_z is not None:
                    descent = start_z - sp['translation']['z']
            if self.use_gelsight:
                self.gelsight_capture_image = True
                self._wait_flag('gelsight_capture_image')

        valid = status == 'contact' and (self.mock or (descent is not None
                                                       and descent >= self.min_descent_m))
        rec = dict(contact_world) if contact_world else {}
        rec.update({
            'stem': stem,
            'plate_pose': contact_plate,
            'pose_at_stop': at_stop,
            'target_plate_mm': [x_mm, y_mm],
            'yaw_rad': yaw_rad, 'yaw_deg': degrees(yaw_rad),
            'start_z_plate_m': start_z, 'descent_m': descent,
            'contact_z_mm': (contact_plate['translation']['z'] * 1000.0
                             if (contact_plate and status == 'contact') else None),
            'valid_contact': bool(valid), 'attempts': attempt,
            'force_threshold_N': self.force_max, 'force_at_stop_N': force_at_stop,
            'duration_s': time.time() - t0,
            'meta': meta or {},
        })
        self.write_json(f'{stem}_pose.json', rec)
        self.run_action(self.pose_goal_client, goal)
        z = rec['contact_z_mm']
        self.get_logger().info(f'{stem}: contact z = {z if z is None else round(z, 3)} mm, '
                               f'F = {force_at_stop:.2f} N, valid = {valid}')
        return rec

    # ------------------------------------------------------------------ camera
    def _latch(self, key):
        def _cb(msg):
            self.rs_latest[key] = msg
        return _cb

    def _grab_one(self, key, typ, suffix) -> object | None:
        sub = self.create_subscription(typ, f'{self.realsense_ns}/{suffix}', self._latch(key),
                                       qos_profile_sensor_data if typ is Image else 10)
        deadline = time.time() + self.realsense_timeout_s
        while time.time() < deadline and key not in self.rs_latest:
            time.sleep(0.02)
        self.destroy_subscription(sub)
        return self.rs_latest.get(key)

    def acquire_image(self) -> tuple[np.ndarray, CameraModel, dict]:
        """Color image + intrinsics + camera-pose info, live or from image_path."""
        info: dict = {}
        if self.image_path:
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(self.image_path)
            folder = os.path.dirname(os.path.abspath(self.image_path))
            cam = CameraModel.from_camera_info_json(os.path.join(folder, 'camera_info.json'))
            tf_path = os.path.join(folder, 'camera_pose.json')
            if os.path.exists(tf_path):
                info = json.load(open(tf_path))
            info['source_image'] = self.image_path
        else:
            goal = self.goal('camera_focus', self.camera_xy[0], self.camera_xy[1],
                             self.standoff_z, 0.0)
            self.get_logger().info(f'Moving camera_focus to plate {self.camera_xy} m, '
                                   f'z={self.standoff_z} m')
            self.run_action(self.pose_goal_client, goal)
            time.sleep(2.0)
            self.rs_latest.clear()
            msg = self._grab_one('color', Image, 'color/image_raw')
            ci = self._grab_one('color_info', CameraInfo, 'color/camera_info')
            if msg is None or ci is None:
                raise RuntimeError(f'no color image / camera_info on {self.realsense_ns}')
            img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cam = CameraModel.from_intrinsics(ci.k[0], ci.k[4], ci.k[2], ci.k[5], list(ci.d),
                                              ci.width, ci.height)
            optical = ci.header.frame_id
            info = {'optical_frame': optical,
                    'plate_to_optical': self.lookup(self.frame_id, optical),
                    'camera_focus_plate': self.lookup(self.frame_id, 'camera_focus'),
                    'image_stamp': {'sec': msg.header.stamp.sec,
                                    'nanosec': msg.header.stamp.nanosec}}
            cv2.imwrite(self.out('color.png'), img)
            with open(self.out('camera_info.json'), 'w') as f:
                json.dump({'color': {'frame_id': optical, 'width': ci.width,
                                     'height': ci.height, 'fx': ci.k[0], 'fy': ci.k[4],
                                     'cx': ci.k[2], 'cy': ci.k[5],
                                     'distortion_model': ci.distortion_model,
                                     'd': list(ci.d)}}, f, indent=2)
            self.write_json('camera_pose.json', info)
        return img, self.resolve_camera_pose(cam, info), info

    def resolve_camera_pose(self, cam: CameraModel, info: dict) -> CameraModel:
        tf_cam = corner_cam = None
        tf = info.get('plate_to_optical')
        if tf:
            t, r = tf['translation'], tf['rotation']
            tf_cam = cam.with_tf([t['x'], t['y'], t['z']], [r['x'], r['y'], r['z'], r['w']])
        if len(self.corners_px) >= 8 and len(self.corners_px) == len(self.corners_mm):
            uv = np.asarray(self.corners_px, float).reshape(-1, 2)
            xy = np.asarray(self.corners_mm, float).reshape(-1, 2)
            R, C = pose_from_plane_points(cam, uv, xy)
            corner_cam = cam.with_pose(R, C, 'corners')
        if tf_cam is not None and corner_cam is not None:
            cmp = compare_poses(tf_cam, corner_cam)
            self.get_logger().info(f'camera pose: TF vs plate corners -> {cmp}')
            self.write_json('camera_pose_compare.json', cmp)
        src = self.camera_pose_source
        chosen = {'tf': tf_cam, 'corners': corner_cam}.get(src) if src != 'auto' else (
            tf_cam if tf_cam is not None else corner_cam)
        if chosen is None:
            raise RuntimeError(
                f'camera_pose_source={src!r} unavailable: need a plate -> optical TF (live '
                'capture) or plate_corners_px + plate_corners_mm parameters')
        self.get_logger().info(f'camera pose from {chosen.pose_source}: '
                               f'centre {np.round(chosen.C_plate_mm, 1)} mm, '
                               f'{chosen.mm_per_px_at():.3f} mm/px at the plate')
        self.write_json('camera_model.json', chosen.summary())
        return chosen

    # ------------------------------------------------------------------ ui
    def preview(self, img: np.ndarray, title: str) -> None:
        if not self.show_preview:
            return
        # Poll instead of waitKey(0): Python only runs the Ctrl+C handler between
        # bytecodes, so a blocking waitKey(0) made the node ignore Ctrl+C until a
        # key was pressed in the (possibly hidden) window. Closing the window also
        # continues.
        self.get_logger().info(f'preview window "{title}": press any key in it (or close it) '
                               'to continue')
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.imshow(title, img)
            while rclpy.ok():
                if cv2.waitKey(50) != -1:
                    break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyWindow(title)
            cv2.waitKey(1)
        except cv2.error as e:  # headless
            self.get_logger().warn(f'preview unavailable: {e}')
        if not rclpy.ok():
            raise KeyboardInterrupt

    def ask(self, question: str) -> bool:
        if not self.confirm:
            return True
        try:
            return input(f'{question} [y/N] ').strip().lower() in ('y', 'yes')
        except EOFError:
            self.get_logger().error('no terminal for confirm; run with confirm:=False to skip')
            return False

    def click_points(self, img: np.ndarray, title: str, max_points: int = 10) -> list:
        """Left-click points on the image; Enter/Space when done, Backspace to undo."""
        pts: list = []
        disp = img.copy()

        def on_mouse(ev, u, v, *_):
            if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < max_points:
                pts.append((float(u), float(v)))

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(title, on_mouse)
        while True:
            disp = img.copy()
            for i, (u, v) in enumerate(pts):
                cv2.drawMarker(disp, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
                cv2.putText(disp, str(i + 1), (int(u) + 6, int(v) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow(title, disp)
            k = cv2.waitKey(30) & 0xFF
            if k in (13, 10, 32):
                break
            if k == 8 and pts:
                pts.pop()
        cv2.destroyWindow(title)
        return pts
