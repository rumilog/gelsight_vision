import os
import cv2
import numpy as np
import rclpy
import threading
import time

from datetime import datetime
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Point, Quaternion
from math import radians
from rclpy.action.client import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from robot_manager_interfaces.action import PoseGoal
from robot_manager_interfaces.srv import Home
from tf_transformations import quaternion_from_euler

from gelsight_vision.gelsight.height_map import HeightMapper, save_height_preview
from gelsight_vision.gelsight.stitch import stitch_session
from gelsight_vision.pipeline import touch_io
from gelsight_vision.pipeline.crop_d555 import crop_rgb_stitched
from gelsight_vision.pipeline.visualize import (
    draw_super_resolution_panel,
    draw_weld_patch_overlay,
    show,
)
from gelsight_vision.tools.find_weld_patch import find_weld_patch, score_patches
from gelsight_vision.tools.super_resolution import super_resolution
from gelsight_vision.tools.touch_patches import touch_patches


class PipelineVisuoTactile(Node):
    def __init__(self):
        super().__init__('pipeline_visuo_tactile')

        _ = self.declare_parameter('ns', 'ur20')
        _ = self.declare_parameter('frame_id', 'plate')
        _ = self.declare_parameter('force_max', 4.0)
        _ = self.declare_parameter('mock', False)
        _ = self.declare_parameter('weld_threshold', 0.5)
        _ = self.declare_parameter('show_preview', True)
        _ = self.declare_parameter(
            'capture_root',
            '/home/ppak/ros2_ws/src/gelsight_vision/captures_visuo_tactile',
        )

        self.ns: str = str(self.get_parameter('ns').value)
        self.frame_id: str = str(self.get_parameter('frame_id').value)
        self.force_max: float = float(self.get_parameter('force_max').value)
        self.mock: bool = bool(self.get_parameter('mock').value)
        self.weld_threshold: float = float(self.get_parameter('weld_threshold').value)
        self.show_preview: bool = bool(self.get_parameter('show_preview').value)

        self.gelsight_frame = None
        self.gelsight_depth = None  # set by gelsight callback after each touch capture
        self.d555_frame = None

        # Capture image flags
        self.gelsight_capture_image = False
        self.gelsight_calibrate = False
        self.d555_capture_image = False

        # The caller sets this before triggering gelsight_capture_image=True so
        # the callback can name files gelsight_<NN>.png matching the dataset
        # convention (capture_index in 1..100).
        self.current_capture_index = 0

        # Shared session folder for captured frames
        capture_root = str(self.get_parameter('capture_root').value)
        session = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.capture_dir = os.path.join(capture_root, session)
        os.makedirs(self.capture_dir, exist_ok=True)
        self.d555_capture_count = 0

        self.height_mapper = HeightMapper()

        self.pose_goal_client: ActionClient = ActionClient(
            self,
            PoseGoal,
            self.ns + "/" + "pose_goal"
        )
        self.pose_goal_client.wait_for_server()

        self.home_client = self.create_client(Home, self.ns + "/" + "home")
        while not self.home_client.wait_for_service():
            time.sleep(1.0)
            self.get_logger().info("waiting for service: " + self.ns + "/home")

        self.cv_bridge = CvBridge()

        # Gelsight Subscriber (baseline calibration + per-touch capture).
        self.gelsight_subscriber = self.create_subscription(
            Image,
            'gelsight_capture/image',
            self.gelsight_listener_callback,
            10
        )

        # D555 RGB Subscriber
        self.d555_subscriber = self.create_subscription(
            Image,
            'gelsight_vision_d555/color/image_raw',
            self.d555_listener_callback,
            10
        )

        # Force / twist / servo plumbing (shared with motion_collect_sample
        # via gelsight_vision.pipeline.touch_io).
        touch_io.setup_force(self, self.ns)
        touch_io.setup_twist(self, self.ns)
        touch_io.setup_servo(self, self.ns)

    # --- Thin wrappers so gelsight/touch.py keeps calling self.move_z(...) etc.
    def move_z(self, speed):
        touch_io.move_z(self, speed, self.frame_id)

    def stop(self):
        touch_io.stop(self, self.frame_id)

    def calibrate_sensor(self, calibration_range: int = 100):
        touch_io.calibrate_sensor(self, calibration_range)

    def gelsight_listener_callback(self, msg):
        if self.gelsight_calibrate:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            baseline_path = os.path.join(self.capture_dir, "gelsight_baseline.png")
            cv2.imwrite(baseline_path, self.height_mapper.crop_native(frame))
            self.height_mapper.calibrate(frame)
            self.get_logger().info(f"gelsight baseline -> {baseline_path}")
            self.gelsight_calibrate = False
            return
        if self.gelsight_capture_image:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.gelsight_frame = frame
            ci = self.current_capture_index
            stem = f"gelsight_{ci:04d}"
            img_path = os.path.join(self.capture_dir, f"{stem}.png")
            depth_path = os.path.join(self.capture_dir, f"{stem}_depth.npy")
            preview_path = os.path.join(self.capture_dir, f"{stem}_depth_preview.png")
            cv2.imwrite(img_path, self.height_mapper.crop_native(frame))
            height = self.height_mapper.image_to_height_tensor(frame).numpy()
            np.save(depth_path, height)
            save_height_preview(height, preview_path)
            self.gelsight_depth = height
            self.get_logger().info(f"gelsight capture {ci} -> {img_path}")
            self.gelsight_capture_image = False

    def d555_listener_callback(self, msg):
        if self.d555_capture_image:
            self.d555_frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.d555_capture_count += 1
            stem = f"d555_{self.d555_capture_count:04d}"
            color_path = os.path.join(self.capture_dir, f"{stem}.png")
            cv2.imwrite(color_path, self.d555_frame)
            self.get_logger().info(f"d555 captured frame -> {color_path}")
            self.d555_capture_image = False

    def run_action(
            self,
            action_client: ActionClient,
            goal_msg,
            show_progress: bool = False
        ):
        feedback_callback = lambda msg: self.get_logger().info(
            f'Progress: {msg.feedback.progress:.1f}%'
        )

        if show_progress:
            result = action_client.send_goal(
                goal_msg,
                feedback_callback=feedback_callback
            )
        else:
            result = action_client.send_goal(goal_msg)

        status = result.status

        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(result.result.message)
            exit(1)


def main(args=None):
    rclpy.init(args=args)
    node = PipelineVisuoTactile()

    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node, ),
        daemon=True
    )
    spin_thread.start()

    try:
        # Capture an untouched-gel frame to calibrate the HeightMapper baseline.
        node.get_logger().info("Capturing gelsight baseline frame for calibration...")
        node.gelsight_calibrate = True
        while node.gelsight_calibrate:
            time.sleep(0.1)
        node.get_logger().info("Gelsight baseline captured and calibrated.")

        # Send request to home robot arm
        request = Home.Request()
        request.speed = 0.1
        node.home_client.call(request)

        # Move to camera_focus pose for the D555 overview.
        q = quaternion_from_euler(radians(0.0), radians(0.0), radians(0.0))
        orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        goal_msg = PoseGoal.Goal()
        goal_msg.target_pose = Pose(
            position=Point(x=0.0, y=0.0, z=0.0),
            orientation=orientation,
        )
        goal_msg.velocity_scaling = 0.1
        goal_msg.acceleration_scaling = 0.1
        goal_msg.frame_id = node.frame_id
        goal_msg.target_id = "camera_focus"
        goal_msg.method = "PTP"

        node.get_logger().info("Moving Camera Focus")
        node.run_action(node.pose_goal_client, goal_msg)
        node.get_logger().info("Finished Moving Camera Focus")

        # Let the arm settle before grabbing the overview.
        time.sleep(5.0)
        node.d555_capture_image = True
        while node.d555_capture_image:
            time.sleep(0.1)

        # Tool 1: find_weld_patch on the cropped overview.
        rgb_stitched_bgr = crop_rgb_stitched(node.d555_frame)
        stitched_path = os.path.join(node.capture_dir, "rgb_stitched.png")
        cv2.imwrite(stitched_path, rgb_stitched_bgr)
        node.get_logger().info(f"rgb_stitched -> {stitched_path}")

        # Tool 1 scores each per-tile crop with the CNN, then thresholds.
        scores = score_patches(node.d555_frame)
        selected = find_weld_patch(scores, threshold=node.weld_threshold)
        node.get_logger().info(
            f"find_weld_patch(threshold={node.weld_threshold}) -> "
            f"{len(selected)} patches: {selected}"
        )

        overlay = draw_weld_patch_overlay(
            rgb_stitched_bgr, scores, threshold=node.weld_threshold,
        )
        overlay_path = os.path.join(node.capture_dir, "rgb_stitched_weld_patches.png")
        cv2.imwrite(overlay_path, overlay)
        node.get_logger().info(f"weld patch overlay -> {overlay_path}")

        if node.show_preview:
            node.get_logger().info("Showing preview window — press any key to continue.")
            show(overlay, title="find_weld_patch (any key to continue)")

        # Tool 2: touch each selected patch and gather per-tile gelsight depth.
        observed = touch_patches(
            node=node,
            capture_indexes=selected,
            orientation=orientation,
            force_max=node.force_max,
            mock=node.mock,
        )
        node.get_logger().info(
            f"touch_patches -> got depth for {len(observed)}/{len(selected)} patches"
        )

        # Partial gelsight stitch of just the touched tiles -- mirrors the
        # dataset's gelsight_stitched panorama but with NaN/zero fill for
        # untouched cells. Useful for eyeballing what Tool 3 actually saw.
        node.get_logger().info("Stitching touched gelsight captures...")
        stitch_session(session_dir=node.capture_dir, touch_x=10, touch_y=10)

        # Tool 3: kick off diffusion inpainting in a thread so the robot can
        # home in parallel (Tool 3 is the long pole on the 3060: ~3-6 min).
        sr_result: dict = {"depth": None, "error": None}

        def _run_super_resolution() -> None:
            try:
                if not observed:
                    return
                rgb_for_sr = cv2.cvtColor(rgb_stitched_bgr, cv2.COLOR_BGR2RGB)
                sr_result["depth"] = super_resolution(
                    rgb_for_sr, observed, seed=42,
                )
            except Exception as exc:  # noqa: BLE001
                sr_result["error"] = exc

        node.get_logger().info("super_resolution: starting (in background)")
        sr_thread = threading.Thread(target=_run_super_resolution, daemon=True)
        sr_thread.start()

        # Send request to home robot arm (runs alongside Tool 3).
        request = Home.Request()
        request.speed = 0.1
        node.home_client.call(request)

        sr_thread.join()
        if sr_result["error"] is not None:
            node.get_logger().error(
                f"super_resolution failed: {sr_result['error']!r}"
            )
        elif sr_result["depth"] is None:
            node.get_logger().warn(
                "super_resolution skipped (no observed patches from Tool 2)"
            )
        else:
            dense = sr_result["depth"]
            dense_npy = os.path.join(node.capture_dir, "dense_depth.npy")
            dense_png = os.path.join(node.capture_dir, "dense_depth_preview.png")
            composite_png = os.path.join(node.capture_dir, "dense_depth_composite.png")
            np.save(dense_npy, dense)
            save_height_preview(dense, dense_png)
            composite = draw_super_resolution_panel(
                rgb_stitched_bgr, list(observed.keys()), dense,
            )
            cv2.imwrite(composite_png, composite)
            node.get_logger().info(
                f"dense depth -> {dense_npy} "
                f"(range=[{dense.min():.3f}, {dense.max():.3f}])"
            )
            node.get_logger().info(f"dense depth preview -> {dense_png}")
            node.get_logger().info(f"dense depth composite -> {composite_png}")

            if node.show_preview:
                node.get_logger().info(
                    "Showing super_resolution preview — press any key to continue."
                )
                show(composite, title="super_resolution (any key to continue)")

    except KeyboardInterrupt:
        node.get_logger().info("Script interrupted by user.")
    finally:
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass
        # spin_thread is daemon=True; bounded join lets normal exits clean up
        # but stops Ctrl-C from hanging when rclpy.spin is mid-callback.
        spin_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
