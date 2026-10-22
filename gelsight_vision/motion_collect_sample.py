import os
import sys
import cv2
import json
import numpy as np
import rclpy
import threading
import time

# IGEV-Stereo lives outside this ROS package. Match the sys.path layout that
# DeepStereo/source_IGEV.sh sets up so both `from igev_stereo import ...` and
# the internal `from core.update import ...` resolve.
_IGEV_ROOT = "/home/ppak/DeepStereo/IGEV/IGEV-Stereo"
for _p in (_IGEV_ROOT, os.path.join(_IGEV_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from igev_stereo import IGEVStereo
from utils.utils import InputPadder

from datetime import datetime
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Point, Quaternion, TwistStamped, WrenchStamped
from gelsight_vision.gelsight.height_map import HeightMapper, save_height_preview
from gelsight_vision.gelsight.stitch import (
    stitch_session,
    stitch_session_anchored,
    write_anchored_ply,
    write_stitched_ply,
)
from gelsight_vision.pipeline.align_clouds import write_combined_ply
from gelsight_vision.gelsight.touch import touch
from math import radians
from moveit_msgs.srv import ServoCommandType
from rclpy.action.client import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from robot_manager_interfaces.action import PoseGoal
from robot_manager_interfaces.srv import Home
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import quaternion_from_euler

class MotionCollectSample(Node):
    def __init__(self):
        super().__init__('motion_collect_sample')

        # Assigns namespace to ur20
        _ = self.declare_parameter('ns', 'ur20')
        _ = self.declare_parameter('frame_id', 'plate')
        _ = self.declare_parameter('force_max', 4.0)
        _ = self.declare_parameter('mock', False)
        _ = self.declare_parameter('touch_x', 1)
        _ = self.declare_parameter('touch_y', 1)
        _ = self.declare_parameter(
            'capture_root',
            # '/home/ppak/HuggingFace/Datasets/Gelsight-Welds/source/v2',
            '/home/ppak/ros2_ws/src/gelsight_vision/captures',
        )
        # RealSense D5xx publishes /depth/image_rect_raw as 16UC1 in raw device
        # units; multiply by depth_scale (meters per raw unit) to get distance.
        # 0.001 is the standard value for D435/D455/D555 stereo depth.
        _ = self.declare_parameter('d555_depth_scale', 0.001)
        # D555 stereo baseline in meters. Sourced from DeepStereo/camera_ros2.py
        # (~95.03 mm). Override via ROS parameter if your unit differs.
        _ = self.declare_parameter('d555_baseline_m', 0.09503433853387833)
        _ = self.declare_parameter(
            'igev_checkpoint',
            os.path.join(_IGEV_ROOT, 'pretrained_models/middlebury/middlebury.pth'),
        )
        _ = self.declare_parameter('igev_iters', 64)

        self.ns: str = str(self.get_parameter('ns').value)
        self.frame_id: str = str(self.get_parameter('frame_id').value)
        self.force_max: float = float(self.get_parameter('force_max').value)
        self.mock: bool = bool(self.get_parameter('mock').value)
        self.touch_x: int = int(self.get_parameter('touch_x').value)
        self.touch_y: int = int(self.get_parameter('touch_y').value)
        self.d555_depth_scale: float = float(
            self.get_parameter('d555_depth_scale').value
        )
        self.d555_baseline_m: float = float(
            self.get_parameter('d555_baseline_m').value
        )
        # Camera_focus target Z in the plate frame. IGEV middlebury needs
        # disparities under ~150 px — at the D555's 95 mm baseline this means
        # standing off ~25 cm from the sample. Edit here to retune.
        self.standoff_z: float = 0.30
        self.igev_checkpoint: str = str(self.get_parameter('igev_checkpoint').value)
        self.igev_iters: int = int(self.get_parameter('igev_iters').value)
        # Bicubic-upsample the stereo pair before IGEV to gain sub-pixel
        # disparity precision. max_disp scales proportionally (192 * factor);
        # values >1.5 exceed IGEV's trained disparity range so the extended
        # cost-volume region has no training signal — treat 2.0+ as risky.
        self.igev_upsample: float = 2.0

        self.gelsight_frame = None
        self.d555_frame = None
        self.d555_depth_frame = None
        # Latched stereo IR frames + intrinsics (populated by subscribers).
        self.infra1_frame = None
        self.infra2_frame = None
        self.infra_camera_info = None
        # Pose snapshotted at the moment the touch force threshold trips.
        # gelsight_listener_callback drains this into a per-touch pose JSON.
        self.pending_touch_pose: dict | None = None

        # Force values
        self.force = 0.0
        self.offset = 0.0

        # Capture image flags
        self.gelsight_capture_image = False
        self.gelsight_calibrate = False
        self.d555_capture_image = False
        # Counter for stereo/IGEV captures (independent of gelsight/d555 counts).
        self.stereo_capture_count = 0

        # Shared session folder for captured frames
        capture_root = str(self.get_parameter('capture_root').value)
        session = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.capture_dir = os.path.join(capture_root, session)
        os.makedirs(self.capture_dir, exist_ok=True)
        self.gelsight_capture_count = 0
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
            sleep(1.0)
            print("waiting for service")
            continue

        # Force Subscriber
        self.force_subscriber = self.create_subscription(
            WrenchStamped, 
            # self.ns + '/force_torque_sensor_broadcaster/wrench_filtered',
            self.ns + '/wrench',
            self.force_callback, 
            10
        )

        self.cv_bridge = CvBridge()

        # TF listener — used to look up the plate -> camera_focus transform at
        # each stereo capture so we can record where the arm actually was.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Gelsight Subscriber
        self.gelsight_subscriber = self.create_subscription(
            Image,
            'gelsight_capture/image',
            self.gelsight_listener_callback,
            10
        )

        # D555 Subscriber
        self.d555_subscriber = self.create_subscription(
            Image,
            'gelsight_vision_d555/color/image_raw',
            self.d555_listener_callback,
            10
        )

        # D555 Depth Subscriber (keeps latest depth frame so it can be saved
        # alongside the color image when the capture trigger fires).
        self.d555_depth_subscriber = self.create_subscription(
            Image,
            'gelsight_vision_d555/depth/image_rect_raw',
            self.d555_depth_listener_callback,
            10
        )

        # D555 stereo IR subscribers — feed IGEV to estimate a dense metric
        # depth map at capture time. The realsense-ros driver publishes IR as
        # 8-bit grayscale on infra1/infra2/image_rect_raw.
        self.infra1_subscriber = self.create_subscription(
            Image,
            'gelsight_vision_d555/infra1/image_rect_raw',
            self.infra1_listener_callback,
            10,
        )
        self.infra2_subscriber = self.create_subscription(
            Image,
            'gelsight_vision_d555/infra2/image_rect_raw',
            self.infra2_listener_callback,
            10,
        )
        # infra1/camera_info -> fx at the current stream resolution, used to
        # turn IGEV disparity into metric depth without re-scaling.
        self.infra_info_subscriber = self.create_subscription(
            CameraInfo,
            'gelsight_vision_d555/infra1/camera_info',
            self.infra_info_callback,
            10,
        )

        # Twist Publisher
        self.twist_publisher = self.create_publisher(
            TwistStamped, 
            self.ns + '/servo_node/delta_twist_cmds', 
            10
        )

        # Enable Servo
        self.servo_client = self.create_client(ServoCommandType, self.ns + '/servo_node/switch_command_type')
        while not self.servo_client.wait_for_service(timeout_sec=1.0): 
            self.get_logger().info("Waiting for service: " + self.ns + '/servo_node/switch_command_type')
        req = ServoCommandType.Request()
        req.command_type = ServoCommandType.Request.TWIST
        future = self.servo_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        # Load IGEV-Stereo once at startup so per-capture inference is fast
        # (~1-2 s on the 3060). Wrapping in DataParallel matches how the
        # middlebury.pth state dict was saved (module.* keys).
        self.get_logger().info(
            f"Loading IGEV-Stereo checkpoint: {self.igev_checkpoint}"
        )
        self.igev_args = self._make_igev_args()
        _igev_model = torch.nn.DataParallel(
            IGEVStereo(self.igev_args), device_ids=[0],
        )
        _igev_model.load_state_dict(torch.load(self.igev_args.restore_ckpt))
        self.igev_model = _igev_model.module.to('cuda').eval()
        self.get_logger().info("IGEV-Stereo model ready.")

    def gelsight_listener_callback(self, msg):
        if self.gelsight_calibrate:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            baseline_path = os.path.join(self.capture_dir, "gelsight_baseline.png")
            cv2.imwrite(baseline_path, self.height_mapper.crop_native(frame))
            self.height_mapper.calibrate(frame)
            print(f"gelsight baseline -> {baseline_path}")
            self.gelsight_calibrate = False
            return
        if self.gelsight_capture_image:
            self.gelsight_frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.gelsight_capture_count += 1
            stem = f"gelsight_{self.gelsight_capture_count:04d}"
            img_path = os.path.join(self.capture_dir, f"{stem}.png")
            depth_path = os.path.join(self.capture_dir, f"{stem}_depth.npy")
            preview_path = os.path.join(self.capture_dir, f"{stem}_depth_preview.png")
            cropped_frame = self.height_mapper.crop_native(self.gelsight_frame)
            cv2.imwrite(img_path, cropped_frame)
            height = self.height_mapper.image_to_height_tensor(self.gelsight_frame).numpy()
            np.save(depth_path, height)
            save_height_preview(height, preview_path)
            print(f"gelsight captured frame -> {img_path}")
            print(f"gelsight depth -> {depth_path}")
            print(f"gelsight depth preview -> {preview_path}")
            # Drain the pose snapshotted by touch() at the force-stop moment.
            if self.pending_touch_pose is not None:
                pose_path = os.path.join(self.capture_dir, f"{stem}_pose.json")
                with open(pose_path, 'w') as f:
                    json.dump(self.pending_touch_pose, f, indent=2)
                print(f"gelsight touch pose -> {pose_path}")
                self.pending_touch_pose = None
            self.gelsight_capture_image = False

    def d555_listener_callback(self, msg):
        if self.d555_capture_image:
            self.d555_frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.d555_capture_count += 1
            stem = f"d555_{self.d555_capture_count:04d}"
            color_path = os.path.join(self.capture_dir, f"{stem}.png")
            depth_path = os.path.join(self.capture_dir, f"{stem}_depth.png")
            depth_npy_path = os.path.join(self.capture_dir, f"{stem}_depth.npy")
            cv2.imwrite(color_path, self.d555_frame)
            print(f"d555 captured frame -> {color_path}")
            if self.d555_depth_frame is not None:
                cv2.imwrite(depth_path, self.d555_depth_frame)
                np.save(depth_npy_path, self.d555_depth_frame)
                print(f"d555 captured depth -> {depth_path}")
                print(f"d555 captured depth -> {depth_npy_path}")
            else:
                print("d555 depth frame not yet received; depth not saved")
            self.d555_capture_image = False

    def d555_depth_listener_callback(self, msg):
        # CV_16UC1 -> uint16 numpy array; raw values are in device units.
        self.d555_depth_frame = self.cv_bridge.imgmsg_to_cv2(
            msg, desired_encoding="16UC1"
        )

    def d555_distance_at(self, x: int, y: int) -> float:
        """Return the distance in meters at pixel (x, y) of the latest D555
        depth frame, or NaN if no frame has been received or the pixel has
        no valid depth (raw 0)."""
        if self.d555_depth_frame is None:
            return float("nan")
        raw = int(self.d555_depth_frame[y, x])
        if raw == 0:
            return float("nan")
        return raw * self.d555_depth_scale

    #### Stereo IR + IGEV
    def infra1_listener_callback(self, msg):
        self.infra1_frame = self.cv_bridge.imgmsg_to_cv2(
            msg, desired_encoding="mono8",
        )

    def infra2_listener_callback(self, msg):
        self.infra2_frame = self.cv_bridge.imgmsg_to_cv2(
            msg, desired_encoding="mono8",
        )

    def infra_info_callback(self, msg):
        self.infra_camera_info = msg

    def capture_and_compute_depth(self) -> None:
        """Snapshot the latched stereo pair, run IGEV, convert disparity to
        metric depth using D555 baseline + infra1 fx, and save all artifacts
        to the session capture dir."""
        if self.infra1_frame is None or self.infra2_frame is None:
            self.get_logger().warn(
                "Stereo frames not yet received; is the D555 driver publishing "
                "infra1/infra2 image_rect_raw?"
            )
            return

        left = self.infra1_frame.copy()
        right = self.infra2_frame.copy()

        self.stereo_capture_count += 1
        stem = f"stereo_{self.stereo_capture_count:04d}"
        left_path = os.path.join(self.capture_dir, f"{stem}_infra1.png")
        right_path = os.path.join(self.capture_dir, f"{stem}_infra2.png")
        cv2.imwrite(left_path, left)
        cv2.imwrite(right_path, right)
        self.get_logger().info(f"stereo pair -> {left_path}, {right_path}")

        t0 = time.time()
        disparity = self._run_igev(left, right)
        self.get_logger().info(
            f"IGEV inference: {time.time() - t0:.2f}s (disp shape {disparity.shape})"
        )

        if self.infra_camera_info is not None:
            fx = float(self.infra_camera_info.k[0])
        else:
            # Fallback matches DeepStereo/camera.py CAMERA_MATRIX[0][0].
            fx = 647.963989
            self.get_logger().warn(
                "infra1/camera_info not received; using fallback fx=647.964"
            )

        depth = (self.d555_baseline_m * fx) / disparity

        disp_npy = os.path.join(self.capture_dir, f"{stem}_disparity.npy")
        depth_npy = os.path.join(self.capture_dir, f"{stem}_depth.npy")
        np.save(disp_npy, disparity)
        np.save(depth_npy, depth)
        self.get_logger().info(f"disparity -> {disp_npy}")
        self.get_logger().info(f"depth (m) -> {depth_npy}")

        cv2.imwrite(
            os.path.join(self.capture_dir, f"{stem}_disparity_preview.png"),
            self._colormap_preview(disparity),
        )
        cv2.imwrite(
            os.path.join(self.capture_dir, f"{stem}_depth_preview.png"),
            self._colormap_preview(depth),
        )

        self._save_point_cloud(depth, left, stem, fx)
        self._save_camera_pose(stem)

        valid = np.isfinite(depth) & (depth > 0.05) & (depth < 2.0)
        if valid.any():
            self.get_logger().info(
                f"Depth stats: min={depth[valid].min():.3f}m, "
                f"max={depth[valid].max():.3f}m, "
                f"median={np.median(depth[valid]):.3f}m, "
                f"std={depth[valid].std():.4f}m, "
                f"valid={100 * valid.mean():.1f}%"
            )
        else:
            self.get_logger().warn(
                "No depth pixels in [0.05, 2.0] m — surface may be out of range."
            )

    def _run_igev(self, left_mono: np.ndarray, right_mono: np.ndarray) -> np.ndarray:
        # Optional bicubic upsample before IGEV → sub-pixel disparity precision.
        # Output disparity is scaled back to native-pixel units so downstream
        # depth math (uses native fx from camera_info) stays unchanged.
        upsample = self.igev_upsample
        if upsample != 1.0:
            h, w = left_mono.shape
            new_size = (int(round(w * upsample)), int(round(h * upsample)))
            left_mono = cv2.resize(left_mono, new_size, interpolation=cv2.INTER_CUBIC)
            right_mono = cv2.resize(right_mono, new_size, interpolation=cv2.INTER_CUBIC)

        # IGEV expects H x W x 3 uint8; broadcast IR grayscale to 3 channels.
        left_rgb = cv2.cvtColor(left_mono, cv2.COLOR_GRAY2RGB)
        right_rgb = cv2.cvtColor(right_mono, cv2.COLOR_GRAY2RGB)

        with torch.no_grad():
            img1 = torch.from_numpy(left_rgb).permute(2, 0, 1).float()[None].to('cuda')
            img2 = torch.from_numpy(right_rgb).permute(2, 0, 1).float()[None].to('cuda')
            padder = InputPadder(img1.shape, divis_by=32)
            img1, img2 = padder.pad(img1, img2)

            disp, _ = self.igev_model(
                img1, img2, iters=self.igev_iters, test_mode=True,
            )
            disp = padder.unpad(disp).squeeze().cpu().numpy().astype(np.float32)

        if upsample != 1.0:
            disp = disp / upsample

        disp[disp <= 0] = np.nan
        disp[np.isinf(disp)] = np.nan
        return disp

    def snapshot_touch_pose(self) -> None:
        """Called from touch() the instant the force threshold trips. Records
        world -> gelsight_tooltip so the tactile height map can later be
        placed in a metric frame comparable to the D555 stereo cloud.

        Using world (not plate) because world is always in TF when the robot
        driver is up — no dependency on set_plate being alive on the right
        ROS_DOMAIN_ID. Convert to plate coords in post if needed."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'world',
                'gelsight_tooltip',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(
                f"touch pose TF lookup world <- gelsight_tooltip "
                f"failed: {e}; pose JSON not written."
            )
            self.pending_touch_pose = None
            return

        t = tf.transform.translation
        r = tf.transform.rotation
        self.pending_touch_pose = {
            'frame_id': tf.header.frame_id,
            'child_frame_id': tf.child_frame_id,
            'stamp': {'sec': tf.header.stamp.sec, 'nanosec': tf.header.stamp.nanosec},
            'translation': {'x': t.x, 'y': t.y, 'z': t.z},
            'rotation': {'x': r.x, 'y': r.y, 'z': r.z, 'w': r.w},
            'force_threshold_N': self.force_max,
            'force_at_stop_N': float(self.force),
        }
        self.get_logger().info(
            f"touch contact @ world ({t.x:.4f}, {t.y:.4f}, {t.z:.4f})m, "
            f"F={self.force:.2f}N (thr={self.force_max})"
        )

    def _save_camera_pose(self, stem: str) -> None:
        """Look up world -> camera_focus at capture time and dump as JSON.
        This is the robot's actual position at the moment of the stereo grab,
        which we'll need later to register tactile touches against the depth."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'world',
                'camera_focus',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(
                f"TF lookup world <- camera_focus failed: {e}; "
                "pose JSON not written."
            )
            return

        t = tf.transform.translation
        r = tf.transform.rotation
        pose_json = {
            'frame_id': tf.header.frame_id,
            'child_frame_id': tf.child_frame_id,
            'stamp': {'sec': tf.header.stamp.sec, 'nanosec': tf.header.stamp.nanosec},
            'translation': {'x': t.x, 'y': t.y, 'z': t.z},
            'rotation': {'x': r.x, 'y': r.y, 'z': r.z, 'w': r.w},
        }
        pose_path = os.path.join(self.capture_dir, f"{stem}_pose.json")
        with open(pose_path, 'w') as f:
            json.dump(pose_json, f, indent=2)
        self.get_logger().info(
            f"camera_focus pose in world: "
            f"z={t.z:.4f}m (x={t.x:.4f}, y={t.y:.4f}) -> {pose_path}"
        )

        # Also save world -> infra1_optical so the stereo PLY (currently in
        # optical frame from backprojection) can be transformed into world
        # and aligned with the anchored gelsight PLY. Frame name comes from
        # the camera_info header — matches whatever the realsense-ros driver
        # publishes on TF, no hardcoding required.
        if self.infra_camera_info is None:
            self.get_logger().warn(
                "infra1 camera_info not yet received; optical pose not written."
            )
            return
        optical_frame = self.infra_camera_info.header.frame_id
        try:
            tf_opt = self.tf_buffer.lookup_transform(
                'world',
                optical_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(
                f"TF lookup world <- {optical_frame} failed: {e}; "
                "optical pose JSON not written."
            )
            return
        ot = tf_opt.transform.translation
        or_ = tf_opt.transform.rotation
        optical_json = {
            'frame_id': tf_opt.header.frame_id,
            'child_frame_id': tf_opt.child_frame_id,
            'stamp': {'sec': tf_opt.header.stamp.sec, 'nanosec': tf_opt.header.stamp.nanosec},
            'translation': {'x': ot.x, 'y': ot.y, 'z': ot.z},
            'rotation': {'x': or_.x, 'y': or_.y, 'z': or_.z, 'w': or_.w},
            'intrinsics': {
                'fx': float(self.infra_camera_info.k[0]),
                'fy': float(self.infra_camera_info.k[4]),
                'cx': float(self.infra_camera_info.k[2]),
                'cy': float(self.infra_camera_info.k[5]),
                'width': int(self.infra_camera_info.width),
                'height': int(self.infra_camera_info.height),
            },
        }
        optical_path = os.path.join(self.capture_dir, f"{stem}_camera_optical_pose.json")
        with open(optical_path, 'w') as f:
            json.dump(optical_json, f, indent=2)
        self.get_logger().info(
            f"{optical_frame} pose in world + intrinsics -> {optical_path}"
        )

    def _save_point_cloud(
        self,
        depth: np.ndarray,
        left_ir: np.ndarray,
        stem: str,
        fx_native: float,
    ) -> None:
        """Backproject depth to 3D and write a binary little-endian PLY,
        colored by the left IR frame. Handles the upsample_factor: depth is
        at the upsampled resolution; the IR frame is at native, so we upsample
        it for coloring and scale the intrinsics to match depth resolution."""
        if self.infra_camera_info is None:
            self.get_logger().warn(
                "infra1/camera_info missing; skipping point cloud."
            )
            return

        k = self.infra_camera_info.k
        fx = float(k[0]); fy = float(k[4])
        cx = float(k[2]); cy = float(k[5])

        h_d, w_d = depth.shape
        h_i, w_i = left_ir.shape
        # Depth resolution = native * upsample_factor. Scale intrinsics
        # proportionally so the backprojection lives in the same physical
        # space as the native calibration.
        if h_d != h_i:
            scale = h_d / h_i
            fx *= scale; fy *= scale
            cx *= scale; cy *= scale
            colors = cv2.resize(left_ir, (w_d, h_d), interpolation=cv2.INTER_LINEAR)
        else:
            colors = left_ir

        v, u = np.meshgrid(np.arange(h_d), np.arange(w_d), indexing='ij')
        z = depth
        # Cull invalid / far pixels — 2m ceiling matches the depth-stats mask.
        valid = np.isfinite(z) & (z > 0.05) & (z < 2.0)
        n = int(valid.sum())
        if n == 0:
            self.get_logger().warn("No valid depth pixels; skipping point cloud.")
            return

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        verts = np.empty(n, dtype=np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
        ]))
        verts['x'] = x[valid].astype(np.float32)
        verts['y'] = y[valid].astype(np.float32)
        verts['z'] = z[valid].astype(np.float32)
        gray = colors[valid].astype(np.uint8)
        verts['r'] = gray; verts['g'] = gray; verts['b'] = gray

        ply_path = os.path.join(self.capture_dir, f"{stem}_cloud.ply")
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        with open(ply_path, 'wb') as f:
            f.write(header.encode('ascii'))
            f.write(verts.tobytes())
        self.get_logger().info(f"point cloud -> {ply_path} ({n} pts)")

    @staticmethod
    def _colormap_preview(arr: np.ndarray) -> np.ndarray:
        safe = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        norm = cv2.normalize(safe, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def _make_igev_args(self):
        checkpoint = self.igev_checkpoint
        # Scale the cost-volume disparity range with the upsample factor so
        # upsampled scenes don't saturate the model. Note: middlebury.pth was
        # trained at max_disp=192, so predictions in the extended range have
        # no training signal — expect degradation beyond ~1.5×.
        max_disp_scaled = int(192 * self.igev_upsample)

        class Args:
            restore_ckpt = checkpoint
            save_numpy = False
            mixed_precision = False
            valid_iters = 32
            hidden_dims = [128] * 3
            corr_implementation = "reg"
            shared_backbone = False
            corr_levels = 2
            corr_radius = 4
            n_downsample = 2
            slow_fast_gru = False
            n_gru_layers = 3
            max_disp = max_disp_scaled
            downsampling = False

        return Args

    #### Movement
    def move_z(self, speed):
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = self.frame_id
        twist_msg.twist.linear.z = speed
        self.twist_publisher.publish(twist_msg)
    def stop(self):
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = self.frame_id
        twist_msg.twist.linear.x = 0.0
        twist_msg.twist.linear.y = 0.0
        twist_msg.twist.linear.z = 0.0
        twist_msg.twist.angular.x = 0.0
        twist_msg.twist.angular.y = 0.0
        twist_msg.twist.angular.z = 0.0
        self.twist_publisher.publish(twist_msg)

    #### Force Sensor
    def force_callback(self, msg: WrenchStamped):
        force = 0
        force += abs(msg.wrench.force.z)
        self.force = abs(abs(force) - abs(self.offset))
    def calibrate_sensor(self, calibration_range = 100):
        self.get_logger().info("Starting sensor calibration")
        self.offset = 0.0
        data = []
        time.sleep(0.1)

        for _ in range(calibration_range):
            data.append(self.force)
            time.sleep(0.01)
        self.offset = sum(data) / len(data)
        time.sleep(0.1)
        self.force = 0 # Potential solution to race condition of node.force >= force_max
        self.get_logger().info(f"Calibration finished! Offset: {self.offset}")

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
    node = MotionCollectSample()

    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node, ),
        daemon=True
    )
    spin_thread.start()

    try:
        # Capture an untouched-gel frame to calibrate the HeightMapper baseline.
        # Still useful to have on hand even though we're not touching yet.
        node.get_logger().info("Capturing gelsight baseline frame for calibration...")
        node.gelsight_calibrate = True
        while node.gelsight_calibrate:
            time.sleep(0.1)
        node.get_logger().info("Gelsight baseline captured and calibrated.")

        # Send request to home robot arm
        request = Home.Request()
        request.speed = 0.1
        node.home_client.call(request)

        q = quaternion_from_euler(radians(0.0), radians(0.0), radians(0.0))
        orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # Move camera_focus to (0, 0, standoff_z) in the plate frame. Tune
        # standoff_z at launch to hone the D555 stereo working distance.
        goal_msg = PoseGoal.Goal()
        goal_msg.target_pose = Pose(
            position=Point(x=0.0, y=0.0, z=node.standoff_z),
            orientation=orientation,
        )
        goal_msg.velocity_scaling = 0.1
        goal_msg.acceleration_scaling = 0.1
        goal_msg.frame_id = node.frame_id
        goal_msg.target_id = "camera_focus"  # Can be any child of tool0. If empty -> tool0 used
        goal_msg.method = "PTP"  # Point-to-Point

        node.get_logger().info(
            f"Moving camera_focus to plate frame (0, 0, {node.standoff_z:.3f})..."
        )
        node.run_action(node.pose_goal_client, goal_msg)
        node.get_logger().info("Camera in position.")

        # Grab a D555 RGB reference frame for the session (useful next to the
        # stereo captures when reviewing standoff quality).
        time.sleep(3.0)
        node.d555_capture_image = True
        time.sleep(2.0)

        # Single stereo capture at camera_focus, then straight into the
        # touch grid. IGEV runs synchronously (~13s at 2x upsample on 3060).
        node.get_logger().info("Capturing stereo pair + running IGEV...")
        node.capture_and_compute_depth()

        # Gelsight tactile touch-grid data collection. Runs after the user
        # exits the interactive stereo loop with 'q'. The full-sensor lateral
        # FOV of the gel surface is 20 x 15 mm; the depth pipeline crops
        # `border_fraction` from each side, so we step the robot by the
        # *cropped* FOV to make adjacent touches butt up in the stitched view.
        gelsight_full_width = 0.020
        gelsight_full_height = 0.015
        keep = node.height_mapper.keep_fraction
        gelsight_width = gelsight_full_width * keep
        gelsight_height = gelsight_full_height * keep
        i_mid = (node.touch_x // 2)
        j_mid = (node.touch_y // 2)
        node.get_logger().info(
            f"Starting {node.touch_x}x{node.touch_y} touch grid "
            f"(step {gelsight_width * 1000:.1f} x {gelsight_height * 1000:.1f} mm)"
        )
        for i in range(node.touch_x, 0, -1):
            x = (-i_mid + i) * gelsight_width
            for j in range(node.touch_y, 0, -1):
                y = (-j_mid + j) * gelsight_height
                node.get_logger().info(f"touch i={i} j={j} -> plate ({x:.4f}, {y:.4f})")
                touch(
                    node=node,
                    frame_id=node.frame_id,
                    mock=node.mock,
                    position=Point(x=x, y=y, z=0.02),
                    orientation=orientation,
                )

        node.get_logger().info("Stitching session captures...")
        stitch_session(
            session_dir=node.capture_dir,
            touch_x=node.touch_x,
            touch_y=node.touch_y,
        )
        write_stitched_ply(
            session_dir=node.capture_dir,
            touch_x=node.touch_x,
            touch_y=node.touch_y,
        )
        # Plate-frame anchored stitch — uses per-touch pose JSONs to place
        # each detrended tile at its measured contact-Z. No-op if pose JSONs
        # aren't present (silent for backward compat, warning printed).
        stitch_session_anchored(
            session_dir=node.capture_dir,
            touch_x=node.touch_x,
            touch_y=node.touch_y,
        )
        write_anchored_ply(
            session_dir=node.capture_dir,
            touch_x=node.touch_x,
            touch_y=node.touch_y,
        )
        # Transform the stereo PLY into world coords and combine with the
        # anchored gelsight PLY. Uses camera_focus pose as camera-frame proxy
        # (see align_clouds.py). No-op if stereo pose JSON is missing.
        write_combined_ply(session_dir=node.capture_dir)

        # Return camera_focus + home the arm.
        _ = node.get_logger().info("Moving Camera Focus")
        _ = node.run_action(node.pose_goal_client, goal_msg)
        _ = node.get_logger().info("Finished Moving Camera Focus")

        request = Home.Request()
        request.speed = 0.1
        node.home_client.call(request)

    except KeyboardInterrupt:
        _ = node.get_logger().info("Script interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join()

if __name__ == '__main__':
    main()

