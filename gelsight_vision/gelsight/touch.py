import math
import time

from geometry_msgs.msg import Pose, Point, Quaternion, TwistStamped, WrenchStamped
from rclpy.node import Node
from robot_manager_interfaces.action import PoseGoal
from tf_transformations import quaternion_from_euler

def touch(
    node: Node,
    frame_id: str,
    mock: bool,
    position: Point,
    orientation: Quaternion,
    dz: float = -0.025,
    force_max: float = 4.0,
):
    gelsight_goal_msg = PoseGoal.Goal()
    gelsight_goal_msg.target_pose = Pose(position=position, orientation=orientation)
    
    gelsight_goal_msg.velocity_scaling = 0.1
    gelsight_goal_msg.acceleration_scaling = 0.1
    gelsight_goal_msg.frame_id = frame_id 
    gelsight_goal_msg.target_id = "gelsight_tooltip"
    gelsight_goal_msg.method = "PTP" # Point-to-Point
    
    # Move gelsight tooltip above sample
    _ = node.get_logger().info("Moving Gelsight Tooltip into Position")
    _ = node.run_action(node.pose_goal_client, gelsight_goal_msg)
    _ = node.get_logger().info("Finished Moving Gelsight Tooltip into Position")
    
    node.calibrate_sensor()

    
    if mock:
        for i in range(1000):
            node.move_z(dz)
            time.sleep(0.005)
    else:
        while True:
            node.move_z(dz)
            print(node.force)
            if node.force >= force_max:
                node.stop()
                break
            time.sleep(0.002)

    # Snapshot the arm pose the instant force-threshold trips. gelsight
    # callback drains this into the per-touch pose JSON later.
    if hasattr(node, 'snapshot_touch_pose'):
        node.snapshot_touch_pose()

    time.sleep(2.0)

    # Gelsight picture take
    node.gelsight_capture_image = True

    while node.gelsight_capture_image:
        time.sleep(0.1)

    # Wait for pose aiction server to be available.
    time.sleep(0.1)

    # Move gelsight tooltip back above sample
    _ = node.get_logger().info("Moving Gelsight Tooltip into Position")
    _ = node.run_action(node.pose_goal_client, gelsight_goal_msg)
    _ = node.get_logger().info("Finished Moving Gelsight Tooltip into Position")

