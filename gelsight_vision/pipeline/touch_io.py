"""Shared ROS2 IO helpers for force-driven touch motions.

Each `setup_*` function takes a Node and bolts on one subscriber / publisher /
client. The op functions (`move_z`, `stop`, `calibrate_sensor`) work against
attributes the setup helpers added.

Intended usage from PipelineVisuoTactile / MotionCollectSample:

    class MyNode(Node):
        def __init__(self):
            super().__init__('my_node')
            ...
            touch_io.setup_force(self, self.ns)
            touch_io.setup_twist(self, self.ns)
            touch_io.setup_servo(self, self.ns)

        def move_z(self, speed): touch_io.move_z(self, speed, self.frame_id)
        def stop(self): touch_io.stop(self, self.frame_id)
        def calibrate_sensor(self, n=100): touch_io.calibrate_sensor(self, n)

so `gelsight/touch.py` keeps calling `node.move_z(...)`, `node.stop()`, etc.
"""

from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import TwistStamped, WrenchStamped
from moveit_msgs.srv import ServoCommandType


def setup_force(node, ns: str) -> None:
    """Add the wrench subscriber + force state to `node`.

    Attaches: `node.force`, `node.offset`, `node.force_subscriber`.
    Topic: `<ns>/wrench`. Reported force is the z-axis magnitude with
    `node.offset` subtracted, so a zero reading after `calibrate_sensor()`
    means "no contact relative to the baseline".
    """
    node.force = 0.0
    node.offset = 0.0

    def _callback(msg: WrenchStamped) -> None:
        magnitude = abs(msg.wrench.force.z)
        node.force = abs(abs(magnitude) - abs(node.offset))

    node.force_subscriber = node.create_subscription(
        WrenchStamped, ns + '/wrench', _callback, 10,
    )


def setup_twist(node, ns: str) -> None:
    """Add the cartesian-twist publisher to `node`.

    Attaches: `node.twist_publisher`. Topic: `<ns>/servo_node/delta_twist_cmds`.
    """
    node.twist_publisher = node.create_publisher(
        TwistStamped, ns + '/servo_node/delta_twist_cmds', 10,
    )


def setup_servo(node, ns: str) -> None:
    """Switch the MoveIt servo node into TWIST command mode.

    Attaches: `node.servo_client`. Blocks until the switch service returns.
    """
    node.servo_client = node.create_client(
        ServoCommandType, ns + '/servo_node/switch_command_type',
    )
    while not node.servo_client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info(
            'Waiting for service: ' + ns + '/servo_node/switch_command_type'
        )
    req = ServoCommandType.Request()
    req.command_type = ServoCommandType.Request.TWIST
    future = node.servo_client.call_async(req)
    rclpy.spin_until_future_complete(node, future)


def move_z(node, speed: float, frame_id: str) -> None:
    """Publish a one-shot z-axis cartesian twist command."""
    twist = TwistStamped()
    twist.header.stamp = node.get_clock().now().to_msg()
    twist.header.frame_id = frame_id
    twist.twist.linear.z = speed
    node.twist_publisher.publish(twist)


def stop(node, frame_id: str) -> None:
    """Publish a zero twist to halt cartesian motion."""
    twist = TwistStamped()
    twist.header.stamp = node.get_clock().now().to_msg()
    twist.header.frame_id = frame_id
    node.twist_publisher.publish(twist)


def calibrate_sensor(node, calibration_range: int = 100) -> None:
    """Average `calibration_range` force samples into `node.offset` so
    subsequent `node.force` readings are relative to no-contact baseline.
    """
    node.get_logger().info('Starting sensor calibration')
    node.offset = 0.0
    time.sleep(0.1)
    samples = []
    for _ in range(calibration_range):
        samples.append(node.force)
        time.sleep(0.01)
    node.offset = sum(samples) / len(samples)
    time.sleep(0.1)
    # Avoid a race where a stale node.force > force_max trips touch() before
    # the first post-calibration sample lands.
    node.force = 0.0
    node.get_logger().info(f'Calibration finished! Offset: {node.offset}')
