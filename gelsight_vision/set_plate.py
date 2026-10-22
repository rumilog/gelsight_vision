import math
import rclpy

from geometry_msgs.msg import Quaternion, TransformStamped
from rclpy.node import Node
from tf_transformations import quaternion_from_euler
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class SetPlate(Node):
    def __init__(self):
        super().__init__('set_plate')

        _ = self.declare_parameter('frame_id', 'plate')
        self.frame_id: str = str(self.get_parameter('frame_id').value)

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = self.frame_id
        t.transform.translation.x = 0.375 # 37.5 cm
        t.transform.translation.y = -0.275 # -27.5 cm
        # t.transform.translation.x = 0.325 # 32.5 cm
        # t.transform.translation.y = -0.025 # -2.5 cm
        # t.transform.translation.x = 0.0
        # t.transform.translation.y = -0.0
        t.transform.translation.z = 0.0
        q = quaternion_from_euler(math.radians(0.0), math.radians(0.0), math.radians(-90.0))
        t.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.tf_static_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = SetPlate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        _ = node.get_logger().info("Script interrupted by user.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
