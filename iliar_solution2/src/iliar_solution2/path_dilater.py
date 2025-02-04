#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dilate a given path by shifting points along the path normal and publish the result.
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion


class PathDilater(Node):
    def __init__(self):
        super().__init__("path_dilater")
        self.declare_parameter("offset", 2.0)  # Corrected parameter name
        self.path_publisher = self.create_publisher(Path, "out_path", 1)
        self.path_subscriber = self.create_subscription(
            Path, "in_path", self.on_path, 1
        )

    def on_path(self, msg: Path):
        """
        Dilate the received path and publish the dilated version.
        """
        dilation_offset = self.get_parameter("offset").get_parameter_value().double_value
        path = [pose.pose.position for pose in msg.poses]

        if len(path) < 2:
            self.get_logger().error("Received path has less than 2 points, skipping dilation.")
            return

        previous_point = path[-1]
        new_positions = []

        for current_point in path:
            dx, dy = current_point.x - previous_point.x, current_point.y - previous_point.y
            norm = math.sqrt(dx**2 + dy**2)
            if norm == 0:
                continue

            # Normal vector
            nx, ny = dy / norm, -dx / norm

            dilated_point = (
                current_point.x + dilation_offset * nx,
                current_point.y + dilation_offset * ny,
                current_point.z,
            )
            new_positions.append(dilated_point)
            previous_point = current_point

        # Publish the dilated path
        dilated_path = Path()
        dilated_path.header = msg.header
        dilated_path.poses = [
            PoseStamped(
                header=msg.header,
                pose=Pose(
                    position=Point(x=x, y=y, z=z),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            )
            for x, y, z in new_positions
        ]
        self.path_publisher.publish(dilated_path)


def main(args=None):
    rclpy.init(args=args)
    node = PathDilater()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
