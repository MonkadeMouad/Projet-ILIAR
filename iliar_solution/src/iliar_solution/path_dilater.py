#! coding: utf-8

"""
Dilate an input and publish it.

Subscriptions:
    - in_path (nav_msgs/Path): The path to dilate

Publications:
    - out_path (nav_msgs/Path): The dilated path

Parameters:
    - offset (float): The offset to dilate the path by

"""

# Standard imports
import sys
import math

# External imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from nav_msgs.msg import Path


class PathDilater(Node):
    """
    Dilate an input path by the given offset. Dilation is performed
    by shifting the vertices along the normal of the path by the given offset.
    """

    def __init__(self, world_file_path):
        super().__init__("path_dilater")
        self.declare_parameter("~offset", 2.0)
        self.path_publisher = self.create_publisher(Path, "out_path", 1)
        self.path_subscriber = self.create_subscription(
            Path, "in_path", self.on_path, 1
        )

    def on_path(self, msg: Path):
        """
        Dilate the path by the given offset and publish the dilated path
        """
        # Read the dilation offset
        dilation_offset = self.get_parameter("~offset").value

        # Extract the poses from the published path
        path = [pstamped.pose.position for pstamped in msg.poses]

        # Dilate the path
        previous_point = path[-1]
        new_positions = []
        for p in path:
            # Current the normalized vector from the previous point to the current point
            dp = (p.x - previous_point.x, p.y - previous_point.y)
            norm_dp = math.sqrt(dp[0] ** 2 + dp[1] ** 2)
            orth_dp = (dp[1] / norm_dp, -dp[0] / norm_dp)

            dilated_point = (
                p.x + dilation_offset * orth_dp[0],
                p.y + dilation_offset * orth_dp[1],
                p.z,
            )
            new_positions.append(dilated_point)
            previous_point = p

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
    """Instantiation node and class."""
    rclpy.init(args=args)

    # Get the world file to parse to publish the road path
    world_file = sys.argv[1]

    # Create the node
    path_dilater = PathDilater(world_file)

    try:
        rclpy.spin(path_dilater)
    except KeyboardInterrupt:
        pass

    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
