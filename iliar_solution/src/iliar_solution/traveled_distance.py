#! coding: utf-8

"""
This node computes the traveled distance of the robot along a path. 
The distance is computed from the projected position of the robot on the path

Subscriptions:
    - path (nav_msgs/Path): The path to follow

Publications:
    - traveled_dist (std_msgs/Float64): The traveled distance along the path
    - curv_abscissa (std_msgs/Float64): The curvilinear abscissa of the robot on the path

Parameters:
    - base_frame (string): The frame of the robot
"""

# External imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException

import geom2d
import numpy as np


class TraveledDistance(Node):
    """
    This node computes the traveled distance of the robot along a path.
    The distance is computed from the projected position of the robot on the path
    """

    def __init__(self):
        super().__init__("traveled_distance")

        # The base frame of the robot
        self.declare_parameter("base_frame", "base_link")

        # The TF objects required to request the transforms
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.last_valid_position = None
        self.total_traveled_distance = 0.0

        # TODO
        # You must instantiante your publisher and subscriber here

    def dist_callback(self, path_msg: Path):
        """Callback to compute the traveled distance along the path."""

        # Extract the 2D (x, y) positions and go into the numpy world
        path = [
            np.array([pstamped.pose.position.x, pstamped.pose.position.y])
            for pstamped in path_msg.poses
        ]
        path_frame = path_msg.header.frame_id


        # Get the base frame of the robot
        base_frame = self.get_parameter("base_frame").get_parameter_value().string_value

        try:
            # Get the pose in the path frame of reference
            trans = self._tf_buffer.lookup_transform(
                path_frame, base_frame, rclpy.time.Time()
            )
            # self.get_logger().info(f"Got transform {trans}")
            car_position = np.array(
                [trans.transform.translation.x, trans.transform.translation.y]
            )
        except LookupException as e:
            self.get_logger().error("failed to get transform {} \n".format(repr(e)))
            return

        # Iterate over the segments in order to determine the
        # projection of the car onto the polygon
        car_projection = None
        min_dist = None
        for p1_idx, (p1, p2) in enumerate(zip(path[:-1], path[1:])):
            # TODO
            pass
        # Do not forget the last segment
        p1 = path[-1]
        p2 = path[0]
        # TODO

        # TODO
        # Update the traveled distance
        # and publish it



def main(args=None):
    """Instantiation node and class."""
    rclpy.init(args=args)

    # Create the node
    traveled_distance = TraveledDistance()

    # Spin the node
    try:
        rclpy.spin(traveled_distance)
    except KeyboardInterrupt:
        pass

    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
