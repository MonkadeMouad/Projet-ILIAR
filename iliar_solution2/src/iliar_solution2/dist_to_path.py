#! coding: utf-8

"""
Compute the distance between the robot and the path. The path is provided as a
sequence of control points. The distance is computed as the distance between
the robot and the segment connecting the two closest points of the path.

Subscriptions:
    - `/path` (nav_msgs/Path): The path to follow

Publications:
    - `/dist_to_path` (std_msgs/Float64): The distance to the path

Parameters:
    - `base_frame` (str): The base frame of the robot

"""

# Standard imports
import sys

# External imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Path
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException
import numpy as np
import geom2d

# from tf_transformations import euler_from_quaternion


class DistToPath(Node):

    def __init__(self):
        super().__init__("dist_to_path")

        # The base frame of the robot
        self.declare_parameter("base_frame", "base_link")

        # Boolean to decide whether or not to compute a signed distance to
        # the path
        self.declare_parameter("signed_distance", False)

        # The TF objects required to request the transforms
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Subscriber: on écoute le path
        self.path_sub = self.create_subscription(
            Path,
            '/lanes/track',        # ou un remapping dans le launch
            self.dist_callback,
            10
        )

        # Publisher: on publie la distance
        self.dist_pub = self.create_publisher(
            Float64,
            '/dist_to_path',       # ou un remapping dans le launch
            10
        )

    def dist_callback(self, msg):
        path = [pstamped.pose for pstamped in msg.poses]
        path_frame = msg.header.frame_id

        base_frame = self.get_parameter("base_frame").value
        signed_distance = self.get_parameter("signed_distance").value

        # On récupère la TF base_link -> path_frame
        try:
            trans = self._tf_buffer.lookup_transform(
                msg.header.frame_id,  # target_frame
                base_frame,           # source_frame
                rclpy.time.Time()
            )
        except LookupException as e:
            self.get_logger().error(f"failed to get transform: {e}")
            return
        # self.get_logger().info(f"Got transform {trans}")
        object_position = trans.transform.translation
        q = trans.transform.rotation
        object_forward = np.array(
            [
                q.w**2 + q.x**2 - q.y**2 - q.z**2,
                   2 * q.w * q.z + 2 * q.x * q.y,
            ]
        )
        

         path_positions = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        if len(path_positions) < 2:
            return

        # Ex. plus proche
        px = object_position.x
        py = object_position.y

        # Trouve l'indice min_idx
        min_idx = 0
        min_dist_2 = float('inf')
        for i, (x, y) in enumerate(path_positions):
            d2 = (x - px)**2 + (y - py)**2
            if d2 < min_dist_2:
                min_dist_2 = d2
                min_idx = i

        # On prend le point avant/après pour former le segment
        prev_i = (min_idx - 1) % len(path_positions)
        next_i = (min_idx + 1) % len(path_positions)
        closest = path_positions[min_idx]
        prev_pt = path_positions[prev_i]
        next_pt = path_positions[next_i]

        # Choisir "second_closest" en fonction de la distance au prev ou next
        dprev = (prev_pt[0] - px)**2 + (prev_pt[1] - py)**2
        dnext = (next_pt[0] - px)**2 + (next_pt[1] - py)**2
        if dprev < dnext:
            second_closest = prev_pt
        else:
            second_closest = next_pt

        # On crée un segment geom2d pour la distance
        seg = geom2d.Segment(geom2d.Point(*closest), geom2d.Point(*second_closest))
        car_point = geom2d.Point(px, py)
        distance = seg.distanceToPoint(car_point)

        # S’il faut un signe, on calcule l’orientation, etc.
        # Pour l’instant, on publie la distance absolue
        self.dist_pub.publish(Float64(data=distance))



def main(args=None):
    """Instantiation node and class."""
    rclpy.init(args=args)

    # Create the node
    dist_to_path = DistToPath()

    # Run
    try:
        rclpy.spin(dist_to_path)
    except KeyboardInterrupt:
        pass

    # End

    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
