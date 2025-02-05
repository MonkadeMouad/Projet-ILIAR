#! coding: utf-8

"""
Extract the road element from a gazebo world file and publish it. The world
file to be parsed is expected to be provided as an argument. For example, in
your xml launch file :


    <node pkg="..." exec="path_publisher" name="..." args="myfile.world">

Subscriptions:

Publications:
    - `/road_path` (nav_msgs/Path)


"""

# Standard imports
import sys
import xml.etree.ElementTree as ET

# External imports
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header


class PathPublisher(Node):
    """
    Extract the road element from a gazebo world file and publish it as a
    nav_msgs/Path message
    """

    def __init__(self, world_file_path):
        super().__init__("path_publisher")

        self.road_points = []
        self.parse_world_file(world_file_path)

        self.publisher_ = self.create_publisher(Path, "road_path", 1)
        self.timer_ = self.create_timer(1.0 / 30.0, self.publish_path)

    def parse_world_file(self, world_file_path):
        """
        Parse the world file to get the road points
        """
        tree = ET.parse(world_file_path)
        root = tree.getroot()

        # Iterate over the points of the road
        self.road_points = []
        for road in root.iter("road"):
            for point in road.iter("point"):
                # self.get_logger().info(f"Point: {point.text}")
                x, y, z = map(float, point.text.split(" "))
                self.road_points.append((x, y, z))
            # Only keep one road
            break

    def publish_path(self):
        """
        Publish the road path on a regular basis. This is triggered
        by a timer.
        """
        base_frame = "world"
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = base_frame

        # Add the path points
        path.poses = [
            PoseStamped(
                header=Header(stamp=path.header.stamp, frame_id=base_frame),
                pose=Pose(
                    position=Point(x=x, y=y, z=z),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            )
            for x, y, z in self.road_points
        ]

        self.publisher_.publish(path)


def main(args=None):
    """Instantiation node and class."""
    rclpy.init(args=args)

    # Get the world file to parse to publish the road path
    world_file = sys.argv[1]

    # Create the node
    path_publisher = PathPublisher(world_file)

    # Run
    try:
        rclpy.spin(path_publisher)
    except KeyboardInterrupt:
        pass

    # End
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
