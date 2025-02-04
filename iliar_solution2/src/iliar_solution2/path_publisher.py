#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract the road element from a Gazebo world file and publish it on `/road_path`.
"""

import sys
import xml.etree.ElementTree as ET
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header


class PathPublisher(Node):
    def __init__(self, world_file_path):
        super().__init__("path_publisher")

        self.road_points = []
        self.parse_world_file(world_file_path)

        self.publisher_ = self.create_publisher(Path, "road_path", 1)
        self.timer_ = self.create_timer(1.0 / 30.0, self.publish_path)

    def parse_world_file(self, world_file_path):
        """
        Parse the Gazebo world file to extract road points.
        """
        try:
            tree = ET.parse(world_file_path)
            root = tree.getroot()

            for road in root.iter("road"):
                for point in road.iter("point"):
                    x, y, z = map(float, point.text.split(" "))
                    self.road_points.append((x, y, z))
                # Process only the first road element
                break

        except Exception as e:
            self.get_logger().error(f"Failed to parse the world file: {e}")

    def publish_path(self):
        """
        Publish the road path as a `nav_msgs/Path` message.
        """
        base_frame = "world"
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = base_frame

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
    rclpy.init(args=args)

    if len(sys.argv) < 2:
        print("Usage: ros2 run <package> path_publisher <world_file>")
        sys.exit(1)

    world_file = sys.argv[1]
    node = PathPublisher(world_file)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
