#! coding: utf-8

"""
This node computes scores from an input temporal serie of float numbers.

Subscriptions:
    - /serie (std_msgs/Float64): Input temporal serie of float numbers.

Publications:
    - /min (std_msgs/Float64): Minimum value of the serie.
    - /max (std_msgs/Float64): Maximum value of the serie.
    - /avg (std_msgs/Float64): Mavg value of the serie.
"""

# External imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class Statistics:

    def __init__(self):
        self.min = None
        self.max = None
        self.avg = None
        self.count = 0
        self.sum = 0

    def update(self, value):
        if self.min is None or value < self.min:
            self.min = value
        if self.max is None or value > self.max:
            self.max = value
        self.count += 1
        self.sum += value
        self.avg = self.sum / self.count


class Scorer(Node):

    def __init__(self):
        super().__init__("scorer")

        self.statistics = Statistics()

        # Create the publishers
        self.pub_min = self.create_publisher(Float64, "/min", 1)
        self.pub_max = self.create_publisher(Float64, "/max", 1)
        self.pub_avg = self.create_publisher(Float64, "/avg", 1)

        # Create the subscription
        self.serie_sub = self.create_subscription(
            msg_type=Float64,
            topic="/serie",
            callback=self.on_value,
            qos_profile=1,
        )

    def on_value(self, msg: Float64):
        value = msg.data

        # Update the statistics
        self.statistics.update(value)

        # Publish the computed scores
        self.pub_min.publish(Float64(data=self.statistics.min))
        self.pub_max.publish(Float64(data=self.statistics.max))
        self.pub_avg.publish(Float64(data=self.statistics.avg))


def main(args=None):
    """Instantiation node and class."""
    rclpy.init(args=args)

    # Create the node
    scorer = Scorer()

    # Spin the node
    try:
        rclpy.spin(scorer)
    except KeyboardInterrupt:
        pass

    scorer.try_shutdown()


if __name__ == "__main__":
    main()
