#! coding: utf-8

"""
Listen to the images and steering command and save them to disk.

Subscriptions:
        - `/audibot/left_camera/image_raw/compressed` (sensor_msgs/CompressedImage)
        - `/audibot/front_camera/image_raw/compressed` (sensor_msgs/CompressedImage)
        - `/audibot/right_camera/image_raw/compressed` (sensor_msgs/CompressedImage)
        - `/audibot/steering_cmd` (std_msgs/Float64)

Publications:

Parameters:

"""

# External imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
import numpy as np
import os
import re

class DatasetRecorder(Node):
    """Listen to the images and steering command and save them to disk.
    This node is used to record the dataset for training the neural network.
    """

    def __init__(self):
        super().__init__("dataset_recorder")

        self.get_logger().info("Recorder started !")

        # We will dump the dataset on drive every 500 samples
        self.chunk_size = 500

        # Directory to save the dataset
        self.save_dir = "ros2_ws/dataset"
        os.makedirs(self.save_dir, exist_ok=True)

        # Determine the starting chunk index by finding the highest existing index
        self.chunk_idx = self.get_highest_chunk_index() + 1

        # These containers temporarily hold the data before the dump
        # on the drive
        self.frames = []
        self.steerings = []

        self.last_steering = None

        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers for images and steering
        self.sub_left = Subscriber(self, CompressedImage, "/audibot/left_camera/image_raw/compressed", qos_profile=qos_profile)
        self.sub_front = Subscriber(self, CompressedImage, "/audibot/front_camera/image_raw/compressed", qos_profile=qos_profile)
        self.sub_right = Subscriber(self, CompressedImage, "/audibot/right_camera/image_raw/compressed", qos_profile=qos_profile)
        self.sub_steering = self.create_subscription(
            Float64, "/audibot/steering_cmd", self.on_steering, qos_profile
        )

        # Synchronizer for the image topics
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_left, self.sub_front, self.sub_right],
            queue_size=10,
            slop=0.1,
        )
        self.sync.registerCallback(self.record_cb)

    def get_highest_chunk_index(self):
        """Find the highest chunk index in the dataset directory."""
        existing_files = [
            f for f in os.listdir(self.save_dir) if re.match(r"chunk_\d+\.npz", f)
        ]
        if not existing_files:
            return -1  # No existing files, start at index 0

        # Extract numeric indices from filenames
        indices = [
            int(re.search(r"chunk_(\d+)\.npz", f).group(1)) for f in existing_files
        ]
        return max(indices)

    def on_steering(self, msg: Float64):
        """Callback for the steering topic."""
        self.last_steering = msg.data

    def record_cb(
        self,
        left_msg: CompressedImage,
        front_msg: CompressedImage,
        right_msg: CompressedImage,
    ):
        """
        Listen to the synchronized images and collect
        both the images and steering command. When the
        chunk size is reached, dump the data on the disk.
        """
        if self.last_steering is None:
            return

        try:
            # Convert the compressed images to OpenCV format
            left_img = self.bridge.compressed_imgmsg_to_cv2(left_msg, "bgr8")
            front_img = self.bridge.compressed_imgmsg_to_cv2(front_msg, "bgr8")
            right_img = self.bridge.compressed_imgmsg_to_cv2(right_msg, "bgr8")

            # Store the steering and the images
            self.frames.append(
                {
                    "left": left_img,
                    "front": front_img,
                    "right": right_img,
                }
            )
            self.steerings.append(self.last_steering)

            if len(self.frames) == self.chunk_size:
                # Dump them on disk
                chunk_path = os.path.join(self.save_dir, f"chunk_{self.chunk_idx}.npz")
                np.savez_compressed(
                    chunk_path,
                    frames=self.frames,
                    steerings=self.steerings,
                )
                self.get_logger().info(f"Saved chunk {self.chunk_idx} to disk.")
                self.chunk_idx += 1

                # Clear temporary storage
                self.steerings = []
                self.frames = []

        except Exception as e:
            self.get_logger().error(f"Failed to process data: {e}")


def main(args=None):
    """Instantiation node and class."""
    rclpy.init(args=args)

    # Create node
    dataset_recorder = DatasetRecorder()

    # Run
    try:
        rclpy.spin(dataset_recorder)
    except KeyboardInterrupt:
        pass

    # end
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
