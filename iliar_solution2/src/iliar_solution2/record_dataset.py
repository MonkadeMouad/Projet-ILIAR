#! coding: utf-8

"""
Listen to the images and steering command and save them to disk.

Subscriptions:
        - `/audibot/left_camera/image_raw/compressed` (sensor_msgs/CompressedImage)
        - `/audibot/front_camera/image_raw/compressed` (sensor_msgs/CompressedImage)
        - `/audibot/right_camera/image_raw/compressed` (sensor_msgs/CompressedImage)
        - `/audibot/steering_cmd` (std_msgs/Float64)

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

import shutil

def atomic_save_npz(file_path, **data):
    """Save an NPZ file by explicitly copying content to avoid extension issues."""
    temp_path = f"{file_path}.tmp.npz"  # Temporary file name to match actual save behavior
    try:
        # Save to the temporary file
        np.savez_compressed(temp_path, **data)

        # Copy the content to the final file
        shutil.copy(temp_path, file_path)
        print(f"Atomic save successful: {file_path}")

    except Exception as e:
        raise RuntimeError(f"Error during atomic save: {e}")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)



class DatasetRecorder(Node):
    """Node to listen to images and steering commands and save them to disk."""

    def __init__(self):
        super().__init__("dataset_recorder")
        self.get_logger().info("Recorder started!")

        # Chunk configuration
        self.chunk_size = 500
        self.frames = []
        self.steerings = []
        self.last_steering = None
        
        # Dataset directory
        self.save_dir = os.path.expanduser("~/ros2_ws/dataset")
        os.makedirs(self.save_dir, exist_ok=True)
        self.chunk_idx = self.get_highest_chunk_index() + 1

        # CvBridge for image conversion
        self.bridge = CvBridge()

        # QoS Profile for subscriptions
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self.sub_left = Subscriber(self, CompressedImage, "/audibot/left_camera/image_raw/compressed", qos_profile=qos_profile)
        self.sub_front = Subscriber(self, CompressedImage, "/audibot/front_camera/image_raw/compressed", qos_profile=qos_profile)
        self.sub_right = Subscriber(self, CompressedImage, "/audibot/right_camera/image_raw/compressed", qos_profile=qos_profile)
        self.sub_steering = self.create_subscription(
            Float64, "/audibot/steering_cmd", self.on_steering, qos_profile
        )

        # Synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_left, self.sub_front, self.sub_right], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.record_cb)

    def get_highest_chunk_index(self):
        """Get the highest chunk index from the dataset directory."""
        existing_files = [
            f for f in os.listdir(self.save_dir) if re.match(r"chunk_\d+\.npz", f)
        ]
        if not existing_files:
            return -1

        indices = [
            int(re.search(r"chunk_(\d+)\.npz", f).group(1)) for f in existing_files
        ]
        return max(indices)

    def on_steering(self, msg: Float64):
        """Callback to store the latest steering command."""
        self.last_steering = msg.data

    def record_cb(self, left_msg, front_msg, right_msg):
        """Callback to process and store synchronized images and steering commands."""
        if self.last_steering is None:
            return

        try:
            # Convert compressed images to OpenCV format
            left_img = self.bridge.compressed_imgmsg_to_cv2(left_msg, "bgr8")
            front_img = self.bridge.compressed_imgmsg_to_cv2(front_msg, "bgr8")
            right_img = self.bridge.compressed_imgmsg_to_cv2(right_msg, "bgr8")

            # Append to temporary storage
            self.frames.append({"left": left_img, "front": front_img, "right": right_img})
            self.steerings.append(self.last_steering)

            if len(self.frames) >= self.chunk_size:
                self.save_chunk()

        except Exception as e:
            self.get_logger().error(f"Failed to process data: {e}")

    def save_chunk(self):
        """Save the current chunk of data to disk."""
        chunk_path = os.path.join(self.save_dir, f"chunk_{self.chunk_idx}.npz")
        try:
            atomic_save_npz(chunk_path, frames=self.frames, steerings=self.steerings)
            self.get_logger().info(f"Saved chunk {self.chunk_idx} to disk.")
            self.chunk_idx += 1

            # Clear temporary storage
            self.frames.clear()
            self.steerings.clear()

        except Exception as e:
            self.get_logger().error(f"Failed to save chunk {self.chunk_idx}: {e}")


def main(args=None):
    """Main function to initialize and run the node."""
    rclpy.init(args=args)

    recorder = DatasetRecorder()
    print("start")
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.get_logger().info("Shutting down recorder.")
        rclpy.shutdown()


if __name__ == "__main__":
    main()