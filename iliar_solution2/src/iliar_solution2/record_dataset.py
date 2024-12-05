#! coding: utf-8

"""
Listen to the images and steering command and save them to disk.

Subscriptions:
        - `/left_image` (sensor_msgs/CompressedImage)
        - `/front_image` (sensor_msgs/CompressedImage)
        - `/right_image` (sensor_msgs/CompressedImage)
        - `/steering` (std_msgs/Float64)

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

class DatasetRecorder(Node):
    """Listen to the images and steering command and save them to disk.
    This node is used to record the dataset for training the neural network.
    """

    def __init__(self):
        super().__init__("dataset_recorder")

        self.get_logger().info("Recorder started !")

        # We will dump the dataset on drive every 500 samples
        self.chunk_size = 500
        self.chunk_idx = 0

        # These containers temporarily hold the data before the dump
        # on the drive
        self.frames = []
        self.steerings = []

        self.last_steering = None

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # TODO
        # Create your subscribers and ApproximateTimeSynchronizer here
        self.camera_left_sub = Subscriber(self,
            CompressedImage,
            '/audibot/left_camera/image_raw/compressed',

            )
        self.camera_right_sub = Subscriber(self,
            CompressedImage,
            '/audibot/right_camera/image_raw/compressed',
       
        )
        self.camera_front_sub = Subscriber(self,
            CompressedImage,
            '/audibot/front_camera/image_raw/compressed',
            
        )
        self.create_subscrition(self,
            Float64,
            '/audibot/steering_cmd',
            self.on_steering,
            qos_profile)
        
        self.bridge = CvBridge()
        self.sync = ApproximateTimeSynchronizer(
            [self.camera_left_sub, self.camera_front_sub, self.camera_right_sub],
            queue_size=10,
            slop=0.1,
        )
        self.sync.registerCallback(self.record_cb)
        
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        
    def on_steering(self, msg: Float64):
        """Callback for the steering topic."""
        self.last_steering = msg.data
        
    def record_cb(self, left_msg, front_msg, right_msg, steering_msg):
        """Callback for the steering topic."""
        last_steering=steering_msg
        # TODO
        left_image = self.image_callback(left_msg)
        front_image = self.image_callback(front_msg)
        right_image = self.image_callback(right_msg)

        # Store images and steering data
        self.frames.append((left_image, front_image, right_image))
        self.steerings.append(self.last_steering)

        # Check chunk size and save
        if len(self.frames) >= self.chunk_size:
            self.save_chunk()
            self.chunk_idx += 1
            self.frames = []
            self.steerings = []

    def save_chunk(self):
        """Save the buffered data to disk."""
        chunk_folder = f"chunk_{self.chunk_idx}"
        os.makedirs(chunk_folder, exist_ok=True)

        file_path = os.path.join(chunk_folder, "data.npz")
        try:
            np.savez_compressed(
            file_path, frames=self.frames, steerings=self.steerings
        )
            self.get_logger().info(
            f"Saved chunk {self.chunk_idx} with {len(self.frames)} samples."
        )
            self.chunk_idx += 1
        except Exception as e:
            self.get_logger().error(f"Failed to save chunk: {e}")



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
