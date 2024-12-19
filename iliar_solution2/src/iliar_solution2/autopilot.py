# External imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
import numpy as np
import cv2
import onnxruntime
import os

class Autopilot(Node):
    def __init__(self):
        super().__init__("autopilot")
        self.get_logger().info("Autopilot node started!")

        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Load the ONNX model
        try:
            self.model = onnxruntime.InferenceSession(os.path.expanduser("~/hammou/iliar_solution2/src/iliar_solution2/dummy_model.onnx"))
            self.get_logger().info("ONNX model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load ONNX model: {e}")

        self.sub_left = Subscriber(self, CompressedImage, "/audibot/left_camera/image_raw/compressed", qos_profile=qos_profile)
        self.sub_front = Subscriber(self, CompressedImage, "/audibot/front_camera/image_raw/compressed", qos_profile=qos_profile)
        self.sub_right = Subscriber(self, CompressedImage, "/audibot/right_camera/image_raw/compressed", qos_profile=qos_profile)

        self.steering_pub = self.create_publisher(Float64, "/audibot/steering_cmd", 10)
        self.throttle_pub = self.create_publisher(Float64, "/audibot/throttle_cmd", 10)

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_left, self.sub_front, self.sub_right], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.image_cb)

    def preprocess_frames(self, input_image):
        resized = cv2.resize(input_image, (128, 128))
        normalized = resized / 255.0
        ordered_img = normalized.transpose(2, 0, 1)
        final_img = ordered_img.astype(np.float32)[np.newaxis, ...]
        return final_img

    def image_cb(self, left_msg, front_msg, right_msg):
        try:
            left_img = self.bridge.compressed_imgmsg_to_cv2(left_msg, "rgb8")
            front_img = self.bridge.compressed_imgmsg_to_cv2(front_msg, "rgb8")
            right_img = self.bridge.compressed_imgmsg_to_cv2(right_msg, "rgb8")

            left_img = self.preprocess_frames(left_img)
            front_img = self.preprocess_frames(front_img)
            right_img = self.preprocess_frames(right_img)

            inputs = {"left": left_img, "forward": front_img, "right": right_img}
            outputs = self.model.run(None, inputs)

            self.get_logger().info(f"Model outputs: {outputs}")

            steering_msg = Float64()
            steering_msg.data = float(outputs[0][0])
            self.steering_pub.publish(steering_msg)

            throttle_msg = Float64()
            throttle_msg.data = 0.8
            self.throttle_pub.publish(throttle_msg)
        except Exception as e:
            self.get_logger().error(f"Error in image_cb: {e}")


def main(args=None):
    rclpy.init(args=args)
    autopilot = Autopilot()
    try:
        rclpy.spin(autopilot)
    except KeyboardInterrupt:
        pass
    finally:
        autopilot.get_logger().info("Shutting down autopilot.")
        rclpy.shutdown()

if __name__ == "__main__":
    main()
