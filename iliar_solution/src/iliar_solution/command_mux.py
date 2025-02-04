import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, UInt8, Bool


class CommandMuxNode(Node):
    """
    CommandMuxNode - Sélectionne les commandes entre téléopération et autopilot
    en fonction de l'état du bouton deadman.
    """

    def __init__(self):
        super().__init__('command_mux_node')

        # État des commandes pour chaque source
        self.commands = {
            "teleop": {"steering": 0.0, "throttle": 0.0, "brake": 0.0, "gear": 0},
            "autopilot": {"steering": 0.0, "throttle": 0.0, "brake": 0.0, "gear": 0},
        }

        # État du bouton deadman
        self.deadman_active = False

        # Publishers finaux
        self.steering_pub = self.create_publisher(Float64, "/audibot/steering_cmd", 10)
        self.throttle_pub = self.create_publisher(Float64, "/audibot/throttle_cmd", 10)
        self.brake_pub = self.create_publisher(Float64, "/audibot/brake_cmd", 10)
        self.gear_pub = self.create_publisher(UInt8, "/audibot/gear_cmd", 10)

        # Subscribers pour la téléopération
        self.create_subscription(Float64, "/teleop/steering_cmd", self.teleop_steering_cb, 10)
        self.create_subscription(Float64, "/teleop/throttle_cmd", self.teleop_throttle_cb, 10)
        self.create_subscription(Float64, "/teleop/brake_cmd", self.teleop_brake_cb, 10)
        self.create_subscription(UInt8, "/teleop/gear_cmd", self.teleop_gear_cb, 10)
        self.create_subscription(Bool, "/teleop/deadman", self.deadman_cb, 10)

        # Subscribers pour l'autopilot
        self.create_subscription(Float64, "/autopilot/steering_cmd", self.autopilot_steering_cb, 10)
        self.create_subscription(Float64, "/autopilot/throttle_cmd", self.autopilot_throttle_cb, 10)

        # Timer pour publier les commandes priorisées
        self.create_timer(0.1, self.publish_priority_commands)

    # Callbacks pour la téléopération
    def teleop_steering_cb(self, msg):
        self.commands["teleop"]["steering"] = msg.data

    def teleop_throttle_cb(self, msg):
        self.commands["teleop"]["throttle"] = msg.data

    def teleop_brake_cb(self, msg):
        self.commands["teleop"]["brake"] = msg.data

    def teleop_gear_cb(self, msg):
        self.commands["teleop"]["gear"] = msg.data

    def deadman_cb(self, msg: Bool):
        """
        Met à jour l'état du bouton deadman.
        """
        self.deadman_active = msg.data
        if self.deadman_active:
            self.get_logger().info("Deadman button pressed: teleop active.")
        else:
            self.get_logger().info("Deadman button released: autopilot active.")

    # Callbacks pour l'autopilot
    def autopilot_steering_cb(self, msg):
        self.commands["autopilot"]["steering"] = msg.data

    def autopilot_throttle_cb(self, msg):
        self.commands["autopilot"]["throttle"] = msg.data

    # Publication des commandes priorisées
    def publish_priority_commands(self):
        """
        Publie les commandes en fonction de la source active.
        """
        source = "teleop" if self.deadman_active else "autopilot"

        # Récupérer les commandes de la source active
        steering_cmd = self.commands[source]["steering"]
        throttle_cmd = self.commands[source]["throttle"]
        brake_cmd = self.commands[source]["brake"]
        gear_cmd = self.commands[source]["gear"]

        # Publier les commandes
        self.steering_pub.publish(Float64(data=steering_cmd))
        self.throttle_pub.publish(Float64(data=throttle_cmd))
        self.brake_pub.publish(Float64(data=brake_cmd))
        self.gear_pub.publish(UInt8(data=gear_cmd))

        # Log des commandes publiées
        self.get_logger().info(
            f"[{source.upper()}] Published - Steering: {steering_cmd}, Throttle: {throttle_cmd}, Brake: {brake_cmd}, Gear: {gear_cmd}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CommandMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down CommandMuxNode.")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
