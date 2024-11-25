#!/usr/bin/env python3


"""teleop_command.py

Ce nœud souscrit à /joy pour écouter l'état des boutons et des axes de la manette,
puis publie les commandes d'accélérateur et de volant sur Gazebo pour contrôler le robot.

"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64

class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_command')

        # Création des Publishers pour les topics /steer et /throttle
        self.steering_pub = self.create_publisher(Float64, 'audibot/steering_cmd', 10)
        self.throttle_pub = self.create_publisher(Float64, 'audibot/throttle_cmd', 10)

        # Création de la Subscription au topic /joy (état de la manette)
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_cb,
            10)

        self.get_logger().info('TeleopNode a été démarré.')

    def joy_cb(self, msg):
        """
        Callback qui est appelé à chaque fois qu'un message est reçu sur le topic /joy.
        Il calcule les commandes à partir des entrées de la manette et les publie.
        """

        # Récupération des valeurs des axes de la manette
        # On suppose que l'axe 0 contrôle la direction (steer) et l'axe 1 contrôle l'accélération (throttle)
        # Ajustez les indices en fonction de votre configuration de manette

        # Contrôle de la direction (steer)
        steer_input = msg.axes[0]  # Axe 0 de la manette (généralement pour la direction)
        steer_command = Float64()
        steer_command.data = steer_input * 60  # Applique un facteur de conversion, ajustez selon vos besoins

        # Contrôle de l'accélération (throttle)
        throttle_input = msg.axes[3]  # Axe 1 de la manette (généralement pour l'accélération)
        throttle_command = Float64()
        throttle_command.data = throttle_input  # Convertir la plage [-1.0, 1.0] à [0.0, 1.0]

        # Publication des commandes sur les topics correspondants
        self.steering_pub.publish(steer_command)
        self.throttle_pub.publish(throttle_command)

        # Optionnel : Afficher les valeurs publiées pour le débogage
        self.get_logger().debug(f'Steer: {steer_command.data}, Throttle: {throttle_command.data}')

def main(args=None):
    rclpy.init(args=args)
    teleop_commandnode = TeleopNode()
    try:
        rclpy.spin(teleop_commandnode)
    except KeyboardInterrupt:
        pass
    finally:
        teleop_commandnode.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
