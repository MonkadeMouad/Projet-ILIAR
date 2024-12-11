#!/usr/bin/env python3

"""teleop_command.py

Ce nœud souscrit à /joy pour écouter l'état des boutons et des axes de la manette,
puis publie les commandes d'accélérateur et de volant sur Gazebo pour contrôler le robot.

"""

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64

class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_command')

        # Création des Publishers pour les topics /steer et /throttle
        self.steering_pub = self.create_publisher(Float64, 'audibot/steering_cmd', 10)
        self.throttle_pub = self.create_publisher(Float64, 'audibot/throttle_cmd', 10)
        self.brake_pub = self.create_publisher(Float64, 'audibot/brake_cmd', 10)

        # Création de la Subscription au topic /joy (état de la manette)
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_cb,
            10)

        # Initialisation de la veille automatique
        self.sleep_mode = False
        self.inactivity_timer = self.create_timer(5.0, self.enter_sleep_mode)
        self.last_activity_time = self.get_clock().now()

        self.get_logger().info('TeleopNode a été démarré.')

    def joy_cb(self, msg):
        """
        Callback qui est appelé à chaque fois qu'un message est reçu sur le topic /joy.
        Il calcule les commandes à partir des entrées de la manette et les publie.
        """

        # Mettre à jour l'heure de la dernière activité
        self.last_activity_time = self.get_clock().now()

        # Sortir du mode veille si nécessaire
        if self.sleep_mode:
            self.get_logger().info('Reprise du mode actif.')
            self.sleep_mode = False

        # Récupération des valeurs des axes de la manette
        steer_input = msg.axes[0]  # Axe 0 de la manette (généralement pour la direction)
        steer_command = Float64()
        steer_command.data = steer_input * 60  # Applique un facteur de conversion, ajustez selon vos besoins

        throttle_input = msg.axes[3]  # Axe 3 de la manette (généralement pour l'accélération)
        brake_input = msg.buttons[1]  # Bouton 1 pour le freinage (par exemple)
        reverse_input = msg.buttons[2]  # Bouton 2 pour la marche arrière (par exemple)

        throttle_command = Float64()
        brake_command = Float64()

        if brake_input:
            # Freinage
            throttle_command.data = 0.0
            brake_command.data = 1.0
        elif reverse_input:
            # Marche arrière
            throttle_command.data = -1.0 * throttle_input
            brake_command.data = 0.0
        else:
            # Accélération normale
            throttle_command.data = throttle_input
            brake_command.data = 0.0

        # Publication des commandes sur les topics correspondants
        self.steering_pub.publish(steer_command)
        self.throttle_pub.publish(throttle_command)
        self.brake_pub.publish(brake_command)

        # Optionnel : Afficher les valeurs publiées pour le débogage
        self.get_logger().debug(f'Steer: {steer_command.data}, Throttle: {throttle_command.data}, Brake: {brake_command.data}')

    def enter_sleep_mode(self):
        """
        Passe en mode veille si aucune activité n'est détectée pendant un certain temps.
        """
        time_since_last_activity = (self.get_clock().now() - self.last_activity_time).nanoseconds / 1e9
        if time_since_last_activity >= 5.0 and not self.sleep_mode:
            self.get_logger().info('Mode veille activé. Aucune activité détectée.')
            self.sleep_mode = True


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
