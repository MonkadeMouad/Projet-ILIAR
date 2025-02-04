"""
TeleopNode - Contrôle minimal du véhicule avec braquage et accélérateur

Ce noeud ROS2 minimal permet de publier des commandes pour l’angle de braquage et l’accélérateur d’un véhicule robotisé en se basant sur les entrées d’une manette de jeu.

### Fonctionnalités :
1. **Souscription :**
   - Topic `/joy` : Reçoit les messages de type `sensor_msgs/msg/Joy` contenant l'état des axes et boutons de la manette.

2. **Publication :**
   - Topic `steering_cmd` : Publie l’angle de braquage (type `std_msgs/msg/Float64`).
   - Topic `throttle_cmd` : Publie l’intensité de l’accélérateur (type `std_msgs/msg/Float64`).

### Méthodes principales :
- `__init__`: Initialise les abonnements et les publications pour le nœud.
- `joy_cb(msg)`: Callback pour traiter les messages du topic `/joy` et publier les commandes correspondantes sur `steering_cmd` et `throttle_cmd`.

###fonctionnement :
- **Angle de braquage (`/audibot/steering_cmd`)** : Contrôlé par l'axe horizontal du joystick gauche (valeur de -1 à 1).  
     Formule : `commande = 10 * valeur_axe`
- **Accélération (`/audibot/throttle_cmd`)** : Contrôlée par l'axe vertical du joystick droit (valeur de 0 à 1).  
     Formule : `commande = valeur_axe`
- **Freinage (`/audibot/brake_cmd`)** : Contrôlée par l'axe vertical du joystick droit (valeur de -1 à 0).  
     Formule : `commande = 8000*-valeur_axe`
- **Marche Avant/arrière (`/audibot/gear_cmd`)** : Contrôlée par les flèches du haut (Marche avant) et du bas (Marche arrière).  
     Formule : `commande = valeur_bouton `
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, UInt8
from sensor_msgs.msg import Joy


class TeleopNode(Node):
    """
    TeleopNode - Noeud pour contrôler un véhicule robotisé à l'aide d'une manette.

    Ce nœud reçoit les données d'une manette via le topic `/joy` et publie les commandes d'angle de braquage et d'accélération sur les topics `/steering_cmd` et `/throttle_cmd`, respectivement.
    """

    def __init__(self):
        """
        Initialise le noeud TeleopNode, les Publishers pour les topics de commande
        et la Subscription au topic `/joy`.

         Paramètres :
         - max_steer : Valeur maximale de la commande de direction (par défaut 1.0).
         - max_throttle : Valeur maximale de la commande d'accélération (par défaut 1.0).
         - steering_axis : Indice de l'axe de la manette utilisé pour la direction (par défaut 0, généralement l'axe gauche de la manette).
         - throttle_axis : Indice de l'axe de la manette utilisé pour l'accélération (par défaut 1, généralement l'axe droit de la manette).
         - max_brake : valeur maximale de la commande de freinage (par défaut 8000)
        """
        super().__init__("teleop_node")

        # Déclarer les paramètres max_steer, max_throttle, steering_axis et throttle_axis
        self.declare_parameter(
            "max_steer", 1.0
        )  # Valeur par défaut 1.0 pour l'angle de braquage
        self.declare_parameter(
            "max_throttle", 1.0
        )  # Valeur par défaut 1.0 pour l'accélération
        self.declare_parameter(
            "steering_axis", 0
        )  # Indice par défaut 0 pour l'axe de direction
        self.declare_parameter(
            "throttle_axis", 1
        )  # Indice par défaut 1 pour l'axe d'accélération
        self.declare_parameter(
            "max_brake", 8000.0
        )  # Plage maximale de freinage par défaut
        self.declare_parameter("deadman_button", 5)  # Indice du bouton deadman

        # Attribut pour suivre l'état du bouton deadman
        self.deadman_state = False  # le bouton n'est pas appuyé

        # Récupérer les valeurs des paramètres
        self.max_steer = (
            self.get_parameter("max_steer").get_parameter_value().double_value
        )
        self.max_throttle = (
            self.get_parameter("max_throttle").get_parameter_value().double_value
        )
        self.max_brake = (
            self.get_parameter("max_brake").get_parameter_value().double_value
        )
        self.steering_axis = (
            self.get_parameter("steering_axis").get_parameter_value().integer_value
        )
        self.throttle_axis = (
            self.get_parameter("throttle_axis").get_parameter_value().integer_value
        )
        self.deadman_button = (
            self.get_parameter("deadman_button").get_parameter_value().integer_value
        )

        # Publishers
        self.steering_pub = self.create_publisher(Float64, "/audibot/steering_cmd", 10)
        self.throttle_pub = self.create_publisher(Float64, "/audibot/throttle_cmd", 10)
        self.brake_pub = self.create_publisher(Float64, "/audibot/brake_cmd", 10)
        self.gear_pub = self.create_publisher(UInt8, "/audibot/gear_cmd", 10)

        # Subscription
        self.joy_sub = self.create_subscription(Joy, "/joy", self.joy_cb, 10)

        # états internes
        self.gear = 0  # 0 pour marche avant, 1 pour marche arrière

        # logs

        self.get_logger().info("TeleopNode has been started.")

    def joy_cb(self, msg: Joy):
        """
        Callback exécuté lorsque des données sont reçues sur le topic `/joy`.
        Calcule et publie les commandes pour le braquage (steering_cmd), l'accélération (throttle_cmd),
        ,le freinage (brake_cmd) et le passage en marche avant/arrière  si le bouton deadman est tenu appuyé.

        Args:
            msg (Joy): Message contenant les données des axes et des boutons de la manette.
        """
        # états internes
        self.deadman_state = (
            msg.buttons[self.deadman_button] == 1
        )  # True si appuyé, False sinon

        if self.deadman_state:
            # Axes du joystick gauche (braquage) et joystick droit (accélération/freinage)
            steering_axis = msg.axes[0]  # Axe horizontal du joystick gauche
            throttle_axis = msg.axes[3]  # Axe vertical du joystick droit
            gear_axis_value = msg.axes[5]  # Axe correspondant aux flèches haut/bas

            # Calcul des commandes
            steering_cmd = (
                self.max_steer * steering_axis
            )  # Mapping de [-1, 1] à [-max_steer, max_steer]

            if throttle_axis < 0:
                brake_cmd = self.max_brake * (
                    -throttle_axis
                )  # Commande de freinage pour throttle_axis négatif
                throttle_cmd = -0.0  # Pas d'accélération lors du freinage
            else:
                brake_cmd = -0.0  # Pas de freinage lorsque throttle_axis est positif
                throttle_cmd = self.max_throttle * throttle_axis

            if gear_axis_value == 1:  # Flèche du haut : Marche avant
                self.gear = 0
            elif gear_axis_value == -1:  # Flèche du bas : Marche arrière
                self.gear = 1

            # Publication des commandes
            # Commande de braquage
            steering_msg = Float64()
            steering_msg.data = steering_cmd
            self.steering_pub.publish(steering_msg)

            # Commande d'accélération
            throttle_msg = Float64()
            throttle_msg.data = throttle_cmd
            self.throttle_pub.publish(throttle_msg)

            # Commande de freinage
            brake_msg = Float64()
            brake_msg.data = brake_cmd
            self.brake_pub.publish(brake_msg)

            # Commande de marche Avant/Arrière
            gear_msg = UInt8()
            gear_msg.data = self.gear
            self.gear_pub.publish(gear_msg)

            # Log des commandes publiées
            self.get_logger().info(
                f"Published steer: {steering_cmd}, throttle: {throttle_cmd}, brake: {brake_cmd}, gear: {self.gear}"
            )
        else:
            # Publier des commandes à 0 pour s'assurer que le véhicule s'arrete lorsque le deadman buton n'est pas préssé
            #self.steering_pub.publish(Float64(data=0.0))
            #self.throttle_pub.publish(Float64(data=0.0))
            #self.brake_pub.publish(Float64(data=8000.0))
            # Si le bouton deadman n'est pas appuyé, arrêter le véhicule
            self.get_logger().info("Deadman button not pressed: stopping the vehicule.")


def main(args=None):
    """Instantiate node and class."""
    rclpy.init(args=args)
    # create node
    teleop_node = TeleopNode()
    # run with default executor

    try:
        rclpy.spin(teleop_node)
    except KeyboardInterrupt:
        teleop_node.get_logger().info("Shutting down TeleopNode.")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()