#! coding: utf-8

"""
This node computes the traveled distance of the robot along a path. 
The distance is computed from the projected position of the robot on the path

Subscriptions:
    - path (nav_msgs/Path): The path to follow

Publications:
    - traveled_dist (std_msgs/Float64): The traveled distance along the path
    - curv_abscissa (std_msgs/Float64): The curvilinear abscissa of the robot on the path

Parameters:
    - base_frame (string): The frame of the robot
"""

# External imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException

import geom2d
import numpy as np


class TraveledDistance(Node):
    """
    This node computes the traveled distance of the robot along a path.
    The distance is computed from the projected position of the robot on the path
    """

    def __init__(self):
        super().__init__("traveled_distance")

        # The base frame of the robot
        self.declare_parameter("base_frame", "base_link")

        # The TF objects required to request the transforms
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.last_valid_position = None
        self.total_traveled_distance = 0.0
        self.total_pub = self.create_publisher(Float64, "/traveled_dist", 10)
        self.curv_pub = self.create_publisher(Float64, "/curv_abscissa", 10)
        # Pour débogage, on publie la projection la plus proche
        self.pub_point = self.create_publisher(Marker, "closest_point", 1)

        self.path_sub = self.create_subscription(
            Path,
            '/lanes/track',
            self.dist_callback,
            10
        )

    def dist_callback(self, path_msg: Path):
        path_frame = path_msg.header.frame_id
        path_2d = [
            (p.pose.position.x, p.pose.position.y)
            for p in path_msg.poses
        ]

        # Récupère position du robot dans path_frame
        base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        try:
            trans = self._tf_buffer.lookup_transform(
                path_frame,
                base_frame,
                rclpy.time.Time()
            )
            car_xy = (trans.transform.translation.x, trans.transform.translation.y)
        except LookupException as e:
            self.get_logger().error(f"Failed transform: {e}")
            return

        # Trouver la projection sur le polygone
        # On teste chaque segment [p[i], p[i+1]]
        best_proj = None
        best_dist2 = float('inf')
        best_s_abscissa = 0.0

        # Pré-calcul de la longueur cumulative pour chaque sommet 
        # => pour obtenir l'abscisse curviligne
        cumul_len = [0.0]
        for i in range(1, len(path_2d)):
            seg_len = geom2d.Point(*path_2d[i-1]).distanceToPoint(geom2d.Point(*path_2d[i]))
            cumul_len.append(cumul_len[-1] + seg_len)
        # Ajouter le dernier segment (fermé)
        seg_len = geom2d.Point(*path_2d[-1]).distanceToPoint(geom2d.Point(*path_2d[0]))
        total_length = cumul_len[-1] + seg_len

        # Parcours des segments
        for i in range(len(path_2d)):
            p1 = path_2d[i]
            p2 = path_2d[(i+1) % len(path_2d)]
            seg = geom2d.Segment(geom2d.Point(*p1), geom2d.Point(*p2))

            proj = seg.projectPoint(geom2d.Point(*car_xy))
            dist2 = proj.distanceToPoint(geom2d.Point(*car_xy))**2
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_proj = proj
                # Calcul abscisse curviligne locale
                # cumul_len[i] => abscisse du point p1
                # alpha = fraction sur le segment p1->p2
                seg_len = seg.length()
                alpha = seg.startPoint().distanceToPoint(proj) / seg_len if seg_len > 1e-9 else 0.0
                best_s_abscissa = cumul_len[i] + alpha * seg_len

        if best_proj is None:
            return

        # Normaliser l'abscisse (0.0 -> 1.0)
        curv_abscissa = best_s_abscissa / total_length

        # Publier le Marker pour le point projeté (débogage)
        marker = Marker()
        marker.header.frame_id = path_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = best_proj.x
        marker.pose.position.y = best_proj.y
        marker.pose.position.z = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.pub_point.publish(marker)

        # Mettre à jour la distance parcourue (en avant)
        # On compare la nouvelle abscisse avec la précédente
        if self.last_curv_abscissa is None:
            self.last_curv_abscissa = curv_abscissa
            # rien à additionner
        else:
            # Faire attention à l'éventuel saut 1.0 -> 0.0 si le robot fait un tour complet
            diff = curv_abscissa - self.last_curv_abscissa
            if diff < -0.5:  
                # On suppose qu'on est repassé de ~1.0 -> ~0.0 => ajout d'un tour
                diff = 1.0 - self.last_curv_abscissa + curv_abscissa
            elif diff > 0.5:
                # Cas improbable (on recule ?)
                # Adaptez la logique si vous souhaitez ignorer la marche arrière
                diff = -(self.last_curv_abscissa + (1.0 - curv_abscissa))

            # On n’ajoute que les déplacements positifs (si on veut “forward only”)
            if diff > 0:
                self.total_traveled_distance += diff * total_length

            self.last_curv_abscissa = curv_abscissa

        # Publication
        self.total_pub.publish(Float64(data=self.total_traveled_distance))
        self.curv_pub.publish(Float64(data=curv_abscissa))



def main(args=None):
    """Instantiation node and class."""
    rclpy.init(args=args)

    # Create the node
    traveled_distance = TraveledDistance()

    # Spin the node
    try:
        rclpy.spin(traveled_distance)
    except KeyboardInterrupt:
        pass

    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
