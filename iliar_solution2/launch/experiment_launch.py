#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemple ROS2 launch file avec PathJoinSubstitution pour éviter l'erreur
'dict' object has no attribute 'perform_substitution'.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

# Substitutions utiles
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition

def generate_launch_description():
    #
    # 1) Déclaration des arguments
    #
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='road.world',
        description='Nom du fichier .world à utiliser pour la simulation'
    )

    gazebo_gui_arg = DeclareLaunchArgument(
        'gazebo_gui',
        default_value='true',
        description='True pour lancer l\'interface graphique Gazebo'
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value='gui.rviz',  # ex: "gui.rviz"
        description='Nom (ou chemin) du fichier de config RViz relatif à config/'
    )

    autopilot_node_arg = DeclareLaunchArgument(
        'autopilot_node',
        default_value='autopilot',
        description='Nom de l\'exécutable pour l\'autopilot'
    )

    teleop_node_arg = DeclareLaunchArgument(
        'teleop_node',
        default_value='teleop_command',
        description='Nom de l\'exécutable pour la téléop'
    )

    #
    # 2) On récupère la LaunchConfiguration associée à chaque argument
    #
    world = LaunchConfiguration('world')
    gazebo_gui = LaunchConfiguration('gazebo_gui')
    rviz_config = LaunchConfiguration('rviz_config')
    autopilot_exec = LaunchConfiguration('autopilot_node')
    teleop_exec = LaunchConfiguration('teleop_node')

    #
    # 3) Noeuds
    #

    # Gazebo server
    # On construit le chemin complet vers le .world en utilisant PathJoinSubstitution
    gazebo_server = Node(
        package='gazebo_ros',
        executable='gzserver',
        name='gazebo_server',
        output='screen',
        arguments=[
            '-s', 'libgazebo_ros_factory.so',
            # Au lieu de world.perform(...), on fait :
            PathJoinSubstitution([
                FindPackageShare('iliar_gazebo'),
                'worlds',
                world  # << c'est la LaunchConfiguration('world')
            ])
        ]
    )

    # Gazebo client (l'interface) : lancé seulement si gazebo_gui == "true"
    gazebo_client = Node(
        package='gazebo_ros',
        executable='gzclient',
        name='gazebo_client',
        output='screen',
        # Condition IfCondition => lance le node si "gazebo_gui" est "true"
        condition=IfCondition(gazebo_gui)
    )

    # Rviz : on construit le chemin complet de config RViz (optionnel)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_node',
        output='screen',
        arguments=[
            '-d',
            PathJoinSubstitution([
                FindPackageShare('iliar_solution2'),
                'config',
                rviz_config
            ])
        ],
        parameters=[{'use_sim_time': True}],
    )

    # Joy node
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Téléop
    teleop_node = Node(
        package='iliar_solution2',
        executable=teleop_exec,  # On lance l'exécutable indiqué par l'argument
        name='teleop_command_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'max_steer': 2.0},
            {'max_throttle': 1.0},
            {'steering_axis': 0},
            {'throttle_axis': 3},
        ]
    )

    # Autopilot
    autopilot_node = Node(
        package='iliar_solution2',
        executable=autopilot_exec,
        name='autopilot_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Command Mux
    command_mux_node = Node(
        package='iliar_solution2',
        executable='command_mux',
        name='command_mux_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Path publisher
    path_publisher_node = Node(
        package='iliar_solution2',
        executable='path_publisher',
        name='path_publisher',
        output='screen',
        # On passe l'argument world comme param supplémentaire, 
        # ou on remappe selon votre code
        arguments=[world],
        parameters=[{'use_sim_time': True}],
    )

    # Path dilater
    path_dilater_node = Node(
        package='iliar_solution2',
        executable='path_dilater',
        name='path_dilater',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'offset': 3.75}
        ],
        remappings=[
            ('in_path', '/road_path'),
            ('out_path', '/lanes/track'),
        ]
    )

    # dist_to_path
    dist_to_path_node = Node(
        package='iliar_solution2',
        executable='dist_to_path',
        name='dist_to_path',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'base_frame': 'base_link'},
            {'signed_distance': False}
        ],
        remappings=[
            ('/path', '/lanes/track'),
            ('/dist_to_path', '/dist_to_path'),
        ]
    )

    # scorer
    scorer_node = Node(
        package='iliar_solution2',
        executable='scorer',
        name='scorer',
        output='screen',
        parameters=[{'use_sim_time': True}],
        remappings=[
            ('/serie', '/dist_to_path'),
            # => scorer lira /dist_to_path et publiera /min, /max, /avg
        ]
    )

    # traveled_distance
    traveled_distance_node = Node(
        package='iliar_solution2',
        executable='traveled_distance',
        name='traveled_distance',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'base_frame': 'base_link'}
        ],
        remappings=[
            ('/path', '/lanes/track'),
            ('/traveled_dist', '/traveled_dist'),
            ('/curv_abscissa', '/curv_abscissa'),
        ]
    )

    #
    # 4) Construction du LaunchDescription
    #
    ld = LaunchDescription()

    # On ajoute d'abord nos arguments
    ld.add_action(world_arg)
    ld.add_action(gazebo_gui_arg)
    ld.add_action(rviz_config_arg)
    ld.add_action(autopilot_node_arg)
    ld.add_action(teleop_node_arg)

    # On ajoute les noeuds
    ld.add_action(gazebo_server)
    ld.add_action(gazebo_client)
    ld.add_action(rviz_node)
    ld.add_action(joy_node)
    ld.add_action(teleop_node)
    ld.add_action(autopilot_node)
    ld.add_action(command_mux_node)

    ld.add_action(path_publisher_node)
    ld.add_action(path_dilater_node)
    ld.add_action(dist_to_path_node)
    ld.add_action(scorer_node)
    ld.add_action(traveled_distance_node)

    return ld
