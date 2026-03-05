#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    model_path = LaunchConfiguration("model").perform(context)
    with open(model_path, "r") as urdf_file:
        robot_description = urdf_file.read()

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0", "0", "0", "0", "0", "0", "base_link", "base_footprint"],
        ),
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=["-entity", "lift2", "-file", model_path],
            output="screen",
        ),
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory("lift2")
    default_model_path = os.path.join(pkg_share, "urdf", "lift2.urdf")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    get_package_share_directory("gazebo_ros"),
                    "launch",
                    "gazebo.launch.py",
                ]
            )
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value=default_model_path,
                description="Absolute path to robot urdf file",
            ),
            gazebo,
            OpaqueFunction(function=launch_setup),
        ]
    )
