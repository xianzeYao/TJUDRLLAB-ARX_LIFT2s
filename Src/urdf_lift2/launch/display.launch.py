#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    model_path = LaunchConfiguration("model").perform(context)
    with open(model_path, "r") as urdf_file:
        robot_description = urdf_file.read()

    return [
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            condition=IfCondition(LaunchConfiguration("gui")),
        ),
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            condition=UnlessCondition(LaunchConfiguration("gui")),
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
        ),
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory("lift2")
    default_model_path = os.path.join(pkg_share, "urdf", "lift2.urdf")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value=default_model_path,
                description="Absolute path to robot urdf file",
            ),
            DeclareLaunchArgument(
                "gui",
                default_value="true",
                description="Use joint_state_publisher_gui if true, else joint_state_publisher",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
