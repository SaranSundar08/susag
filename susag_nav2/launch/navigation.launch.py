# Copyright (c) 2021 Juan Miguel Jimeno
#
# Licensed under the Apache License, Version 2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    TextSubstitution,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


# 1:1 to match every sandbox result and the Gazebo world launch
# (susag_new_model/launch/gazebo_barn.launch.py); see the comment there.
BARN_SCALE = 1.0
BARN_MAPS_DIR = (
    f'/home/saran/robohouse_ws/src/BARN_dataset/scaled_{BARN_SCALE:g}/map_files'
)


def generate_launch_description():

    nav2_launch_path = PathJoinSubstitution(
        [FindPackageShare('nav2_bringup'), 'launch', 'bringup_launch.py']
    )

    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare('susag_nav2'), 'rviz', 'susag_nav.rviz']
    )

    world_idx = LaunchConfiguration('world_idx')
    default_map_path = [
        TextSubstitution(text=f'{BARN_MAPS_DIR}/yaml_'),
        world_idx,
        TextSubstitution(text='.yaml'),
    ]

    nav2_config_path = PathJoinSubstitution(
        [FindPackageShare('susag_nav2'), 'param', 'navigation_tgmppi_tight.yaml']
    )

    return LaunchDescription([

        DeclareLaunchArgument(
            name='sim',
            default_value='false',
            description='Use simulation time if true'
        ),

        DeclareLaunchArgument(
            name='rviz',
            default_value='true',
            description='Run RViz'
        ),

        DeclareLaunchArgument(
            name='world_idx',
            default_value='0',
            description='BARN world/map index; uses the 1:1 (scaled_1) map'
        ),

        DeclareLaunchArgument(
            name='map',
            default_value=default_map_path,
            description='Optional full map YAML override'
        ),

        DeclareLaunchArgument(
            name='nav2_params',
            default_value=nav2_config_path,
            description='Full path to the nav2 params YAML file'
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch_path),
            launch_arguments={
                'map': LaunchConfiguration('map'),
                'use_sim_time': LaunchConfiguration('sim'),
                'params_file': LaunchConfiguration('nav2_params')
            }.items()
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path],
            condition=IfCondition(LaunchConfiguration('rviz')),
            parameters=[
                {'use_sim_time': LaunchConfiguration('sim')}
            ]
        )
    ])
