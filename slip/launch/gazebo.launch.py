import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_slip = get_package_share_directory('slip')
    urdf_file = os.path.join(pkg_slip, 'urdf', 'slip.urdf')
    world_file = os.path.join(pkg_slip, 'worlds', 'slip_world.world')

    robot_description = ParameterValue(Command(['xacro ', urdf_file]), value_type=str)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py'
            ])
        ]),
        launch_arguments={'world': world_file}.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }]
    )

    # Delay spawn until Gazebo is fully up
    spawn_entity = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-file', urdf_file,
                    '-entity', 'slip',
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '0.11',
                ],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
        gazebo,
        robot_state_publisher,
        spawn_entity,
    ])
