import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share_dir = get_package_share_directory('susag_teleop')
    config_file = os.path.join(package_share_dir, 'config', 'max_teleop.yaml')

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[config_file],
        output='screen',
    )

    teleop_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy_node',
        parameters=[config_file],
        remappings=[
            ('/cmd_vel', '/cmd_vel'),
        ],
        output='screen',
    )

    return LaunchDescription([
        joy_node,
        teleop_node,
    ])
