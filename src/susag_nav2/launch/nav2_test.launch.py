import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

ROS_DISTRO = os.environ.get('ROS_DISTRO', '')

def generate_launch_description():
    # Launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')

    # Default paths (use PathJoinSubstitution — no os.path.join)
    default_map = PathJoinSubstitution([
        FindPackageShare('susag_nav2'), 'map', 'susag_map.yaml'
    ])

    # Choose params path depending on ROS_DISTRO
    if ROS_DISTRO == 'humble':
        default_params = PathJoinSubstitution([
            FindPackageShare('susag_nav2'), 'param', 'test_config.yaml'
        ])
    else:
        default_params = PathJoinSubstitution([
            FindPackageShare('susag_nav2'), 'param', 'test_config.yaml'
        ])

    nav2_bringup_launch = PathJoinSubstitution([
        FindPackageShare('nav2_bringup'), 'launch', 'bringup_launch.py'
    ])

    rviz_config = PathJoinSubstitution([
        FindPackageShare('susag_nav2'), 'rviz', 'susag_nav.rviz'
    ])

    return LaunchDescription([
        # Declare args (default_value can be a Substitution)
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=TextSubstitution(text='false'),
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Full path to map yaml'
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Full path to Nav2 params yaml'
        ),


        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_bringup_launch),
            launch_arguments={
                'map': map_file,
                'use_sim_time': use_sim_time,
                'params_file': params_file,
                'autostart': 'true',                 # <- be explicit
                # 'use_respawn': 'false',            # optional
                # 'use_composition': 'False',        # optional (Humble default is False)
            }.items(),
        ),

        # Explicit lifecycle manager to force nodes ACTIVE
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'autostart': True,
                'bond_timeout': 0.0,
                'node_names': [
                    'map_server',
                    'amcl',
                    'planner_server',
                    'controller_server',
                    'behavior_server',
                    'bt_navigator',
                    'waypoint_follower',
                    'velocity_smoother',
                    'collision_monitor',
                    'docking_server',
                ],
            }],
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),
    ])
