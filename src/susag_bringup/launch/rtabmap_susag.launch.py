from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os 

def generate_launch_description():
    delete_db = LaunchConfiguration('delete_db_on_start')
    return LaunchDescription([
        DeclareLaunchArgument(
            'delete_db_on_start',
            default_value='true',
            description='delete RTABmap database on start'
        ),

        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            output="screen",
            parameters=[
                # Core
                {"use_sim_time": True},
                {"frame_id": "base_link"},
                {"odom_frame_id": "odom"},
                {"map_frame_id": "map"},
                # Subscriptions
                {"subscribe_rgb": True},
                {"subscribe_depth": True},
                {"subscribe_scan": True},
                {"approx_sync": True},
                # Topics (match your list)
                {"rgb_topic": "/susag/depth/depth_camera/image_raw"},
                {"depth_topic": "/susag/depth/depth_camera/depth/image_raw"},
                {"camera_info_topic": "/susag/depth/depth_camera/camera_info"},
                {"scan_topic": "/scan"},
                # Tuning (can also live in YAML)
                {"Cloud/MaxDepth": "6.0"},
                {"Cloud/VoxelSize": "0.05"},
            ],
            arguments=[
                "--delete_db_on_start"  # toggled by the arg
            ] if str(delete_db.perform({})).lower() == "true" else []
        ),
    ])