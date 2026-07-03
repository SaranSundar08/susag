from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    pkg_share = get_package_share_directory('susag_robot_description')
    params_file = os.path.join(pkg_share, 'config', 'rtabmap_params.yaml')

    # 1) Sync RGB + Depth + CameraInfo -> RGBD
    rgbd_sync = Node(
        package='rtabmap_sync',
        executable='rgbd_sync',
        name='rgbd_sync',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'approx_sync': True,
        }],
        remappings=[
            ('rgb/image',        '/susag/depth/depth_camera/image_raw'),
            ('depth/image',      '/susag/depth/depth_camera/depth/image_raw'),
            ('rgb/camera_info',  '/susag/depth/depth_camera/camera_info'),
            ('rgbd_image',       '/rgbd_image'),  # output
        ]
    )

    # 2) RTAB-Map SLAM node
    rtabmap_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[params_file],
        remappings=[
            ('rgbd_image', '/rgbd_image'),
            ('odom',       '/odom'),       # from rgbd_odometry
        ]
    )

    # 3) 3D point cloud map (global)
    cloud_map_node = Node(
        package='rtabmap_util',
        executable='point_cloud_xyzrgb',
        name='cloud_map',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'frame_id': 'map',   # global map frame
            'voxel_size': 0.05,
        }],
        remappings=[
            ('cloud',      '/cloud_map'),
            ('rgbd_image', '/rgbd_image'),
            ('odom',       '/odom'),
        ]
    )

    return LaunchDescription([
        rgbd_sync,
        rtabmap_node,
        cloud_map_node
    ])
