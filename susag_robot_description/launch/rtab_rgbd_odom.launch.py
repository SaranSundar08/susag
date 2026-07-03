from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    rgbd_odom = Node(
        package='rtabmap_odom',          
        executable='rgbd_odometry',
        name='rgbd_odometry',
        output='screen',
        parameters=[{
            'use_sim_time': True,

            # Frames: adapt if your base frame is different
            'frame_id': 'base_link',     # your robot base frame
            'odom_frame_id': 'odom',
            'publish_tf': True,         # let robot_localization handle TF later
            'guess_frame_id': 'base_link',

            # VO tuning (can tweak later)
            'Odom/Strategy': '0',
            'Vis/MinInliers': '15',
            'Vis/MaxFeatures': '1000',
        }],
        remappings=[
            ('rgb/image',       '/susag/depth/depth_camera/image_raw'),
            ('depth/image',     '/susag/depth/depth_camera/depth/image_raw'),
            ('rgb/camera_info', '/susag/depth/depth_camera/camera_info'),
        ]
    )

    return LaunchDescription([rgbd_odom])
