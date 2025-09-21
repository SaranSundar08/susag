from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('susag_bringup')

    pcl_params = os.path.join(pkg_share, 'config', 'grid_map_pcl.yaml')
    viz_params = os.path.join(pkg_share, 'config', 'visualization.yaml')

    # Run the PCL loader node (this reads your point cloud and builds grid map)
    pcl_node = Node(
        package="grid_map_pcl",
        executable="grid_map_pcl_loader_node",
        name="grid_map_pcl",
        parameters=[pcl_params],
        output="screen"
    )

    # Visualization node
    viz_node = Node(
        package="grid_map_visualization",
        executable="grid_map_visualization",
        name="grid_map_visualization",
        parameters=[viz_params],
        output="screen"
    )

    # Optional RViz
    rviz_cfg = os.path.join(pkg_share, 'config', 'grid_map_basic.rviz')
    rviz_args = ['-d', rviz_cfg] if os.path.exists(rviz_cfg) else []
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=rviz_args,
        output="screen"
    )

    return LaunchDescription([pcl_node, viz_node, rviz_node])
