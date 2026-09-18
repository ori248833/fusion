import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # Use the calibration file installed with this package.
    config_file = os.path.join(
        get_package_share_directory('fs_fusion_box'),
        'config',
        'fusion_box.yaml'
    )

    return LaunchDescription([
        Node(
            package='fs_fusion_box',
            executable='fusion_box_node',
            name='fusion_box_node',
            output='screen',

            parameters=[config_file],

            remappings=[
                ('/perception/camera/cones_custom', '/yolo/cones'),
            ]
        )
    ])
