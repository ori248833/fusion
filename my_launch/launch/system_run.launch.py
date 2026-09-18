import os
import logging
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

main_package_path = get_package_share_directory('my_launch')
lidar_package_path = get_package_share_directory('lidar_cone_detector')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_launch_description():
    
    # ==========================================
    # 1. 配置 2D YOLO 节点
    # ==========================================
    yolo_params_path = os.path.join(
        main_package_path, 
        'configs', 
        'yolo_detector.yaml'
    )
    
    yolo_node = Node(
        package='cone_detector',       
        executable='yolo_detector',    
        name='yolo_detector',
        output='screen',
        parameters=[yolo_params_path]
    )

    # ==========================================
    # 2. 配置 3D 雷达 Launch 文件
    # ==========================================
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(lidar_package_path, 'launch', 'infer_trt_cuda.launch.py')
        )
    )            

    # ==========================================
    # 3. 配置 融合节点 (fs_fusion_box)
    # ==========================================
    config_file = os.path.join(
        main_package_path,
        'configs',
        'fusion_box.yaml'
    )

    fusion_node = Node(
        package='fs_fusion_box',
        executable='fusion_box_node',
        name='fusion_box_node',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('/perception/camera/cones_custom', '/yolo/cones'),
        ]
    )

    # ==========================================
    # 4. 组合启动列表
    # ==========================================
    launch_list = [
        yolo_node,  
        lidar_launch,
        fusion_node,
    ]

    return LaunchDescription(launch_list)
