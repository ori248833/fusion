import os
import logging
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
    with open(yolo_params_path, encoding='utf-8') as config_handle:
        yolo_settings = yaml.safe_load(config_handle)['/yolo_detector']['ros__parameters']
    
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
        ),
        launch_arguments={
            'input_topic': LaunchConfiguration('input_topic'),
            'enable_latency_test': LaunchConfiguration('lidar_enable_latency_test'),
            'latency_log_every_n': LaunchConfiguration('lidar_latency_log_every_n'),
            'queue_log_interval_sec': LaunchConfiguration('lidar_queue_log_interval_sec'),
        }.items(),
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

    with open(config_file, encoding='utf-8') as config_handle:
        fusion_settings = yaml.safe_load(config_handle)['/fusion_box_node']['ros__parameters']
    depth_settings = fusion_settings.get('depth_fallback', {})
    depth_mode = depth_settings.get('mode', 'off')
    if depth_mode not in ('off', 'preview', 'map'):
        raise ValueError('depth_fallback.mode must be off, preview or map')
    depth_nodes = []
    if depth_mode != 'off':
        depth_nodes.append(Node(
            package='my_launch',
            executable='depth_cone_fallback',
            name='depth_cone_fallback',
            output='screen',
            parameters=[{
                'depth_topic': depth_settings['depth_topic'],
                'max_range_m': depth_settings['max_range_m'],
                'min_yolo_confidence': depth_settings['min_yolo_confidence'],
                'lidar_frame': fusion_settings['lidar_frame'],
                'image_width': fusion_settings['image_width'],
                'image_height': fusion_settings['image_height'],
                'intrinsics': [fusion_settings['camera_matrix'][key]
                               for key in ('fx', 'fy', 'cx', 'cy')],
                'lidar_to_camera_matrix': fusion_settings['lidar_to_camera_matrix'],
            }],
        ))

    # ==========================================
    # 4. 组合启动列表
    # ==========================================
    monitor_config = os.path.join(main_package_path, 'configs', 'frame_monitor.yaml')
    monitor_node = Node(
        package='my_launch',
        executable='frame_monitor',
        name='frame_monitor',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_frame_monitor')),
        parameters=[monitor_config, {
            'output_root': ParameterValue(
                LaunchConfiguration('frame_monitor_output_root'), value_type=str),
            'lidar_topic': ParameterValue(LaunchConfiguration('input_topic'), value_type=str),
            'camera_topic': str(yolo_settings['image_topic']),
            'camera_compressed': bool(yolo_settings.get('use_compressed', False)),
            'yolo_max_fps': float(yolo_settings.get('max_fps', 30.0)),
            'use_sim_time': ParameterValue(
                LaunchConfiguration('frame_monitor_use_sim_time'), value_type=bool),
            'config_files': [yolo_params_path, config_file, monitor_config],
        }],
        # Large reports can need more than launch's normal shutdown grace period.
        sigterm_timeout='60',
        sigkill_timeout='120',
    )

    launch_list = [
        DeclareLaunchArgument('enable_frame_monitor', default_value='true'),
        DeclareLaunchArgument(
            'frame_monitor_output_root',
            default_value=str(Path.home() / 'perception_runs')),
        DeclareLaunchArgument('frame_monitor_use_sim_time', default_value='false'),
        DeclareLaunchArgument('input_topic', default_value='/lidar_points'),
        DeclareLaunchArgument('lidar_enable_latency_test', default_value='false'),
        DeclareLaunchArgument('lidar_latency_log_every_n', default_value='10'),
        DeclareLaunchArgument('lidar_queue_log_interval_sec', default_value='5.0'),
        monitor_node,
        yolo_node,  
        lidar_launch,
        *depth_nodes,
        fusion_node,
    ]

    return LaunchDescription(launch_list)
