from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_centerhead_engine = PathJoinSubstitution([
        FindPackageShare('lidar_cone_detector'),
        'models',
        'pointpillars_centerhead_full_int8_qat.engine',
    ])

    def arg(name, default, description):
        return DeclareLaunchArgument(name, default_value=default, description=description)

    return LaunchDescription([
        # backbone_head_engine_path: CenterHead-only TensorRT engine。默认只使用包内 models 目录。
        # 如果要临时对齐 FP16，可显式切到同目录 pointpillars_centerhead_fp16.engine。
        arg(
            'backbone_head_engine_path',
            default_centerhead_engine,
            'CenterHead TensorRT engine path. Defaults to packaged models/pointpillars_centerhead_full_int8_qat.engine.'),

        # input_topic: 输入 PointCloud2 话题。当前默认 Hesai 点云 /lidar_points。
        # 如果驱动能发布 point_step=16 的紧凑 XYZI topic，切过去通常能降低 pinned_copy/H2D。
        arg(
            'input_topic', '/lidar_points',
            'Input sensor_msgs/PointCloud2 topic. Prefer compact XYZI topic if available.'),

        # score_thresh: CenterHead heatmap 分数阈值。越高误检越少、后处理越快，但可能漏检。
        # my_cone_anchorfree.yaml 的默认是 0.30；低到 0.05 会显著增加 TopK/NMS 压力。
        arg(
            'score_thresh', '0.3',
            'Detection score threshold. Higher is faster/fewer false positives; lower improves recall but costs NMS.'),

        # nms_thresh: BEV rotated NMS IoU 阈值。越低抑制越强，重复框更少；越高保留更多近邻框。
        # 锥桶通常 0.1 比较激进且快；如果相邻锥桶被互相抑制，可小幅提高。
        arg(
            'nms_thresh', '0.1',
            'Rotated BEV NMS IoU threshold. Increase if nearby true cones suppress  each other.'),

        # enable_tracking: 是否把检测到的锥桶做简单全局去重并保存 txt。
        # 正式低延迟运行建议 false；需要赛道锥桶记录时再打开。
        arg(
            'enable_tracking', 'false',
            'Enable simple global cone dedup/tracking and TXT saving. Disable for lowest latency.'),

        # publish_markers: 是否发布 RViz MarkerArray。RViz 可视化方便但会增加 pub/CPU 开销。
        # 上车只给下游算法用 bbox 时建议 false。
        arg(
            'publish_markers', 'true',
            'Publish RViz visualization markers. Disable to reduce publish overhead.'),

        # enable_latency_test: 是否拆分打印各阶段耗时。打开会有 CUDA event 同步，适合定位瓶颈；
        # 最终测真实性能/p95 时建议 false，并配合 tegrastats 或外部统计。
        arg(
            'enable_latency_test', 'false',
            'Print stage latency. Useful for profiling but adds synchronization; disable for final timing.'),

        # latency_log_every_n: 延迟日志打印频率，每 N 帧打印一次。越大日志扰动越小。
        # 逐帧采样设 1；正式长跑建议 10/30。
        arg(
            'latency_log_every_n', '10',
            'Print latency every N frames. Use 1 for detailed profiling, 10/30 for lower log overhead.'),

        # latency_print_interval: latency_log_every_n 的兼容别名/覆盖项。0 表示不用它。
        # 如果外部脚本已经用这个名字，可以设成 10/30；否则保持 0。
        arg(
            'latency_print_interval', '0',
            'Optional override for latency_log_every_n. 0 means disabled.'),

        # max_voxels: CUDA voxelization 允许的最大 voxel 数。
        # my_cone_anchorfree.yaml TEST 为 30000；如果现场 voxel 数稳定低于 20k，可调小一点省显存。
        arg(
            'max_voxels', '30000',
            'Maximum voxels kept per frame. Must fit engine profile; larger is safer but can cost memory/time.'),

        # bev_clear_mode: spatial_features 清零策略。
        # occupied/prev_cells 只清上一帧写过的 cell，通常最快；full 每帧清约 13.1MB，最稳但慢；
        # auto 在 prev voxel 数超过阈值时全清。
        arg(
            'bev_clear_mode', 'occupied',
            'BEV clear mode: occupied/prev_cells, full, or auto. occupied usually minimizes clear cost.'),

        # bev_clear_auto_full_threshold: auto 模式阈值，prev voxel 数 >= 阈值时改全图清空。
        # anchor-free 新范围下 voxel 数可能和旧模型不同；建议先用 occupied，auto 可从 12000 起步。
        arg(
            'bev_clear_auto_full_threshold', '12000',
            'Auto BEV clear threshold. Raise if auto switches to full clear too often.'),

        # marker_lifetime_sec: RViz marker 保留时间。越短越不容易残影，越长越平滑。
        # publish_markers:=false 时无影响。
        arg(
            'marker_lifetime_sec', '0.5',
            'RViz marker lifetime in seconds. Only applies when publish_markers is true.'),

        # gpu_preprocess: 是否用 CUDA 直接解析 PointCloud2 raw bytes 并 voxelize。
        # 正式必须 true；设 false 会退回 CPU XYZI vector 路径，便于排查但更慢。
        arg(
            'gpu_preprocess', 'true',
            'Use CUDA raw PointCloud2 parsing + voxelization. Disable only for fallback/debug.'),

        # fixed_hesai_layout: 对 Hesai 固定字段布局走快速路径，避免每帧扫描 fields。
        # 当前点云 x/y/z/intensity 在 0/4/8/12 时应保持 true；换雷达/字段布局异常时设 false。
        arg(
            'fixed_hesai_layout', 'true',
            'Assume Hesai x/y/z/intensity float32 offsets 0/4/8/12 after first validation.'),

        # compact_raw_h2d: Hesai point_step=26 时只把前 16 字节 XYZI 打包进 pinned buffer。
        # direct_raw_h2d:=true 时会跳过 compact，改为整帧 raw 一次性 memcpy，通常能压低 CPU copy 尖峰。
        arg(
            'compact_raw_h2d', 'true',
            'Compact PointCloud2 raw to XYZI before H2D when layout allows it.'),

        # direct_raw_h2d: 跳过 CPU 逐点 compact，整帧 raw bytes 直接 H2D，由 GPU 按 offset 读 XYZI。
        # Hesai point_step=26 下会多传一些 H2D 字节，但把 32 万次小 stride copy 变成一次 memcpy；
        # 当前 Orin 日志主要抖在 pinned_copy，默认打开它更适合稳定 p95。
        arg(
            'direct_raw_h2d', 'true',
            'Copy full raw PointCloud2 to GPU without CPU compacting. Usually reduces pinned_copy spikes.'),

        # clear_voxel_buffers: 是否每帧清空完整 d_voxels/d_coords。
        # CUDA VFE 只读取 voxel_num_points 范围内的真实点，padding 区域不参与计算，
        # 所以默认 false 可以少清约 10MB/帧；排查输入残留时可临时设 true。
        arg(
            'clear_voxel_buffers', 'false',
            'Clear full voxel/coord buffers every frame. Usually keep false to reduce voxel_reset bandwidth.'),

        # fast_pointcloud_parser: CPU fallback 路径的快速解析开关。
        # gpu_preprocess:=true 且布局正常时不会走到；保留 true 即可。
        arg(
            'fast_pointcloud_parser', 'true',
            'Use optimized CPU parser in fallback path. Normally unused when gpu_preprocess is true.'),

        # cpu_range_filter: CPU fallback 解析时是否先按 point_cloud_range 过滤。
        # 能减少 CPU fallback 的点数；GPU preprocess 正常时无影响。
        arg(
            'cpu_range_filter', 'true',
            'Apply range filtering in CPU fallback parser. Normally unused in GPU path.'),

        # cls_output_is_logits: CenterHead hm 输出是否是 raw logits。当前导出工具默认输出 raw hm。
        # 如果你导出的 engine 已经内置 sigmoid，需要设 false，否则分数会被二次 sigmoid。
        arg(
            'cls_output_is_logits', 'true',
            'Whether CenterHead hm output is logits. Set false if exported engine already applies sigmoid.'),

        # feature_map_stride: CenterHead FEATURE_MAP_STRIDE。my_cone_anchorfree.yaml 里是 2。
        # 如果 backbone/head 导出结构变了，decode 的真实坐标步长也必须同步改。
        arg(
            'feature_map_stride', '2',
            'CenterHead feature map stride. Must match DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE.'),

        # filter_invalid_boxes: 发布前过滤 NaN、超范围中心、极端尺寸框。
        # 正式建议 true；如果调试低分 raw 输出想看所有框，可临时 false。
        arg(
            'filter_invalid_boxes', 'true',
            'Filter NaN/out-of-range/unphysical boxes before NMS/publish.'),

        # nms_pre_max_size: CPU reference 后处理的 pre-NMS TopK。
        # debug/full CPU 路径用它；越大更保守但 NMS 更慢。GPU 路径另有 gpu_nms_pre_max_size。
        arg(
            'nms_pre_max_size', '4096',
            'CPU/reference pre-NMS TopK. Larger keeps more candidates but costs CPU NMS.'),

        # nms_post_max_size: NMS 后最多发布多少个检测框。
        # 锥桶实际数量远小于 500；调小可保护异常低阈值时的发布开销。
        arg(
            'nms_post_max_size', '500',
            'Maximum detections after NMS. Lower to cap publish/postprocess cost under noisy thresholds.'),

        # use_gpu_postprocess: 是否启用 GPU score threshold / TopK / rotated NMS。
        # 正式 split 模式建议 true；需要和旧 CPU 后处理逐项对齐时设 false。
        arg(
            'use_gpu_postprocess', 'true',
            'Use GPU threshold/sort/NMS. Disable for CPU reference/fallback debugging.'),

        # gpu_nms_pre_max_size: GPU NMS 的 pre-NMS TopK，直接影响 rotated IoU 计算量 O(K^2)。
        # 锥桶场景最终框很少，512 更利于 p95；低阈值查漏检时可临时调到 1024/2048。
        arg(
            'gpu_nms_pre_max_size', '512',
            'GPU pre-NMS TopK. Smaller is faster; increase if low-threshold recall debugging needs more candidates.'),

        # gpu_post_max_candidates: GPU threshold 后候选 buffer 容量。
        # 正常 score_thresh=0.30 下 2048 通常够用并降低排序开销；如果日志提示 overflow，可增到 4096/8192 或提高阈值。
        arg(
            'gpu_post_max_candidates', '2048',
            'GPU candidate buffer size after thresholding. Increase if overflow warning appears.'),

        # 下面把 launch 参数转成 ROS 参数；数值/布尔参数用 ParameterValue 固定类型，
        # 避免 ROS2 把字符串默认值按错误类型传给 C++ 节点。
        Node(
            package='lidar_cone_detector',
            executable='ros2_infer_trt_cuda',
            name='lidar_cone_detector_trt_cuda',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'backbone_head_engine_path': LaunchConfiguration('backbone_head_engine_path'),
                'input_topic': LaunchConfiguration('input_topic'),
                'score_thresh': ParameterValue(LaunchConfiguration('score_thresh'), value_type=float),
                'nms_thresh': ParameterValue(LaunchConfiguration('nms_thresh'), value_type=float),
                'enable_tracking': ParameterValue(LaunchConfiguration('enable_tracking'), value_type=bool),
                'publish_markers': ParameterValue(LaunchConfiguration('publish_markers'), value_type=bool),
                'enable_latency_test': ParameterValue(LaunchConfiguration('enable_latency_test'), value_type=bool),
                'latency_log_every_n': ParameterValue(LaunchConfiguration('latency_log_every_n'), value_type=int),
                'latency_print_interval': ParameterValue(
                    LaunchConfiguration('latency_print_interval'), value_type=int),
                'max_voxels': ParameterValue(LaunchConfiguration('max_voxels'), value_type=int),
                'bev_clear_mode': LaunchConfiguration('bev_clear_mode'),
                'bev_clear_auto_full_threshold': ParameterValue(
                    LaunchConfiguration('bev_clear_auto_full_threshold'), value_type=int),
                'marker_lifetime_sec': ParameterValue(LaunchConfiguration('marker_lifetime_sec'), value_type=float),
                'gpu_preprocess': ParameterValue(LaunchConfiguration('gpu_preprocess'), value_type=bool),
                'fixed_hesai_layout': ParameterValue(LaunchConfiguration('fixed_hesai_layout'), value_type=bool),
                'compact_raw_h2d': ParameterValue(LaunchConfiguration('compact_raw_h2d'), value_type=bool),
                'direct_raw_h2d': ParameterValue(LaunchConfiguration('direct_raw_h2d'), value_type=bool),
                'clear_voxel_buffers': ParameterValue(
                    LaunchConfiguration('clear_voxel_buffers'), value_type=bool),
                'fast_pointcloud_parser': ParameterValue(LaunchConfiguration('fast_pointcloud_parser'), value_type=bool),
                'cpu_range_filter': ParameterValue(LaunchConfiguration('cpu_range_filter'), value_type=bool),
                'cls_output_is_logits': ParameterValue(LaunchConfiguration('cls_output_is_logits'), value_type=bool),
                'feature_map_stride': ParameterValue(
                    LaunchConfiguration('feature_map_stride'), value_type=int),
                'filter_invalid_boxes': ParameterValue(LaunchConfiguration('filter_invalid_boxes'), value_type=bool),
                'nms_pre_max_size': ParameterValue(LaunchConfiguration('nms_pre_max_size'), value_type=int),
                'nms_post_max_size': ParameterValue(LaunchConfiguration('nms_post_max_size'), value_type=int),
                'use_gpu_postprocess': ParameterValue(
                    LaunchConfiguration('use_gpu_postprocess'), value_type=bool),
                'gpu_nms_pre_max_size': ParameterValue(
                    LaunchConfiguration('gpu_nms_pre_max_size'), value_type=int),
                'gpu_post_max_candidates': ParameterValue(
                    LaunchConfiguration('gpu_post_max_candidates'), value_type=int),
            }],
        ),
    ])
