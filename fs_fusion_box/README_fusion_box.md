# Fusion Box 融合节点使用说明

本文对应以下两个优化文件：

```text
fs_fusion_box_node.cpp
fs_fusion_box_math.cpp
```

节点采用“LiDAR 实时输出、Camera 异步补色”的结构：LiDAR 决定目标是否存在、目标位置和 Map 发布时刻；YOLO 只为 LiDAR 已经检测到的锥桶提供颜色。相机延迟或漏检不会阻塞、删除或新增 LiDAR 锥桶。

## 1. 设计目标

优化后的节点遵循以下原则：

1. LiDAR 约 10 Hz，每一帧到达后立即发布 Map。
2. YOLO 约 15～20 Hz，可以晚到，但超过 0.5 秒的结果不再使用。
3. Camera 按原始采集时间戳查找对应的历史 LiDAR 帧。
4. 颜色写入内部 Track，由后续 LiDAR 帧继承。
5. YOLO 单帧漏检不会把已有颜色重新变为 `UNKNOWN`。
6. Camera 不能增加、删除或修改 LiDAR 目标位置。
7. 下层 SLAM 使用的输出话题、消息格式和 header 语义保持不变。
8. 3D 可视化不再调用；2D 可视化可以在运行时开关。

## 2. 文件职责

### `fs_fusion_box_node.cpp`

负责：

- ROS 2 参数、订阅和发布；
- LiDAR 每帧实时 Map 发布；
- 历史 LiDAR 缓存；
- 短期锥桶 Track；
- 相机结果延迟检查和时间戳配对；
- 异步融合任务线程；
- Camera 与 Map 颜色的显式映射；
- Track 颜色更新和冲突确认；
- 2D 可视化开关；
- 周期健康日志和 Ctrl+C 最终统计。

### `fs_fusion_box_math.cpp`

负责：

- LiDAR 3D 框投影到相机图像；
- 计算投影框和 YOLO 框的 IoU；
- 底边纵向门控；
- 横向中心门控；
- 构造 LiDAR→YOLO 候选匹配；
- 按 IoU 进行严格一对一分配；
- 根据 Camera 颜色和 LiDAR `label` 生成融合颜色。

## 3. 输入与输出接口

为了兼容下层 SLAM，以下接口保持不变。

### LiDAR 输入

```text
Topic: /detected_cones_bbox
Type:  lidar_cone_detector/msg/ThreeDConeArray
```

节点使用：

- `header.stamp`：LiDAR 采集时间；
- `header.frame_id`：LiDAR 坐标系；
- `cones[].center.x/y/z`：锥桶中心；
- `cones[].size.x/y/z`：3D 框尺寸；
- `cones[].yaw`：3D 框朝向；
- `cones[].label`：锥桶大小类别。

LiDAR 标签约定保持不变：

```text
label = 1 → 小锥桶
label = 2 → 大锥桶
```

### Camera 输入

```text
Topic: /perception/camera/cones_custom
Type:  cone_interfaces/msg/ConeArray
```

节点使用：

- `header.stamp`：原始图像采集时间，而不是 YOLO 完成时间；
- `cones[].center.x/y`：YOLO 框中心像素坐标；
- `cones[].size.x/y`：YOLO 框宽度和高度；
- `cones[].confidence`：YOLO 置信度；
- `cones[].color`：由 YOLO 节点映射后的 Camera 颜色枚举。

### SLAM 输出

```text
Topic: /perception/fusion/map
Type:  drd25_msgs/msg/Map
```

输出规则：

```text
Map.header = 当前 LiDAR 帧 header
Map.track  = 当前 LiDAR 帧的全部锥桶
```

每个输出锥桶：

```text
x     = 当前 LiDAR center.x
y     = 当前 LiDAR center.y
color = Track 已确认颜色；没有颜色时为 UNKNOWN
```

Camera 不修改 `x/y`，也不会产生 Camera-only 锥桶。

## 4. 编译前要求

两个 `.cpp` 文件需要放回原 ROS 2 包中对应的源码位置。项目还必须包含：

```text
include/fs_fusion_box/fs_fusion_box_math.hpp
include/fs_fusion_box/visualizer.hpp
```

以及以下消息包或依赖：

```text
rclcpp
rcl_interfaces
Eigen3
OpenCV
vision_msgs
cone_interfaces
lidar_cone_detector
drd25_msgs
```

`fs_fusion_box_math.hpp` 中的融合函数声明必须与实现一致：

```cpp
FusionResult fuse_measurements(
    const std::vector<ProjectedBox>& projected_boxes,
    const vision_msgs::msg::Detection3DArray::ConstSharedPtr& lidar_msg,
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr& camera_msg,
    double overlap_threshold);
```

如果旧头文件中还存在五参数版本，需要同步改为以上四参数声明。

`FusionResult` 至少需要保留：

```cpp
std::vector<drd25_msgs::msg::Cone> fused_cones;
std::vector<size_t> unmatched_camera_indices;
```

`Visualizer` 仍需提供：

```cpp
publishSyntheticView(
    camera_msg,
    lidar_msg,
    params,
    projected_boxes,
    matches,
    track_ids,
    final_colors,
    decisions);
```

节点已经不再调用：

```cpp
publishFusedCones(...)
```

## 5. 编译指令

在 ROS 2 工作空间根目录执行：

```bash
source /opt/ros/<ROS_DISTRO>/setup.bash
colcon build --packages-select fs_fusion_box --symlink-install
source install/setup.bash
```

如果包名不是 `fs_fusion_box`，使用实际包名：

```bash
colcon list
```

查看包注册的可执行程序：

```bash
ros2 pkg executables fs_fusion_box
```

## 6. 启动方法

如果可执行程序名称为 `fusion_box_node`：

```bash
ros2 run fs_fusion_box fusion_box_node
```

使用参数文件：

```bash
ros2 run fs_fusion_box fusion_box_node --ros-args \
  --params-file <PATH_TO_CONFIG>/fusion_box.yaml
```

也可以从 launch 文件启动：

```bash
ros2 launch <PACKAGE_NAME> <LAUNCH_FILE>.launch.py
```

启动后检查接口：

```bash
ros2 node info /fusion_box_node
ros2 topic hz /detected_cones_bbox
ros2 topic hz /perception/camera/cones_custom
ros2 topic hz /perception/fusion/map
ros2 topic echo /perception/fusion/map
```

正常情况下：

```text
/perception/fusion/map 频率 ≈ LiDAR 输入频率 ≈ 10 Hz
```

## 7. 完整工作逻辑

### 7.1 LiDAR 实时链路

每一帧 LiDAR 到达后：

```text
LiDAR callback
  → 清理过期 Track
  → 当前锥桶与已有 Track 做最近邻关联
  → 为没有匹配的锥桶创建新 Track
  → 从 Track 读取颜色
  → 构造 drd25_msgs/msg/Map
  → 立即发布 /perception/fusion/map
  → 保存 LiDAR 历史帧和对应 track_id
  → 清理过期历史帧
```

Map 发布不经过相机融合任务队列，因此相机投影、匹配、可视化或工作线程积压都不会阻塞 LiDAR 发布。

旧代码中的雷达任务清队列逻辑已经移除。现在只有 Camera 补色任务允许 latest-only。

### 7.2 内部 Track

每个 Track 保存：

```text
id
最近一次 LiDAR x/y
最近一次 LiDAR 时间戳
已经确认的颜色
待确认的冲突颜色
冲突颜色连续次数
最近一次颜色时间戳
```

当前帧锥桶和已有 Track 使用全局距离候选进行一对一关联：

```text
1. 计算当前所有锥桶与所有有效 Track 的 x/y 距离
2. 删除超过 track_match_distance 的候选
3. 按距离从小到大排序
4. 一个锥桶和一个 Track 都只能使用一次
5. 未关联锥桶创建新 Track
```

当前实现直接使用 LiDAR `center.x/y`。因此 `track_match_distance` 的单位必须与 LiDAR 坐标单位一致。默认值 `1.50` 假设坐标单位为米。

如果车辆高速运动且 LiDAR 坐标系随车运动，短期最近邻可能出现关联错误。当前版本没有引入 TF/里程计，以避免改变现有系统接口；比赛前需要根据实际速度和锥桶间距测试 `track_match_distance`。

### 7.3 LiDAR 历史缓存

每个历史帧保存：

```text
LiDAR header.stamp
原始 ThreeDConeArray
该帧每个锥桶对应的 track_id
```

历史缓存按时间清理，而不是仅按帧数清理：

```text
当前 LiDAR 时间 - 历史帧时间 > lidar_history_duration
→ 删除历史帧
```

默认保存 `0.60 s`，用于覆盖允许的 `0.50 s` Camera 延迟并留出处理余量。

### 7.4 Camera 延迟检查

Camera 到达后计算：

```text
camera_age = 当前 ROS 时间 - camera.header.stamp
```

规则：

```text
camera_age > max_camera_result_age
→ 直接丢弃

camera_age > camera_age_warn_threshold
→ 仍处理，但计入高延迟统计
```

默认值：

```text
正常目标：≤ 0.40 s
硬上限：  ≤ 0.50 s
```

计算 Camera age 时，当前 ROS 时钟必须与图像 header 使用同一时间体系。如果系统使用仿真时钟，相关节点应统一设置：

```yaml
use_sim_time: true
```

### 7.5 Camera 与历史 LiDAR 时间配对

对每个有效 Camera 结果，在历史缓存中寻找：

```text
best_lidar = argmin |camera_stamp - lidar_stamp|
```

只有满足：

```text
|camera_stamp - best_lidar_stamp| <= max_sync_diff
```

才进入空间匹配。

`max_sync_diff` 和 `max_camera_result_age` 含义不同：

| 参数 | 限制内容 |
|---|---|
| `max_sync_diff` | 两个传感器的采集时间差 |
| `max_camera_result_age` | YOLO 结果晚到的总时间 |

不要因为 YOLO 处理慢而把 `max_sync_diff` 改成 0.5 秒。

### 7.6 Camera 补色工作线程

有效 Camera 任务被送入独立工作线程：

```text
历史 LiDAR
  → 转为 Detection3DArray
  → 3D 框投影到图像
Camera ConeArray
  → 显式颜色映射
  → 转为 Detection2DArray
  → 雷达框搜索 YOLO 框
  → 更新对应 Track 颜色
```

Camera 任务队列只保留最新有效任务。工作线程来不及处理时，旧 Camera 补色任务可以被更新结果替代，但 LiDAR Map 发布不会丢帧。

### 7.7 颜色何时出现在 Map 中

Camera 匹配完成后只更新内部 Track，不额外发布重复 Map：

```text
Camera 匹配成功
→ Track.color 更新
→ 下一帧 LiDAR 到达
→ 当前 LiDAR 位置 + Track 已确认颜色
→ 发布 Map
```

LiDAR 为 10 Hz 时，颜色更新后通常还需等待最多约 `0.1 s` 才会出现在输出 Map 中。

## 8. 雷达主导的空间匹配

融合函数首先为每一个 LiDAR 检测创建输出锥桶：

```text
x/y   = LiDAR 位置
color = UNKNOWN
```

然后每个有效 LiDAR 投影框主动搜索 YOLO 框。

### 第一级：投影有效性

LiDAR 框必须满足：

- 有角点位于相机前方；
- 投影框与图像区域相交；
- 裁剪后投影面积大于 0。

无效投影不会删除 LiDAR 锥桶，只是本次无法获得 Camera 颜色。

### 第二级：底边纵向门控

```text
LiDAR bottom = box.y + box.height
Camera bottom = box.y + box.height
```

动态阈值：

```text
Ty = max(0.25 × LiDAR 投影框高度, 5 px)
```

要求：

```text
|Camera bottom - LiDAR bottom| <= Ty
```

### 第三级：横向中心门控

动态阈值：

```text
Tx = max(1.5 × LiDAR 投影框宽度, 15 px)
```

要求：

```text
|Camera center_x - LiDAR center_x| <= Tx
```

### 第四级：IoU

通过几何门控后计算：

```text
IoU = intersection / union
```

只有：

```text
IoU > overlap_threshold
```

才成为候选关系。

### 严格一对一分配

所有候选按照 IoU 从高到低排序：

```text
LiDAR 已经使用 → 跳过
Camera 已经使用 → 跳过
否则接受该候选
```

因此不会再出现多个 LiDAR 框同时使用同一个 YOLO 框。

### YOLO 漏检

如果某个 LiDAR 框没有找到 YOLO 框：

```text
Track 已有颜色 → 下一次 LiDAR 输出继续沿用
Track 没有颜色 → 按 LiDAR 标签输出 UNKNOWN_SMALL（label=1）或 UNKNOWN_BIG（label=2）
```

LiDAR 锥桶本身永远保留。

## 9. Camera 颜色显式映射

`cone_interfaces/msg/Cone` 和 `drd25_msgs/msg/Cone` 属于两个消息包，不能假设颜色枚举数字相同。

节点执行明确转换：

```text
cone_interfaces::Cone::BLUE
→ drd25_msgs::Cone::BLUE

cone_interfaces::Cone::RED
→ drd25_msgs::Cone::RED

cone_interfaces::Cone::YELLOW_SMALL
→ drd25_msgs::Cone::YELLOW_SMALL

cone_interfaces::Cone::YELLOW_BIG
→ drd25_msgs::Cone::YELLOW_BIG

其他值
→ drd25_msgs::Cone::UNKNOWN
```

映射发生在 `convert_camera()`，随后才把 Map 颜色值写入中间 `class_id` 字符串。因此 `fuse_measurements()` 读取到的是 `drd25_msgs` 的颜色编号。

如果出现 UNKNOWN 映射，周期日志会显示：

```text
camera detections mapped to UNKNOWN; verify message enums
```

此时应检查：

```bash
ros2 interface show cone_interfaces/msg/Cone
ros2 interface show drd25_msgs/msg/Cone
ros2 topic echo /perception/camera/cones_custom
```

## 10. 黄色大小处理

LiDAR `label` 逻辑保持原有约定。

Camera 匹配为黄色时：

```text
LiDAR label = 1 → YELLOW_SMALL
LiDAR label = 2 → YELLOW_BIG
其他 label      → 使用 Camera 映射后的黄色类别
```

蓝色和红色直接使用 Camera 映射后的颜色。

## 11. Track 颜色更新与冲突

颜色更新规则：

```text
Track 当前 UNKNOWN
→ 立即接受有效 Camera 颜色

新颜色与 Track 当前颜色相同
→ 保持颜色并更新时间

新颜色与 Track 当前颜色冲突
→ 先记录候选颜色
→ 连续达到 color_conflict_confirmations 次后才覆盖
```

默认：

```text
color_conflict_confirmations = 2
```

这样可以避免一次 YOLO 误分类立即覆盖已经稳定的 Track 颜色。

Track 颜色不会因为 Camera 某一帧漏检而清空。只有 Track 超过 `track_timeout` 未再次被 LiDAR 看到时，整个 Track 才会被删除。

## 12. 可视化

### 默认比赛模式

```yaml
enable_visualization: false
```

关闭后不会：

- 构造可视化数据；
- 复制融合锥桶数组；
- 唤醒可视化线程处理数据；
- 调用 OpenCV 可视化；
- 发布 2D 合成图。

### 运行时开启

```bash
ros2 param set /fusion_box_node enable_visualization true
```

运行时关闭：

```bash
ros2 param set /fusion_box_node enable_visualization false
```

降低可视化频率：

```bash
ros2 param set /fusion_box_node visualization_every_n 5
```

表示每 5 个可视化候选任务处理一次。

当前只保留：

```cpp
visualizer_->publishSyntheticView(...);
```

黑色画布上的逐目标信息包括：

- LiDAR 帧时间戳；
- 每个 LiDAR/YOLO 框对应的 Track ID；
- LiDAR 编号与 YOLO 编号；
- 最终融合颜色；
- YOLO 置信度；
- 匹配对的实际 IoU；
- 匹配框中心之间的关联线；
- 历史颜色、颜色冲突和未匹配的紧凑状态标记。

可视化直接复用融合计算使用的 `projected_boxes` 和 `matches`，不再单独
重复计算 LiDAR 投影，因此显示结果与实际匹配逻辑保持一致。

以下 3D 可视化已经从节点调用链删除：

```cpp
visualizer_->publishFusedCones(...);
```

可视化线程使用条件变量，没有新数据时保持睡眠，不再每 5 ms 轮询。

## 13. 参数说明

### 原有参数

| 参数 | 默认值 | 当前作用 | 修改后是否立即生效 |
|---|---:|---|---|
| `lidar_frame` | `hesai_lidar` | Visualizer 使用的 LiDAR frame | 需重启 |
| `overlap_threshold` | `0.60` | IoU 接受阈值 | 需重启 |
| `image_width` | `640` | 投影图像宽度 | 需重启 |
| `image_height` | `480` | 投影图像高度 | 需重启 |
| `camera_matrix.fx` | `500.0` | 相机内参 fx | 需重启 |
| `camera_matrix.fy` | `500.0` | 相机内参 fy | 需重启 |
| `camera_matrix.cx` | `320.0` | 相机内参 cx | 需重启 |
| `camera_matrix.cy` | `240.0` | 相机内参 cy | 需重启 |
| `dist_coeffs` | `[0,0,0,0,0]` | 保存畸变参数 | 需重启 |
| `lidar_to_camera_matrix` | 空 | 4×4 LiDAR→Camera 外参，Row-major | 需重启 |
| `max_sync_diff` | `0.05` | Camera 与 LiDAR 最大采集时间差 | 需重启 |
| `sync_window` | `0.04` | 兼容旧配置，当前新流程未使用 | — |
| `lidar_window_size` | `10` | 兼容旧配置，已改为按时间缓存 | — |
| `camera_window_size` | `10` | 兼容旧配置，当前新流程未使用 | — |

注意：当前投影函数使用针孔模型 `K` 和 `T_l2c`，虽然会读取 `dist_coeffs`，但没有调用 OpenCV 畸变投影函数。如果输入图像已经是 rectified 图像，这是合理的；不要对 rectified 图像再次应用畸变。

### 新增内部参数

| 参数 | 默认值 | 说明 | 修改后是否立即生效 |
|---|---:|---|---|
| `lidar_history_duration` | `0.60` s | 历史 LiDAR 保存时间 | 需重启 |
| `max_camera_result_age` | `0.50` s | Camera 结果允许晚到的硬上限 | 需重启 |
| `camera_age_warn_threshold` | `0.40` s | Camera 高延迟统计阈值 | 需重启 |
| `track_match_distance` | `1.50` | LiDAR 锥桶与 Track 的最大 x/y 距离 | 需重启 |
| `track_timeout` | `1.00` s | Track 未更新后的删除时间 | 需重启 |
| `color_conflict_confirmations` | `2` | 冲突颜色覆盖前需要的连续确认次数 | 需重启 |
| `enable_visualization` | `false` | 2D 可视化总开关 | 是 |
| `visualization_every_n` | `1` | 可视化抽帧间隔 | 是 |
| `health_log_interval` | `5.0` s | 周期健康日志间隔 | 需重启 |

当前代码只有可视化相关参数注册了运行时更新回调。其他参数即使执行 `ros2 param set`，内部缓存值也不会立即改变，需要重启节点。

## 14. 推荐参数文件

```yaml
fusion_box_node:
  ros__parameters:
    lidar_frame: hesai_lidar

    image_width: 640
    image_height: 480
    camera_matrix.fx: 500.0
    camera_matrix.fy: 500.0
    camera_matrix.cx: 320.0
    camera_matrix.cy: 240.0
    dist_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]

    # 必须替换为实际 4x4 Row-major 外参
    lidar_to_camera_matrix: []

    overlap_threshold: 0.60
    max_sync_diff: 0.05

    lidar_history_duration: 0.60
    max_camera_result_age: 0.50
    camera_age_warn_threshold: 0.40

    track_match_distance: 1.50
    track_timeout: 1.00
    color_conflict_confirmations: 2

    enable_visualization: false
    visualization_every_n: 1
    health_log_interval: 5.0
```

不要直接使用空的 `lidar_to_camera_matrix` 参加比赛。必须填写实际标定矩阵。

## 15. 周期健康日志

默认每 5 秒打印一次：

```text
Health | lidar rx/pub 10.0/10.0 Hz (500/500), camera 17.8 Hz
accepted=420 expired=3 no_history=5
age p50/p95/max 82.1/191.4/403.2 ms
sync_p95=34.1 ms fusion_p95=4.5 ms
tracks=18 colored=86.2% updates=201 conflicts=2
```

字段含义：

| 字段 | 含义 |
|---|---|
| `lidar rx/pub` | 当前周期 LiDAR 接收/发布频率 |
| `(received/published)` | 启动以来 LiDAR 接收/发布总数 |
| `camera Hz` | Camera 输入频率 |
| `accepted` | 通过延迟和时间同步检查的 Camera 帧数 |
| `expired` | 超过 0.5 秒而丢弃的 Camera 帧数 |
| `no_history` | 没有找到时间差合格历史 LiDAR 的 Camera 帧数 |
| `age p50/p95/max` | Camera 结果年龄分布 |
| `sync_p95` | Camera/LiDAR 采集时间差 P95 |
| `fusion_p95` | 投影、空间匹配和颜色更新耗时 P95 |
| `tracks` | 当前有效 Track 数 |
| `colored` | 已发布 LiDAR 锥桶中有颜色的累计比例 |
| `updates` | Track 颜色接受或刷新次数 |
| `conflicts` | 新 Camera 颜色与已有 Track 颜色冲突次数 |

关键检查：

```text
LiDAR received 必须等于 LiDAR published
```

不相等时节点会输出 WARN。

## 16. Ctrl+C 最终统计

节点退出时会：

```text
running = false
唤醒融合条件变量
唤醒可视化条件变量
join 融合线程
join 可视化线程
打印最终摘要
```

最终摘要包括：

- 运行时长；
- LiDAR 接收/发布数量；
- LiDAR 内部损失数量；
- Camera 接收/接受数量；
- Camera 过期和无历史配对数量；
- 高延迟 Camera 数量；
- 被新 Camera 任务替代的旧任务数量；
- Camera age P50/P95/最大值；
- 同步时间差 P95；
- 融合处理耗时 P95；
- 最大融合任务队列深度；
- 累计有颜色锥桶比例；
- 颜色更新和冲突数；
- 显式映射为 UNKNOWN 的数量。

## 17. 常见问题排查

### Map 频率低于 LiDAR

检查：

```bash
ros2 topic hz /detected_cones_bbox
ros2 topic hz /perception/fusion/map
```

以及日志中的：

```text
LiDAR received/published
```

正常情况下两者总数应一致。

### Camera 大量 expired

说明 YOLO 端到端延迟超过：

```text
max_camera_result_age = 0.5 s
```

检查 YOLO timing 日志中的推理时间和 e2e 时间，不建议直接扩大时间同步阈值。

### Camera 大量 no_history

可能原因：

- Camera 与 LiDAR 时钟不一致；
- `max_sync_diff` 太小；
- Camera 结果到达时对应 LiDAR 已离开历史缓存；
- Camera header 被错误替换成推理完成时间；
- `lidar_history_duration` 小于 Camera 实际延迟。

### 所有颜色都是 UNKNOWN

依次检查：

```bash
ros2 topic echo /perception/camera/cones_custom
ros2 interface show cone_interfaces/msg/Cone
ros2 interface show drd25_msgs/msg/Cone
```

查看周期日志中的：

```text
Unknown color mappings
```

如果 UNKNOWN 映射为 0，但最终仍然全 UNKNOWN，应继续检查：

- 外参矩阵；
- Camera 图像尺寸；
- 内参；
- `sync_p95`；
- `overlap_threshold`；
- 底边和横向门控是否通过。

### 颜色出现串锥或跳变

可能原因：

- `track_match_distance` 太大；
- LiDAR 坐标随车辆运动，短期最近邻关联错误；
- 相邻锥桶投影框重叠；
- 标定误差；
- YOLO 分类本身冲突。

可以：

- 适当减小 `track_match_distance`；
- 保持 `color_conflict_confirmations >= 2`；
- 开启 2D 可视化检查投影；
- 核对固定坐标系或后续接入车辆运动补偿。

### 黄色大小错误

检查 LiDAR 输入：

```bash
ros2 topic echo /detected_cones_bbox
```

确认：

```text
label 1 = small
label 2 = big
```

节点不会重新定义这个约定。

## 18. 比赛前检查清单

1. 两个 `.cpp` 与 `fs_fusion_box_math.hpp` 函数签名一致。
2. `colcon build` 无编译错误。
3. `lidar_to_camera_matrix` 为实际 4×4 Row-major 标定矩阵。
4. Camera 输入 header 保留原图采集时间戳。
5. Camera 和 LiDAR 使用相同时钟体系。
6. `/perception/fusion/map` 与 LiDAR 都约为 10 Hz。
7. LiDAR received 与 published 总数一致。
8. Camera age P95 明显小于 500 ms。
9. `sync_p95` 小于 `max_sync_diff`。
10. Camera UNKNOWN 显式映射数量为 0。
11. 蓝、红、小黄、大黄分别做一次实际消息验证。
12. `track_match_distance` 与 LiDAR 坐标单位和车辆速度匹配。
13. 比赛模式关闭 `enable_visualization`。
14. Ctrl+C 后两个工作线程能退出并打印最终摘要。
