# Lidar Cone Detector (ROS 2)

这个包现在只保留 C++/CUDA/TensorRT 推理链路。

- `infer_trt_cuda.launch.py`：C++/CUDA/TensorRT 生产版本，默认走 split pipeline。
- 当前默认高速路线：CUDA voxelization -> fused CUDA VFE/scatter -> CenterHead INT8 512 engine -> GPU postprocess。
- 这版代码按 `my_cone_anchorfree.yaml` 的 CenterPoint/CenterHead 结构解码，不再按旧 AnchorHead anchor decode 走。
- 当前运行只依赖包内 `models/pointpillars_centerhead_int8.engine` 和编译进代码的 VFE fused weights。

## C++/CUDA TensorRT 运行

```bash
cd ~/perception_PointPillars
source install/setup.bash
ros2 launch lidar_cone_detector infer_trt_cuda.launch.py
```

常用参数：

```bash
ros2 launch lidar_cone_detector infer_trt_cuda.launch.py \
  input_topic:=/lidar_points \
  score_thresh:=0.30 \
  nms_thresh:=0.1 \
  enable_latency_test:=true
```

`enable_latency_test` 打开后会打印每一帧的拆分延时：
PointCloud2 布局解析、pinned host copy、H2D、CUDA voxelization、TensorRT GPU、D2H、后处理、发布和总耗时。

RViz2 可视化框默认发布到 `/detected_cones_markers`。比赛上车时如果只需要算法输出，
可以关闭 MarkerArray 构造和发布：

```bash
publish_markers:=false
```

默认开启 `gpu_preprocess:=true`，此时节点不会先在 CPU 上生成 XYZI 中间 vector，而是把
PointCloud2 的原始 `data` 复制到 pinned host buffer，再异步拷到 GPU，由一个 CUDA kernel
直接完成 raw buffer 解析、range filter 和 voxelization。如果遇到不连续 row layout 或缺少
`x/y/z` 字段的点云，会自动退回 CPU parser。调试时可以用：

```bash
gpu_preprocess:=false
```

默认开启 `fixed_hesai_layout:=true`。节点会在第一帧确认字段是
`x/y/z/intensity float32` 且偏移为 `0/4/8/12`，之后每帧直接使用这个固定布局，
不再重复扫描 PointCloud2 fields。如果换了其他雷达或 topic，临时加：

```bash
fixed_hesai_layout:=false
```

默认开启 `direct_raw_h2d:=true`。对于当前 Hesai `point_step=26` 的点云，节点把整帧
PointCloud2 raw bytes 一次性拷到 pinned buffer/GPU，由 CUDA kernel 按字段偏移读取 XYZI。
这会比只拷 XYZI 多传一些 H2D 字节，但能避免 32 万点的 CPU stride compact 小拷贝，通常更利于
压低 `pinned_copy` p95/p99。

如果想做 A/B 对比“少传 H2D 字节”和“少做 CPU packing”，可以关闭 direct raw，启用 compact：

```bash
direct_raw_h2d:=false compact_raw_h2d:=true
```

此时如果 `x/y/z/intensity` 是前 16 字节 `float32`，节点只打包 XYZI，不搬 ring/timestamp。

voxelizer 默认 `clear_voxel_buffers:=false`，不会每帧清空完整 `d_voxels/d_coords`。CUDA VFE
只读取 `voxel_num_points` 范围内的真实点，padding 区域不参与计算，因此可以少清约 10MB/帧。若要排查
输入残留，可临时恢复旧行为：

```bash
clear_voxel_buffers:=true
```

split pipeline 默认使用 `bev_clear_mode:=occupied`，只清上一帧写过的 BEV cell，避免每帧清空
`64x320x160` 的整张 spatial feature。若要对比全图清空或自动策略，可以分别使用：

```bash
bev_clear_mode:=full
bev_clear_mode:=auto bev_clear_auto_full_threshold:=12000
```

延迟日志默认每 10 帧打印一次，减少终端 I/O 对 p95/p99 的干扰。需要逐帧统计时可以加：

```bash
latency_log_every_n:=1
```

注意 `enable_latency_test:=true` 会为多段 CUDA 计时插入 event 同步，适合定位瓶颈，但会轻微扰动
真实端到端延迟。最终上车确认 p95/p99 时，建议再跑一组：

```bash
enable_latency_test:=false
```

split pipeline 默认开启 `use_gpu_postprocess:=true`。正式运行时，TensorRT 的 CenterHead
`hm/center/center_z/dim/rot` 会留在 GPU 上完成 heatmap threshold、local-max、TopK/sort、
box decode 和 rotated NMS mask/selection，CPU 只拿最终检测框，不再整帧 D2H raw output。若要和旧 CPU
后处理对齐或排查问题，可以临时关闭：

```bash
use_gpu_postprocess:=false
```

GPU candidate buffer 默认 `gpu_post_max_candidates:=2048`。正常 `score_thresh:=0.30` 下通常够用，
同时比 8192 少做一些排序工作。如果低阈值导致 buffer 溢出，日志会提示 raw candidates 超过 buffer，可以调大：

```bash
gpu_post_max_candidates:=4096
```

GPU NMS 还有一个独立的预筛 TopK，默认 `gpu_nms_pre_max_size:=512`。它只影响 GPU 后处理路径，
CPU reference 仍使用 `nms_pre_max_size`。锥桶场景最终框很少，512 更有利于 p95；如果你为了极低阈值调试、
担心漏掉低分候选，可以临时调大：

```bash
gpu_nms_pre_max_size:=2048
```

如果 Hesai driver 能发布紧凑 XYZI 点云，建议把 `input_topic` 切到 `point_step=16`、字段为
`x/y/z/intensity float32` 的 topic。节点启动后会在日志里打印 `compact_xyzi_step16 yes/no`，
用来确认当前 topic 是否已经是紧凑格式。

`score_thresh` 默认使用训练配置 `my_cone_anchorfree.yaml` 里的 `0.30`。如果设成 `0.05`，低置信度 heatmap 峰值会大量进入 NMS，RViz 里容易出现满屏乱框，后处理延时也会明显变大。

当前 anchor-free 关键形状：

- 点云范围：`[0, -19.2, -3, 19.2, 19.2, 1]`
- voxel：`[0.12, 0.12, 4.0]`
- CUDA scatter 输出：`spatial_features [1,64,320,160]`
- CenterHead 输出：`hm [1,2,160,80]`, `center [1,2,160,80]`, `center_z [1,1,160,80]`, `dim [1,3,160,80]`, `rot [1,2,160,80]`
- label 映射：`1 -> Cone`, `2 -> BigCone`

默认 engine 路径来自 ROS 包内 share 目录：`models/pointpillars_centerhead_int8.engine`。
如果要临时切到 FP16 对齐，可以显式传入 `backbone_head_engine_path:=.../models/pointpillars_centerhead_fp16.engine`。

注意：TensorRT `.engine` 通常不能跨机器和 TensorRT 版本通用。Orin NX 上建议用同一份 Q/DQ ONNX 在 Orin 的 TensorRT 10.3 环境重新构建 engine。
