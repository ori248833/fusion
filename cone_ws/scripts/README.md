# cone_ws/scripts 工具说明

本目录包含5个独立工具。它们分别用于TensorRT engine构建与测速、模型综合评测、
PT模型导出ONNX，以及通过rosbag检查YOLO ROS节点。

## 1. 工具总览

| 文件 | 主要用途 | 是否读取YAML | 是否使用rosbag | 主要输出 |
|---|---|---:|---:|---|
| `benchmark_engine.py` | 单独测试TensorRT engine耗时 | 否 | 否 | JSON、CSV、engine信息 |
| `build_int8_engine.py` | 将PT导出并构建混合精度INT8 engine | 否 | 否 | ONNX、engine、构建报告 |
| `evaluate_models.py` | 比较模型/分辨率的速度与检测效果 | 可选 | 是 | 一个汇总JSON |
| `export_onnx.py` | 将`.pt`导出为无NMS的ONNX | 否 | 否 | `.onnx`模型 |
| `orin_validate.sh` | 启动YOLO节点并检查输出话题频率 | 是 | 是 | 日志目录和summary.txt |

## 2. 运行环境

以下命令都建议在Orin上、`cone_ws`目录中执行：

```bash
cd "/home/dian/fusion拉取/fusion/cone_ws"
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Python工具依赖见项目根目录的`requirements.txt`。TensorRT engine必须在兼容的
TensorRT/CUDA设备上运行，因此engine测速和engine评测应在实际部署的Orin上完成。

---

## 3. `benchmark_engine.py`

### 作用

只测TensorRT engine本身，不经过：

- ROS和DDS；
- rosbag；
- 相机图像解码；
- letterbox和颜色转换；
- Ultralytics NMS；
- `/yolo/cones`消息构造与发布。

它报告两组数据：

1. `compute`：输入输出已经在GPU时的TensorRT执行时间；
2. `pipeline`：H2D输入复制、TensorRT执行和D2H输出复制的总时间。

每组包含`mean/p50/p90/p95/p99/max`，并分别提供CUDA事件测得的`gpu`、
主机墙钟`host`和TensorRT入队`enqueue`耗时。

### 是否使用YAML

不使用任何项目YAML。模型路径、测试次数和动态输入尺寸全部通过命令行参数传入。

### 用法1：固定960输入的engine完整测速

```bash
python3 scripts/benchmark_engine.py \
  --engine models/aggressive_0.engine \
  --warmup 100 \
  --iterations 1000 \
  --output-dir runs/engine_benchmark/aggressive_960
```

固定输入engine会自动读取输入尺寸，不需要额外传`--input-shape`。

### 用法2：动态输入engine只测计算并保存原始样本

```bash
python3 scripts/benchmark_engine.py \
  --engine models/dynamic.engine \
  --input-shape 1x3x640x640 \
  --warmup 100 \
  --iterations 1000 \
  --compute-only \
  --raw-samples \
  --output-dir runs/engine_benchmark/dynamic_640
```

### 输出

指定的输出目录中包含：

- `benchmark.json`：环境、engine哈希、输入输出、统计结果；
- `latencies.csv`：每次迭代的耗时；
- `engine_info.json`：TensorRT EngineInspector层信息，运行环境支持时生成。

重点看：

- 纯engine延迟：`summary.compute.gpu`；
- 包含传输的延迟：`summary.pipeline.gpu`；
- 稳定性：`p95/p99/max`，不要只看平均值。

---

## 4. `evaluate_models.py`

### 作用

让一个或多个模型在同一段rosbag图像上运行，并比较不同输入分辨率。它包含两部分：

1. rosbag速度评测；
2. 可选的带标签数据集准确率评测。

在脚本顶部的`MODEL_CANDIDATES`中，每一项代表一个“模型＋分辨率”候选。每个候选
都会从rosbag开头重新读取相同图像，保证速度和检测分布可以公平比较。

### 是否使用YAML

分两种情况：

#### `DATASET_YAML = None`

不需要`cone_ws/configs`中的YAML，可以得到：

- `model.predict()`的mean、p50、p95、p99、max；
- Ultralytics预处理、推理、后处理耗时；
- GPU到CPU检测框复制耗时；
- 每帧检测数量、置信度和类别分布；
- 小框检测数量和比例。

这些检测统计没有人工真值，不能称为Precision、Recall或mAP。

#### `DATASET_YAML = 'configs/cone_bag_2026.yaml'`

脚本会读取该数据集YAML，并额外运行Ultralytics validation，得到：

- Precision；
- Recall；
- mAP50、mAP75、mAP50-95；
- BLUE、RED、YELLOW_BIG、YELLOW_SMALL每类指标；
- 混淆矩阵；
- 已成功匹配检测框的颜色准确率。

使用前必须把`configs/cone_bag_2026.yaml`中的`path`改成当前机器上带标签数据集的
真实位置。YAML指向的是已经整理成YOLO格式的图片和标签，不是原始rosbag目录。

### 用法1：比较同一个PT模型的960、800和640输入

先修改脚本顶部的配置区：

```python
MODEL_CANDIDATES = [
    {'name': 'best_pt_960', 'path': 'models/best.pt', 'imgsz': 960},
    {'name': 'best_pt_800', 'path': 'models/best.pt', 'imgsz': 800},
    {'name': 'best_pt_640', 'path': 'models/best.pt', 'imgsz': 640},
]
ROS_BAG_PATH = '/path/to/rosbag_directory'
IMAGE_TOPIC = '/zed/zed_node/rgb/color/rect/image'
DATASET_YAML = 'configs/cone_bag_2026.yaml'
OUTPUT_JSON = 'runs/model_evaluation/best_resolution.json'
```

然后直接运行，不传任何参数：

```bash
python3 scripts/evaluate_models.py
```

这里的第一项`best_pt_960`是比较基准。

### 用法2：比较PT和INT8 engine

```python
MODEL_CANDIDATES = [
    {'name': 'pt_960', 'path': 'models/best.pt', 'imgsz': 960},
    {'name': 'int8_960', 'path': 'models/aggressive_0.engine', 'imgsz': 960},
]
ROS_BAG_PATH = '/path/to/rosbag_directory'
DATASET_YAML = 'configs/cone_bag_2026.yaml'
OUTPUT_JSON = 'runs/model_evaluation/pt_vs_int8_960.json'
```

```bash
python3 scripts/evaluate_models.py
```

固定输入尺寸的TensorRT engine只能使用构建时的尺寸。不要用960 engine测试640输入；
应先构建640 engine，再把该候选的`imgsz`设为640。

### 输出JSON重点字段

- `results[].bag.timing_ms.model_wall`：完整模型调用耗时；
- `results[].bag.timing_ms.ultralytics_inference`：Ultralytics报告的推理耗时；
- `results[].bag.prediction_observations`：无真值的bag检测统计；
- `results[].validation.overall`：总体P/R/mAP；
- `results[].validation.per_class`：每种锥桶类别指标；
- `results[].validation.matched_color_accuracy`：匹配框颜色准确率；
- `comparison.deltas`：相对第一个成功候选的速度和准确率变化。

`model_p50_change_percent`为负表示更快；Recall和mAP变化以百分点表示，正数更好。

---

## 5. `export_onnx.py`

### 作用

使用Ultralytics把`.pt`模型导出为固定输入尺寸的ONNX。导出时：

- `nms=False`，输出仍是YOLO原始检测头结果；
- `dynamic=False`，输入尺寸固定；
- `half=False`，以FP32导出，后续由TensorRT负责FP16/INT8构建；
- 默认opset为11。

对于当前4类YOLOv8模型，960输入通常输出`[1, 8, 18900]`：4个框坐标通道、
4个类别通道和18900个候选位置。ONNX不包含最终NMS。

### 是否使用YAML

不使用YAML，也不使用rosbag。模型、分辨率和输出位置都通过命令行传入。

### 用法1：导出960 ONNX

```bash
python3 scripts/export_onnx.py \
  --model models/best.pt \
  --imgsz 960 \
  --opset 11 \
  --output models/best_960.onnx
```

### 用法2：导出并简化640 ONNX

```bash
python3 scripts/export_onnx.py \
  --model models/best.pt \
  --imgsz 640 \
  --opset 11 \
  --simplify \
  --output models/best_640.onnx
```

导出结束后脚本会打印ONNX输入、输出shape和opset。`--simplify`可能需要额外安装
ONNX简化相关依赖。

---

## 6. `build_int8_engine.py`

### 作用

一次完成以下流程：

1. 使用Ultralytics将`.pt`导出成固定输入尺寸ONNX；
2. 用真实相机图片执行MinMax INT8校准；
3. 按配置的分层精度策略构建INT8＋FP16 TensorRT engine；
4. 保存构建配置、模型哈希、engine输入输出和实际层格式检查报告。

它不读取YAML和rosbag。所有设置都在脚本顶部的“用户配置区”。校准图片不需要标签，
但应覆盖实际运行中的远近锥桶、不同光照和赛道背景。当前YOLO节点没有裁剪ROI，所以
`CROP_TOP_RATIO`应保持`0.0`。

### 精度策略

- `aggressive`：检测头`cv2/cv3`卷积分支尽量保留INT8，stem和DFL/解码部分回退FP16；
- `conservative`：整个检测头回退FP16，通常精度更稳；
- `full_int8`：不主动保护层，由TensorRT尽量选择INT8；
- `custom`：只按照`CUSTOM_FP16_PATTERNS`和`CUSTOM_FP32_PATTERNS`设置。

### 用法1：构建640 aggressive engine

修改`scripts/build_int8_engine.py`顶部配置：

```python
PT_MODEL_PATH = 'models/best.pt'
ONNX_OUTPUT_PATH = 'models/best_int8_640.onnx'
ENGINE_OUTPUT_PATH = 'models/best_int8_640.engine'
CALIBRATION_IMAGE_DIR = '/home/dian/calibration_images'
CALIBRATION_CACHE_PATH = 'runs/tensorrt_build/best_int8_640.cache'
BUILD_REPORT_PATH = 'runs/tensorrt_build/best_int8_640_report.json'
LAYER_INSPECTOR_PATH = 'runs/tensorrt_build/best_int8_640_layers.json'
INPUT_HEIGHT = 640
INPUT_WIDTH = 640
PRECISION_PROFILE = 'aggressive'
```

然后运行：

```bash
python3 scripts/build_int8_engine.py
```

### 用法2：构建960 conservative engine

```python
ONNX_OUTPUT_PATH = 'models/best_int8_960_conservative.onnx'
ENGINE_OUTPUT_PATH = 'models/best_int8_960_conservative.engine'
CALIBRATION_CACHE_PATH = 'runs/tensorrt_build/best_int8_960.cache'
BUILD_REPORT_PATH = 'runs/tensorrt_build/best_int8_960_report.json'
LAYER_INSPECTOR_PATH = 'runs/tensorrt_build/best_int8_960_layers.json'
INPUT_HEIGHT = 960
INPUT_WIDTH = 960
PRECISION_PROFILE = 'conservative'
```

```bash
python3 scripts/build_int8_engine.py
```

改变模型、输入尺寸、裁剪比例或校准图片后，脚本会根据缓存元数据自动拒绝不匹配的
旧校准缓存。`BUILD_REPORT_PATH`记录请求的分层精度规则；`LAYER_INSPECTOR_PATH`
记录TensorRT融合后实际采用的层格式。实际部署前仍需用`evaluate_models.py`验证Recall、
mAP和颜色准确率，再用`benchmark_engine.py`测量纯engine延迟。

---

## 7. `orin_validate.sh`

### 作用

这是YOLO ROS节点的快速功能检查脚本。它会：

1. 加载YOLO参数YAML并启动`yolo_detector`；
2. 用`--clock`播放指定rosbag中的图像话题；
3. 等待一条`/yolo/cones`消息；
4. 使用`ros2 topic hz`统计输出频率；
5. 可选统计`/yolo/debug_image`频率；
6. 保存检测器、bag、话题频率和结论日志。

它只检查节点能否工作以及输出频率，不计算Precision、Recall或mAP。

### 是否使用YAML

默认使用仓库中主系统的`../my_launch/configs/yolo_detector.yaml`，从而让单独验证和
`system_run.launch.py`使用同一套YOLO参数。也可以通过`--params-file`指定其他文件。
脚本会从该YAML读取：

- `model_path`；
- `image_topic`。

其他YOLO参数由ROS节点加载。脚本会显式设置`use_sim_time=true`，与rosbag的
`--clock`配合。

脚本现在默认根据自身位置确定`cone_ws`目录，因此移动整个工作空间后不需要修改脚本
中的用户名或绝对路径。

### 用法1：使用默认YOLO参数检查输出频率

```bash
bash scripts/orin_validate.sh \
  --bag "/path/to/rosbag_directory" \
  --duration 30
```

### 用法2：指定参数文件、话题、播放速率和日志目录

```bash
bash scripts/orin_validate.sh \
  --bag "/path/to/rosbag_directory" \
  --params-file ../my_launch/configs/yolo_detector.yaml \
  --image-topic /zed/zed_node/rgb/color/rect/image \
  --bag-rate 1.0 \
  --duration 60 \
  --check-debug \
  --log-dir runs/orin_validation/manual_test
```

### 输出

默认输出目录：

```text
runs/orin_validation_YYYYMMDD_HHMMSS/
```

主要文件：

- `detector.log`：YOLO节点日志和分段耗时；
- `bag.log`：rosbag播放日志；
- `cones_once.log`：收到的第一条检测消息；
- `hz_cones.log`：`/yolo/cones`频率；
- `hz_debug.log`：调试图频率，仅`--check-debug`时生成；
- `summary.txt`：路径、平均频率和PASS/FAIL结论。

---

## 8. 应该选择哪个脚本

| 想回答的问题 | 使用工具 |
|---|---|
| engine自身到底需要多少毫秒？ | `benchmark_engine.py` |
| 960、800、640哪个速度与效果更合适？ | `evaluate_models.py` |
| PT和INT8 engine损失了多少精度、快了多少？ | `evaluate_models.py` |
| 如何把PT导出成某个固定分辨率的ONNX？ | `export_onnx.py` |
| 如何把PT构建成分层量化的INT8 engine？ | `build_int8_engine.py` |
| YOLO ROS节点能否启动、输出频率是多少？ | `orin_validate.sh` |
| ROS相机时间戳到检测结果的端到端延迟是多少？ | 启动YOLO节点并查看其timing日志 |

## 9. YAML关系总结

| YAML | 被哪个脚本使用 | 是否自动使用 |
|---|---|---|
| `configs/cone_bag_2026.yaml` | `evaluate_models.py` | `DATASET_YAML`指向它时 |

`cone_ws/configs`目前只保留模型评测用的`cone_bag_2026.yaml`。YOLO节点运行参数统一
使用`my_launch/configs/yolo_detector.yaml`。

`evaluate_models.py`的rosbag速度评测与YOLO节点参数YAML没有关系。模型路径、
`imgsz`、`conf`和`iou`都来自脚本顶部的配置区；只有带标签准确率评测会读取
`DATASET_YAML`指定的数据集YAML。
