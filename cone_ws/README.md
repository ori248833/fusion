# Cone WS (ROS2 + YOLOv8)

**作用**：从 rosbag 生成训练集 → 训练 YOLO → 在 ROS2 中推理并可视化。

下面是**最短、能跑通**的说明。

---

## 0) Python 依赖（固定 PyTorch 版本）

项目训练环境固定使用：
- `torch==2.0.1`
- `torchvision==0.15.2`
- `torchaudio==2.0.2`

安装：
```bash
python3 -m pip install -r /home/juziwei/cone_ws/requirements.txt
```

---

## 1) 目录结构

- rosbag 原始数据：`/home/juziwei/cone_ws/rosbag/`
- 处理后的数据集：`/home/juziwei/cone_ws/data/`
- 训练脚本：`/home/juziwei/cone_ws/scripts/`
- 预训练权重：`/home/juziwei/cone_ws/models/`
- 训练输出：`/home/juziwei/cone_ws/runs/`

类别（4 类训练）：
- 0: BLUE
- 1: RED
- 2: YELLOW_BIG
- 3: YELLOW_SMALL

说明：
- 训练和数据集统一按 4 类
- `Cone.msg` 里仍保留 `UNKNOWN` 作为运行时兜底值，避免对外接口突变

---

## 2) 数据处理（自动遍历所有 rosbag）

```bash
conda deactivate
source /opt/ros/humble/setup.bash
source /home/juziwei/cone_ws/install/setup.bash

python3 /home/juziwei/cone_ws/scripts/bag_to_yolo.py \
  --bag-dir /home/juziwei/cone_ws/rosbag \
  --model-path /home/juziwei/cone_ws/runs/cone_yolov8n_clean_fixed3_gpu/weights/best.pt \
  --dataset-name cone_manual_expand \
  --image-topic /camera1/image_raw \
  --conf 0.5 \
  --val-ratio 0.2 \
  --split-mode hash \
  --every-n 6 \
  --log-every 50
```

输出目录：
`/home/juziwei/cone_ws/data/cone_manual_expand_YYYYMMDD_HHMMSS/`

命名规则：
- `--dataset-name` 现在作为目录前缀使用
- 实际目录会自动追加日期时间，方便区分每次抽帧结果
- 如果你确实要写死完整目录名，可以直接传 `--out-root`

只抽原图给人工标注：
```bash
python3 /home/juziwei/cone_ws/scripts/bag_to_yolo.py \
  --bag-dir /home/juziwei/cone_ws/rosbag \
  --dataset-name cone_raw_manual \
  --image-topic /camera1/image_raw \
  --every-n 6 \
  --extract-only
```

输出目录类似：
`/home/juziwei/cone_ws/data/cone_raw_manual_YYYYMMDD_HHMMSS/images/`

把 Roboflow 导出的已标注数据重新按 bag/时间段切成 train/val：
```bash
python3 /home/juziwei/cone_ws/scripts/split_dataset.py \
  --src /home/juziwei/cone_ws/data/cone_detection_last.v1i.yolov8 \
  --dataset-name cone_manual_split \
  --val-ratio 0.2 \
  --group-chunk-size 50
```

在重新训练前筛查可疑标注：
```bash
python3 /home/juziwei/cone_ws/scripts/review_dataset.py \
  --dataset-root /home/juziwei/cone_ws/data/cone_detection_last.v1i.yolov8 \
  --large-box 0.12 \
  --max-print 50
```

---

## 3) 训练（遍历 /data 下所有数据集）

```bash
conda deactivate
source /opt/ros/humble/setup.bash
source /home/juziwei/cone_ws/install/setup.bash

python3 /home/juziwei/cone_ws/scripts/train_yolov8n.py \
  --data-root /home/juziwei/cone_ws/data
```

训练输出：`/home/juziwei/cone_ws/runs/cone_yolov8nX/weights/best.pt`

Roboflow 导出：
- 选择 `Object Detection`
- 选择 `YOLOv8`
- 支持 `train/valid/test` 结构，训练脚本会自动对齐到项目里的 4 类顺序
- 默认增强已经收敛到更适合前视相机锥桶检测的配置；需要时可用命令行参数覆盖

---

## 4) 推理节点（发布话题）

```bash
conda deactivate
source /opt/ros/humble/setup.bash
source /home/ori/cone_ws/install/setup.bash

ros2 run cone_detector yolo_detector --ros-args \
  --params-file /home/ori/cone_ws/configs/yolo_detector.yaml
```

频率/性能常用参数：
- 参数都在 `configs/yolo_detector.yaml`，改完重启节点即可
- 想跑满输入（例如 30Hz）测试：把 `max_fps` 调大（如 `30.0`）
- 只要锥桶结果更快：关闭或降采样 debug 图（`publish_debug_image: false` 或 `debug_image_every_n: 2`）
- 用 rosbag 的压缩图输入：把 `image_topic` 改成 `/camera1/image_compressed` 且 `use_compressed: true`

发布话题：
- `/yolo/debug_image`
- `/yolo/cones`

---

## 5) 播放 rosbag（让节点有输入）

选择**可用**的 rosbag（损坏的包不能播）：
```bash
ros2 bag play /home/juziwei/cone_ws/rosbag/rosbag2_2026_01_28-17_29_35 \
  --clock --topics /camera1/image_raw
```

---

## 6) RViz2 可视化

```bash
rviz2 -d /home/juziwei/cone_ws/rviz/yolo_debug_image.rviz
```

---

## 常见问题

- conda 报错可忽略，但务必用系统 Python 3.10（ROS2 Humble）
- rosbag 播放报 `database disk image is malformed` → 该包损坏，换其他包

---

## 7) Orin 快速验证（功能 + 频率）

在 Orin 上执行：

```bash
conda deactivate
source /opt/ros/humble/setup.bash
source /home/juziwei/cone_ws/install/setup.bash

bash /home/juziwei/cone_ws/scripts/orin_validate.sh \
  --bag /home/juziwei/cone_ws/rosbag/rosbag2_2026_01_28-17_29_35 \
  --duration 20 \
  --check-debug
```

脚本会自动：
- 启动 `yolo_detector`
- 回放指定 rosbag 的图像话题
- 统计 `/yolo/cones`（可选 `/yolo/debug_image`）频率
- 输出 `PASS/FAIL` 结论和日志目录

日志默认输出到：
`/home/juziwei/cone_ws/runs/orin_validation_YYYYMMDD_HHMMSS/`
