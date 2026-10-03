#!/usr/bin/env python3
"""Compare YOLO models and input resolutions on a ROS 2 bag.

The bag pass measures deployment-oriented latency and prediction statistics on
exactly the same image stream for every candidate.  A ROS bag normally has no
ground-truth annotations, so true accuracy metrics are optional and come from
the Ultralytics dataset YAML selected in the user configuration section.

The script writes one JSON report to the path selected in the user configuration
section below.  It is intended to run without command-line arguments in the
ROS 2 / Ultralytics environment on the target Jetson.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


# =============================================================================
# 用户配置区：只需要修改这一段，然后运行 python3 scripts/evaluate_models.py
# =============================================================================

# 每一项就是一个独立候选。第一项成功的候选会作为comparison的对比基准。
# PT模型可以使用不同imgsz；固定尺寸TensorRT engine的imgsz必须与构建尺寸一致。
MODEL_CANDIDATES = [
    {
        'name': 'best_pt_960',
        'path': 'models/best.pt',
        'imgsz': 960,
    },
    {
        'name': 'best_pt_800',
        'path': 'models/best.pt',
        'imgsz': 800,
    },
    {
        'name': 'best_pt_640',
        'path': 'models/best.pt',
        'imgsz': 640,
    },
]

# 用于速度测试的ROS 2 bag。必须改成实际路径。
ROS_BAG_PATH = '/path/to/rosbag_directory'

# 图像话题。设为None时，脚本自动选择bag中的第一个Image/CompressedImage话题。
IMAGE_TOPIC = '/zed/zed_node/rgb/color/rect/image'

# 带标签的Ultralytics数据集YAML。
# 设为None时只测试bag速度和检测分布，不计算Precision/Recall/mAP。
DATASET_YAML = None
VALIDATION_SPLIT = 'val'

# 最终汇总JSON。相对路径以cone_ws目录为基准。
OUTPUT_JSON = 'runs/model_evaluation/best_resolution.json'

# bag推理参数，应尽量与正式YOLO节点保持一致。
DEVICE = '0'
CONFIDENCE_THRESHOLD = 0.1
NMS_IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 300

# 速度采样参数。
WARMUP_FRAMES = 20
MAX_MEASURED_FRAMES = 500  # 0表示测试bag中的全部图像。
FRAME_STEP = 1             # 1表示每张图都测，2表示每两张取一张。

# 无真值bag上的“小框/远处目标代理指标”，单位为原图像素。
# 该指标不是远处锥桶召回率；真正Recall必须使用带标签数据集。
SMALL_BOX_HEIGHT_PX = 20.0

# Ultralytics validation参数。
VALIDATION_BATCH = 1
VALIDATION_WORKERS = 4
VALIDATION_CONFIDENCE = 0.001

# =============================================================================
# 用户配置区结束：通常不需要修改下面的代码
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


IMAGE_MESSAGE_TYPES = {
    'sensor_msgs/msg/Image',
    'sensor_msgs/msg/CompressedImage',
}


@dataclass(frozen=True)
class Candidate:
    name: str
    model_path: Path
    imgsz: int


@dataclass
class BagSource:
    reader: object
    topic: str
    topic_type: str
    message_class: object


def resolve_config_path(value) -> Optional[Path]:
    if value is None or str(value).strip() == '':
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_user_configuration():
    if not MODEL_CANDIDATES:
        raise ValueError('MODEL_CANDIDATES cannot be empty')

    candidates: List[Candidate] = []
    used_names = set()
    for index, configured in enumerate(MODEL_CANDIDATES):
        if not isinstance(configured, dict):
            raise TypeError(f'MODEL_CANDIDATES[{index}] must be a dict')
        name = str(configured.get('name', '')).strip()
        model_path = resolve_config_path(configured.get('path'))
        imgsz = int(configured.get('imgsz', 0))
        if not name:
            raise ValueError(f'MODEL_CANDIDATES[{index}] has an empty name')
        if name in used_names:
            raise ValueError(f'Duplicate candidate name: {name}')
        if model_path is None or not model_path.is_file():
            raise FileNotFoundError(f'Model not found for {name}: {model_path}')
        if imgsz <= 0 or imgsz % 32 != 0:
            raise ValueError(f'{name} imgsz must be a positive multiple of 32')
        used_names.add(name)
        candidates.append(Candidate(name=name, model_path=model_path, imgsz=imgsz))

    args = SimpleNamespace(
        bag=resolve_config_path(ROS_BAG_PATH),
        image_topic=IMAGE_TOPIC or None,
        data=resolve_config_path(DATASET_YAML),
        split=str(VALIDATION_SPLIT),
        output=resolve_config_path(OUTPUT_JSON),
        device=str(DEVICE),
        conf=float(CONFIDENCE_THRESHOLD),
        iou=float(NMS_IOU_THRESHOLD),
        max_det=int(MAX_DETECTIONS),
        warmup=int(WARMUP_FRAMES),
        max_frames=int(MAX_MEASURED_FRAMES),
        frame_step=int(FRAME_STEP),
        small_box_height=float(SMALL_BOX_HEIGHT_PX),
        val_batch=int(VALIDATION_BATCH),
        val_workers=int(VALIDATION_WORKERS),
        val_conf=float(VALIDATION_CONFIDENCE),
    )
    if args.bag is None or not args.bag.exists():
        raise FileNotFoundError(f'Bag not found: {args.bag}')
    if args.data is not None and not args.data.is_file():
        raise FileNotFoundError(f'Dataset YAML not found: {args.data}')
    if args.output is None or args.output.suffix.lower() != '.json':
        raise ValueError('OUTPUT_JSON must point to a .json file')
    if args.split not in ('train', 'val', 'test'):
        raise ValueError("VALIDATION_SPLIT must be 'train', 'val', or 'test'")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError('CONFIDENCE_THRESHOLD must be in [0, 1]')
    if not 0.0 <= args.val_conf <= 1.0:
        raise ValueError('VALIDATION_CONFIDENCE must be in [0, 1]')
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError('NMS_IOU_THRESHOLD must be in [0, 1]')
    if args.max_det < 1:
        raise ValueError('MAX_DETECTIONS must be >= 1')
    if args.warmup < 0 or args.max_frames < 0:
        raise ValueError('WARMUP_FRAMES and MAX_MEASURED_FRAMES must be >= 0')
    if args.frame_step < 1:
        raise ValueError('FRAME_STEP must be >= 1')
    if args.small_box_height <= 0.0:
        raise ValueError('SMALL_BOX_HEIGHT_PX must be > 0')
    if args.val_batch < 1 or args.val_workers < 0:
        raise ValueError('VALIDATION_BATCH must be >= 1 and workers must be >= 0')
    return args, candidates


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def storage_identifier(bag_path: Path) -> str:
    metadata_path = bag_path / 'metadata.yaml' if bag_path.is_dir() else None
    if metadata_path and metadata_path.is_file():
        text = metadata_path.read_text(encoding='utf-8', errors='replace')
        match = re.search(r'^\s*storage_identifier:\s*["\']?([^\s"\']+)', text, re.M)
        if match:
            return match.group(1)
    if bag_path.suffix.lower() == '.mcap':
        return 'mcap'
    return 'sqlite3'


def open_bag_source(bag_path: Path, requested_topic: Optional[str]) -> BagSource:
    try:
        import rosbag2_py
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:
        raise RuntimeError(
            'ROS 2 Python bag modules are unavailable. Source the ROS 2 and '
            'workspace setup files before running this script.'
        ) from error

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path), storage_id=storage_identifier(bag_path)
    )
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)
    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }

    if requested_topic:
        if requested_topic not in topic_types:
            raise ValueError(
                f'Image topic {requested_topic!r} is not in the bag. '
                f'Available topics: {sorted(topic_types)}'
            )
        selected_topic = requested_topic
        selected_type = topic_types[selected_topic]
        if selected_type not in IMAGE_MESSAGE_TYPES:
            raise ValueError(
                f'Topic {selected_topic!r} has type {selected_type!r}; expected '
                'sensor_msgs/msg/Image or sensor_msgs/msg/CompressedImage'
            )
    else:
        image_topics = [
            (name, msg_type)
            for name, msg_type in topic_types.items()
            if msg_type in IMAGE_MESSAGE_TYPES
        ]
        if not image_topics:
            raise ValueError('No Image or CompressedImage topic was found in the bag')
        selected_topic, selected_type = sorted(image_topics)[0]

    reader.set_filter(rosbag2_py.StorageFilter(topics=[selected_topic]))
    return BagSource(
        reader=reader,
        topic=selected_topic,
        topic_type=selected_type,
        message_class=get_message(selected_type),
    )


def iter_images(
    source: BagSource,
    frame_step: int,
) -> Iterator[Tuple[object, int]]:
    try:
        import cv2
        import numpy as np
        from cv_bridge import CvBridge
        from rclpy.serialization import deserialize_message
    except ImportError as error:
        raise RuntimeError(
            'OpenCV/cv_bridge/rclpy dependencies are unavailable. Source the '
            'ROS 2 and workspace setup files before running this script.'
        ) from error

    bridge = CvBridge()
    image_index = 0
    while source.reader.has_next():
        topic, serialized, timestamp_ns = source.reader.read_next()
        if topic != source.topic:
            continue
        image_index += 1
        if (image_index - 1) % frame_step != 0:
            continue
        message = deserialize_message(serialized, source.message_class)
        if source.topic_type == 'sensor_msgs/msg/CompressedImage':
            image = cv2.imdecode(
                np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise ValueError(
                    f'Failed to decode compressed image at bag timestamp {timestamp_ns}'
                )
        else:
            image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        yield image, int(timestamp_ns)


def finite_float(value) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(values, ratio: float) -> float:
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=np.float64), ratio))


def summarize_ms(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    mean = float(sum(values) / len(values))
    return {
        'count': len(values),
        'mean_ms': mean,
        'p50_ms': percentile(values, 50),
        'p90_ms': percentile(values, 90),
        'p95_ms': percentile(values, 95),
        'p99_ms': percentile(values, 99),
        'max_ms': float(max(values)),
        'fps_from_mean': float(1000.0 / mean) if mean > 0.0 else 0.0,
    }


def summarize_values(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    return {
        'count': len(values),
        'mean': float(sum(values) / len(values)),
        'p50': percentile(values, 50),
        'p95': percentile(values, 95),
        'min': float(min(values)),
        'max': float(max(values)),
    }


def synchronize_cuda(torch_module) -> None:
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
    except Exception:
        # Some exported backends do not expose CUDA through the installed torch.
        pass


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def evaluate_bag(candidate: Candidate, model, args) -> Dict[str, object]:
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError('NumPy and PyTorch are required for bag evaluation') from error

    source = open_bag_source(args.bag, args.image_topic)
    model_wall_ms: List[float] = []
    extract_ms: List[float] = []
    pipeline_ms: List[float] = []
    stage_times: Dict[str, List[float]] = {
        'preprocess': [],
        'inference': [],
        'postprocess': [],
    }
    confidences: List[float] = []
    detections_per_frame: List[float] = []
    class_counts: Counter = Counter()
    small_detection_count = 0
    frames_with_detections = 0
    warmed = 0
    measured = 0

    print(
        f'[{candidate.name}] bag={args.bag} topic={source.topic} '
        f'type={source.topic_type}'
    )
    for image, _timestamp_ns in iter_images(source, args.frame_step):
        synchronize_cuda(torch)
        model_start = time.perf_counter()
        results = model.predict(
            source=image,
            imgsz=candidate.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )
        synchronize_cuda(torch)
        model_end = time.perf_counter()

        extract_start = time.perf_counter()
        result = results[0] if results else None
        if result is None or result.boxes is None or len(result.boxes) == 0:
            detections = np.empty((0, 6), dtype=np.float32)
        else:
            detections = result.boxes.data.detach().cpu().numpy()
        extract_end = time.perf_counter()

        if warmed < args.warmup:
            warmed += 1
            continue

        wall_value = (model_end - model_start) * 1000.0
        extract_value = (extract_end - extract_start) * 1000.0
        model_wall_ms.append(wall_value)
        extract_ms.append(extract_value)
        pipeline_ms.append(wall_value + extract_value)

        if result is not None and isinstance(getattr(result, 'speed', None), dict):
            for stage in stage_times:
                value = finite_float(result.speed.get(stage))
                if value is not None:
                    stage_times[stage].append(value)

        detection_count = int(detections.shape[0])
        detections_per_frame.append(float(detection_count))
        if detection_count:
            frames_with_detections += 1
            if detections.shape[1] >= 6:
                for row in detections:
                    confidence = float(row[4])
                    detected_class = int(row[5])
                    confidences.append(confidence)
                    class_counts[detected_class] += 1
                    if float(row[3] - row[1]) <= args.small_box_height:
                        small_detection_count += 1

        measured += 1
        if measured % 50 == 0:
            print(f'[{candidate.name}] measured {measured} bag frames')
        if args.max_frames > 0 and measured >= args.max_frames:
            break

    if measured == 0:
        raise RuntimeError(
            'No measured frames were produced. Check the image topic, bag, '
            'frame step, warm-up count, and max-frames setting.'
        )

    names = getattr(model, 'names', {})
    named_class_counts = {
        class_name(names, int(index)): int(count)
        for index, count in sorted(class_counts.items())
    }
    total_detections = int(sum(class_counts.values()))
    return {
        'bag_path': str(args.bag.resolve()),
        'image_topic': source.topic,
        'message_type': source.topic_type,
        'warmup_frames': warmed,
        'measured_frames': measured,
        'frame_step': args.frame_step,
        'timing_ms': {
            'model_wall': summarize_ms(model_wall_ms),
            'boxes_gpu_to_cpu': summarize_ms(extract_ms),
            'model_plus_extract': summarize_ms(pipeline_ms),
            'ultralytics_preprocess': summarize_ms(stage_times['preprocess']),
            'ultralytics_inference': summarize_ms(stage_times['inference']),
            'ultralytics_postprocess': summarize_ms(stage_times['postprocess']),
        },
        'prediction_observations': {
            'ground_truth_available': False,
            'warning': (
                'These bag statistics describe model output and are not '
                'accuracy, recall, or mAP because the bag has no labels.'
            ),
            'total_detections': total_detections,
            'frames_with_detections': frames_with_detections,
            'frames_with_detections_ratio': frames_with_detections / measured,
            'detections_per_frame': summarize_values(detections_per_frame),
            'confidence': summarize_values(confidences),
            'class_counts': named_class_counts,
            'small_box_height_threshold_px': args.small_box_height,
            'small_detection_count': small_detection_count,
            'small_detection_ratio': (
                small_detection_count / total_detections
                if total_detections > 0
                else 0.0
            ),
        },
    }


def to_list(value) -> Optional[List[float]]:
    if value is None:
        return None
    try:
        if hasattr(value, 'detach'):
            value = value.detach().cpu()
        if hasattr(value, 'tolist'):
            value = value.tolist()
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def validation_metrics(candidate: Candidate, model, args) -> Dict[str, object]:
    if args.data is None:
        return {
            'available': False,
            'reason': 'DATASET_YAML is None, so labeled validation was skipped.',
        }

    print(
        f'[{candidate.name}] validating labeled data={args.data} '
        f'split={args.split}'
    )
    with tempfile.TemporaryDirectory(prefix='yolo_model_eval_') as temp_dir:
        metrics = model.val(
            data=str(args.data),
            split=args.split,
            imgsz=candidate.imgsz,
            batch=args.val_batch,
            conf=args.val_conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            workers=args.val_workers,
            rect=False,
            plots=False,
            save=False,
            save_json=False,
            project=temp_dir,
            name=candidate.name,
            exist_ok=True,
            verbose=False,
        )

    box = metrics.box
    names = getattr(metrics, 'names', getattr(model, 'names', {}))
    class_indices = to_list(getattr(box, 'ap_class_index', None)) or []
    precision = to_list(getattr(box, 'p', None)) or []
    recall = to_list(getattr(box, 'r', None)) or []
    ap50 = to_list(getattr(box, 'ap50', None)) or []
    ap = to_list(getattr(box, 'ap', None)) or []
    per_class = {}
    for offset, raw_class_index in enumerate(class_indices):
        class_index = int(raw_class_index)
        per_class[class_name(names, class_index)] = {
            'class_id': class_index,
            'precision': precision[offset] if offset < len(precision) else None,
            'recall': recall[offset] if offset < len(recall) else None,
            'map50': ap50[offset] if offset < len(ap50) else None,
            'map50_95': ap[offset] if offset < len(ap) else None,
        }

    color_accuracy = None
    confusion_matrix = None
    raw_confusion = getattr(getattr(metrics, 'confusion_matrix', None), 'matrix', None)
    if raw_confusion is not None:
        try:
            import numpy as np

            matrix = np.asarray(raw_confusion, dtype=np.float64)
            confusion_matrix = matrix.tolist()
            class_count = len(names)
            core = matrix[:class_count, :class_count]
            matched = float(core.sum())
            if matched > 0.0:
                color_accuracy = float(np.trace(core) / matched)
        except (TypeError, ValueError):
            confusion_matrix = None

    speed = {
        str(key): finite_float(value)
        for key, value in getattr(metrics, 'speed', {}).items()
    }
    return {
        'available': True,
        'dataset_yaml': str(args.data.resolve()),
        'split': args.split,
        'val_conf': args.val_conf,
        'iou': args.iou,
        'overall': {
            'precision': finite_float(getattr(box, 'mp', None)),
            'recall': finite_float(getattr(box, 'mr', None)),
            'map50': finite_float(getattr(box, 'map50', None)),
            'map75': finite_float(getattr(box, 'map75', None)),
            'map50_95': finite_float(getattr(box, 'map', None)),
        },
        'per_class': per_class,
        'matched_color_accuracy': color_accuracy,
        'confusion_matrix': confusion_matrix,
        'ultralytics_speed_ms_per_image': speed,
        'notes': [
            'The summary precision and recall are selected from the PR curve; '
            'mAP uses detections retained above val_conf.',
            'matched_color_accuracy excludes background false positives and '
            'missed ground-truth objects; inspect per-class precision/recall too.',
            'True far-cone recall requires distance or object-size annotations '
            'and is not inferred from unlabeled bag detections.',
        ],
    }


def successful_metric(result: Dict[str, object], path: Sequence[str]):
    value = result
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return finite_float(value)


def add_comparison(results: List[Dict[str, object]]) -> Dict[str, object]:
    successful = [item for item in results if item.get('status') == 'ok']
    if not successful:
        return {'baseline': None, 'deltas': {}}
    baseline = successful[0]
    base_latency = successful_metric(
        baseline, ('bag', 'timing_ms', 'model_wall', 'p50_ms')
    )
    base_map = successful_metric(
        baseline, ('validation', 'overall', 'map50_95')
    )
    base_recall = successful_metric(
        baseline, ('validation', 'overall', 'recall')
    )
    deltas = {}
    for item in successful:
        latency = successful_metric(item, ('bag', 'timing_ms', 'model_wall', 'p50_ms'))
        current_map = successful_metric(item, ('validation', 'overall', 'map50_95'))
        current_recall = successful_metric(item, ('validation', 'overall', 'recall'))
        deltas[item['candidate']] = {
            'model_p50_change_percent': (
                (latency / base_latency - 1.0) * 100.0
                if latency is not None and base_latency and base_latency > 0.0
                else None
            ),
            'map50_95_change_percentage_points': (
                (current_map - base_map) * 100.0
                if current_map is not None and base_map is not None
                else None
            ),
            'recall_change_percentage_points': (
                (current_recall - base_recall) * 100.0
                if current_recall is not None and base_recall is not None
                else None
            ),
        }
    return {
        'baseline': baseline['candidate'],
        'interpretation': (
            'Negative model_p50_change_percent is faster. Positive accuracy '
            'percentage-point changes are better.'
        ),
        'deltas': deltas,
    }


def print_result_summary(result: Dict[str, object]) -> None:
    name = result['candidate']
    if result.get('status') != 'ok':
        print(f'[{name}] FAILED: {result.get("error")}')
        return
    latency = successful_metric(result, ('bag', 'timing_ms', 'model_wall', 'p50_ms'))
    latency_p95 = successful_metric(
        result, ('bag', 'timing_ms', 'model_wall', 'p95_ms')
    )
    recall = successful_metric(result, ('validation', 'overall', 'recall'))
    map_value = successful_metric(result, ('validation', 'overall', 'map50_95'))
    pieces = [
        f'[{name}]',
        f'model p50={latency:.2f} ms' if latency is not None else 'model p50=n/a',
        f'p95={latency_p95:.2f} ms' if latency_p95 is not None else 'p95=n/a',
        f'recall={recall:.4f}' if recall is not None else 'recall=n/a',
        f'mAP50-95={map_value:.4f}' if map_value is not None else 'mAP50-95=n/a',
    ]
    print(' | '.join(pieces))


def main() -> int:
    try:
        args, candidates = load_user_configuration()
    except (TypeError, ValueError, FileNotFoundError) as error:
        print(f'Configuration error: {error}', file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError as error:
        print(f'Ultralytics is unavailable: {error}', file=sys.stderr)
        return 2

    report: Dict[str, object] = {
        'schema_version': 1,
        'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'environment': {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'ultralytics': package_version('ultralytics'),
            'torch': package_version('torch'),
            'tensorrt': package_version('tensorrt'),
        },
        'configuration': {
            'models': [
                {
                    'name': item.name,
                    'path': str(item.model_path),
                    'imgsz': item.imgsz,
                }
                for item in candidates
            ],
            'bag': str(args.bag),
            'image_topic_requested': args.image_topic,
            'data': str(args.data) if args.data else None,
            'split': args.split,
            'device': args.device,
            'conf': args.conf,
            'val_conf': args.val_conf,
            'iou': args.iou,
            'max_det': args.max_det,
            'warmup': args.warmup,
            'max_frames': args.max_frames,
            'frame_step': args.frame_step,
            'small_box_height_px': args.small_box_height,
        },
        'results': [],
    }

    for candidate in candidates:
        print(
            f'\n=== Evaluating {candidate.name}: '
            f'{candidate.model_path} @ {candidate.imgsz} ==='
        )
        result: Dict[str, object] = {
            'candidate': candidate.name,
            'model_path': str(candidate.model_path),
            'imgsz': candidate.imgsz,
        }
        try:
            model = YOLO(str(candidate.model_path))
            result['bag'] = evaluate_bag(candidate, model, args)
            result['validation'] = validation_metrics(candidate, model, args)
            result['status'] = 'ok'
        except Exception as error:
            result['status'] = 'error'
            result['error'] = f'{type(error).__name__}: {error}'
        report['results'].append(result)
        print_result_summary(result)

    report['comparison'] = add_comparison(report['results'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'\nSummary written to: {args.output}')

    failed = sum(item.get('status') != 'ok' for item in report['results'])
    if failed:
        print(
            f'{failed} candidate(s) failed; successful candidates remain in the report.',
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
