#!/usr/bin/env python3
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import traceback

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from ultralytics import YOLO

from cone_interfaces.msg import Cone, ConeArray


@dataclass(frozen=True)
class PendingFrame:
    """One latest-frame slot entry handed from ROS to the inference worker."""

    message: object
    callback_start: float
    callback_ros: float


class YOLOConeDetector(Node):
    def __init__(self):
        super().__init__('yolo_cone_detector')

        self.declare_parameter(
            'model_path',
            'src/cone_ws/models/best_v5_scale_ft_int8_640_aggressive_no0.engine',
        )
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('max_fps', 30.0)
        # Race/performance mode defaults: no drawing, conversion, or debug publish.
        self.declare_parameter('enable_visualization', False)
        # Backward-compatible alias. Either switch can enable visualization.
        self.declare_parameter('publish_debug_image', False)
        self.declare_parameter('debug_image_every_n', 1)
        self.declare_parameter('detect_log_interval', 0)
        self.declare_parameter('enable_timing', False)
        self.declare_parameter('timing_log_interval', 100)
        self.declare_parameter('timing_window_size', 100)
        self.declare_parameter('timing_warmup_frames', 10)
        self.declare_parameter('imgsz', 640)

        model_path = self._resolve_model_path(
            str(self.get_parameter('model_path').value)
        )
        self.conf_thresh = float(self.get_parameter('conf_threshold').value)
        self.use_compressed = bool(self.get_parameter('use_compressed').value)
        self.max_fps = float(self.get_parameter('max_fps').value)
        self.enable_visualization = bool(
            self.get_parameter('enable_visualization').value
            or self.get_parameter('publish_debug_image').value
        )
        self.debug_image_every_n = max(
            1, int(self.get_parameter('debug_image_every_n').value)
        )
        self.detect_log_interval = int(
            self.get_parameter('detect_log_interval').value
        )
        self.enable_timing = bool(self.get_parameter('enable_timing').value)
        self.timing_log_interval = max(
            1, int(self.get_parameter('timing_log_interval').value)
        )
        self.timing_window_size = max(
            1, int(self.get_parameter('timing_window_size').value)
        )
        self.timing_warmup_frames = max(
            0, int(self.get_parameter('timing_warmup_frames').value)
        )
        self.imgsz = int(self.get_parameter('imgsz').value)

        self.bridge = CvBridge()
        if not model_path.exists():
            raise FileNotFoundError(f'Model path not found: {model_path}')
        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        self.model = YOLO(str(model_path))

        self.frame_count = 0
        self.detect_log_count = 0
        self.timing_samples = deque(maxlen=self.timing_window_size)
        self._timing_lock = threading.Lock()

        # The ROS subscription callback only replaces this one-slot buffer.
        # TensorRT runs in a separate worker so the executor remains responsive
        # to new images, /clock, parameter updates, and shutdown.
        self._latest_condition = threading.Condition()
        self._latest_frame = None
        self._stop_worker_event = threading.Event()
        self._last_inference_start = 0.0
        self._input_frame_count = 0
        self._overwritten_frame_count = 0
        self._worker_error_count = 0
        self._worker_thread = None

        latest_only_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        image_topic = str(self.get_parameter('image_topic').value)
        if self.use_compressed:
            self.image_sub = self.create_subscription(
                CompressedImage,
                image_topic,
                self.compressed_image_callback,
                latest_only_qos,
            )
        else:
            self.image_sub = self.create_subscription(
                Image, image_topic, self.image_only_callback, latest_only_qos
            )

        # Use a reliable QoS for the debug stream so RViz2 and other generic
        # image viewers can subscribe without requiring a manual Best Effort
        # QoS override. The input and detection topics remain Best Effort.
        debug_image_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.debug_image_pub = self.create_publisher(
            Image, '/yolo/debug_image', debug_image_qos
        )
        self.cone_pub = self.create_publisher(
            ConeArray, '/yolo/cones', latest_only_qos
        )
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        self.class_color_map = {
            0: Cone.BLUE,
            1: Cone.RED,
            2: Cone.YELLOW_BIG,
            3: Cone.YELLOW_SMALL,
        }
        mode = 'debug' if self.enable_visualization else 'race/performance'
        self.get_logger().info(f'YOLO Cone Detector initialized ({mode} mode)')
        if self.enable_timing:
            self.get_logger().info(
                'Timing enabled: reporting p50/p95/max over the latest '
                f'{self.timing_window_size} processed frames after '
                f'{self.timing_warmup_frames} warm-up frames'
            )
        self._worker_thread = threading.Thread(
            target=self._inference_worker,
            name='yolo-inference-worker',
            daemon=True,
        )
        self._worker_thread.start()
        self.get_logger().info(
            'Latest-frame worker started: subscription callbacks only replace '
            'one pending frame; inference runs outside the ROS executor'
        )

    @staticmethod
    def _resolve_model_path(configured_path: str) -> Path:
        """Resolve a model relative to the installed package when needed."""
        requested = Path(configured_path).expanduser()
        candidates = [requested]

        if not requested.is_absolute():
            # Supports `ros2 run` / launch after setup.py installs the engine.
            try:
                candidates.append(
                    Path(get_package_share_directory('cone_detector')) / requested
                )
            except PackageNotFoundError:
                pass

            # Supports a source-tree run before colcon install.
            source_workspace = Path(__file__).resolve().parents[3]
            candidates.append(source_workspace / requested)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return requested

    def _on_parameters_changed(self, parameters):
        updates = {parameter.name: parameter.value for parameter in parameters}
        try:
            if 'enable_visualization' in updates:
                self.enable_visualization = bool(updates['enable_visualization'])
            if 'publish_debug_image' in updates:
                self.enable_visualization = bool(updates['publish_debug_image'])
            if 'enable_timing' in updates:
                self.enable_timing = bool(updates['enable_timing'])
            if 'debug_image_every_n' in updates:
                self.debug_image_every_n = max(
                    1, int(updates['debug_image_every_n'])
                )
            if 'timing_log_interval' in updates:
                self.timing_log_interval = max(
                    1, int(updates['timing_log_interval'])
                )
            if 'timing_window_size' in updates:
                self.timing_window_size = max(
                    1, int(updates['timing_window_size'])
                )
                with self._timing_lock:
                    self.timing_samples = deque(
                        self.timing_samples, maxlen=self.timing_window_size
                    )
            if 'timing_warmup_frames' in updates:
                self.timing_warmup_frames = max(
                    0, int(updates['timing_warmup_frames'])
                )
            if 'detect_log_interval' in updates:
                self.detect_log_interval = int(updates['detect_log_interval'])
            if 'max_fps' in updates:
                self.max_fps = float(updates['max_fps'])
                with self._latest_condition:
                    self._latest_condition.notify_all()
            return SetParametersResult(successful=True)
        except (TypeError, ValueError) as error:
            return SetParametersResult(successful=False, reason=str(error))

    def _enqueue_latest_frame(self, image_msg) -> None:
        pending = PendingFrame(
            message=image_msg,
            callback_start=time.perf_counter(),
            callback_ros=self.get_clock().now().nanoseconds * 1e-9,
        )
        with self._latest_condition:
            if self._stop_worker_event.is_set():
                return
            self._input_frame_count += 1
            if self._latest_frame is not None:
                self._overwritten_frame_count += 1
            self._latest_frame = pending
            self._latest_condition.notify()

    def _take_latest_frame_when_due(self):
        """Wait for a frame and apply max_fps while retaining only the newest."""
        with self._latest_condition:
            while not self._stop_worker_event.is_set():
                if self._latest_frame is None:
                    self._latest_condition.wait()
                    continue

                max_fps = self.max_fps
                if max_fps > 0.0 and self._last_inference_start > 0.0:
                    due_time = self._last_inference_start + 1.0 / max_fps
                    remaining = due_time - time.perf_counter()
                    if remaining > 0.0:
                        # wait() releases the lock. New arrivals can replace
                        # the pending frame while the rate limiter is waiting.
                        self._latest_condition.wait(timeout=remaining)
                        continue

                pending = self._latest_frame
                self._latest_frame = None
                return pending
        return None

    def _inference_worker(self) -> None:
        while not self._stop_worker_event.is_set():
            pending = self._take_latest_frame_when_due()
            if pending is None:
                break

            worker_start = time.perf_counter()
            self._last_inference_start = worker_start
            try:
                self._process_image(pending, worker_start)
            except Exception as error:
                self._worker_error_count += 1
                self.get_logger().error(
                    f'Error in YOLO inference worker: {error}\n'
                    f'{traceback.format_exc()}'
                )

    def _stop_inference_worker(self) -> None:
        worker = self._worker_thread
        if worker is None:
            return
        self._stop_worker_event.set()
        with self._latest_condition:
            self._latest_frame = None
            self._latest_condition.notify_all()
        if worker is not threading.current_thread():
            worker.join(timeout=5.0)
        if worker.is_alive():
            self.get_logger().warning(
                'YOLO inference worker did not stop within 5 seconds'
            )
        self._worker_thread = None

    def destroy_node(self):
        self._stop_inference_worker()
        return super().destroy_node()

    def _wants_debug_image(self) -> bool:
        return (
            self.enable_visualization
            and self.frame_count % self.debug_image_every_n == 0
        )

    def _maybe_log_detected(self, detected_count: int) -> None:
        if detected_count <= 0 or self.detect_log_interval <= 0:
            return
        self.detect_log_count += 1
        if self.detect_log_count % self.detect_log_interval == 0:
            self.get_logger().info(f'Detected {detected_count} cones')

    @staticmethod
    def _extract_boxes(results):
        if not results or results[0] is None or results[0].boxes is None:
            return None
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None

        # One GPU -> CPU synchronization/copy instead of separate xyxy/conf/cls copies.
        data = boxes.data.detach().cpu().numpy()
        if data.ndim != 2 or data.shape[1] < 6:
            return None
        return data[:, :4], data[:, 4], data[:, 5].astype(np.int32, copy=False)

    @staticmethod
    def _decode_compressed(msg: CompressedImage):
        image = cv2.imdecode(
            np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError('cv2.imdecode returned None for CompressedImage')
        return image

    @staticmethod
    def _clip_bbox(bbox, width: int, height: int):
        x1, y1, x2, y2 = bbox.astype(np.int32, copy=False)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return (
            max(0, min(int(x1), width - 1)),
            max(0, min(int(y1), height - 1)),
            max(0, min(int(x2), width - 1)),
            max(0, min(int(y2), height - 1)),
        )

    def _map_color(self, cls_id):
        return self.class_color_map.get(cls_id, Cone.UNKNOWN)

    def _get_class_name(self, cls_id):
        if isinstance(self.model.names, dict):
            return self.model.names.get(cls_id, str(cls_id))
        if isinstance(self.model.names, list) and 0 <= cls_id < len(self.model.names):
            return self.model.names[cls_id]
        return str(cls_id)

    def _build_cone_array(self, header, extracted):
        cone_array = ConeArray()
        # Preserve acquisition time/frame exactly; never replace with completion time.
        cone_array.header = header
        if extracted is None:
            return cone_array

        xyxy, confs, classes = extracted
        for bbox, conf, cls_id in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = bbox
            width = max(0.0, float(x2 - x1))
            height = max(0.0, float(y2 - y1))
            cone = Cone()
            cone.center.x = float(x1) + width * 0.5
            cone.center.y = float(y1) + height * 0.5
            cone.center.z = 0.0
            cone.size.x = width
            cone.size.y = height
            cone.size.z = 0.0
            cone.confidence = float(conf)
            cone.color = self._map_color(int(cls_id))
            cone_array.cones.append(cone)
        return cone_array

    def _draw_detections(self, image, extracted):
        if extracted is None:
            return image
        height, width = image.shape[:2]
        xyxy, confs, classes = extracted
        for bbox, conf, cls_id in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = self._clip_bbox(bbox, width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            color = (0, 255, 0)
            label = f'{self._get_class_name(int(cls_id))} {float(conf):.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return image

    def _publish_debug_image(self, cv_image, extracted, header):
        # This entire allocation/drawing/conversion path is unreachable in race mode.
        debug_image = self._draw_detections(cv_image.copy(), extracted)
        if not debug_image.flags['C_CONTIGUOUS']:
            debug_image = np.ascontiguousarray(debug_image)
        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
        debug_msg.header = header
        self.debug_image_pub.publish(debug_msg)

    @staticmethod
    def _stamp_to_seconds(header) -> float:
        return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9

    @staticmethod
    def _valid_age_ms(current_seconds: float, source_seconds: float) -> float:
        age_ms = (current_seconds - source_seconds) * 1000.0
        # A negative or extremely large value means that the camera stamp and
        # this node are not using the same clock. Keep the internal timings,
        # but do not report a misleading source-to-node latency.
        if 0.0 <= age_ms < 60_000.0:
            return age_ms
        return float('nan')

    @staticmethod
    def _model_speed_ms(results, model_wall_ms: float):
        values = {
            'preprocess': float('nan'),
            'inference': float('nan'),
            'postprocess': float('nan'),
            'model_other': float('nan'),
        }
        if not results or results[0] is None:
            return values
        speed = getattr(results[0], 'speed', None)
        if not isinstance(speed, dict):
            return values

        for key in ('preprocess', 'inference', 'postprocess'):
            raw_value = speed.get(key)
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values[key] = value

        parts = [values[key] for key in ('preprocess', 'inference', 'postprocess')]
        if all(np.isfinite(value) for value in parts):
            values['model_other'] = max(0.0, model_wall_ms - sum(parts))
        return values

    @staticmethod
    def _format_timing_stat(samples, key: str) -> str:
        values = np.asarray(
            [sample[key] for sample in samples], dtype=np.float64
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 'p50=n/a p95=n/a max=n/a (clock mismatch)'
        return (
            f'p50={np.percentile(values, 50):.1f} '
            f'p95={np.percentile(values, 95):.1f} '
            f'max={np.max(values):.1f}'
        )

    def _record_and_maybe_log_timing(self, header, marks):
        if not self.enable_timing:
            return
        if self.frame_count <= self.timing_warmup_frames:
            return

        source_stamp = self._stamp_to_seconds(header)
        before_callback = self._valid_age_ms(
            marks['callback_ros'], source_stamp
        )
        total = (marks['published'] - marks['callback_start']) * 1000.0
        processing = (marks['published'] - marks['worker_start']) * 1000.0
        sample = {
            'before_callback': before_callback,
            'worker_wait': (
                marks['worker_start'] - marks['callback_start']
            ) * 1000.0,
            'convert': (
                marks['converted'] - marks['worker_start']
            ) * 1000.0,
            'preprocess': marks['model_speed']['preprocess'],
            'inference': marks['model_speed']['inference'],
            'postprocess': marks['model_speed']['postprocess'],
            'model_other': marks['model_speed']['model_other'],
            'model': (marks['model_end'] - marks['model_start']) * 1000.0,
            'extract': (marks['extracted'] - marks['model_end']) * 1000.0,
            'processing': processing,
            'total': total,
            # Combine source-to-callback ROS time with monotonic in-node time.
            # This remains valid when rosbag /clock does not advance during
            # worker inference.
            'e2e': before_callback + total,
        }
        with self._timing_lock:
            self.timing_samples.append(sample)
            if self.frame_count % self.timing_log_interval != 0:
                return
            samples = list(self.timing_samples)

        with self._latest_condition:
            input_frames = self._input_frame_count
            overwritten_frames = self._overwritten_frame_count
            pending_frames = int(self._latest_frame is not None)

        lines = [
            f'YOLO timing over latest {len(samples)} processed frames (ms):',
            '  frames          '
            f'received={input_frames} processed={self.frame_count} '
            f'overwritten={overwritten_frames} pending={pending_frames} '
            f'worker_errors={self._worker_error_count}',
        ]
        labels = (
            ('before_callback', 'source stamp -> callback'),
            ('worker_wait', 'callback -> inference worker'),
            ('convert', 'worker -> OpenCV image'),
            ('preprocess', 'Ultralytics preprocess'),
            ('inference', 'TensorRT inference reported by Ultralytics'),
            ('postprocess', 'Ultralytics postprocess/NMS'),
            ('model_other', 'model() wall time not in speed fields'),
            ('model', 'Ultralytics model()'),
            ('extract', 'boxes GPU -> CPU'),
            ('processing', 'worker -> cones publish'),
            ('total', 'callback -> cones publish'),
            ('e2e', 'source stamp -> cones publish'),
        )
        for key, label in labels:
            lines.append(
                f'  {key:<15} {self._format_timing_stat(samples, key)} '
                f'[{label}]'
            )
        self.get_logger().info(
            '\n'.join(lines)
        )

    def _process_image(self, pending: PendingFrame, worker_start: float):
        image_msg = pending.message
        if self.use_compressed:
            cv_image = self._decode_compressed(image_msg)
        else:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        if not cv_image.flags['C_CONTIGUOUS']:
            cv_image = np.ascontiguousarray(cv_image)
        converted_time = time.perf_counter()

        self.frame_count += 1
        model_start = time.perf_counter()
        results = self.model(
            cv_image,
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            verbose=False,
        )
        model_end = time.perf_counter()
        model_wall_ms = (model_end - model_start) * 1000.0
        model_speed = self._model_speed_ms(results, model_wall_ms)
        extracted = self._extract_boxes(results)
        extracted_time = time.perf_counter()

        count = 0 if extracted is None else int(extracted[0].shape[0])
        self._maybe_log_detected(count)
        cone_array = self._build_cone_array(image_msg.header, extracted)
        self.cone_pub.publish(cone_array)
        published_time = time.perf_counter()

        # Record /yolo/cones timing before optional debug-image work so that
        # visualization cannot inflate the detector output latency.
        self._record_and_maybe_log_timing(
            image_msg.header,
            {
                'callback_start': pending.callback_start,
                'callback_ros': pending.callback_ros,
                'worker_start': worker_start,
                'converted': converted_time,
                'model_start': model_start,
                'model_end': model_end,
                'model_speed': model_speed,
                'extracted': extracted_time,
                'published': published_time,
            },
        )

        if self._wants_debug_image():
            self._publish_debug_image(cv_image, extracted, image_msg.header)

    def image_only_callback(self, image_msg: Image):
        self._enqueue_latest_frame(image_msg)

    def compressed_image_callback(self, image_msg: CompressedImage):
        self._enqueue_latest_frame(image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOConeDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
