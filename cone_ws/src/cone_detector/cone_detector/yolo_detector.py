#!/usr/bin/env python3
from pathlib import Path
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
        self.declare_parameter('timing_log_interval', 30)
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
        self.imgsz = int(self.get_parameter('imgsz').value)

        self.bridge = CvBridge()
        if not model_path.exists():
            raise FileNotFoundError(f'Model path not found: {model_path}')
        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        self.model = YOLO(str(model_path))

        self.frame_count = 0
        self.detect_log_count = 0
        self.last_process_wall_time = 0.0

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
            if 'detect_log_interval' in updates:
                self.detect_log_interval = int(updates['detect_log_interval'])
            if 'max_fps' in updates:
                self.max_fps = float(updates['max_fps'])
            return SetParametersResult(successful=True)
        except (TypeError, ValueError) as error:
            return SetParametersResult(successful=False, reason=str(error))

    def _should_process(self) -> bool:
        if self.max_fps <= 0:
            return True
        now = time.monotonic()
        if now - self.last_process_wall_time < 1.0 / self.max_fps:
            return False
        self.last_process_wall_time = now
        return True

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

    def _maybe_log_timing(self, header, marks):
        if not self.enable_timing:
            return
        if self.frame_count % self.timing_log_interval != 0:
            return
        convert_ms = (marks['converted'] - marks['start']) * 1000.0
        infer_ms = (marks['inferred'] - marks['converted']) * 1000.0
        extract_ms = (marks['extracted'] - marks['inferred']) * 1000.0
        build_publish_ms = (marks['published'] - marks['extracted']) * 1000.0
        total_ms = (marks['published'] - marks['start']) * 1000.0

        now_ros = self.get_clock().now().nanoseconds * 1e-9
        source_stamp = self._stamp_to_seconds(header)
        e2e_ms = (now_ros - source_stamp) * 1000.0
        e2e_text = f' e2e={e2e_ms:.1f}ms' if 0.0 <= e2e_ms < 60_000.0 else ''
        self.get_logger().info(
            f'timing convert={convert_ms:.1f}ms infer={infer_ms:.1f}ms '
            f'extract={extract_ms:.1f}ms build+publish={build_publish_ms:.1f}ms '
            f'total={total_ms:.1f}ms{e2e_text}'
        )

    def _process_image(self, image_msg, cv_image, start_time):
        if not cv_image.flags['C_CONTIGUOUS']:
            cv_image = np.ascontiguousarray(cv_image)
        converted_time = time.perf_counter()

        self.frame_count += 1
        results = self.model(
            cv_image,
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            verbose=False,
        )
        inferred_time = time.perf_counter()
        extracted = self._extract_boxes(results)
        extracted_time = time.perf_counter()

        count = 0 if extracted is None else int(extracted[0].shape[0])
        self._maybe_log_detected(count)
        cone_array = self._build_cone_array(image_msg.header, extracted)
        self.cone_pub.publish(cone_array)

        if self._wants_debug_image():
            self._publish_debug_image(cv_image, extracted, image_msg.header)
        published_time = time.perf_counter()
        self._maybe_log_timing(
            image_msg.header,
            {
                'start': start_time,
                'converted': converted_time,
                'inferred': inferred_time,
                'extracted': extracted_time,
                'published': published_time,
            },
        )

    def image_only_callback(self, image_msg: Image):
        if not self._should_process():
            return
        start_time = time.perf_counter()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            self._process_image(image_msg, cv_image, start_time)
        except Exception as error:
            self.get_logger().error(
                f'Error in image callback: {error}\n{traceback.format_exc()}'
            )

    def compressed_image_callback(self, image_msg: CompressedImage):
        if not self._should_process():
            return
        start_time = time.perf_counter()
        try:
            self._process_image(
                image_msg, self._decode_compressed(image_msg), start_time
            )
        except Exception as error:
            self.get_logger().error(
                f'Error in compressed-image callback: {error}\n{traceback.format_exc()}'
            )


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
