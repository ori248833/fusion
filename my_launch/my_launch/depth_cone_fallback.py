"""Estimate near-field cone candidates from paired YOLO and ZED frames."""

from collections import OrderedDict
from math import ceil, floor, hypot, isfinite

import numpy as np


def depth_image_view(message):
    """Read an Image payload without copying the full frame."""
    # The ZED wrapper can label its OpenNI millimeter stream as mono16.
    encoding = {
        '32FC1': ('f4', 1.0),
        '16UC1': ('u2', 0.001),
        'mono16': ('u2', 0.001),
    }.get(message.encoding)
    if encoding is None:
        raise ValueError(f'unsupported depth encoding {message.encoding!r}')
    scalar, meters_per_unit = encoding
    dtype = np.dtype(('>' if message.is_bigendian else '<') + scalar)
    if (message.width <= 0 or message.height <= 0
            or message.step < message.width * dtype.itemsize
            or len(message.data) < message.step * message.height):
        raise ValueError('invalid depth Image dimensions, step or data length')
    return np.ndarray(
        (message.height, message.width), dtype=dtype, buffer=memoryview(message.data),
        strides=(message.step, dtype.itemsize),
    ), meters_per_unit


def estimate_cone_xy(image, meters_per_unit, bbox, intrinsics, camera_to_lidar, max_range):
    """Estimate the cone base XY using body depth and the box's bottom-center pixel."""
    u, v, width, height = bbox
    if not all(map(isfinite, bbox)) or min(width, height) < 8:
        return None
    left, right = u - width / 2, u + width / 2
    top, bottom = v - height / 2, v + height / 2
    if left <= 0 or top <= 0 or right >= image.shape[1] - 1 or bottom >= image.shape[0] - 1:
        return None  # Clipped boxes do not provide a reliable base pixel.

    # Sample a narrow strip inside the cone body, above its contact with the ground.
    columns = np.linspace(ceil(u - 0.12 * width), floor(u + 0.12 * width), 9, dtype=int)
    rows = np.linspace(ceil(top + 0.45 * height), floor(top + 0.75 * height), 9, dtype=int)
    sample = image[np.ix_(np.unique(rows), np.unique(columns))].astype(np.float64)
    sample *= meters_per_unit
    valid = sample[np.isfinite(sample) & (sample > 0)]
    min_support = max(6, ceil(sample.size * 0.60))
    if valid.size < min_support:
        return None
    # Require a dominant depth cluster so mixed foreground/background is rejected.
    median = float(np.median(valid))
    inliers = valid[np.abs(valid - median) <= 0.12]
    if inliers.size < min_support:
        return None
    depth = float(np.median(inliers))
    fx, fy, cx, cy = intrinsics
    camera_point = [(u - cx) * depth / fx, (bottom - cy) * depth / fy, depth, 1.0]
    point = camera_to_lidar @ camera_point
    x, y = float(point[0]), float(point[1])
    if x > 0 and hypot(x, y) <= max_range:
        return x, y
    return None


def _frame_key(message):
    header = message.header
    if not header.frame_id or (header.stamp.sec == 0 and header.stamp.nanosec == 0):
        return None
    return header.stamp.sec, header.stamp.nanosec, header.frame_id


class FramePairs:
    """Match either arrival order by exact acquisition stamp and optical frame."""

    def __init__(self, capacity=8):
        self.capacity = capacity
        self.depth = OrderedDict()
        self.boxes = OrderedDict()

    def receive(self, message, is_depth):
        key = _frame_key(message)
        if key is None:
            return None
        queue, other = (self.depth, self.boxes) if is_depth else (self.boxes, self.depth)
        match = other.pop(key, None)
        if match is not None:
            return (message, match) if is_depth else (match, message)
        queue[key] = message
        queue.move_to_end(key)
        if len(queue) > self.capacity:
            queue.popitem(last=False)
        return None


def main(args=None):
    """Run separately from both YOLO inference and LiDAR-driven Map publication."""
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    from cone_interfaces.msg import Cone as CameraCone, ConeArray
    from drd25_msgs.msg import Cone, Map

    class DepthConeFallback(Node):
        def __init__(self):
            super().__init__('depth_cone_fallback')
            defaults = {
                'depth_topic': '/zed/zed_node/depth/depth_registered',
                'max_range_m': 4.0,
                'min_yolo_confidence': 0.50,
                'lidar_frame': 'hesai_lidar',
                'image_width': 0,
                'image_height': 0,
                'intrinsics': [0.0] * 4,
                'lidar_to_camera_matrix': [0.0] * 16,
            }
            settings = {
                key: self.declare_parameter(key, value).value
                for key, value in defaults.items()
            }
            self.max_range = settings['max_range_m']
            self.min_confidence = settings['min_yolo_confidence']
            self.lidar_frame = settings['lidar_frame']
            self.shape = settings['image_height'], settings['image_width']
            self.intrinsics = settings['intrinsics']
            matrix = np.asarray(settings['lidar_to_camera_matrix']).reshape(4, 4)
            rotation = matrix[:3, :3]
            if (len(self.intrinsics) != 4 or not all(map(isfinite, self.intrinsics))
                    or min(self.intrinsics[:2]) <= 0 or min(self.shape) <= 0
                    or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3], [0, 0, 0, 1])
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=0.01)
                    or np.linalg.det(rotation) <= 0 or not self.lidar_frame
                    or not isfinite(self.max_range) or self.max_range <= 0
                    or not 0.0 <= self.min_confidence <= 1.0):
                raise ValueError('invalid depth fallback calibration or thresholds')
            self.camera_to_lidar = np.linalg.inv(matrix)
            self.pairs = FramePairs()
            self.colors = {
                CameraCone.BLUE: Cone.BLUE,
                CameraCone.RED: Cone.RED,
                CameraCone.YELLOW_BIG: Cone.YELLOW_BIG,
                CameraCone.YELLOW_SMALL: Cone.YELLOW_SMALL,
            }
            qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.publisher = self.create_publisher(Map, '/perception/camera/depth_candidates', qos)
            self.depth_sub = self.create_subscription(
                Image, settings['depth_topic'], lambda msg: self.receive(msg, True), qos)
            self.boxes_sub = self.create_subscription(
                ConeArray, '/yolo/cones', lambda msg: self.receive(msg, False), qos)
            self.get_logger().info(
                f'Depth candidates enabled within {self.max_range:.2f} m in {self.lidar_frame}')

        def receive(self, message, is_depth):
            pair = self.pairs.receive(message, is_depth)
            if pair is None:
                return
            depth, boxes = pair
            output = Map()
            output.header.stamp = boxes.header.stamp
            output.header.frame_id = self.lidar_frame
            try:
                image, scale = depth_image_view(depth)
                if image.shape != self.shape:
                    raise ValueError(
                        f'depth shape {image.shape} differs from calibration {self.shape}')
            except ValueError as error:
                self.get_logger().warning(str(error), throttle_duration_sec=5.0)
                self.publisher.publish(output)
                return
            for box in boxes.cones:
                color = self.colors.get(box.color)
                if color is None or not self.min_confidence <= box.confidence <= 1.0:
                    continue
                position = estimate_cone_xy(
                    image, scale, (box.center.x, box.center.y, box.size.x, box.size.y),
                    self.intrinsics, self.camera_to_lidar, self.max_range)
                if position is not None:
                    output.track.append(Cone(x=position[0], y=position[1], color=color))
            self.publisher.publish(output)

    rclpy.init(args=args)
    node = None
    try:
        node = DepthConeFallback()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
