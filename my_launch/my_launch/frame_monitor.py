"""Observe perception topics, checkpoint data and update reports while running."""

import os
from pathlib import Path
import shutil
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, PointCloud2

from cone_interfaces.msg import ConeArray
from drd25_msgs.msg import Map
from lidar_cone_detector.msg import ThreeDConeArray

from .frame_recording import create_run_directory, FrameRecorder, sync_directory
from .frame_report import generate_report, PeriodicReports


class FrameMonitor(Node):
    """Use raw subscriptions so Python never decodes image/point-cloud arrays."""

    def __init__(self):
        super().__init__('frame_monitor')
        defaults = {
            'output_root': str(Path.home() / 'perception_runs'),
            'lidar_topic': '/lidar_points',
            'camera_topic': '/zed/zed_node/rgb/color/rect/image',
            'camera_compressed': False,
            'lidar_detection_topic': '/detected_cones_bbox',
            'yolo_topic': '/yolo/cones',
            'fusion_topic': '/perception/fusion/map',
            'subscription_depth': 10,
            'writer_queue_capacity': 20000,
            'sync_interval_seconds': 0.1,
            'report_interval_seconds': 5.0,
            'yolo_max_fps': 20.0,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.declare_parameter('config_files', [''])
        params = {key: self.get_parameter(key).value for key in defaults}
        camera_type = CompressedImage if params['camera_compressed'] else Image
        specifications = (
            ('lidar_raw', PointCloud2, params['lidar_topic']),
            ('camera_raw', camera_type, params['camera_topic']),
            ('lidar_detection', ThreeDConeArray, params['lidar_detection_topic']),
            ('yolo', ConeArray, params['yolo_topic']),
            ('fusion', Map, params['fusion_topic']),
        )
        run_dir = create_run_directory(params['output_root'])
        config_dir = run_dir / 'configs'
        config_dir.mkdir()
        config_errors = []
        for index, filename in enumerate(self.get_parameter('config_files').value):
            if filename:
                try:
                    source = Path(filename)
                    destination = config_dir / f'{index}_{source.name}'
                    shutil.copy2(source, destination)
                    with destination.open('r+b') as config_handle:
                        os.fsync(config_handle.fileno())
                    sync_directory(config_dir)
                except OSError as error:
                    config_errors.append(str(error))
        self.recorder = FrameRecorder(run_dir, {
            'parameters': params,
            'use_sim_time': self.get_parameter('use_sim_time').value,
            'topics': {key: topic for key, _, topic in specifications},
            'config_copy_errors': config_errors,
            'qos': 'BEST_EFFORT / VOLATILE / KEEP_LAST',
            'time_zero': 'monitor initialization; earlier startup frames are unobserved',
        }, capacity=int(params['writer_queue_capacity']),
            sync_interval=float(params['sync_interval_seconds']))
        self.reports = None
        self._monitor_subscriptions = []
        self._warned = False
        self._report_warned = False
        try:
            self.reports = PeriodicReports(
                self.recorder, interval=float(params['report_interval_seconds']))
            qos = QoSProfile(
                depth=max(1, int(params['subscription_depth'])),
                reliability=ReliabilityPolicy.BEST_EFFORT)
            for key, message_type, topic in specifications:
                self._monitor_subscriptions.append(self.create_subscription(
                    message_type, topic, self._callback_for(key), qos, raw=True))
            self.create_timer(2.0, self._check_recording)
        except Exception:
            if self.reports is not None:
                self.reports.close()
            self.recorder.close('subscription_initialization_failed')
            raise
        self.get_logger().info(f'Frame recording directory: {run_dir}')
        self.get_logger().info(
            'Time zero is monitor startup. Data is fsynced and reports are updated '
            'periodically; saving does not require Ctrl+C.')

    def _callback_for(self, stream):
        def callback(data):
            received = time.monotonic_ns()
            system = time.time_ns()
            self.recorder.record(
                stream, data, self.get_clock().now().nanoseconds,
                received_ns=received, receive_system_ns=system)
        return callback

    def _check_recording(self):
        if not self._warned and (self.recorder.error or any(self.recorder.dropped.values())):
            self._warned = True
            self.get_logger().error(
                'Frame monitor recording is incomplete (writer failure or full queue). '
                'See run.json; this is a monitor issue, not proof of perception drops.')
        if not self._report_warned and (self.reports.error or self.reports.recovery_errors):
            self._report_warned = True
            self.get_logger().error(
                f'Report update/recovery error: {self.reports.error or ""} '
                f'{self.reports.recovery_errors}. Raw CSV recording is independent.')

    def finish(self, reason):
        """Close the writer even if the ROS context has already been shut down."""
        self.reports.close()
        self.recorder.close(reason)
        report = generate_report(self.recorder.run_dir)
        print(f'Frame-rate report saved: {report}', flush=True)


def main(args=None):
    """Run until normal launch shutdown; leave CSV recoverable after a hard kill."""
    rclpy.init(args=args)
    node = None
    reason = 'shutdown'
    try:
        node = FrameMonitor()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        reason = 'interrupt_or_context_shutdown'
    except Exception:
        reason = 'monitor_error'
        raise
    finally:
        try:
            if node is not None:
                node.finish(reason)
        finally:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
