"""Exercise depth fallback geometry and frame pairing without ROS or a camera."""

from types import SimpleNamespace
import unittest

import numpy as np

from my_launch.depth_cone_fallback import depth_image_view, estimate_cone_xy, FramePairs


def message(stamp, frame='left_optical'):
    return SimpleNamespace(header=SimpleNamespace(
        frame_id=frame, stamp=SimpleNamespace(sec=10, nanosec=stamp)))


class DepthConeFallbackTest(unittest.TestCase):
    def test_depth_formats_and_row_padding(self):
        formats = (
            ('32FC1', '<f4', [2.5, 3.5], 1.0),
            ('16UC1', '>u2', [2500, 3500], 0.001),
            ('mono16', '<u2', [2500, 3500], 0.001),
        )
        for encoding, dtype, values, expected_scale in formats:
            with self.subTest(encoding=encoding):
                payload = np.array(values, dtype=dtype).tobytes() + b'\0\0'
                depth = SimpleNamespace(
                    width=2, height=2, step=len(payload), encoding=encoding,
                    is_bigendian=dtype.startswith('>'), data=payload * 2)
                image, scale = depth_image_view(depth)
                self.assertEqual(scale, expected_scale)
                np.testing.assert_allclose(image * scale, [[2.5, 3.5], [2.5, 3.5]])
                depth.data = depth.data[:-1]
                with self.assertRaises(ValueError):
                    depth_image_view(depth)

    def test_frame_pairing_requires_same_stamp_and_optical_frame(self):
        pairs = FramePairs(capacity=2)
        boxes, depth = message(1), message(1)
        following_depth = message(2)
        self.assertIsNone(pairs.receive(boxes, False))
        self.assertIsNone(pairs.receive(message(1, 'right_optical'), True))
        self.assertIsNone(pairs.receive(following_depth, True))
        self.assertEqual(pairs.receive(depth, True), (depth, boxes))
        boxes = message(2)
        self.assertEqual(pairs.receive(boxes, False), (following_depth, boxes))
        for stamp in (3, 4, 5):
            self.assertIsNone(pairs.receive(message(stamp), True))
        self.assertIsNone(pairs.receive(message(3), False))

    def estimate(self, image, bbox=(60, 50, 20, 40), camera_x_offset=0.10):
        camera_to_lidar = np.array([
            [0, 0, 1, camera_x_offset],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ])
        return estimate_cone_xy(image, 1.0, bbox, (100, 100, 50, 50),
                                camera_to_lidar, 4.0)

    def test_planar_range_gate_after_transform(self):
        image = np.full((100, 100), 3.0, dtype=np.float32)
        np.testing.assert_allclose(self.estimate(image), (3.1, -0.3))
        image.fill(3.9)
        self.assertIsNone(self.estimate(image))  # Depth < 4 m, but planar range > 4 m.
        image.fill(4.0)
        self.assertEqual(self.estimate(image, (50, 50, 20, 40), 0.0), (4.0, 0.0))

    def test_depth_support_and_clipped_boxes(self):
        image = np.full((100, 100), 3.0)
        image[:, 58] = np.inf
        np.testing.assert_allclose(self.estimate(image), (3.1, -0.3))
        image[:, 60:] = 5.0
        self.assertIsNone(self.estimate(image))
        image.fill(0.0)
        self.assertIsNone(self.estimate(image))
        image.fill(2.0)
        self.assertIsNone(self.estimate(image, (60, 80, 20, 40)))
