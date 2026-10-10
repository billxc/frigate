"""Regression tests for external detector initialization and request recovery."""

import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import cv2
import numpy as np

from frigate.detectors.jpeg import encode_jpeg
from frigate.detectors.plugins.zmq_ipc import ZmqIpcDetector


def make_detector() -> ZmqIpcDetector:
    """Create a detector with a mocked peer, without opening a socket."""
    detector = ZmqIpcDetector.__new__(ZmqIpcDetector)
    detector.detector_config = SimpleNamespace(
        jpeg_quality=None,
        model=SimpleNamespace(model_type=SimpleNamespace(name="yologeneric")),
        linger_ms=0,
    )
    detector.router = Mock()
    detector.router.detect.return_value = np.zeros((20, 6), np.float32)
    return detector


class TestZmqTransport(TestCase):
    def test_raw_payload_and_peer_result_are_forwarded_to_router(self):
        detector = make_detector()
        image = np.zeros((1, 8, 8, 3), np.uint8)
        result = detector.detect_raw(image)
        header, payload = detector.router.detect.call_args.args[0]
        self.assertEqual(json.loads(header)["shape"], list(image.shape))
        self.assertEqual(payload, image.tobytes())
        self.assertIs(result, detector.router.detect.return_value)


class TestJpegTransport(TestCase):
    def test_encoder_preserves_size_and_rgb_channel_order(self):
        rgb = np.full((16, 16, 3), (240, 20, 30), np.uint8)
        encoded = encode_jpeg(rgb, quality=95)
        decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        self.assertEqual(decoded.shape, rgb.shape)
        self.assertLessEqual(np.max(np.abs(decoded.astype(int) - rgb)), 3)

    def test_optional_jpeg_payload_keeps_uint8_header_and_marks_encoding(self):
        detector = make_detector()
        detector.detector_config.jpeg_quality = 95
        image = np.full((1, 16, 16, 3), 100, np.uint8)
        detector.detect_raw(image)
        header, payload = detector.router.detect.call_args.args[0]
        header = json.loads(header)
        self.assertEqual(header["encoding"], "jpeg")
        self.assertEqual(header["dtype"], "uint8")
        self.assertEqual(header["shape"], list(image.shape))
        self.assertTrue(payload.startswith(b"\xff\xd8"))
