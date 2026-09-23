"""Regression tests for external detector initialization and request recovery."""

import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import cv2
import numpy as np
import zmq

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
    detector._model_ready = True
    detector._model_name = "test.onnx"
    detector._zero_result = np.zeros((20, 6), np.float32)
    detector._socket = Mock()
    detector._socket.recv_multipart.return_value = [detector._zero_result.tobytes()]
    detector._create_socket = Mock()
    detector._initialize_model = Mock()
    return detector


class TestZmqRecovery(TestCase):
    def test_failed_initialization_is_retried_on_next_request(self):
        detector = make_detector()
        detector._model_ready = False
        detector._initialize_model.side_effect = lambda: setattr(
            detector, "_model_ready", True
        )
        result = detector.detect_raw(np.zeros((1, 8, 8, 3), np.uint8))
        detector._create_socket.assert_called_once()
        detector._initialize_model.assert_called_once()
        detector._socket.send_multipart.assert_called_once()
        self.assertEqual(result.shape, (20, 6))

    def test_timeout_marks_peer_unready_without_nested_initialization(self):
        detector = make_detector()
        detector._socket.recv_multipart.side_effect = zmq.Again()
        result = detector.detect_raw(np.zeros((1, 8, 8, 3), np.uint8))
        self.assertFalse(detector._model_ready)
        self.assertIs(result, detector._zero_result)
        detector._initialize_model.assert_not_called()

    def test_availability_query_does_not_extend_socket_timeout(self):
        detector = make_detector()
        detector._socket.recv_multipart.return_value = [
            json.dumps({"model_available": True, "model_loaded": True}).encode()
        ]
        self.assertTrue(detector._check_and_transfer_model())
        detector._socket.setsockopt.assert_not_called()


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
        header, payload = detector._socket.send_multipart.call_args.args[0]
        header = json.loads(header)
        self.assertEqual(header["encoding"], "jpeg")
        self.assertEqual(header["dtype"], "uint8")
        self.assertEqual(header["shape"], list(image.shape))
        self.assertTrue(payload.startswith(b"\xff\xd8"))
