"""Small real-ZMQ protocol tests; no Frigate/Windows service is touched."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import zmq

from frigate.detectors.zmq_failover import FailoverDetector


class Server:
    def __init__(self, model, label):
        self.model, self.label = model, label
        self.mode = "ok"
        self.calls = 0
        self.models = []
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        self.ready.wait()

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        self.endpoint = f"tcp://127.0.0.1:{port}"
        self.ready.set()
        while not self.stop.is_set():
            if not socket.poll(20):
                continue
            frames = socket.recv_multipart()
            h = json.loads(frames[0])
            self.calls += 1
            if self.mode == "timeout":
                time.sleep(0.15)
            if h.get("model_request"):
                self.models.append(h["model_name"])
                ok = self.mode != "unloaded" and h["model_name"] == self.model
                socket.send_json(
                    {
                        "model_available": ok,
                        "model_loaded": ok,
                        "model_name": self.model,
                    }
                )
            else:
                rows = np.zeros((20, 6), np.float32)
                if self.mode != "empty":
                    rows[0] = [self.label, 0.9, 0.1, 0.1, 0.8, 0.8]
                header = {"shape": [20, 6], "dtype": "float32"}
                if self.mode == "error":
                    header["error"] = "simulated inference failure"
                payload = b"bad" if self.mode == "malformed" else rows.tobytes()
                socket.send_multipart([json.dumps(header).encode(), payload])
        socket.close()
        context.term()

    def close(self):
        self.stop.set()
        self.thread.join()


class FailoverTests(unittest.TestCase):
    def setUp(self):
        self.primary = Server("yolo26m.onnx", 0)
        self.fallback = Server("yolov9t.onnx", 16)
        self.temp = tempfile.TemporaryDirectory()
        self.status = Path(self.temp.name) / "status.json"
        self.router = FailoverDetector(
            [
                (self.primary.endpoint, self.primary.model),
                (self.fallback.endpoint, self.fallback.model),
            ],
            timeout_ms=60,
            retry_interval_s=0.2,
            status_path=self.status,
        )
        self.frames = [b'{"model_type":"yolo-generic"}', b"test image"]

    def tearDown(self):
        self.router.close()
        self.primary.close()
        self.fallback.close()
        self.temp.cleanup()

    def test_empty_is_success_then_error_fallback_and_recovery(self):
        self.router.detect(self.frames)
        self.primary.mode = "empty"
        self.assertFalse(self.router.detect(self.frames).any())
        self.assertEqual(self.fallback.calls, 0)
        self.primary.mode = "error"
        self.assertEqual(self.router.detect(self.frames)[0, 0], 16)
        count = self.primary.calls
        self.router.detect(self.frames)
        self.assertEqual(self.primary.calls, count)  # no repeated timeout each frame
        self.primary.mode = "ok"
        time.sleep(0.21)
        self.assertEqual(self.router.detect(self.frames)[0, 0], 0)
        self.assertEqual(json.loads(self.status.read_text())["state"], "primary")
        self.assertEqual(set(self.primary.models), {"yolo26m.onnx"})
        self.assertEqual(set(self.fallback.models), {"yolov9t.onnx"})

    def test_cold_primary_timeout_and_return_to_primary(self):
        self.primary.mode = "timeout"
        start = time.monotonic()
        self.assertEqual(self.router.detect(self.frames)[0, 0], 16)
        self.assertLess(time.monotonic() - start, 0.18)
        self.primary.mode = "ok"
        time.sleep(0.21)
        self.assertEqual(self.router.detect(self.frames)[0, 0], 0)

    def test_malformed_and_both_unavailable_then_fallback_recovers(self):
        self.primary.mode = "malformed"
        self.fallback.mode = "unloaded"
        self.assertFalse(self.router.detect(self.frames).any())
        self.assertEqual(json.loads(self.status.read_text())["state"], "unavailable")
        self.fallback.mode = "ok"
        time.sleep(0.21)
        self.assertEqual(self.router.detect(self.frames)[0, 0], 16)
        self.assertEqual(json.loads(self.status.read_text())["model"], "yolov9t.onnx")


if __name__ == "__main__":
    unittest.main()
