"""Priority ZMQ inference routing for predeployed, protocol-compatible models."""

import json
import logging
import time
from pathlib import Path

import numpy as np
import zmq

logger = logging.getLogger(__name__)


class Backend:
    def __init__(self, context, endpoint, model_name, timeout_ms):
        self.context = context
        self.endpoint = endpoint
        self.model_name = model_name
        self.timeout_ms = timeout_ms
        self.socket = None
        self.ready = False
        self.retry_at = 0.0

    def close(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.ready = False

    def _remaining(self, deadline):
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            raise TimeoutError("inference request deadline exceeded")
        return remaining

    def _exchange(self, frames, deadline):
        self.socket.setsockopt(zmq.SNDTIMEO, self._remaining(deadline))
        self.socket.send_multipart(frames)
        self.socket.setsockopt(zmq.RCVTIMEO, self._remaining(deadline))
        return self.socket.recv_multipart()

    def infer(self, frames):
        deadline = time.monotonic() + self.timeout_ms / 1000
        if self.socket is None:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.IMMEDIATE, 1)
            self.socket.connect(self.endpoint)
        if not self.ready:
            request = {"model_request": True, "model_name": self.model_name}
            reply = self._exchange([json.dumps(request).encode()], deadline)
            state = json.loads(reply[0])
            if not (
                isinstance(state, dict)
                and state.get("model_available")
                and state.get("model_loaded")
                and state.get("model_name") == self.model_name
            ):
                raise RuntimeError(f"predeployed model not ready: {self.model_name}")
            # No uploads: each endpoint must keep its own configured model.
            self.ready = True
        reply = self._exchange(frames, deadline)
        if len(reply) == 2:
            header = json.loads(reply[0])
            if not isinstance(header, dict):
                raise ValueError("invalid detection response header")
            if header.get("error"):
                raise RuntimeError(header["error"])
            if header.get("shape") != [20, 6] or header.get("dtype") != "float32":
                raise ValueError("invalid detection response shape/dtype")
            payload = reply[1]
        elif len(reply) == 1 and len(reply[0]) == 480:
            payload = reply[0]
        else:
            raise ValueError("invalid detection response frames")
        rows = np.frombuffer(payload, dtype=np.float32).reshape(20, 6)
        if not np.isfinite(rows).all():
            raise ValueError("non-finite detection response")
        return rows


class FailoverDetector:
    def __init__(
        self, endpoints, timeout_ms=500, retry_interval_s=10, status_path=None
    ):
        self.context = zmq.Context()
        self.backends = [
            Backend(self.context, endpoint, model, timeout_ms)
            for endpoint, model in endpoints
        ]
        self.retry_interval_s = retry_interval_s
        self.status_path = Path(status_path) if status_path else None
        self.active = None
        self.zero = np.zeros((20, 6), dtype=np.float32)
        self._status("starting")

    def _status(self, state):
        self.state = state
        record = {
            "state": state,
            "changed_at": time.time(),
            "endpoint": self.active.endpoint if self.active else None,
            "model": self.active.model_name if self.active else None,
        }
        logger.info("ZMQ detector route: %s", record)
        if self.status_path:
            try:
                temporary = self.status_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(record))
                temporary.replace(self.status_path)
            except OSError:
                logger.exception("Could not write detector route status")

    def detect(self, frames):
        for backend in self.backends:
            if time.monotonic() < backend.retry_at:
                continue
            try:
                detections = backend.infer(frames)
            except (
                zmq.ZMQError,
                TimeoutError,
                ValueError,
                RuntimeError,
                KeyError,
                IndexError,
            ) as exc:
                logger.warning(
                    "ZMQ backend failed: %s model=%s: %s",
                    backend.endpoint,
                    backend.model_name,
                    exc,
                )
                backend.close()
                backend.retry_at = time.monotonic() + self.retry_interval_s
                continue
            # Valid empty detections are success, not a reason to switch models.
            if self.active is not backend:
                self.active = backend
                self._status("primary" if backend is self.backends[0] else "fallback")
            return detections
        if self.state != "unavailable":
            self.active = None
            self._status("unavailable")
        return self.zero

    def close(self):
        for backend in self.backends:
            backend.close()
        self.context.term()
