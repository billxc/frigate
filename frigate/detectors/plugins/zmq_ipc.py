"""Frigate 0.18 ZMQ JPEG detector with priority failover; models are predeployed."""

import json
import os
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig
from frigate.detectors.jpeg import encode_jpeg
from frigate.detectors.zmq_failover import FailoverDetector

DETECTOR_KEY = "zmq"


class ZmqDetectorConfig(BaseDetectorConfig):
    model_config = ConfigDict(title="ZMQ with priority failover")
    type: Literal[DETECTOR_KEY]
    endpoint: str = "ipc:///tmp/cache/zmq_detector"
    request_timeout_ms: int = Field(default=500, gt=0)
    jpeg_quality: int | None = Field(default=None, ge=1, le=100)
    linger_ms: Literal[0] = 0
    fallback_endpoint: str | None = None
    fallback_model_name: str | None = None
    retry_interval_s: float = Field(default=10, gt=0)
    status_path: str | None = None

    @model_validator(mode="after")
    def paired_fallback(self):
        if bool(self.fallback_endpoint) != bool(self.fallback_model_name):
            raise ValueError(
                "fallback_endpoint and fallback_model_name must be set together"
            )
        return self


class ZmqIpcDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: ZmqDetectorConfig):
        super().__init__(detector_config)
        endpoints = [
            (detector_config.endpoint, os.path.basename(detector_config.model.path))
        ]
        if detector_config.fallback_endpoint:
            endpoints.append(
                (detector_config.fallback_endpoint, detector_config.fallback_model_name)
            )
        self.router = FailoverDetector(
            endpoints,
            detector_config.request_timeout_ms,
            detector_config.retry_interval_s,
            detector_config.status_path,
        )

    def detect_raw(self, tensor_input):
        header = {
            "shape": list(tensor_input.shape),
            "dtype": tensor_input.dtype.name,
            "model_type": self.detector_config.model.model_type.name,
        }
        if self.detector_config.jpeg_quality is not None:
            payload = encode_jpeg(
                tensor_input[0], quality=self.detector_config.jpeg_quality
            )
            header["encoding"] = "jpeg"
        else:
            payload = tensor_input.tobytes(order="C")
        return self.router.detect([json.dumps(header).encode(), payload])

    def __del__(self):
        if hasattr(self, "router"):
            self.router.close()
