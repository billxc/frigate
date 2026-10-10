"""Sample archived video with the configured Frigate object detector."""

from pathlib import Path


class DetectionUnavailable(RuntimeError):
    """The configured detector could not analyze the recording."""


def collect_objects(samples, tracked, thresholds):
    """Summarize each detected label without claiming multi-object tracking."""
    objects = {}
    for offset, detections in samples:
        for label, score, (top, left, bottom, right) in detections:
            if label not in tracked or score < thresholds[label]:
                continue
            box = [float(left), float(top), float(right - left), float(bottom - top)]
            if label not in objects:
                objects[label] = {
                    "label": label,
                    "score": score,
                    "box": box,
                    "snapshot_offset": offset,
                    "first_offset": offset,
                    "last_offset": offset,
                    "samples": 1,
                }
            else:
                obj = objects[label]
                obj["last_offset"] = offset
                obj["samples"] += 1
                if score > obj["score"]:
                    obj.update(score=score, box=box, snapshot_offset=offset)
    return [objects[label] for label in sorted(objects)]


def analyze_recording(config, camera: str, path: Path, duration: float) -> dict:
    """Detect one sample per second using the existing detector plugin."""
    import cv2
    import numpy as np

    from frigate.object_detection.base import LocalObjectDetector

    detector_name, detector_config = next(iter(config.detectors.items()))
    if detector_config.type != "zmq":
        raise DetectionUnavailable("Archive analysis currently requires a ZMQ detector")
    detector = LocalObjectDetector(detector_config=detector_config)
    detector.labels = config.model.merged_labelmap
    objects_config = config.cameras[camera].objects
    if not objects_config.track:
        detector.detect_api.router.close()
        return {
            "status": "complete",
            "detector": detector_name,
            "sample_fps": 1,
            "frames": 0,
            "objects": [],
        }
    thresholds = {
        label: max(
            objects_config.filters[label].min_score,
            objects_config.filters[label].threshold,
        )
        for label in objects_config.track
    }
    capture = cv2.VideoCapture(str(path))
    samples = []
    try:
        if not capture.isOpened():
            raise DetectionUnavailable("Could not decode imported recording")
        for offset in range(int(duration) + 1):
            if offset >= duration - 0.1:
                break
            capture.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
            ok, frame = capture.read()
            if not ok:
                if samples and offset >= duration - 1:
                    break
                raise DetectionUnavailable("Could not decode detection sample")
            pixels = frame
            if detector_config.model.input_pixel_format == "rgb":
                pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif detector_config.model.input_pixel_format != "bgr":
                raise DetectionUnavailable("Archive analysis requires RGB or BGR input")
            resized = cv2.resize(
                pixels, (detector_config.model.width, detector_config.model.height)
            )
            tensor = np.expand_dims(resized, axis=0)
            detections = detector.detect(tensor, threshold=min(thresholds.values()))
            # The configured failover plugin returns zero rows on an outage.
            # That is not evidence that the archived scene contains no objects.
            if (
                hasattr(detector.detect_api, "router")
                and detector.detect_api.router.state == "unavailable"
            ):
                raise DetectionUnavailable("All inference backends unavailable")
            samples.append((offset, detections))
    finally:
        capture.release()
        if hasattr(detector.detect_api, "router"):
            detector.detect_api.router.close()
    return {
        "status": "complete",
        "detector": detector_name,
        "sample_fps": 1,
        "frames": len(samples),
        "objects": collect_objects(samples, objects_config.track, thresholds),
    }
