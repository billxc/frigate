"""Import an existing external event recording into native Frigate storage."""

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, ValidationError

from frigate.api.auth import require_camera_access, require_role
from frigate.api.defs.tags import Tags
from frigate.api.external_detection import DetectionUnavailable, analyze_recording
from frigate.const import CLIPS_DIR, RECORD_DIR, THUMB_DIR
from frigate.models import Event, Recordings, ReviewSegment

logger = logging.getLogger(__name__)
router = APIRouter(tags=[Tags.events])
_import_lock = threading.Lock()


class ExternalEventMetadata(BaseModel):
    camera: str = Field(max_length=20)
    source: Literal["mijia-cloud-event"] = "mijia-cloud-event"
    source_event_id: str = Field(min_length=1, max_length=128)
    start_time: float = Field(gt=0)
    reported_duration: float = Field(gt=0, le=600)
    label: str = Field(default="doorbell", min_length=1, max_length=20)
    sub_label: str | None = Field(default=None, max_length=100)
    description: str | None = None
    playback_codec: Literal["copy", "h264"] = "copy"
    recording_timezone: Literal["utc", "local"] = "utc"
    detect_objects: bool = True


def _probe(config, path: Path) -> dict:
    output = subprocess.check_output(
        [
            config.ffmpeg.ffprobe_path,
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        timeout=20,
    )
    return json.loads(output)


def _run_ffmpeg(config, *args: str) -> None:
    subprocess.run(
        [
            config.ffmpeg.ffmpeg_path,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-protocol_whitelist",
            "file",
            *args,
        ],
        capture_output=True,
        check=True,
        timeout=60,
    )


def _apply_detection(config, metadata, event_id, recording_path, duration):
    """Attach sampled object events to an imported recording and its Review."""
    result = analyze_recording(config, metadata.camera, recording_path, duration)
    ids = [event_id]
    for obj in result["objects"]:
        suffix = hashlib.sha256(obj["label"].encode()).hexdigest()[:6]
        object_id = f"{event_id}-{suffix}"
        ids.append(object_id)
        snapshot_time = metadata.start_time + obj["snapshot_offset"]
        clean = Path(CLIPS_DIR) / f"{metadata.camera}-{object_id}-clean.png"
        thumb = Path(THUMB_DIR) / metadata.camera / f"{object_id}.webp"
        _run_ffmpeg(
            config,
            "-ss",
            str(obj["snapshot_offset"]),
            "-i",
            str(recording_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=640:-2",
            str(clean),
        )
        _run_ffmpeg(
            config,
            "-i",
            str(clean),
            "-frames:v",
            "1",
            "-vf",
            "scale=320:-2",
            str(thumb),
        )
        Event.insert(
            id=object_id,
            camera=metadata.camera,
            label=obj["label"],
            sub_label=metadata.sub_label,
            start_time=metadata.start_time + obj["first_offset"],
            end_time=metadata.start_time + min(obj["last_offset"] + 1, duration),
            false_positive=False,
            zones=[],
            thumbnail=None,
            has_clip=True,
            has_snapshot=True,
            retain_indefinitely=False,
            data={
                "type": "object",
                "source": metadata.source,
                "external_parent_id": event_id,
                "sampled_detection": True,
                "score": obj["score"],
                "top_score": obj["score"],
                "box": obj["box"],
                "region": [0, 0, 1, 1],
                "snapshot_clean": True,
                "snapshot_frame_time": snapshot_time,
            },
        ).on_conflict_ignore().execute()
    parent = Event.get_by_id(event_id)
    Event.update(data={**parent.data, "object_detection": result}).where(
        Event.id == event_id
    ).execute()
    review = ReviewSegment.get_by_id(event_id)
    ReviewSegment.update(
        data={
            **review.data,
            "detections": ids,
            "objects": [metadata.label, *[obj["label"] for obj in result["objects"]]],
        }
    ).where(ReviewSegment.id == event_id).execute()
    Recordings.update(objects=sum(obj["samples"] for obj in result["objects"])).where(
        Recordings.id == event_id
    ).execute()
    return result


def _store_import(config, metadata: ExternalEventMetadata, payload: bytes) -> dict:
    digest = hashlib.sha256(
        f"{metadata.camera}/{metadata.source}/{metadata.source_event_id}".encode()
    ).hexdigest()[:12]
    event_id = f"{metadata.start_time:.3f}-{digest}"
    with _import_lock, Event._meta.database.connection_context():
        previous = Event.get_or_none(Event.id == event_id)
        if previous is not None:
            if previous.data.get("external_source_id") != metadata.source_event_id:
                raise ValueError("External event identifier conflict")
            if previous.data.get("playback_codec", "copy") != metadata.playback_codec:
                raise ValueError("Existing import uses a different playback codec")
            recording = Recordings.get_or_none(Recordings.id == event_id)
            review = ReviewSegment.get_or_none(ReviewSegment.id == event_id)
            if (
                recording is not None
                and review is not None
                and Path(recording.path).is_file()
            ):
                detection = previous.data.get("object_detection")
                if metadata.detect_objects and detection is None:
                    detection = _apply_detection(
                        config,
                        metadata,
                        event_id,
                        Path(recording.path),
                        recording.duration,
                    )
                return {
                    "success": True,
                    "event_id": event_id,
                    "review_id": event_id,
                    "duplicate": True,
                    "object_detection": detection,
                }

        stamp = (
            datetime.fromtimestamp(metadata.start_time).astimezone()
            if metadata.recording_timezone == "local"
            else datetime.fromtimestamp(metadata.start_time, UTC)
        )
        recording_path = (
            Path(RECORD_DIR)
            / stamp.strftime("%Y-%m-%d/%H")
            / metadata.camera
            / f"{stamp:%M.%S}-{digest}.mp4"
        )
        clean_path = Path(CLIPS_DIR) / f"{metadata.camera}-{event_id}-clean.png"
        review_thumb = (
            Path(CLIPS_DIR) / "review" / f"thumb-{metadata.camera}-{event_id}.webp"
        )
        event_thumb = Path(THUMB_DIR) / metadata.camera / f"{event_id}.webp"
        for parent in {
            recording_path.parent,
            clean_path.parent,
            review_thumb.parent,
            event_thumb.parent,
        }:
            parent.mkdir(parents=True, exist_ok=True)

        # Keep staging on the media filesystem, not the /tmp/cache tmpfs, so
        # publishing into recordings/clips uses an atomic same-device rename.
        staging_root = Path(CLIPS_DIR) / ".external-import"
        staging_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="external-import-", dir=staging_root
        ) as directory:
            staging = Path(directory)
            uploaded = staging / "uploaded.mp4"
            uploaded.write_bytes(payload)
            source_info = _probe(config, uploaded)
            video = next(
                s for s in source_info["streams"] if s["codec_type"] == "video"
            )
            duration = float(source_info["format"]["duration"])
            if (
                not 0 < duration <= 600
                or abs(duration - metadata.reported_duration) > 1
            ):
                raise ValueError("Recording duration differs from source metadata")
            stored = staging / "recording.mp4"
            codec_args = ["-c", "copy"]
            if metadata.playback_codec == "h264":
                codec_args = [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-vf",
                    "scale=w='min(1280,iw)':h=-2",
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    "2",
                    "-bf",
                    "0",
                    "-c:a",
                    "aac",
                    "-ar",
                    "48000",
                    "-ac",
                    "1",
                ]
            elif video["codec_name"] == "hevc":
                codec_args += ["-tag:v", "hvc1"]
            _run_ffmpeg(
                config,
                "-i",
                str(uploaded),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                *codec_args,
                "-movflags",
                "+faststart",
                str(stored),
            )
            duration = float(_probe(config, stored)["format"]["duration"])
            clean = staging / "clean.png"
            thumb = staging / "thumb.webp"
            _run_ffmpeg(
                config,
                "-i",
                str(stored),
                "-frames:v",
                "1",
                "-vf",
                "scale=640:-2",
                str(clean),
            )
            _run_ffmpeg(
                config,
                "-i",
                str(clean),
                "-frames:v",
                "1",
                "-vf",
                "scale=320:-2",
                str(thumb),
            )
            os.replace(stored, recording_path)
            os.replace(clean, clean_path)
            event_thumb.write_bytes(thumb.read_bytes())
            os.replace(thumb, review_thumb)

        end_time = metadata.start_time + duration
        event_data = {
            "type": "api",
            "source": metadata.source,
            "external_source_id": metadata.source_event_id,
            "reported_duration": metadata.reported_duration,
            "playback_codec": metadata.playback_codec,
            "recording_timezone": metadata.recording_timezone,
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "description": metadata.description,
            "score": 0,
            "top_score": 0,
            "snapshot_clean": True,
            "snapshot_frame_time": metadata.start_time,
        }
        # Frigate serializes writes through its existing queue database. Publish
        # the Review last, after the media and recording/event indexes are ready.
        # Stable identifiers let a repeat request finish a partial import safely.
        Recordings.insert(
            id=event_id,
            camera=metadata.camera,
            path=str(recording_path),
            start_time=metadata.start_time,
            end_time=end_time,
            duration=duration,
            segment_size=recording_path.stat().st_size / (1024 * 1024),
            motion=0,
            objects=0,
            regions=0,
        ).on_conflict_ignore().execute()
        Event.insert(
            id=event_id,
            camera=metadata.camera,
            label=metadata.label,
            sub_label=metadata.sub_label,
            start_time=metadata.start_time,
            end_time=end_time,
            false_positive=False,
            zones=[],
            thumbnail=None,
            has_clip=True,
            has_snapshot=True,
            retain_indefinitely=False,
            data=event_data,
        ).on_conflict_ignore().execute()
        ReviewSegment.insert(
            id=event_id,
            camera=metadata.camera,
            start_time=metadata.start_time,
            end_time=end_time,
            severity="alert",
            thumb_path=str(review_thumb),
            data={
                "detections": [event_id],
                "objects": [metadata.label],
                "verified_objects": [],
                "sub_labels": [metadata.sub_label] if metadata.sub_label else [],
                "zones": [],
                "audio": [],
                "thumb_time": metadata.start_time,
                "metadata": None,
            },
        ).on_conflict_ignore().execute()
        detection = None
        if metadata.detect_objects:
            detection = _apply_detection(
                config, metadata, event_id, recording_path, duration
            )
        logger.info(
            "Imported external recording %s for camera %s", event_id, metadata.camera
        )
        return {
            "success": True,
            "event_id": event_id,
            "review_id": event_id,
            "duplicate": False,
            "start_time": metadata.start_time,
            "end_time": end_time,
            "duration": duration,
            "object_detection": detection,
        }


@router.post(
    "/events/import",
    dependencies=[Depends(require_role(["admin"]))],
    summary="Import an external event recording",
    description="Import one existing MP4 using its original event time. Creates native Event, Review and Recordings indexes, snapshots and playback media. Repeating the same source event is idempotent. No live capture is started.",
)
async def import_external_event(
    request: Request,
    metadata: Annotated[str, Form(description="JSON ExternalEventMetadata")],
    video: Annotated[UploadFile, File(description="Existing MP4 recording")],
):
    """Receive an existing recording and publish its native Frigate indexes."""
    try:
        parsed = ExternalEventMetadata.model_validate_json(metadata)
    except ValidationError:
        raise HTTPException(422, "Invalid external event metadata") from None
    if parsed.camera not in request.app.frigate_config.cameras:
        raise HTTPException(404, "Camera is not configured")
    await require_camera_access(parsed.camera, request=request)
    if video.size is not None and video.size > 64 * 1024 * 1024:
        raise HTTPException(413, "Recording exceeds 64 MiB")
    payload = await video.read()
    if len(payload) > 64 * 1024 * 1024:
        raise HTTPException(413, "Recording exceeds 64 MiB")
    try:
        return await asyncio.to_thread(
            _store_import, request.app.frigate_config, parsed, payload
        )
    except DetectionUnavailable:
        raise HTTPException(
            503, "Object detection unavailable; import can be retried"
        ) from None
    except (ValueError, StopIteration):
        raise HTTPException(400, "Invalid recording or metadata") from None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.exception("Unable to import external recording for %s", parsed.camera)
        raise HTTPException(500, "Unable to import external recording") from None
