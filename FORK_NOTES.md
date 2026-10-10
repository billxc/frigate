# Local Frigate patch stack

Branch: `local/nas-patches-v0.18` in `billxc/frigate`.
Base: Frigate v0.18.0, `77a66e75c61862b048a07c1295877f4b31343504`.

This branch collects the local source patches as separate commits. It is not
rebased onto the current upstream development architecture, and it is not an
upstream pull request. Existing fork branches are preserved.

## Commit order

| Commit | Concern |
| --- | --- |
| `2ebaffb` | Retry ZMQ peer initialization and honor availability-query timeout |
| `2b8500a` | Optional in-process OpenCV JPEG transport for uint8 RGB images |
| `e617e78` | Avoid output WebSocket races that interrupt preview generation |
| `2ed6667` | Priority inference failover with predeployed per-backend models |
| `5f4a8ea` | Asia/Shanghai recording directory and filename convention |
| `e964471` | Administrator-only existing-event recording import API |
| `2424346` | Optional local-time directory convention for imported recordings |

The first two commits retain the earlier single-peer prototype history. The
failover commit replaces that plugin's implementation, while retaining optional
JPEG transport. Its focused tests replace assertions about the old plugin's
internal socket state with router/transport tests.

## Current source coverage

The local deployment mounts eight source files, covered by this branch:

- `frigate/detectors/plugins/zmq_ipc.py`
- `frigate/detectors/zmq_failover.py`
- `frigate/detectors/jpeg.py`
- `frigate/record/maintainer.py`
- `frigate/output/output.py`
- `frigate/output/ws_auth.py`
- `frigate/api/event.py`
- `frigate/api/external_event.py`

Failover sources were copied from the active local implementation and formatted
with Ruff. The JPEG helper retains the existing fork's equivalent implementation,
with type annotation, docstring and formatting differences from its mounted copy.
The recording, output and import source files match the deployed copies.

### Priority failover

The primary backend is preferred, an unavailable backend is retried after the
configured interval, and valid empty detections are successful inference. Each
endpoint owns a predeployed model; this router does not upload or replace model
files. The receiver must implement the matching model-ready and detection
response protocol. Endpoint addresses and model filenames are configured by the
operator, not embedded in this branch.

### JPEG transport

JPEG is optional and lossy. `jpeg_quality` controls encoding and the header uses
`encoding=jpeg`. The receiver must support this extension. Sender uint8 RGB
layout describes transport pixels, not INT8 model quantization. There is no
capability negotiation. The receiver changes are maintained separately in
`billxc/apple-silicon-detector`.

### Recording timezone

Only disk directory/filename naming changes to Asia/Shanghai. Database epoch
timestamps and media timing remain unchanged. Existing recordings are not
migrated by this patch.

### Historical-event import

`POST /api/events/import` receives an existing decoded MP4 and JSON metadata.
It checks administrator role and camera access, preserves the original event
time/source identifier, and creates native Event, Review and Recordings indexes,
clean snapshots and thumbnails. Stable identifiers make repeats idempotent.

The optional H.264/AAC playback derivative improves browser compatibility. The
original helper archive is separate and is not modified. `recording_timezone`
selects UTC or service-local disk naming; it does not change event epoch times.
Media staging is on the media filesystem, not the cache tmpfs. Writes use the
existing queued ORM and publish Review last, not a multi-table transaction.

The import-only camera remains disabled for capture. No timelapse previews or
new MQTT/WebSocket notifications are generated. Changing codec policy on an
existing import is rejected instead of silently overwriting it. Automatic cloud
polling and synchronization belong to the standalone helper, not this fork.

## Validation and publication scope

Ruff checks cover newly collected failover/transport tests and the import module;
Python source compilation and focused deployment-source comparisons are used
for this collection. Historical isolated and production import tests returned
success, preserved the original timestamp and returned duplicate=true on repeat.

Focused tests are in `frigate/test/test_zmq_failover.py`,
`frigate/test/test_zmq_ipc.py` and `frigate/test/test_output_ws_auth.py`.
The host does not have a complete Frigate test runtime; collection-time unit
execution stopped at missing transitive dependencies. No full-suite pass or
upstream compatibility is claimed. The generated API auth specification is
carried with the import commit, not edited manually.

Only source code, focused tests and documentation are published. No credentials,
configuration from a running installation, database, model weights, media,
receiver deployment scripts or private runtime artifacts are included. No live
service is modified by assembling or publishing this branch.

This work is AI-assisted. Human review and the upstream contribution/AI policy
remain required before any upstream submission.
