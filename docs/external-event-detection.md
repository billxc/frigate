# Object detection for imported historical recordings

The external-event importer defaults `detect_objects` to true. Set it to false
to retain storage-only behavior. The multipart metadata field is JSON; this flag
does not add a new route or change its administrator/camera authorization.

## Processing

- Sample the stored recording at one frame per second.
- Use the configured ZMQ detector plugin and its existing priority failover.
  The first configured detector is selected. This implementation is limited to
  ZMQ with RGB/BGR input; it does not initialize a second hardware detector.
- Apply configured tracked labels and the higher of each label's `min_score`
  and `threshold` to each sample.
- Collect a label-level summary with first/last sampled offsets and the best
  confidence/frame. This is not individual-object tracking or median-confidence
  confirmation. Multiple objects with the same label are summarized together.
- Create stable child Event IDs, real scores, relative boxes and best-frame
  clean snapshots/thumbnails. Retain the parent cloud event and recording.
- Add child IDs and labels to the existing Review and save the detection summary
  under the parent's `data.object_detection`. The API returns the same summary.

Original timestamps are retained by adding sample offsets to the historical
start time. Nothing is injected into the live capture pipeline.

Repeating an older storage-only import with `detect_objects=true` performs the
missing analysis without creating another recording. Completed analysis is not
repeated. Detector outage returns 503; media/parent records may already exist,
so the same import can be retried. Partial child inserts use stable IDs.

## Limits and validation

This version uses confidence and tracked-label filters only. It does not apply
motion masks, object masks, area/ratio filters, zones, full tracking, GenAI
captions or timelapse previews. The child snapshot stores a clean image and box
metadata, not a burned-in bounding box. The sampled intervals are approximate.

Processing is synchronous in the existing importer worker and import lock.
Long recordings can exceed a client's HTTP timeout; there is no background-job
API yet. The standalone client's `--skip-detection` selects storage-only import.

Two focused tests cover label filtering, confidence, best-frame/offset selection
and empty results. Ruff and source compilation are checked. Real video inference,
production backfill and native child-event rendering are not yet verified. This
change is source-only and does not update the running deployment.
