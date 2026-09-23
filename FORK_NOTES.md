# Experimental remote inference patches

Branch: `local/remote-inference-v0.18`

Base: Frigate **v0.18.0**, commit `77a66e75c61862b048a07c1295877f4b31343504`.
This branch reproduces the locally tested implementation for inspection. It is
not a port to the current 0.19 development architecture or an upstream-ready PR.

## Separate concerns

1. **ZMQ recovery:** retry initialization after a peer failure and use the
   configured timeout for model availability queries. Model file transfer keeps
   its existing timeout. The previously closed, unmerged upstream PR
   [#24080](https://github.com/blakeblackshear/frigate/pull/24080) addresses a
   related problem and should be considered before proposing another fix.
2. **Optional JPEG transport:** encode contiguous uint8 RGB images in process
   with OpenCV, then send the existing multipart request with `encoding=jpeg`.
   The default raw tensor path remains available. No encoder subprocess is used.

The matching receiver changes are in
[billxc/apple-silicon-detector](https://github.com/billxc/apple-silicon-detector/tree/local/remote-inference).
That branch supports both raw tensors and the optional JPEG request.

## Relevant configuration fields

```yaml
detectors:
  remote:
    type: zmq
    endpoint: tcp://detector.example:5555
    request_timeout_ms: 2000
    jpeg_quality: 95

model:
  path: /config/model_cache/model.onnx
  width: 640
  height: 640
  model_type: yolo-generic
  input_pixel_format: rgb
  input_tensor: nhwc
  input_dtype: int
  labelmap_path: /labelmap/coco-80.txt
```

This is a partial example, not a complete camera configuration. Image dimensions
must match the model used in the paired setup.

The dtype/layout above describe the sender's image before JPEG encoding, **not
an INT8-quantized model**. The paired receiver decodes RGB and builds normalized
NCHW float32 input. This prototype convention was tested with YOLOv9-t; it must
not be assumed to work for arbitrary model preprocessing. An upstream design
should separate transport representation from model input requirements.

JPEG is lossy. Its quality is explicit, and an older receiver does not understand
this extension. There is no capability negotiation. To return to raw tensors,
remove `jpeg_quality` and restore the model's original dtype/layout together.

## Validation and scope

Focused tests:

```sh
python3 -u -m unittest frigate.test.test_zmq_ipc
```

All five focused tests passed in the pinned Frigate 0.18 runtime. The tests mock
the peer and check retries, timeout handling, JPEG framing, shape and RGB channel
order. Hardware inference and camera playback are not exercised
by these unit tests. Earlier separate deployment experiments exercised actual
JPEG transport and a remote DirectML model. The full upstream test suite and
0.19 compatibility have not been validated for this branch. New helper/test
files pass the focused Ruff checks. The original ZMQ module has pre-existing
broad-exception lint findings; these patches reduce them from 12 to 8 without
bundling an unrelated exception-handling rewrite.

This branch adds no credentials, model weights, recordings, local addresses,
SSH tunnels, Windows scheduled tasks, or deployment manifests. The implementation
was AI-assisted; human review and the upstream contribution/AI policies still
apply before any upstream submission.
