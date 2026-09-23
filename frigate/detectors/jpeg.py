"""In-process CPU JPEG encoding of the detector's uint8 HWC RGB image."""

import cv2
from numpy import ndarray


def encode_jpeg(rgb: ndarray, *, quality: int) -> bytes:
    """Encode uint8 HWC RGB pixels as a non-progressive 4:2:0 JPEG."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
            cv2.IMWRITE_JPEG_OPTIMIZE,
            0,
            cv2.IMWRITE_JPEG_PROGRESSIVE,
            0,
        ],
    )
    if not ok:
        raise RuntimeError("OpenCV JPEG encoding failed")
    return encoded.tobytes()
