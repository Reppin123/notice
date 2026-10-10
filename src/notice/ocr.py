"""On-device text reading + face check (Apple Vision). No key, no model call, ~100-200 ms per crop."""
from __future__ import annotations

import cv2
import numpy as np
import Quartz
import Vision
from Foundation import NSData


def _cgimage(bgr: np.ndarray):
    ok, png = cv2.imencode(".png", bgr)
    data = NSData.dataWithBytes_length_(png.tobytes(), len(png))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    return Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)


def read(bgr: np.ndarray) -> tuple[str, float]:
    """(text top-to-bottom, largest face area as a fraction of the image)."""
    img = _cgimage(bgr)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, {})
    text_req = Vision.VNRecognizeTextRequest.alloc().init()
    text_req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    text_req.setUsesLanguageCorrection_(True)
    face_req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    ok, err = handler.performRequests_error_([text_req, face_req], None)
    lines = []
    for obs in text_req.results() or []:
        cand = obs.topCandidates_(1)
        if cand and cand[0].confidence() >= 0.3:
            box = obs.boundingBox()  # normalized, origin bottom-left
            lines.append((-(box.origin.y + box.size.height), box.origin.x, cand[0].string()))
    lines.sort()
    face = max((f.boundingBox().size.width * f.boundingBox().size.height for f in face_req.results() or []), default=0.0)
    return " ".join(l[2] for l in lines), float(face)
