"""One on-device Apple Vision pass per crop: text, QR/barcodes, image labels, faces.
No key, no model call (~100-200 ms warm)."""
from __future__ import annotations

import cv2
import numpy as np
import Quartz
import Vision
from Foundation import NSData

# Labels too generic to help the agent ("a structure", "an indoor scene")
_GENERIC = {"structure", "material", "indoor", "outdoor", "wood_processed", "document", "text", "screenshot",
            "printed_page", "paper", "illustrations", "art", "graphic", "pattern", "rectangle", "consumer_electronics"}


LANGS = ["en-US"]


def _cgimage(bgr: np.ndarray):
    """Raw RGBA straight into a CGImage (~2 ms; a PNG round-trip was 50-60 ms)."""
    rgba = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA))
    h, w = rgba.shape[:2]
    provider = Quartz.CGDataProviderCreateWithCFData(NSData.dataWithBytes_length_(rgba.tobytes(), rgba.nbytes))
    return Quartz.CGImageCreate(w, h, 8, 32, w * 4, Quartz.CGColorSpaceCreateDeviceRGB(),
                                Quartz.kCGImageAlphaNoneSkipLast, provider, None, False,
                                Quartz.kCGRenderingIntentDefault)


def analyze(bgr: np.ndarray) -> dict:
    """{'text', 'conf', 'codes', 'labels', 'face'}; text top-to-bottom, conf = mean OCR confidence,
    face = largest face area fraction."""
    img = _cgimage(bgr)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, {})
    text_req = Vision.VNRecognizeTextRequest.alloc().init()
    text_req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    text_req.setUsesLanguageCorrection_(True)
    text_req.setRecognitionLanguages_(LANGS)  # unset, Vision mixed Cyrillic into English labels
    face_req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    code_req = Vision.VNDetectBarcodesRequest.alloc().init()
    label_req = Vision.VNClassifyImageRequest.alloc().init()
    handler.performRequests_error_([text_req, face_req, code_req, label_req], None)

    lines = []
    for obs in text_req.results() or []:
        cand = obs.topCandidates_(1)
        if cand and cand[0].confidence() >= 0.3:
            box = obs.boundingBox()  # normalized, origin bottom-left
            lines.append((-(box.origin.y + box.size.height), box.origin.x, cand[0].string(), cand[0].confidence()))
    lines.sort()
    codes = []
    for c in code_req.results() or []:
        p = c.payloadStringValue()
        if p and p not in codes:
            codes.append(str(p))
    labels = [str(o.identifier()).replace("_", " ") for o in (label_req.results() or [])
              if o.confidence() >= 0.35 and o.identifier() not in _GENERIC][:3]
    faces = [_box(f) for f in face_req.results() or []]
    face = max((w * h for _, _, w, h in faces), default=0.0)
    text = " ".join(l[2] for l in lines)
    conf = sum(l[3] for l in lines) / len(lines) if lines else 0.0
    if sum(ch.isalnum() for ch in text) < 3:  # OCR noise like "*" or "|"
        text, conf = "", 0.0
    return {"text": text, "conf": float(conf), "codes": codes, "labels": labels, "face": float(face), "faces": faces}


def codes(bgr: np.ndarray) -> list[str]:
    """QR/barcodes only (for a wider area around the gaze: codes sit beside the text you read)."""
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(_cgimage(bgr), {})
    req = Vision.VNDetectBarcodesRequest.alloc().init()
    handler.performRequests_error_([req], None)
    out = []
    for c in req.results() or []:
        p = c.payloadStringValue()
        if p and str(p) not in out:
            out.append(str(p))
    return out


def read(bgr: np.ndarray) -> tuple[str, float]:
    a = analyze(bgr)
    return a["text"], a["face"]


def _box(obs) -> tuple[float, float, float, float]:
    """Vision box (normalized, origin bottom-left) -> (x, y, w, h) normalized, origin top-left."""
    b = obs.boundingBox()
    return (b.origin.x, 1 - b.origin.y - b.size.height, b.size.width, b.size.height)


def find_faces(bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(_cgimage(bgr), {})
    req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    handler.performRequests_error_([req], None)
    return [_box(f) for f in req.results() or []]


def blur_faces(bgr: np.ndarray, faces=None) -> int:
    """Blur every detected face in place (bystanders in the venue). Returns how many."""
    faces = find_faces(bgr) if faces is None else faces
    h, w = bgr.shape[:2]
    for fx, fy, fw, fh in faces:
        x0, y0 = max(int((fx - fw * 0.2) * w), 0), max(int((fy - fh * 0.3) * h), 0)
        x1, y1 = min(int((fx + fw * 1.2) * w), w), min(int((fy + fh * 1.2) * h), h)
        if x1 > x0 and y1 > y0:
            roi = bgr[y0:y1, x0:x1]
            k = max(15, (max(roi.shape[:2]) // 2) | 1)
            bgr[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    return len(faces)
