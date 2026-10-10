"""Non-VLM visual evidence via Apple's Vision framework (on-device, ships with macOS, no model license).

- detect_people(): face and human-body rectangles (supporting evidence for subject count).
- feature_distance(): VNFeaturePrint distance between two images (lower = more similar).
- face_distance(): feature-print distance between the largest face crops of two images.

These are SUPPORTING signals. A small distance is not proof of identity and a large one is not proof
of a different person; thresholds come from `cinema benchmark --stage validators`.
"""
from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from PIL import Image


def _handler(path: Path):
    import Vision
    from Foundation import NSURL
    return Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(path)), None)


def detect_people(path: Path, min_conf: float = 0.5) -> dict:
    import Vision
    faces = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    humans = Vision.VNDetectHumanRectanglesRequest.alloc().init()
    try:
        humans.setUpperBodyOnly_(False)
    except AttributeError:
        pass
    ok, err = _handler(path).performRequests_error_([faces, humans], None)
    if not ok:
        raise RuntimeError(f"Vision failed on {path}: {err}")

    def boxes(req):
        out = []
        for r in req.results() or []:
            if r.confidence() < min_conf:
                continue
            b = r.boundingBox()
            out.append({"x": b.origin.x, "y": b.origin.y, "w": b.size.width, "h": b.size.height,
                        "conf": round(float(r.confidence()), 3)})
        return out

    f, h = boxes(faces), boxes(humans)
    return {"faces": f, "humans": h, "face_count": len(f), "human_count": len(h)}


@lru_cache(maxsize=512)
def _featureprint(path: str, mtime_ns: int):
    import Vision
    req = Vision.VNGenerateImageFeaturePrintRequest.alloc().init()
    ok, err = _handler(Path(path)).performRequests_error_([req], None)
    if not ok or not req.results():
        raise RuntimeError(f"feature print failed on {path}: {err}")
    return req.results()[0]


def _vector(obs):
    """Feature-print embedding as a numpy vector (pyobjc does not expose computeDistance's float* out-arg)."""
    import numpy as np
    data = bytes(obs.data())
    dtype = np.float32 if obs.elementType() == 1 else np.float64  # VNElementTypeFloat=1, Double=2
    return np.frombuffer(data, dtype=dtype)[: obs.elementCount()].astype(np.float64)


def feature_distance(a: Path, b: Path) -> float:
    """Euclidean distance between Vision feature prints (the metric VNFeaturePrintObservation uses)."""
    import numpy as np
    va = _vector(_featureprint(str(a), a.stat().st_mtime_ns))
    vb = _vector(_featureprint(str(b), b.stat().st_mtime_ns))
    return round(float(np.linalg.norm(va - vb)), 4)


def largest_face_crop(path: Path, out: Path, margin: float = 0.35) -> Path | None:
    people = detect_people(path)
    if not people["faces"]:
        return None
    f = max(people["faces"], key=lambda b: b["w"] * b["h"])
    im = Image.open(path).convert("RGB")
    W, H = im.size
    # Vision uses normalized, bottom-left-origin coordinates.
    x0, y0 = f["x"] * W, (1 - f["y"] - f["h"]) * H
    w, h = f["w"] * W, f["h"] * H
    box = (max(0, x0 - margin * w), max(0, y0 - margin * h), min(W, x0 + w * (1 + margin)),
           min(H, y0 + h * (1 + margin)))
    im.crop(tuple(int(v) for v in box)).resize((224, 224), Image.LANCZOS).save(out)
    return out


def face_distance(a: Path, b: Path) -> float | None:
    with tempfile.TemporaryDirectory() as td:
        ca = largest_face_crop(a, Path(td) / "a.png")
        cb = largest_face_crop(b, Path(td) / "b.png")
        if not ca or not cb:
            return None
        return feature_distance(ca, cb)
