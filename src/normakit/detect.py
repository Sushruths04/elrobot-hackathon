"""Object detection for pick-and-place: JPEG/frame in -> compact JSON detections out.

Two modes:
  * closed-set COCO (YOLO11n, fast):    Detector(classes=["cup", "bottle"])
  * open-vocabulary (YOLO-World, text): Detector(prompts=["red bull can", "cup"])
COCO has `cup` and `bottle` but no "can" -> use prompts for the Red Bull can.

Image source is not our business here: frames come from Station (`usbvideo/<id>` JPEGs),
a webcam or files - see tools/detect_live.py. Pixel -> table XY is vision.py's homography.
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parents[2] / "models"
COCO_MODEL = "yolo11n.pt"
OPEN_VOCAB_MODEL = "yolov8s-worldv2.pt"

# Our demo objects (whitelist): label=synonym|synonym. Tune the phrases, not the labels.
# Measured 2026-11-21 (tape-measure photo): every extra class/phrase LOWERS scores (alone 0.43 ->
# with 'bottle' 0.11). Appearance phrases ('black device' 0.59) beat names. Keep lists short.
DEMO_OBJECTS = [
    "duplo=lego duplo brick|toy building block|plastic toy brick|lego",
    "can=soda can|aluminum can|energy drink can|drink can|tin can",
    "bottle=bottle|plastic bottle|water bottle",
]


@dataclass
class Detection:
    label: str
    conf: float
    box: list[float]        # x1, y1, x2, y2 in pixels
    center_px: list[float]  # box centre (x, y) in pixels
    bottom_px: list[float]  # bottom-centre: where the object touches the table (better for homography)
    area_frac: float        # box area / image area
    phrase: str = ""         # which prompt phrase fired (for tuning synonyms)

    def to_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else [round(x, 1) for x in v] if isinstance(v, list) else v)
                for k, v in asdict(self).items()}


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def to_rgb(img) -> np.ndarray:
    """JPEG bytes, path or ndarray (RGB) -> RGB ndarray."""
    if isinstance(img, np.ndarray):
        return img
    from PIL import Image
    src = io.BytesIO(img) if isinstance(img, (bytes, bytearray)) else img
    return np.array(Image.open(src).convert("RGB"))


class Detector:
    def __init__(self, classes: list[str] | None = None, prompts: list[str] | None = None,
                 conf: float = 0.35, model: str | None = None, imgsz: int = 640):
        from ultralytics import YOLO
        MODELS.mkdir(exist_ok=True)
        # prompts: "label=phrase1|phrase2" -> several phrases, one output label (whitelist)
        self.alias = {}
        for p in prompts or []:
            p = p.strip()
            if not p:
                continue
            label, _, phrases = p.partition("=") if "=" in p else (p, "", p)
            for ph in phrases.split("|"):
                if ph.strip():
                    self.alias[ph.strip()] = label.strip()
        self.prompts = list(self.alias)
        name = model or (OPEN_VOCAB_MODEL if self.prompts else COCO_MODEL)
        self.model = YOLO(str(MODELS / name) if not Path(name).exists() else name)  # downloads into models/
        self.device, self.conf, self.imgsz = _device(), conf, imgsz
        self.class_ids = None
        if self.prompts:
            self.model.set_classes(self.prompts)
        elif classes:
            names = {v: k for k, v in self.model.names.items()}
            missing = [c for c in classes if c not in names]
            if missing:
                raise ValueError(f"not COCO classes: {missing}; use prompts=[...] instead. "
                                 f"e.g. {sorted(names)[:12]}...")
            self.class_ids = [names[c] for c in classes]

    def detect(self, img) -> list[Detection]:
        rgb = to_rgb(img)
        h, w = rgb.shape[:2]
        # ultralytics expects BGR for ndarrays
        r = self.model.predict(rgb[:, :, ::-1], conf=self.conf, classes=self.class_ids, imgsz=self.imgsz,
                               device=self.device, verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            name = r.names[int(k)]
            out.append(Detection(label=self.alias.get(name, name), phrase=name, conf=float(c), box=[x1, y1, x2, y2],
                                 center_px=[(x1 + x2) / 2, (y1 + y2) / 2], bottom_px=[(x1 + x2) / 2, y2],
                                 area_frac=float((x2 - x1) * (y2 - y1) / (w * h))))
        return _dedupe(sorted(out, key=lambda d: -d.conf))

    def find(self, img, label: str) -> dict:
        """Best detection of `label`, as compact JSON with a readable error (AGENTS.md §6)."""
        dets = self.detect(img)
        hits = [d for d in dets if d.label == label]
        if hits:
            return {"ok": True, **hits[0].to_dict()}
        seen = sorted({d.label for d in dets})
        return {"ok": False, "error": f"no '{label}' found; visible: {', '.join(seen) or 'nothing'}"}


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def _dedupe(dets, thr=0.6):
    """Synonym prompts can box the same object twice under one label: keep the most confident."""
    keep = []
    for d in dets:
        if all(not (k.label == d.label and _iou(k.box, d.box) > thr) for k in keep):
            keep.append(d)
    return keep


def annotate(rgb: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """Draw boxes + labels; returns a BGR image for cv2.imshow / imwrite."""
    import cv2
    img = np.ascontiguousarray(rgb[:, :, ::-1])
    for d in dets:
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.circle(img, tuple(map(int, d.bottom_px)), 4, (0, 0, 255), -1)
        cv2.putText(img, f"{d.label} {d.conf:.2f}", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    return img
