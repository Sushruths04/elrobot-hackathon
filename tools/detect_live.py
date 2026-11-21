#!/usr/bin/env python3
"""
Live object detection (YOLO) - read-only, never moves the arm.

  .venv/bin/python tools/detect_live.py                                  # Mac webcam, COCO cup+bottle
  .venv/bin/python tools/detect_live.py --demo                           # our objects: duplo, can, bottle
  .venv/bin/python tools/detect_live.py --prompts "can=soda can|drink can,cup"   # open-vocabulary (YOLO-World)
  .venv/bin/python tools/detect_live.py --source photo1.jpg photo2.jpg    # files -> snapshots/det_*.jpg
  .venv/bin/python tools/detect_live.py --source station [--camera cam0]  # Station camera (usbvideo queue)
  add --no-show for headless (JSON lines only)

Every frame's detections are appended to logs/detections-<time>.jsonl. Press q in the window to quit, s to save the raw frame to snapshots/.
Webcam: dev only, while we have no Station camera (AGENTS.md: on the rig, cameras go through Station).
"""
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from normakit.detect import DEMO_OBJECTS, Detector, annotate, to_rgb  # noqa: E402


def frames(a):
    import cv2
    src = a.source
    if src and src[0] == "station":
        from normakit.station_backend import StationBackend
        b = StationBackend(a.host)
        b.start()
        t0 = time.time()
        while True:
            try:
                yield "station", to_rgb(b.frame_jpeg(a.camera))
            except RuntimeError as e:
                if time.time() - t0 > 10:
                    sys.exit(f"station camera: {e}")
            time.sleep(0.03)
    elif src and not src[0].startswith("webcam"):
        for p in src:
            yield p, to_rgb(p)
    else:
        idx = int(src[0].split(":")[1]) if src and ":" in src[0] else 0
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            sys.exit(f"cannot open webcam {idx} (macOS: allow camera access for your terminal app in "
                     "System Settings -> Privacy & Security -> Camera)")
        while True:
            ok, bgr = cap.read()
            if not ok:
                sys.exit("webcam returned no frame")
            yield f"webcam:{idx}", bgr[:, :, ::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs="*", default=["webcam"], help="webcam[:N] | station | image files")
    ap.add_argument("--classes", default="cup,bottle", help="COCO class names (closed-set YOLO11n)")
    ap.add_argument("--prompts", default="", help="comma-separated prompts -> open-vocabulary YOLO-World; "
                    "'label=phrase1|phrase2' maps synonyms to one label")
    ap.add_argument("--demo", action="store_true", help="our 3 objects: duplo, can, bottle (detect.DEMO_OBJECTS)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--model", help="override weights file")
    ap.add_argument("--camera", default="cam0", help="station camera (cam0 or usbvideo/<id>)")
    ap.add_argument("--host", default="localhost:8888")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    prompts = DEMO_OBJECTS if a.demo else [p for p in a.prompts.split(",") if p.strip()]
    det = Detector(classes=None if prompts else a.classes.split(","), prompts=prompts, conf=a.conf, model=a.model)
    (REPO / "logs").mkdir(exist_ok=True)
    log = open(REPO / "logs" / f"detections-{time.strftime('%Y%m%d-%H%M%S')}.jsonl", "a")
    is_files = a.source and a.source[0] not in ("station",) and not a.source[0].startswith("webcam")
    import cv2
    n, t0 = 0, time.time()
    for name, rgb in frames(a):
        t = time.time()
        dets = det.detect(rgb)
        ms = (time.time() - t) * 1000
        rec = {"t": round(t, 3), "src": name, "shape": list(rgb.shape[:2]), "ms": round(ms, 1),
               "dets": [d.to_dict() for d in dets]}
        log.write(json.dumps(rec) + "\n"); log.flush()
        n += 1
        if is_files or a.no_show or n % 15 == 1:
            print(json.dumps({k: rec[k] for k in ("src", "ms")} | {"dets": [(d.label, round(d.conf, 2), d.phrase) for d in dets]}))
        if is_files:
            out = REPO / "snapshots" / f"det_{Path(name).stem}.jpg"
            out.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(out), annotate(rgb, dets))
            print("  ->", out)
            continue
        if not a.no_show:
            img = annotate(rgb, dets)
            summary = "  ".join(f"{d.label} {d.conf:.2f}" for d in dets) or "nothing"
            cv2.putText(img, f"{n / (time.time() - t0):.1f} fps {ms:.0f} ms | {summary}", (8, img.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.imshow("detect_live (q = quit)", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):  # save the raw frame (for offline debugging / training data)
                out = REPO / "snapshots" / f"frame_{time.strftime('%H%M%S')}.jpg"
                out.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(out), rgb[:, :, ::-1])
                print("saved", out, flush=True)
    print("log:", log.name)


if __name__ == "__main__":
    main()
