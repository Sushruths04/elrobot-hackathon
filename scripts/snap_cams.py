"""Read-only: save one frame from every Station camera to run/cam_<name>.jpg and print sizes."""
import io, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from PIL import Image
from normakit.station_backend import StationBackend
b = StationBackend("localhost:8888"); b.start(); time.sleep(2)
print("cameras:", b.cameras())
for c in b.cameras():
    if c.startswith("cam"):
        jpg = None
        for _ in range(40):
            try:
                jpg = b.frame_jpeg(c); break
            except Exception:
                time.sleep(0.25)
        if jpg is None:
            print(c, "ERROR no frame after 10 s"); continue
        time.sleep(1.5); jpg = b.frame_jpeg(c)       # let auto-exposure settle
        im = Image.open(io.BytesIO(jpg)); import numpy as np
        p = ROOT / f"run/{c}_frame.jpg"; p.write_bytes(jpg)
        print(f"{c}: {len(jpg)} bytes, {im.size}, mean brightness {np.asarray(im.convert('L')).mean():.0f}/255 -> {p.relative_to(ROOT)}")
sys.stdout.flush(); os._exit(0)
