"""First thing on site: verify the live Station link. Prints what it sees; moves NOTHING unless --nudge.

    ./bin/station --web --tcp --config run/station.yaml     (terminal 1; Linux needs: sudo usermod -a -G dialout $USER)
    .venv/bin/python scripts/smoke_station.py [--host localhost:8888] [--nudge]
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normakit.station_backend import StationBackend

ap = argparse.ArgumentParser(); ap.add_argument("--host", default="localhost:8888"); ap.add_argument("--nudge", action="store_true")
a = ap.parse_args()
b = StationBackend(a.host); b.start()
time.sleep(2)
st = b.state()
print("bus:", b._bus)
for j in st.joints.values():
    print(f" motor {j.id}: norm={j.norm:.3f} goal={j.goal_norm:.3f} raw={j.raw} range=[{j.lo},{j.hi}] I={j.current_ma}mA T={j.temp_c}C")
print("cameras:", b.cameras())
for c in b.cameras():
    if c.startswith("cam"):
        try:
            jpg = b.frame_jpeg(c); Path(f"run/smoke_{c}.jpg").write_bytes(jpg)
            from PIL import Image; import io
            print(f" {c}: {len(jpg)} bytes, size {Image.open(io.BytesIO(jpg)).size} -> run/smoke_{c}.jpg")
        except Exception as e:
            print(f" {c}: {e}")
if a.nudge:
    j = st.joints[2]; tgt = min(0.9, max(0.1, j.norm + 0.05))
    print("nudging joint 2 ->", round(tgt, 3)); b.send_goals({2: tgt}, 0.2); time.sleep(1.5)
    print("now:", round(b.state().joints[2].norm, 3), "(sending it back)"); b.send_goals({2: j.norm}, 0.2)
