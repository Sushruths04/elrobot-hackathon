#!/usr/bin/env python3
"""Latency benchmark: python server/bench.py --url http://A100_HOST:8000 --n 50"""
import argparse
import base64
import io
import os
import time

import requests


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--no-image", action="store_true")
    a = ap.parse_args()
    headers = {}
    if os.environ.get("POLICY_API_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["POLICY_API_TOKEN"]
    body = {"state": [0.5] * 8, "task": "bench"}
    if not a.no_image:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), (90, 120, 150)).save(buf, "JPEG", quality=85)
        body["image_jpeg_b64"] = base64.b64encode(buf.getvalue()).decode()
    s = requests.Session()
    print("health:", s.get(a.url + "/health", headers=headers, timeout=10).json())
    s.post(a.url + "/act", json=body, headers=headers, timeout=60)  # warmup
    e2e, srv = [], []
    for _ in range(a.n):
        t0 = time.perf_counter()
        r = s.post(a.url + "/act", json=body, headers=headers, timeout=30)
        r.raise_for_status()
        e2e.append((time.perf_counter() - t0) * 1000)
        srv.append(r.json()["server_ms"])
    print(f"n={a.n} chunk={len(r.json()['action_chunk'])}x{len(r.json()['action_chunk'][0])}")
    print(f"end-to-end ms: p50={pct(e2e,.5):.1f} p95={pct(e2e,.95):.1f} max={max(e2e):.1f}")
    print(f"server ms:     p50={pct(srv,.5):.1f} p95={pct(srv,.95):.1f} max={max(srv):.1f}")


if __name__ == "__main__":
    main()
