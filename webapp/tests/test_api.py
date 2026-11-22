from fastapi.testclient import TestClient


def test_root_ok(client):
    r = client.get("/")
    assert r.status_code == 200


def test_status(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    d = r.json()
    assert d["backend"] == "sim"
    assert "anthropic_key" in d
    assert d["human_confirmed"] is True  # sim needs no confirmation


def test_estop_latches_until_release(client, sim_arm):
    try:
        r = client.post("/api/estop")
        assert r.status_code == 200 and r.json()["estopped"] is True
        res = sim_arm.guard.move({1: 0.5})
        assert res["ok"] is False  # Guard latch: the web layer cannot bypass it
        r = client.post("/api/release")
        assert r.status_code == 200 and r.json()["ok"] is True
        res = sim_arm.guard.move({1: 0.5})
        assert res["ok"] is True
    finally:
        sim_arm.guard.release()


def test_confirm_human_endpoint(station_gated_app):
    c = TestClient(station_gated_app)
    d = c.get("/api/status").json()
    assert d["backend"] == "station"
    assert d["human_confirmed"] is False
    r = c.post("/api/confirm-human")
    assert r.status_code == 200 and r.json()["ok"] is True
    d = c.get("/api/status").json()
    assert d["human_confirmed"] is True
    assert d["human_window_s"] > 0


def test_ws_state_and_frame(client):
    got_state = got_frame = False
    with client.websocket_connect("/ws") as ws:
        for _ in range(40):  # timeout guard: bounded reads, server pushes every ~0.5 s
            msg = ws.receive_json()
            if msg.get("type") == "state":
                joints = msg["joints"]["joints"]
                if all(str(i) in joints for i in range(1, 9)):
                    got_state = True
            if msg.get("type") == "frame" and msg.get("jpeg_b64"):
                got_frame = True
            if got_state and got_frame:
                break
    assert got_state, "no state telemetry received"
    assert got_frame, "no camera frame received"
