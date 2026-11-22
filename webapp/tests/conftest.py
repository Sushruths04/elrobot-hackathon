import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEBAPP = os.path.dirname(_HERE)
_REPO = os.path.dirname(_WEBAPP)
for p in (os.path.join(_REPO, "src"), _WEBAPP):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient  # noqa: E402

import app as webapp_app  # noqa: E402
from normakit.mcp_server import build  # noqa: E402
from safety import HumanGate  # noqa: E402


@pytest.fixture(scope="session")
def sim_arm():
    arm = build("sim", "")
    yield arm
    arm.guard.release()
    arm.backend.stop_backend()


@pytest.fixture()
def app(sim_arm):
    return webapp_app.create_app(sim_arm, "sim")


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def station_gated_app(sim_arm):
    # sim backend under a station-mode gate: exercises the gate with no hardware
    return webapp_app.create_app(sim_arm, "station", HumanGate("station"))
