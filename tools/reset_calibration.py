#!/usr/bin/env python3
"""
Reset calibration on one bus - same commands as the web UI "Reset" button:
stop mirroring + ResetCalibration (unlocks EEPROM and resets every servo -> offsets cleared,
torque OFF, arm goes LIMP). Support the arm before running this!

  python3 tools/reset_calibration.py --bus 5B79016336
"""
import argparse, asyncio, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estop  # noqa: E402  (sets up repo paths)
from estop import new_station_client, send_commands, st3215, commands, drivers  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", required=True)
    ap.add_argument("--host", default="localhost")
    a = ap.parse_args()
    client = await new_station_client(a.host, logging.getLogger("reset"))
    rc = st3215.Command(target_bus_serial=a.bus, reset_calibration=st3215.ResetCalibrationCommand(reset=True))
    await send_commands(client, [estop.stop_mirror_cmd(a.bus)])
    await send_commands(client, [commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=rc.encode())])
    print(f"reset-calibration sent to bus {a.bus}; servos reboot for a few seconds")


if __name__ == "__main__":
    asyncio.run(main())
