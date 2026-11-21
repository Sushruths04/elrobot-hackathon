"""Start (or stop) Station's built-in AUTOMATIC calibration on one bus - the same command as the web button. THE ARM MOVES INTO ITS STOPS (~3 min).
A person must support the arm, the area must be clear and the servo power plug within reach.
    .venv/bin/python scripts/auto_calibrate.py --bus leader --dry-run     # build the command, send nothing
    .venv/bin/python scripts/auto_calibrate.py --bus leader --yes         # start
    .venv/bin/python scripts/auto_calibrate.py --bus leader --stop        # stop it (torque goes off: arm may drop)"""
import argparse, asyncio, json, logging, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import station_backend as sb
from station_py import new_station_client, send_commands
from target.gen_python.protobuf.drivers.st3215 import st3215
from target.gen_python.protobuf.station import commands as station_commands, drivers
ap = argparse.ArgumentParser(); ap.add_argument("--bus", required=True, help="leader | follower | serial"); ap.add_argument("--yes", action="store_true")
ap.add_argument("--stop", action="store_true"); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
B = json.loads((ROOT / "run/buses.json").read_text()); serial = B.get(a.bus, a.bus)
cmd = st3215.Command(target_bus_serial=serial, **({"stop_auto_calibrate": st3215.StopAutoCalibrateCommand(stop=True)} if a.stop else {"auto_calibrate": st3215.AutoCalibrateCommand(calibrate=True)}))
pack = station_commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())
print(f"{'STOP' if a.stop else 'START'} auto-calibration on bus {serial} ({a.bus}); command bytes: {len(cmd.encode())}")
if a.dry_run or not (a.yes or a.stop): print("nothing sent" + ("" if a.dry_run else " (add --yes to start)")); sys.stdout.flush(); os._exit(0)
async def main():
    c = await new_station_client("localhost:8888", logging.getLogger("autocal")); await send_commands(c, [pack]); print("command sent", flush=True); await asyncio.sleep(1.0); os._exit(0)
asyncio.run(main())
