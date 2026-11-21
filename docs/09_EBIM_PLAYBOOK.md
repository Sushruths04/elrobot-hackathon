# Working style carried over from the owner's EBIM real-robot project (generic lessons only)

These are the rules the owner enforced on a previous Claude-Code-driven robot project. They apply here unchanged.

1. **Build the loop, don't be the loop.** The whole task is ONE process with the decision loop inside it (`scripts/autorun.py`). Do not chain scripts by hand and decide between runs; separate scripts also pay start-up cost each time. Claude (API) is called for *perception only*; decisions and retries live in code.
2. **Look at the camera image first.** Before theorising about a failure, capture and view the RGB frame (`scripts/snap_cams.py`). Two past failures were settled in 30 s this way.
3. **Calibration is not hardcoding.** Fixed properties of an object (grip depth, jaw opening) are calibration: keep them in a JSON, taught by a teleop demonstration. Anything about *where* something is must be measured live from the image every run, never a constant.
4. **Never judge a grasp from the jaw gap or current alone.** Confirm a hold visually (camera + Claude). Current is telemetry only. A past run called a successful pick a failure because it trusted the gripper readout.
5. **Close fully, wait until the jaw stops, wait 2 s more, only then lift.** Implemented in `Arm.close_gripper` (`POST_CLOSE_WAIT_S`).
6. **Ask the owner for a teleop demonstration** when a pose or grip geometry is in question. Here we have a leader arm: a 5-second demo beats hours of measuring. Record the pose from live joints.
7. **Report before editing working code**: evidence first (what changed since the last good run), then wait for approval. Never delete anything that cannot be regenerated.
8. **Be terse.** Tools over prose, conclusions first.
9. **The organisers' spec is the source of truth** (here: the NormaCore track brief). Check it before optimising any gate.

## Mapping to this project (RGB camera + ElRobot)
`look (RGB) -> Claude locates object pixel -> table homography (calibrate_camera.py --table) -> IK reach -> close, settle, lift -> Claude verifies held -> locate target -> place -> Claude verifies placed -> log`. No depth camera: object height comes from the table plane and a calibrated grip depth.
Status (2026-11-21 15:50): loop runs end-to-end in the simulator (colour perception); the sim grasp itself still fails (`timeout reaching goal`, contact at table height), so nothing is verified on the real arm. `--vision claude` is written but never run (needs `ANTHROPIC_API_KEY` in the shell).
