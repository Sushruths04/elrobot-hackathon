"""Hand-guided demo with the gripper OPEN: the human moves the follower to the tissue; the gripper is closed on the human's word (via armd /gripper)."""
import sys
sys.argv += ["--open-gripper", "--max", "900", "--countdown", "5"]
exec(compile(open(__file__.replace("tempo_pinch_demo.py", "tempo_hand_demo.py")).read(), "tempo_hand_demo.py", "exec"))
