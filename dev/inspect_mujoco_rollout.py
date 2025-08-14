# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import inspect

import mujoco.rollout

print("Rollout location:", inspect.getfile(mujoco.rollout.Rollout))
print("Methods:", [m for m in dir(mujoco.rollout.Rollout) if not m.startswith("_")])

# Check if we can see the source
try:
    source = inspect.getsource(mujoco.rollout.Rollout)
    print("Source available:", len(source), "characters")
except Exception:  # noqa: S110
    print("Source not available (compiled)")

# Check the rollout method specifically
try:
    rollout_source = inspect.getsource(mujoco.rollout.Rollout.rollout)
    print("Rollout method source:")
    print(rollout_source[:500], "..." if len(rollout_source) > 500 else "")
except Exception:  # noqa: S110
    print("Rollout method source not available (compiled)")
