# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from copy import deepcopy

import mujoco
import numpy as np
from mujoco.rollout import Rollout

import judo_cpp
from judo import MODEL_PATH

XML_PATH = str(MODEL_PATH / "xml/spot_locomotion.xml")

num_threads = 64
batch_size = num_threads
time_steps = 100


def mujoco_rollout(models: list, datas: list, full_states: np.ndarray, controls: np.ndarray) -> None:
    """Test MuJoCo rollout performance."""
    """Run Mujoco rollout."""
    rollout_obj = Rollout(nthread=num_threads)
    for _i in range(10):
        start_time = time.time()
        states_raw, sensors = rollout_obj.rollout(models, datas, full_states, controls)
        end_time = time.time()
        print("Python Rollout time:", end_time - start_time)
    states = np.array(states_raw)[..., 1:]
    print("States shape:", states.shape)
    return states


def cpp_rollout(models: list, datas: list, x0_batched: np.ndarray, controls: np.ndarray) -> None:
    """Test C++ rollout performance."""
    """Run C++ rollout."""
    for _ in range(10):
        start_time = time.time()
        states_cpp, sensors_cpp, inputs_cpp = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)
        end_time = time.time()
        print("C++ Rollout time:", end_time - start_time)
    return states_cpp


def create_test_data() -> tuple:
    """Create test data for benchmarking."""
    """Create test data for Mujoco rollout."""
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    models = [deepcopy(model) for _ in range(batch_size)]
    datas = [mujoco.MjData(m) for m in models]

    x0 = np.zeros(model.nq + model.nv)
    x0[: model.nq] = model.qpos0
    x0_batched = np.tile(x0, (batch_size, 1))
    full_states = np.concatenate([time.time() * np.ones((batch_size, 1)), x0_batched], axis=-1)
    controls = np.random.randn(batch_size, time_steps, model.nu) * 0.1

    return model, models, datas, x0_batched, full_states, controls


print("=== Testing C++ FIRST (cold performance) ===")
model, models, datas, x0_batched, full_states, controls = create_test_data()
cpp_states = cpp_rollout(models, datas, x0_batched, controls)
python_states = mujoco_rollout(models, datas, full_states, controls)

print("\n=== Testing Python FIRST (like benchmark_spot.py) ===")
model, models, datas, x0_batched, full_states, controls = create_test_data()
python_states = mujoco_rollout(models, datas, full_states, controls)
cpp_states = cpp_rollout(models, datas, x0_batched, controls)

# Compare results
if python_states is not None and cpp_states is not None:
    diff = np.linalg.norm(python_states - cpp_states)
    print(f"\nDifference between Python and C++ states: {diff}")
