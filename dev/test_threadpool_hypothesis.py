# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from copy import deepcopy

import mujoco
import numpy as np
from mujoco.rollout import Rollout

import judo_cpp
from judo import MODEL_PATH

XML_PATH = str(MODEL_PATH / "xml/spot_locomotion.xml")


def test_threadpool_hypothesis():
    """Test if threadpool management affects performance stability"""
    num_threads = 64
    batch_size = num_threads
    time_steps = 100

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    models = [deepcopy(model) for _ in range(batch_size)]
    datas = [mujoco.MjData(m) for m in models]

    x0 = np.zeros(model.nq + model.nv)
    x0[: model.nq] = model.qpos0
    x0_batched = np.tile(x0, (batch_size, 1))
    full_states = np.concatenate([time.time() * np.ones((batch_size, 1)), x0_batched], axis=-1)
    controls = np.random.randn(batch_size, time_steps, model.nu) * 0.1

    print("=== Testing Python Rollout (Persistent ThreadPool) ===")

    # Create persistent threadpool
    rollout_obj = Rollout(nthread=num_threads)

    python_times = []
    for i in range(20):
        start_time = time.time()
        states_raw, sensors = rollout_obj.rollout(models, datas, full_states, controls)
        end_time = time.time()

        elapsed = end_time - start_time
        python_times.append(elapsed)
        print(f"Python Persistent {i + 1:2d}: {elapsed:.4f}s")

    print("\n=== Testing Python Rollout (Fresh ThreadPool Each Time) ===")

    python_fresh_times = []
    for i in range(20):
        # Create new rollout object each time
        fresh_rollout = Rollout(nthread=num_threads)

        start_time = time.time()
        states_raw, sensors = fresh_rollout.rollout(models, datas, full_states, controls)
        end_time = time.time()

        elapsed = end_time - start_time
        python_fresh_times.append(elapsed)
        print(f"Python Fresh {i + 1:2d}: {elapsed:.4f}s")

        fresh_rollout.close()  # Clean up

    print("\n=== Testing C++ Rollout (Manual Threading) ===")

    cpp_times = []
    for i in range(20):
        start_time = time.time()
        states_cpp, sensors_cpp, inputs_cpp = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)
        end_time = time.time()

        elapsed = end_time - start_time
        cpp_times.append(elapsed)
        print(f"C++ Manual {i + 1:2d}: {elapsed:.4f}s")

    # Analysis
    python_times = np.array(python_times)
    python_fresh_times = np.array(python_fresh_times)
    cpp_times = np.array(cpp_times)

    print("\n=== Stability Analysis ===")
    print("Python Persistent ThreadPool:")
    print(f"  Mean: {python_times.mean():.4f}s ± {python_times.std():.4f}s")
    print(f"  CV: {python_times.std() / python_times.mean() * 100:.1f}%")

    print("\nPython Fresh ThreadPool:")
    print(f"  Mean: {python_fresh_times.mean():.4f}s ± {python_fresh_times.std():.4f}s")
    print(f"  CV: {python_fresh_times.std() / python_fresh_times.mean() * 100:.1f}%")

    print("\nC++ Manual Threading:")
    print(f"  Mean: {cpp_times.mean():.4f}s ± {cpp_times.std():.4f}s")
    print(f"  CV: {cpp_times.std() / cpp_times.mean() * 100:.1f}%")

    print("\n=== Threading Hypothesis ===")
    print("If threadpool management is the cause:")
    print(f"  Python persistent should be most stable: {python_times.std() / python_times.mean() * 100:.1f}% CV")
    print(f"  Python fresh should be less stable: {python_fresh_times.std() / python_fresh_times.mean() * 100:.1f}% CV")
    print(f"  C++ manual should be least stable: {cpp_times.std() / cpp_times.mean() * 100:.1f}% CV")


if __name__ == "__main__":
    # System warm-up first
    print("Warming up system...")
    for _ in range(5):
        _ = np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)
        time.sleep(0.1)
    time.sleep(2)

    test_threadpool_hypothesis()
