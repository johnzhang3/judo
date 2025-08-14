# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from copy import deepcopy

import mujoco
import numpy as np
from mujoco.rollout import Rollout

import judo_cpp
from judo import MODEL_PATH

XML_PATH = str(MODEL_PATH / "xml/spot_locomotion.xml")


def test_persistent_thread_pool() -> None:
    """Test the new persistent thread pool implementation vs original C++ and Python."""
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

    # System warm-up
    print("Warming up system...")
    for _ in range(5):
        _ = np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)
        time.sleep(0.1)
    time.sleep(2)

    print("=== Testing Python Rollout (Persistent ThreadPool) ===")
    rollout_obj = Rollout(nthread=num_threads)

    # Warm-up
    for _ in range(5):
        _ = rollout_obj.rollout(models, datas, full_states, controls)

    python_times = []
    for i in range(15):
        start_time = time.time()
        states_raw, sensors = rollout_obj.rollout(models, datas, full_states, controls)
        end_time = time.time()

        elapsed = end_time - start_time
        python_times.append(elapsed)
        print(f"Python {i + 1:2d}: {elapsed:.4f}s")

    print("\n=== Testing C++ Original (OpenMP) ===")

    # Warm-up
    for _ in range(5):
        _ = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)

    cpp_original_times = []
    for i in range(15):
        start_time = time.time()
        states_cpp, sensors_cpp, inputs_cpp = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)
        end_time = time.time()

        elapsed = end_time - start_time
        cpp_original_times.append(elapsed)
        print(f"C++ Original {i + 1:2d}: {elapsed:.4f}s")

    print("\n=== Testing C++ Persistent ThreadPool ===")

    # Warm-up
    for _ in range(5):
        _ = judo_cpp.persistent_cpp_rollout(models, datas, x0_batched, controls)

    cpp_persistent_times = []
    for i in range(15):
        start_time = time.time()
        states_persistent, sensors_persistent, inputs_persistent = judo_cpp.persistent_cpp_rollout(
            models, datas, x0_batched, controls
        )
        end_time = time.time()

        elapsed = end_time - start_time
        cpp_persistent_times.append(elapsed)
        print(f"C++ Persistent {i + 1:2d}: {elapsed:.4f}s")

    # Analysis
    python_times = np.array(python_times)
    cpp_original_times = np.array(cpp_original_times)
    cpp_persistent_times = np.array(cpp_persistent_times)

    print("\n=== Performance & Stability Comparison ===")

    print("Python (Persistent ThreadPool):")
    print(f"  Mean: {python_times.mean():.4f}s ± {python_times.std():.4f}s")
    print(f"  CV: {python_times.std() / python_times.mean() * 100:.1f}%")

    print("\nC++ Original (OpenMP):")
    print(f"  Mean: {cpp_original_times.mean():.4f}s ± {cpp_original_times.std():.4f}s")
    print(f"  CV: {cpp_original_times.std() / cpp_original_times.mean() * 100:.1f}%")

    print("\nC++ Persistent ThreadPool:")
    print(f"  Mean: {cpp_persistent_times.mean():.4f}s ± {cpp_persistent_times.std():.4f}s")
    print(f"  CV: {cpp_persistent_times.std() / cpp_persistent_times.mean() * 100:.1f}%")

    print("\n=== Stability Improvement ===")
    original_cv = cpp_original_times.std() / cpp_original_times.mean() * 100
    persistent_cv = cpp_persistent_times.std() / cpp_persistent_times.mean() * 100
    python_cv = python_times.std() / python_times.mean() * 100

    print(f"C++ Original CV: {original_cv:.1f}%")
    print(f"C++ Persistent CV: {persistent_cv:.1f}%")
    print(f"Python CV: {python_cv:.1f}%")
    print(f"Stability improvement: {original_cv / persistent_cv:.1f}x")
    print(f"C++ Persistent vs Python stability: {persistent_cv / python_cv:.1f}x")

    print("\n=== Performance Comparison ===")
    cpp_orig_speedup = python_times.mean() / cpp_original_times.mean()
    cpp_pers_speedup = python_times.mean() / cpp_persistent_times.mean()
    print(f"C++ Original vs Python: {cpp_orig_speedup:.2f}x faster")
    print(f"C++ Persistent vs Python: {cpp_pers_speedup:.2f}x faster")
    print(f"C++ Persistent vs C++ Original: {cpp_original_times.mean() / cpp_persistent_times.mean():.2f}x")

    # Verify correctness
    print("\n=== Correctness Check ===")
    states_python = np.array(states_raw)[..., 1:]  # Remove time column
    diff_orig = np.linalg.norm(states_python - states_cpp)
    diff_pers = np.linalg.norm(states_python - states_persistent)
    print(f"Python vs C++ Original difference: {diff_orig:.2e}")
    print(f"Python vs C++ Persistent difference: {diff_pers:.2e}")
    print(f"Results are {'✓ identical' if diff_pers < 1e-10 else '✗ different'}")


if __name__ == "__main__":
    test_persistent_thread_pool()

    # Clean up
    print("\nCleaning up thread pool...")
    judo_cpp.shutdown_thread_pool()
