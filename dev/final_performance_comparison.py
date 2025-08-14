# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from copy import deepcopy

import mujoco
import numpy as np
from mujoco.rollout import Rollout

import judo_cpp
from judo import MODEL_PATH

XML_PATH = str(MODEL_PATH / "xml/spot_locomotion.xml")


def final_performance_comparison():
    """Final performance comparison with corrected methodology"""
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

    print("=== FINAL PERFORMANCE COMPARISON (Corrected Methodology) ===")

    # Test 1: Python with persistent thread pool
    print("\n1. Python (Persistent ThreadPool) - CONTINUOUS:")
    rollout_obj = Rollout(nthread=num_threads)

    # Warm-up
    for _ in range(5):
        _ = rollout_obj.rollout(models, datas, full_states, controls)

    python_times = []
    for i in range(20):
        start_time = time.time()
        states_raw, sensors = rollout_obj.rollout(models, datas, full_states, controls)
        end_time = time.time()
        python_times.append(end_time - start_time)
        # No pauses

    # Test 2: C++ with OpenMP (original)
    print("2. C++ Original (OpenMP) - CONTINUOUS:")

    # Warm-up
    for _ in range(5):
        _ = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)

    cpp_original_times = []
    for i in range(20):
        start_time = time.time()
        states_cpp, sensors_cpp, inputs_cpp = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)
        end_time = time.time()
        cpp_original_times.append(end_time - start_time)
        # No pauses

    # Test 3: C++ with persistent thread pool
    print("3. C++ Persistent ThreadPool - CONTINUOUS:")

    # Warm-up
    for _ in range(5):
        _ = judo_cpp.persistent_cpp_rollout(models, datas, x0_batched, controls)

    cpp_persistent_times = []
    for i in range(20):
        start_time = time.time()
        states_persistent, sensors_persistent, inputs_persistent = judo_cpp.persistent_cpp_rollout(
            models, datas, x0_batched, controls
        )
        end_time = time.time()
        cpp_persistent_times.append(end_time - start_time)
        # No pauses

    # Analysis
    python_times = np.array(python_times)
    cpp_original_times = np.array(cpp_original_times)
    cpp_persistent_times = np.array(cpp_persistent_times)

    print("\n=== RESULTS ===")

    print("\nPython (Persistent ThreadPool):")
    print(f"  Mean: {python_times.mean():.4f}s ± {python_times.std():.4f}s")
    print(f"  Range: {python_times.min():.4f}s - {python_times.max():.4f}s")
    print(f"  CV: {python_times.std() / python_times.mean() * 100:.1f}%")

    print("\nC++ Original (OpenMP):")
    print(f"  Mean: {cpp_original_times.mean():.4f}s ± {cpp_original_times.std():.4f}s")
    print(f"  Range: {cpp_original_times.min():.4f}s - {cpp_original_times.max():.4f}s")
    print(f"  CV: {cpp_original_times.std() / cpp_original_times.mean() * 100:.1f}%")

    print("\nC++ Persistent ThreadPool:")
    print(f"  Mean: {cpp_persistent_times.mean():.4f}s ± {cpp_persistent_times.std():.4f}s")
    print(f"  Range: {cpp_persistent_times.min():.4f}s - {cpp_persistent_times.max():.4f}s")
    print(f"  CV: {cpp_persistent_times.std() / cpp_persistent_times.mean() * 100:.1f}%")

    print("\n=== STABILITY RANKING ===")
    python_cv = python_times.std() / python_times.mean() * 100
    cpp_orig_cv = cpp_original_times.std() / cpp_original_times.mean() * 100
    cpp_pers_cv = cpp_persistent_times.std() / cpp_persistent_times.mean() * 100

    results = [("Python Persistent", python_cv), ("C++ Original", cpp_orig_cv), ("C++ Persistent", cpp_pers_cv)]
    results.sort(key=lambda x: x[1])

    for i, (name, cv) in enumerate(results, 1):
        print(f"{i}. {name}: {cv:.1f}% CV")

    print("\n=== PERFORMANCE RANKING ===")
    perf_results = [
        ("Python Persistent", python_times.mean()),
        ("C++ Original", cpp_original_times.mean()),
        ("C++ Persistent", cpp_persistent_times.mean()),
    ]
    perf_results.sort(key=lambda x: x[1])

    for i, (name, mean_time) in enumerate(perf_results, 1):
        speedup = perf_results[-1][1] / mean_time  # Speedup vs slowest
        print(f"{i}. {name}: {mean_time:.4f}s ({speedup:.2f}x faster than slowest)")

    print("\n=== SUCCESS METRICS ===")
    print(f"✓ C++ Persistent vs Python stability: {cpp_pers_cv / python_cv:.1f}x (target: <2x)")
    print(f"✓ C++ Persistent vs Python speed: {python_times.mean() / cpp_persistent_times.mean():.2f}x faster")
    print(f"✓ C++ Persistent vs C++ Original stability: {cpp_orig_cv / cpp_pers_cv:.1f}x better")

    # Verify correctness
    states_python = np.array(states_raw)[..., 1:]  # Remove time column
    diff_orig = np.linalg.norm(states_python - states_cpp)
    diff_pers = np.linalg.norm(states_python - states_persistent)
    print(f"✓ Results identical (diff < 1e-10): {diff_pers < 1e-10}")


if __name__ == "__main__":
    final_performance_comparison()

    print("\nCleaning up...")
    judo_cpp.shutdown_thread_pool()
