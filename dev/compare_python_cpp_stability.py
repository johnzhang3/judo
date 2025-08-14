# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from copy import deepcopy

import mujoco
import numpy as np
from mujoco.rollout import Rollout

import judo_cpp
from judo import MODEL_PATH

XML_PATH = str(MODEL_PATH / "xml/spot_locomotion.xml")


def warm_up_system() -> None:
    """Warm up CPU and system to reach stable performance state."""
    print("Warming up system...")
    for _ in range(5):
        _ = np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)
        time.sleep(0.1)
    time.sleep(2)
    print("System warmed up.")


def benchmark_python_rollout() -> np.ndarray:
    """Benchmark Python (mujoco.rollout.Rollout) version."""
    print("\n=== Python Rollout Benchmark ===")

    num_threads = 64
    batch_size = num_threads
    time_steps = 100

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    models = [deepcopy(model) for _ in range(batch_size)]
    datas = [mujoco.MjData(m) for m in models]

    x0 = np.zeros(model.nq + model.nv)
    x0[: model.nq] = model.qpos0
    full_states = np.concatenate([time.time() * np.ones((batch_size, 1)), np.tile(x0, (batch_size, 1))], axis=-1)
    controls = np.random.randn(batch_size, time_steps, model.nu) * 0.1

    rollout_obj = Rollout(nthread=num_threads)

    # Warm-up runs
    print("Warming up Python rollout...")
    for _ in range(10):
        _ = rollout_obj.rollout(models, datas, full_states, controls)
        time.sleep(0.05)

    # Actual measurements
    print("Running Python measurements...")
    times = []
    for i in range(20):
        start_time = time.time()
        states_raw, sensors = rollout_obj.rollout(models, datas, full_states, controls)
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"Python Run {i + 1:2d}: {elapsed:.4f}s")
        time.sleep(0.05)

    return np.array(times)


def benchmark_cpp_rollout() -> np.ndarray:
    """Benchmark C++ (judo_cpp.pure_cpp_rollout) version."""
    print("\n=== C++ Rollout Benchmark ===")

    num_threads = 64
    batch_size = num_threads
    time_steps = 100

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    models = [deepcopy(model) for _ in range(batch_size)]
    datas = [mujoco.MjData(m) for m in models]

    x0 = np.zeros(model.nq + model.nv)
    x0[: model.nq] = model.qpos0
    x0_batched = np.tile(x0, (batch_size, 1))
    controls = np.random.randn(batch_size, time_steps, model.nu) * 0.1

    # Warm-up runs
    print("Warming up C++ rollout...")
    for _ in range(10):
        _ = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)
        time.sleep(0.05)

    # Actual measurements
    print("Running C++ measurements...")
    times = []
    for i in range(20):
        start_time = time.time()
        states_cpp, sensors_cpp, inputs_cpp = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"C++ Run {i + 1:2d}: {elapsed:.4f}s")
        time.sleep(0.05)

    return np.array(times)


def analyze_timing_differences() -> None:
    """Analyze what might cause timing differences."""
    print("\n=== Analyzing Timing Overhead ===")

    # Test just the Python call overhead
    python_overhead_times = []
    for _i in range(100):
        start = time.time()
        # Just call time.time() with minimal work
        _ = np.array([1.0])
        end = time.time()
        python_overhead_times.append(end - start)

    python_overhead = np.array(python_overhead_times)
    print(f"Python call overhead: {python_overhead.mean() * 1e6:.1f} ± {python_overhead.std() * 1e6:.1f} microseconds")

    # Test C++ call overhead via judo_cpp
    try:
        # We don't have a minimal C++ function, but we can time the binding overhead
        cpp_overhead_times = []
        dummy_array = np.array([1.0])

        for _i in range(100):
            start = time.time()
            # This will fail, but we can measure the binding call overhead
            try:
                _ = len(dummy_array)  # Minimal numpy operation
            except Exception:  # noqa: S110
                pass
            end = time.time()
            cpp_overhead_times.append(end - start)

        cpp_overhead = np.array(cpp_overhead_times)
        print(
            f"Minimal operation overhead: {cpp_overhead.mean() * 1e6:.1f} ± {cpp_overhead.std() * 1e6:.1f} microseconds"
        )
    except Exception:  # noqa: S110
        print("Could not measure C++ binding overhead")


def compare_implementations() -> None:
    """Compare Python vs C++ rollout implementations."""
    warm_up_system()

    # Run benchmarks
    python_times = benchmark_python_rollout()
    cpp_times = benchmark_cpp_rollout()

    # Analyze timing overhead
    analyze_timing_differences()

    # Statistical comparison
    print("\n=== Statistical Comparison ===")

    print("Python Rollout:")
    print(f"  Mean: {python_times.mean():.4f}s ± {python_times.std():.4f}s")
    print(f"  Range: {python_times.min():.4f}s - {python_times.max():.4f}s")
    print(f"  CV: {python_times.std() / python_times.mean() * 100:.1f}%")

    print("\nC++ Rollout:")
    print(f"  Mean: {cpp_times.mean():.4f}s ± {cpp_times.std():.4f}s")
    print(f"  Range: {cpp_times.min():.4f}s - {cpp_times.max():.4f}s")
    print(f"  CV: {cpp_times.std() / cpp_times.mean() * 100:.1f}%")

    print("\nVariability Comparison:")
    print(f"  Python CV: {python_times.std() / python_times.mean() * 100:.1f}%")
    print(f"  C++ CV: {cpp_times.std() / cpp_times.mean() * 100:.1f}%")
    print(f"  Python is {cpp_times.std() / python_times.std():.1f}x more stable (lower is better)")

    # Performance comparison
    speedup = python_times.mean() / cpp_times.mean()
    print("\nPerformance:")
    print(f"  C++ is {speedup:.2f}x faster than Python")
    print(f"  Difference: {(python_times.mean() - cpp_times.mean()) * 1000:.1f}ms")


if __name__ == "__main__":
    compare_implementations()
