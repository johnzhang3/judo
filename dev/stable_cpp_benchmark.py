# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from copy import deepcopy

import mujoco
import numpy as np

import judo_cpp
from judo import MODEL_PATH

XML_PATH = str(MODEL_PATH / "xml/spot_locomotion.xml")


def warm_up_system() -> None:
    """Warm up CPU and system to reach stable performance state."""
    print("Warming up system...")

    # Create minimal workload to boost CPU frequency
    for _ in range(5):
        # Simple CPU-intensive operation
        _ = np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)
        time.sleep(0.1)

    # Wait for frequency scaling to settle
    time.sleep(2)
    print("System warmed up.")


def run_stable_benchmark() -> None:
    """Run benchmark with warm-up and statistical robustness."""
    # Setup
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

    # 1. System warm-up
    warm_up_system()

    # 2. Benchmark warm-up (don't measure these)
    print("Warming up benchmark...")
    for _ in range(10):
        _ = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)
        time.sleep(0.05)  # Brief pause

    # 3. Actual measurements
    print("Running measurements...")
    times = []
    for i in range(15):
        start_time = time.time()
        states_cpp, sensors_cpp, inputs_cpp = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"Run {i + 1:2d}: {elapsed:.4f}s")

        # Small pause to prevent thermal throttling
        time.sleep(0.1)

    # 4. Statistical analysis
    times = np.array(times)

    # Remove outliers (beyond 2 std devs)
    mean_time = times.mean()
    std_time = times.std()
    mask = np.abs(times - mean_time) <= 2 * std_time
    clean_times = times[mask]

    print("\n=== Results ===")
    print("All measurements:")
    print(f"  Mean: {times.mean():.4f}s ± {times.std():.4f}s")
    print(f"  Range: {times.min():.4f}s - {times.max():.4f}s")
    print(f"  CV: {times.std() / times.mean() * 100:.1f}%")

    if len(clean_times) < len(times):
        print(f"\nAfter removing {len(times) - len(clean_times)} outliers:")
        print(f"  Mean: {clean_times.mean():.4f}s ± {clean_times.std():.4f}s")
        print(f"  Range: {clean_times.min():.4f}s - {clean_times.max():.4f}s")
        print(f"  CV: {clean_times.std() / clean_times.mean() * 100:.1f}%")

    # Recommend using the stable measurement
    stable_time = clean_times.mean() if len(clean_times) >= 10 else times[5:].mean()
    print(f"\n=== Recommended stable performance: {stable_time:.4f}s ===")

    return stable_time


if __name__ == "__main__":
    print("Running stable C++ benchmark...")
    stable_time = run_stable_benchmark()

    print(f"\nFor consistent comparisons, use: {stable_time * 1000:.1f}ms per rollout")
