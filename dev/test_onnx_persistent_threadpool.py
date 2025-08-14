# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from copy import deepcopy

import mujoco
import numpy as np

import judo_cpp
from judo import MODEL_PATH

XML_PATH = str(MODEL_PATH / "xml/spot_locomotion.xml")
ONNX_PATH = "identity_network_spot.onnx"  # Use the correct Spot identity network


def comprehensive_onnx_performance_test():
    """Test the persistent thread pool with ONNX interleaved rollouts"""
    num_threads = 64
    batch_size = num_threads
    time_steps = 100
    inference_frequency = 5  # Run ONNX inference every 5 steps

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    models = [deepcopy(model) for _ in range(batch_size)]
    datas = [mujoco.MjData(m) for m in models]

    x0 = np.zeros(model.nq + model.nv)
    x0[: model.nq] = model.qpos0
    x0_batched = np.tile(x0, (batch_size, 1))
    controls = np.random.randn(batch_size, time_steps, model.nu) * 0.1

    # System warm-up
    print("Warming up system...")
    for _ in range(5):
        _ = np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)
        time.sleep(0.1)
    time.sleep(2)

    print("=== COMPREHENSIVE ONNX INTERLEAVED ROLLOUT PERFORMANCE TEST ===")
    print(f"Configuration: {batch_size} rollouts, {time_steps} steps, ONNX every {inference_frequency} steps")

    # Test 1: Original OpenMP ONNX Interleaved Rollout
    print("\n1. Original ONNX Interleaved (OpenMP):")

    # Warm-up
    try:
        for _ in range(3):
            _ = judo_cpp.onnx_interleave_rollout(models, datas, x0_batched, controls, ONNX_PATH, inference_frequency)
    except Exception as e:
        print(f"   ONNX model not found or invalid: {e}")
        print("   Skipping ONNX tests - using pure rollouts instead")

        # Fall back to pure rollouts for comparison
        print("\nFalling back to pure rollout comparison...")
        test_pure_rollout_comparison()
        return

    original_onnx_times = []
    for i in range(10):
        start_time = time.time()
        states_orig, sensors_orig, inputs_orig, inferences_orig = judo_cpp.onnx_interleave_rollout(
            models, datas, x0_batched, controls, ONNX_PATH, inference_frequency
        )
        end_time = time.time()
        original_onnx_times.append(end_time - start_time)
        print(f"   Original ONNX {i + 1:2d}: {original_onnx_times[-1]:.4f}s")

    # Test 2: Persistent Thread Pool ONNX Interleaved Rollout
    print("\n2. Persistent ONNX Interleaved (Thread Pool):")

    # Warm-up
    for _ in range(3):
        _ = judo_cpp.persistent_onnx_interleave_rollout(
            models, datas, x0_batched, controls, ONNX_PATH, inference_frequency
        )

    persistent_onnx_times = []
    for i in range(10):
        start_time = time.time()
        states_pers, sensors_pers, inputs_pers, inferences_pers = judo_cpp.persistent_onnx_interleave_rollout(
            models, datas, x0_batched, controls, ONNX_PATH, inference_frequency
        )
        end_time = time.time()
        persistent_onnx_times.append(end_time - start_time)
        print(f"   Persistent ONNX {i + 1:2d}: {persistent_onnx_times[-1]:.4f}s")

    # Test 3: Pure rollouts for baseline
    print("\n3. Pure Rollouts (No ONNX) for comparison:")

    # Original pure
    for _ in range(3):
        _ = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)

    pure_original_times = []
    for i in range(10):
        start_time = time.time()
        _ = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)
        end_time = time.time()
        pure_original_times.append(end_time - start_time)

    # Persistent pure
    for _ in range(3):
        _ = judo_cpp.persistent_cpp_rollout(models, datas, x0_batched, controls)

    pure_persistent_times = []
    for i in range(10):
        start_time = time.time()
        _ = judo_cpp.persistent_cpp_rollout(models, datas, x0_batched, controls)
        end_time = time.time()
        pure_persistent_times.append(end_time - start_time)

    # Analysis
    original_onnx_times = np.array(original_onnx_times)
    persistent_onnx_times = np.array(persistent_onnx_times)
    pure_original_times = np.array(pure_original_times)
    pure_persistent_times = np.array(pure_persistent_times)

    print("\n=== PERFORMANCE ANALYSIS ===")

    print("\nOriginal ONNX Interleaved (OpenMP):")
    print(f"  Mean: {original_onnx_times.mean():.4f}s ± {original_onnx_times.std():.4f}s")
    print(f"  CV: {original_onnx_times.std() / original_onnx_times.mean() * 100:.1f}%")

    print("\nPersistent ONNX Interleaved (Thread Pool):")
    print(f"  Mean: {persistent_onnx_times.mean():.4f}s ± {persistent_onnx_times.std():.4f}s")
    print(f"  CV: {persistent_onnx_times.std() / persistent_onnx_times.mean() * 100:.1f}%")

    print("\nPure Original (OpenMP):")
    print(f"  Mean: {pure_original_times.mean():.4f}s ± {pure_original_times.std():.4f}s")
    print(f"  CV: {pure_original_times.std() / pure_original_times.mean() * 100:.1f}%")

    print("\nPure Persistent (Thread Pool):")
    print(f"  Mean: {pure_persistent_times.mean():.4f}s ± {pure_persistent_times.std():.4f}s")
    print(f"  CV: {pure_persistent_times.std() / pure_persistent_times.mean() * 100:.1f}%")

    print("\n=== ONNX OVERHEAD ANALYSIS ===")
    orig_overhead = original_onnx_times.mean() - pure_original_times.mean()
    pers_overhead = persistent_onnx_times.mean() - pure_persistent_times.mean()

    print(
        f"ONNX overhead (Original): {orig_overhead * 1000:.1f}ms ({orig_overhead / pure_original_times.mean() * 100:.1f}% increase)"
    )
    print(
        f"ONNX overhead (Persistent): {pers_overhead * 1000:.1f}ms ({pers_overhead / pure_persistent_times.mean() * 100:.1f}% increase)"
    )

    print("\n=== THREAD POOL BENEFITS ===")
    onnx_speedup = original_onnx_times.mean() / persistent_onnx_times.mean()
    onnx_stability = original_onnx_times.std() / persistent_onnx_times.std()

    print(f"ONNX interleaved speedup: {onnx_speedup:.2f}x")
    print(f"ONNX interleaved stability improvement: {onnx_stability:.1f}x")

    # Verify correctness
    print("\n=== CORRECTNESS VERIFICATION ===")
    states_diff = np.linalg.norm(states_orig - states_pers)
    inferences_diff = np.linalg.norm(inferences_orig - inferences_pers)
    print(f"States difference: {states_diff:.2e}")
    print(f"Inferences difference: {inferences_diff:.2e}")
    print(f"Results identical: {states_diff < 1e-10 and inferences_diff < 1e-10}")


def test_pure_rollout_comparison():
    """Fallback test if ONNX model is not available"""
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

    print("=== PURE ROLLOUT THREAD POOL COMPARISON ===")

    # Original
    original_times = []
    for i in range(15):
        start = time.time()
        _ = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)
        end = time.time()
        original_times.append(end - start)
        print(f"Original {i + 1:2d}: {original_times[-1]:.4f}s")

    # Persistent
    persistent_times = []
    for i in range(15):
        start = time.time()
        _ = judo_cpp.persistent_cpp_rollout(models, datas, x0_batched, controls)
        end = time.time()
        persistent_times.append(end - start)
        print(f"Persistent {i + 1:2d}: {persistent_times[-1]:.4f}s")

    original_times = np.array(original_times)
    persistent_times = np.array(persistent_times)

    print(
        f"\nOriginal: {original_times.mean():.4f}s ± {original_times.std():.4f}s (CV: {original_times.std() / original_times.mean() * 100:.1f}%)"
    )
    print(
        f"Persistent: {persistent_times.mean():.4f}s ± {persistent_times.std():.4f}s (CV: {persistent_times.std() / persistent_times.mean() * 100:.1f}%)"
    )
    print(f"Speedup: {original_times.mean() / persistent_times.mean():.2f}x")
    print(f"Stability improvement: {original_times.std() / persistent_times.std():.1f}x")


if __name__ == "__main__":
    comprehensive_onnx_performance_test()

    print("\nCleaning up...")
    judo_cpp.shutdown_thread_pool()
