# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from copy import deepcopy

import mujoco
import numpy as np

import judo_cpp
from judo import MODEL_PATH

XML_PATH = str(MODEL_PATH / "xml/spot_locomotion.xml")


def diagnose_persistent_threadpool():
    """Diagnose why persistent thread pool shows instability"""
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

    print("=== Diagnosing Persistent Thread Pool Performance ===")

    # Test 1: Short bursts vs continuous
    print("\n1. Testing burst vs continuous execution:")

    # Burst execution (like our previous test)
    print("Burst execution (with pauses):")
    times_burst = []
    for i in range(10):
        start = time.time()
        _ = judo_cpp.persistent_cpp_rollout(models, datas, x0_batched, controls)
        end = time.time()
        times_burst.append(end - start)
        print(f"  Burst {i + 1}: {times_burst[-1]:.4f}s")
        time.sleep(0.1)  # Small pause

    # Continuous execution (no pauses)
    print("Continuous execution (no pauses):")
    times_continuous = []
    for i in range(10):
        start = time.time()
        _ = judo_cpp.persistent_cpp_rollout(models, datas, x0_batched, controls)
        end = time.time()
        times_continuous.append(end - start)
        print(f"  Continuous {i + 1}: {times_continuous[-1]:.4f}s")
        # No pause

    print(f"Burst CV: {np.std(times_burst) / np.mean(times_burst) * 100:.1f}%")
    print(f"Continuous CV: {np.std(times_continuous) / np.mean(times_continuous) * 100:.1f}%")

    # Test 2: Fresh thread pool each time
    print("\n2. Testing fresh thread pool for each call:")
    times_fresh = []
    for i in range(10):
        judo_cpp.shutdown_thread_pool()  # Force fresh thread pool
        start = time.time()
        _ = judo_cpp.persistent_cpp_rollout(models, datas, x0_batched, controls)
        end = time.time()
        times_fresh.append(end - start)
        print(f"  Fresh {i + 1}: {times_fresh[-1]:.4f}s")

    print(f"Fresh thread pool CV: {np.std(times_fresh) / np.mean(times_fresh) * 100:.1f}%")

    # Test 3: Compare thread pool sizes
    print("\n3. Testing different thread pool sizes:")
    for threads in [32, 64, 128]:
        print(f"Testing with {threads} threads:")
        judo_cpp.shutdown_thread_pool()

        # Create models/data for this thread count
        test_models = (
            models[:threads]
            if threads <= len(models)
            else models + [deepcopy(model) for _ in range(threads - len(models))]
        )
        test_datas = (
            datas[:threads]
            if threads <= len(datas)
            else datas + [mujoco.MjData(model) for _ in range(threads - len(datas))]
        )
        test_x0 = np.tile(x0, (threads, 1))
        test_controls = np.random.randn(threads, time_steps, model.nu) * 0.1

        times_thread_test = []
        for i in range(5):
            start = time.time()
            _ = judo_cpp.persistent_cpp_rollout(test_models, test_datas, test_x0, test_controls)
            end = time.time()
            times_thread_test.append(end - start)
            print(f"    {threads} threads run {i + 1}: {times_thread_test[-1]:.4f}s")

        print(f"  {threads} threads CV: {np.std(times_thread_test) / np.mean(times_thread_test) * 100:.1f}%")

    # Summary
    print("\n=== Summary ===")
    print(
        f"Burst execution (with pauses): {np.mean(times_burst):.4f}s ± {np.std(times_burst):.4f}s (CV: {np.std(times_burst) / np.mean(times_burst) * 100:.1f}%)"
    )
    print(
        f"Continuous execution (no pauses): {np.mean(times_continuous):.4f}s ± {np.std(times_continuous):.4f}s (CV: {np.std(times_continuous) / np.mean(times_continuous) * 100:.1f}%)"
    )
    print(
        f"Fresh thread pool each time: {np.mean(times_fresh):.4f}s ± {np.std(times_fresh):.4f}s (CV: {np.std(times_fresh) / np.mean(times_fresh) * 100:.1f}%)"
    )


if __name__ == "__main__":
    diagnose_persistent_threadpool()
    judo_cpp.shutdown_thread_pool()
