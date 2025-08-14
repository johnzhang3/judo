# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import os
import time
from copy import deepcopy

import mujoco
import numpy as np
import psutil

import judo_cpp
from judo import MODEL_PATH

XML_PATH = str(MODEL_PATH / "xml/spot_locomotion.xml")


def get_system_info():
    """Get current system performance indicators"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    load_avg = os.getloadavg()

    print(f"CPU Usage: {cpu_percent:.1f}%")
    print(f"Memory Usage: {memory.percent:.1f}% ({memory.available // (1024**3):.1f}GB available)")
    print(f"Load Average: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")

    # Get CPU frequency if available
    try:
        freq = psutil.cpu_freq()
        if freq:
            print(f"CPU Frequency: {freq.current:.0f}MHz (max: {freq.max:.0f}MHz)")
    except:
        pass

    return {"cpu_percent": cpu_percent, "memory_percent": memory.percent, "load_avg": load_avg[0]}


def run_cpp_benchmark(num_iterations=20):
    """Run a controlled C++ benchmark with system monitoring"""
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

    print("=== System Info Before Benchmark ===")
    sys_info_before = get_system_info()

    times = []
    for i in range(num_iterations):
        start_time = time.time()
        states_cpp, sensors_cpp, inputs_cpp = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"Iteration {i + 1:2d}: {elapsed:.4f}s")

        # Brief pause to let system settle
        time.sleep(0.1)

    print("\n=== System Info After Benchmark ===")
    sys_info_after = get_system_info()

    # Statistics
    times = np.array(times)
    print("\n=== Performance Statistics ===")
    print(f"Mean: {times.mean():.4f}s")
    print(f"Std:  {times.std():.4f}s")
    print(f"Min:  {times.min():.4f}s")
    print(f"Max:  {times.max():.4f}s")
    print(f"CV:   {times.std() / times.mean() * 100:.1f}%")  # Coefficient of variation

    # Identify outliers (more than 2 std devs from mean)
    outliers = np.abs(times - times.mean()) > 2 * times.std()
    if outliers.any():
        print(f"Outliers: {np.where(outliers)[0] + 1}")

    return times, sys_info_before, sys_info_after


if __name__ == "__main__":
    print("Running C++ performance variability analysis...")
    print("This will take about 30 seconds...\n")

    times, before, after = run_cpp_benchmark()

    print("\n=== System Changes ===")
    print(f"CPU Usage Change: {after['cpu_percent'] - before['cpu_percent']:.1f}%")
    print(f"Memory Usage Change: {after['memory_percent'] - before['memory_percent']:.1f}%")
    print(f"Load Average Change: {after['load_avg'] - before['load_avg']:.2f}")
