#!/usr/bin/env python3
# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.


import time
from copy import deepcopy

import mujoco
import numpy as np

import judo_cpp
from judo import MODEL_PATH

XML_PATH = str(MODEL_PATH / "xml/spot_locomotion.xml")

num_threads = 64
batch_size = num_threads
time_steps = 100

model = mujoco.MjModel.from_xml_path(XML_PATH)


def test_fresh_models():
    """Test C++ rollout with fresh models/data"""
    print("=== Testing with FRESH models/data ===")
    models = [deepcopy(model) for _ in range(batch_size)]
    datas = [mujoco.MjData(m) for m in models]

    x0 = np.zeros(model.nq + model.nv)
    x0[: model.nq] = model.qpos0
    x0_batched = np.tile(x0, (batch_size, 1))
    controls = np.random.randn(batch_size, time_steps, model.nu) * 0.1

    for i in range(5):
        start_time = time.time()
        states_cpp, sensors_cpp, inputs_cpp = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)
        end_time = time.time()
        print(f"Fresh C++ Rollout time: {end_time - start_time:.4f}")


def test_reused_models():
    """Test C++ rollout with reused models/data (after Python rollout)"""
    print("\n=== Testing with REUSED models/data (after Python rollout) ===")
    from mujoco.rollout import Rollout

    models = [deepcopy(model) for _ in range(batch_size)]
    datas = [mujoco.MjData(m) for m in models]
    rollout_obj = Rollout(nthread=num_threads)

    x0 = np.zeros(model.nq + model.nv)
    x0[: model.nq] = model.qpos0
    x0_batched = np.tile(x0, (batch_size, 1))
    full_states = np.concatenate([time.time() * np.ones((batch_size, 1)), x0_batched], axis=-1)
    controls = np.random.randn(batch_size, time_steps, model.nu) * 0.1

    # First run Python rollout (modifies the datas)
    print("Running Python rollout first...")
    states_raw, sensors = rollout_obj.rollout(models, datas, full_states, controls)

    # Now test C++ rollout with the modified datas
    print("Now running C++ rollout with same models/datas...")
    for i in range(5):
        start_time = time.time()
        states_cpp, sensors_cpp, inputs_cpp = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)
        end_time = time.time()
        print(f"Reused C++ Rollout time: {end_time - start_time:.4f}")

    rollout_obj.close()


if __name__ == "__main__":
    test_fresh_models()
    test_reused_models()
