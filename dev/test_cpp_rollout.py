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

model = mujoco.MjModel.from_xml_path(XML_PATH)
models = [deepcopy(model) for _ in range(batch_size)]
datas = [mujoco.MjData(m) for m in models]
# print(model.qpos0)
rollout_obj = Rollout(nthread=num_threads)

x0 = np.zeros(model.nq + model.nv)
x0[: model.nq] = model.qpos0
x0_batched = np.tile(x0, (batch_size, 1))
full_states = np.concatenate([time.time() * np.ones((batch_size, 1)), x0_batched], axis=-1)
controls = np.random.randn(batch_size, time_steps, model.nu) * 0.1


def mujoco_rollout(models, datas, full_states, controls):
    for i in range(10):
        start_time = time.time()
        states_raw, sensors = rollout_obj.rollout(models, datas, full_states, controls)
        end_time = time.time()
        print("python Rollout time:", end_time - start_time)
    states = np.array(states_raw)[..., 1:]
    print("States shape:", states.shape)


# model = mujoco.MjModel.from_xml_path(XML_PATH)
# models = [deepcopy(model) for _ in range(batch_size)]
# datas = [mujoco.MjData(m) for m in models]
# x0 = np.zeros(model.nq + model.nv)
# x0[:model.nq] = model.qpos0
# x0_batched = np.tile(x0, (batch_size, 1))


def cpp_rollout(models, datas, x0_batched, controls):
    for i in range(10):
        # controls = np.random.randn(batch_size, time_steps, model.nu) * 0.1
        start_time = time.time()
        states_cpp, sensors_cpp, inputs_cpp = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)
        end_time = time.time()
        print("C++ Rollout time:", end_time - start_time)

    # compare states and states_cpp
    # diff = np.linalg.norm(states - states_cpp)
    # print("Difference between states and states_cpp:", diff)


mujoco_rollout(models, datas, full_states, controls)
cpp_rollout(models, datas, x0_batched, controls)
