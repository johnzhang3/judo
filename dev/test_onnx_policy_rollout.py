# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import argparse
import time
from copy import deepcopy

import mujoco
import numpy as np

import judo_cpp
from judo import MODEL_PATH


def build_models(batch_size: int, xml_path: str) -> tuple[list, list, int, int, int]:
    model = mujoco.MjModel.from_xml_path(xml_path)
    models = [deepcopy(model) for _ in range(batch_size)]
    datas = [mujoco.MjData(m) for m in models]
    return models, datas, model.nq + model.nv, model.nu, model.nsensordata


def main() -> None:
    parser = argparse.ArgumentParser(description="Test persistent ONNX policy rollout")
    parser.add_argument("--onnx_model_path", type=str, required=True, help="Path to ONNX policy model")
    parser.add_argument("--xml", type=str, default=str(MODEL_PATH / "xml/spot_locomotion.xml"), help="MuJoCo XML path")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of parallel rollouts (threads)")
    parser.add_argument("--horizon", type=int, default=128, help="Number of simulation steps")
    parser.add_argument("--state_history", type=int, default=10, help="State history length maintained per thread")
    parser.add_argument("--action_history", type=int, default=5, help="Action history length maintained per thread")
    parser.add_argument("--infer_every", type=int, default=1, help="Inference frequency (every N steps)")
    args = parser.parse_args()

    models, datas, state_dim, action_dim, sensor_dim = build_models(args.batch_size, args.xml)

    # Initial state x0 for each rollout: concatenate qpos (from qpos0) and qvel (zeros)
    x0 = np.zeros(state_dim, dtype=np.float64)
    # Acquire a representative model to read qpos0
    ref_model = models[0]
    x0[: ref_model.nq] = ref_model.qpos0
    x0_batched = np.tile(x0, (args.batch_size, 1))

    # Run twice to exercise persistent thread pool behavior and per-thread buffers
    for run_idx in range(2):
        t0 = time.time()
        states, actions, sensors = judo_cpp.persistent_onnx_policy_rollout(
            models,
            datas,
            x0_batched,
            args.horizon,
            args.onnx_model_path,
            args.state_history,
            args.action_history,
            args.infer_every,
        )
        dt = time.time() - t0

        # Shape checks
        assert states.shape == (args.batch_size, args.horizon + 1, state_dim), (
            "states shape mismatch",
            states.shape,
            (args.batch_size, args.horizon + 1, state_dim),
        )
        assert actions.shape == (args.batch_size, args.horizon, action_dim), (
            "actions shape mismatch",
            actions.shape,
            (args.batch_size, args.horizon, action_dim),
        )
        assert sensors.shape[0:2] == (args.batch_size, args.horizon), (
            "sensors batch/time mismatch",
            sensors.shape,
        )

        print(
            f"run={run_idx} ok | states {states.shape} actions {actions.shape} sensors {sensors.shape} | time {dt:.3f}s",
            flush=True,
        )

    # Explicitly shutdown the persistent pool if desired
    judo_cpp.shutdown_thread_pool()


if __name__ == "__main__":
    main()
