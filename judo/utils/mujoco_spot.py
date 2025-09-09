from typing import Callable, Literal

import numpy as np
from mujoco import MjData, MjModel

from judo_cpp import rollout_spot, sim_spot


class RolloutBackend:
    """Spot-specific backend that runs ONNX policy each step."""

    def __init__(self, num_threads: int, backend: Literal["mujoco"], task_to_sim_ctrl: Callable) -> None:
        self.backend = backend
        if self.backend == "mujoco":
            self.rollout_func = rollout_spot
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        self.task_to_sim_ctrl = task_to_sim_ctrl
        self.previous_policy_output = np.zeros((1, 12))

    def rollout(
        self,
        model_data_pairs: list[tuple[MjModel, MjData]],
        x0: np.ndarray,
        controls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ms, ds = zip(*model_data_pairs, strict=True)
        ms = list(ms)
        ds = list(ds)

        nq = ms[0].nq
        nv = ms[0].nv
        nu = ms[0].nu

        x0_batched = np.tile(x0, (len(ms), 1))
        # processed_controls = self.task_to_sim_ctrl(controls)
        assert x0_batched.shape[-1] == nq + nv
        assert x0_batched.ndim == 2
        assert controls.ndim == 3

        controls = self.task_to_sim_ctrl(controls)
        states, sensors = self.rollout_func(ms, ds, x0_batched, controls)
        return np.array(states), np.array(sensors)


class SimBackend:
    """Single-step sim using Spot ONNX policy for controls."""

    def __init__(self, task_to_sim_ctrl: Callable) -> None:
        self.task_to_sim_ctrl = task_to_sim_ctrl

    def sim(self, sim_model: MjModel, sim_data: MjData, sim_controls: np.ndarray) -> None:
        # processed_ctrl = self.task_to_sim_ctrl(sim_controls)
        x0 = np.concatenate([sim_data.qpos, sim_data.qvel])
        controls = self.task_to_sim_ctrl(sim_controls)
        controls = controls.flatten()
        sim_spot(sim_model, sim_data, x0, controls)


