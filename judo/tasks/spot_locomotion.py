# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.tasks.base import Task, TaskConfig
from judo.tasks.cost_functions import (
    quadratic_norm,
    smooth_l1_norm,
)
# from judo.utils.mujoco_cpp import RolloutBackend, SimBackend
from judo.utils.mujoco_spot import RolloutBackend, SimBackend

XML_PATH = str(MODEL_PATH / "xml/spot_locomotion.xml")


@dataclass
class SpotLocomotionConfig(TaskConfig):
    """Configuration for Spot Locomotion task."""

    w_upright: float = 10.0  # Weight for upright orientation reward
    w_height: float = 5.0  # Weight for maintaining proper height
    w_lateral: float = 1.0  # Weight for staying centered laterally
    w_velocity: float = 0.1  # Weight for velocity penalty
    w_control: float = 0.1  # Weight for control effort penalty
    p_upright: float = 0.01  # Smoothing parameter for upright reward
    p_height: float = 0.1  # Smoothing parameter for height reward
    p_lateral: float = 0.1  # Smoothing parameter for lateral position


class SpotLocomotion(Task[SpotLocomotionConfig]):
    """Spot Locomotion task."""

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the spot locomotion task."""
        super().__init__(model_path, sim_model_path=sim_model_path)
        self.RolloutBackend = RolloutBackend
        self.SimBackend = SimBackend
        self.reset()

    @property
    def nu(self) -> int:
        """Number of control inputs."""
        # 13 = torso_vel(3) + arm_cmd(7) + torso_pos(3)
        # leg_cmd(12) are zeroed and populated in the rollout wrapper
        return 13

    @property
    def ctrlrange(self) -> np.ndarray:
        """Control limits for this task."""
        return np.array([[-np.inf, np.inf]] * 13)

    def task_to_sim_ctrl(self, controls: np.ndarray) -> np.ndarray:
        """Maps the controls from the optimizer to the controls used in the simulation."""
        return controls[..., :19]

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotLocomotionConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """
        Compute reward for spot locomotion task.

        Rewards the robot for standing upright, staying centered,
        and minimizing unnecessary motion and control effort.
        """
        batch_size = states.shape[0]

        # Extract base position (x, y, z) and quaternion (w, x, y, z)
        base_pos = states[..., :3]  # x, y, z position
        base_quat = states[..., 3:7]  # w, x, y, z quaternion
        base_lin_vel = states[..., 26:29]  # linear velocity
        base_ang_vel = states[..., 29:32]  # angular velocity
        joint_vel = states[..., 32:]  # joint velocities

        # Upright reward: penalize deviation from upright orientation
        # For upright stance, we want the quaternion close to [1,0,0,0] (no rotation)
        # We use the w component of quaternion - closer to 1 means more upright
        upright_rew = -config.w_upright * smooth_l1_norm(base_quat[..., 0] - 1, config.p_upright).sum(-1)

        # Height reward: penalize deviation from desired height (around 0.7m)
        desired_height = 0.7
        height_rew = -config.w_height * smooth_l1_norm(base_pos[..., 2] - desired_height, config.p_height).sum(-1)

        # Lateral position reward: penalize drifting in x-y plane
        lateral_distance = np.linalg.norm(base_pos[..., :2], axis=-1)  # Distance from origin in x-y plane
        lateral_rew = -config.w_lateral * smooth_l1_norm(lateral_distance, config.p_lateral).sum(-1)

        # Velocity penalty: penalize excessive motion
        velocity_rew = -config.w_velocity * (
            quadratic_norm(base_lin_vel).sum(-1)
            + quadratic_norm(base_ang_vel).sum(-1)
            + quadratic_norm(joint_vel).sum(-1)
        )

        # Control effort penalty
        control_rew = -config.w_control * quadratic_norm(controls).sum(-1)

        assert upright_rew.shape == (batch_size,)
        assert height_rew.shape == (batch_size,)
        assert lateral_rew.shape == (batch_size,)
        assert velocity_rew.shape == (batch_size,)
        assert control_rew.shape == (batch_size,)

        return upright_rew + height_rew + lateral_rew + velocity_rew + control_rew

    def reset(self) -> None:
        """Resets the spot locomotion task."""
        # Reset the height of the robot to a reasonable starting position
        self.data.qpos[2] = 1.5  # Set z position to 1.5 meters
        mujoco.mj_forward(self.model, self.data)

    def optimizer_warm_start(self) -> np.ndarray:
        """Returns a warm start for the optimizer."""
        return np.zeros(self.nu)
