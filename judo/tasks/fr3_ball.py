# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.tasks.base import Task, TaskConfig
from judo.tasks.cost_functions import (
    quadratic_norm,
)

XML_PATH = str(MODEL_PATH / "xml/fr3_ball.xml")
QPOS_HOME = np.array(
    [
        0.7, 0, 0.72, 1, 0, 0, 0,  # object
        0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854,  # arm
        0.04, 0.04,  # gripper, equality constrained
    ]
)  # fmt: skip


@dataclass
class FR3BallConfig(TaskConfig):
    """Configuration for FR3 Ball Dribbling task."""

    # Dribbling reward weights
    w_ball_height: float = 5.0  # Reward for ball at target height
    w_ball_velocity: float = 2.0  # Reward for proper ball velocity direction
    w_contact_timing: float = 8.0  # Reward for hitting ball at right time
    w_horizontal_stability: float = 1.0  # Keep ball near robot
    w_rhythmic_motion: float = 3.0  # Encourage consistent dribbling rhythm
    w_ee_ball_distance: float = 1.0  # Keep end-effector reasonably close
    w_control: float = 0.01  # Small control penalty

    # Task parameters
    target_ball_height: float = 0.8  # Target height for ball (meters)
    contact_threshold: float = 0.15  # Distance threshold for contact (meters)
    max_horizontal_drift: float = 0.3  # Maximum allowed horizontal drift (meters)
    target_dribble_frequency: float = 2.0  # Target dribbles per second


class FR3Ball(Task[FR3BallConfig]):
    """FR3 Ball Dribbling task."""

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the FR3 ball dribbling task."""
        super().__init__(model_path, sim_model_path=sim_model_path)

        # Get sensor indices for end-effector position
        self.grasp_site_adr = self.get_sensor_start_index("trace_grasp_site")

        # State indices for ball (freejoint: 3 pos + 4 quat = 7 dofs)
        self.ball_pos_slice = slice(0, 3)  # x, y, z position
        self.ball_quat_slice = slice(3, 7)  # quaternion

        # Velocity indices (after all position dofs)
        ball_vel_start = self.model.nq  # Start of velocity section
        self.ball_vel_slice = slice(ball_vel_start, ball_vel_start + 3)  # linear velocity
        self.ball_angvel_slice = slice(ball_vel_start + 3, ball_vel_start + 6)  # angular velocity

        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: FR3BallConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """
        Compute reward for FR3 ball dribbling task.

        Comprehensive reward for basketball dribbling behavior including:
        - Ball height maintenance (periodic up-down motion)
        - Proper velocity direction (up when low, down when high)
        - Contact timing (hit ball when at peak height)
        - Horizontal stability (keep ball near robot)
        - Rhythmic consistency (regular dribbling intervals)
        """
        batch_size, sequence_length = states.shape[0], states.shape[1]

        # Extract ball state
        ball_pos = states[..., self.ball_pos_slice]  # (batch, seq, 3)
        ball_vel = states[..., self.ball_vel_slice]  # (batch, seq, 3)
        ball_height = ball_pos[..., 2]  # z-coordinate
        ball_vel_z = ball_vel[..., 2]  # vertical velocity

        # Get end-effector position from sensors
        ee_pos = sensors[..., self.grasp_site_adr : self.grasp_site_adr + 3]  # (batch, seq, 3)

        # === 1. Ball Height Reward ===
        # Encourage ball to oscillate around target height
        height_error = np.abs(ball_height - config.target_ball_height)
        height_reward = -config.w_ball_height * height_error

        # === 2. Ball Velocity Direction Reward ===
        # Reward proper velocity direction based on height
        height_diff = ball_height - config.target_ball_height
        # When ball is above target, reward downward velocity (negative)
        # When ball is below target, reward upward velocity (positive)
        desired_vel_direction = -np.sign(height_diff)
        velocity_alignment = ball_vel_z * desired_vel_direction
        # Only reward when velocity is in correct direction
        velocity_reward = config.w_ball_velocity * np.maximum(velocity_alignment, 0)

        # === 3. Contact Timing Reward ===
        # Reward end-effector being close to ball when ball is at good height
        ee_ball_distance = np.linalg.norm(ee_pos - ball_pos, axis=-1)
        is_contact = ee_ball_distance < config.contact_threshold

        # Contact should happen when ball is at or above target height
        good_contact_height = ball_height >= (config.target_ball_height - 0.1)
        good_contact = is_contact & good_contact_height
        contact_reward = config.w_contact_timing * good_contact.astype(float)

        # === 4. Horizontal Stability Reward ===
        # Keep ball within reasonable horizontal distance from robot base
        robot_base_pos = np.array([0.0, 0.0])  # Robot base at origin
        ball_horizontal_pos = ball_pos[..., :2]  # x, y coordinates
        horizontal_distance = np.linalg.norm(ball_horizontal_pos - robot_base_pos, axis=-1)
        stability_penalty = np.maximum(horizontal_distance - config.max_horizontal_drift, 0)
        stability_reward = -config.w_horizontal_stability * stability_penalty

        # === 5. Rhythmic Motion Reward ===
        # Encourage consistent timing between ball peaks
        # Simple approach: reward ball bouncing (low height + upward velocity)
        is_low = ball_height < 0.2  # Ball is close to ground
        is_moving_up = ball_vel_z > 0.5  # Ball has significant upward velocity
        bounce_indicator = is_low & is_moving_up
        rhythm_reward = config.w_rhythmic_motion * bounce_indicator.astype(float)

        # === 6. End-effector Distance Reward (Secondary) ===
        # Weak reward to keep end-effector reasonably close
        ee_distance_reward = -config.w_ee_ball_distance * ee_ball_distance

        # === 7. Control Effort Penalty ===
        control_penalty = -config.w_control * quadratic_norm(controls)

        assert height_reward.shape == (batch_size, sequence_length)
        assert velocity_reward.shape == (batch_size, sequence_length)
        assert contact_reward.shape == (batch_size, sequence_length)
        assert stability_reward.shape == (batch_size, sequence_length)
        assert rhythm_reward.shape == (batch_size, sequence_length)
        assert ee_distance_reward.shape == (batch_size, sequence_length)
        assert control_penalty.shape == (batch_size, sequence_length)

        # === Combine All Rewards ===
        # Sum over time dimension to get reward per batch
        total_reward = (
            height_reward.sum(axis=-1)
            + velocity_reward.sum(axis=-1)
            + contact_reward.sum(axis=-1)
            + stability_reward.sum(axis=-1)
            + rhythm_reward.sum(axis=-1)
            + ee_distance_reward.sum(axis=-1)
            + control_penalty.sum(axis=-1)
        )

        return total_reward

    def _detect_ball_bounce_events(self, ball_height: np.ndarray, ball_vel_z: np.ndarray) -> np.ndarray:
        """
        Detect ball bounce events based on height and velocity changes.

        Args:
            ball_height: Ball height over time (batch, seq)
            ball_vel_z: Ball vertical velocity over time (batch, seq)

        Returns:
            bounce_events: Boolean array indicating bounce events (batch, seq)
        """
        # Simple bounce detection: ball is low and has significant upward velocity
        height_threshold = 0.25  # Consider "low" if below this height
        velocity_threshold = 0.3  # Minimum upward velocity to count as bounce

        is_low = ball_height < height_threshold
        is_bouncing_up = ball_vel_z > velocity_threshold

        # Combined bounce indicator
        bounce_events = is_low & is_bouncing_up

        return bounce_events

    def reset(self) -> None:
        """Resets the FR3 ball dribbling task."""
        # Reset robot to home position with gripper open
        # Set joint positions: 7 arm joints + 2 gripper joints (frozen in open position)
        self.reset_command = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])  # Only arm joints
        self.data.qpos[:] = QPOS_HOME
        self.data.qvel[:] = 0.0
        # Only set controls for arm joints (7), gripper is frozen via equality constraints
        self.data.ctrl[:7] = self.reset_command
        mujoco.mj_forward(self.model, self.data)
