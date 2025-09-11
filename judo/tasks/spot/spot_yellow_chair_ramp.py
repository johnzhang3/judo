# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from judo.utils.indexing import get_pos_indices, get_sensor_indices
from judo import MODEL_PATH
from judo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)
from judo.tasks.spot.spot_base import GOAL_POSITIONS, SpotBase, SpotBaseConfig

XML_PATH = str(MODEL_PATH / "xml/spot_components/spot_yellow_chair_ramp.xml")

Z_AXIS = np.array([0.0, 0.0, 1.0])
RESET_OBJECT_POSE = np.array([3, 0, 0.275, 1, 0, 0, 0])
# annulus object position sampling
RADIUS_MIN = 1.0
RADIUS_MAX = 2.0
USE_LEGS = False

@dataclass
class SpotYellowChairRampConfig(SpotBaseConfig):
    """Config for the spot box manipulation task."""

    goal_position: np.ndarray = field(default_factory=lambda: GOAL_POSITIONS().origin)
    w_orientation: float = 15.0
    w_torso_proximity: float = 0.1
    w_gripper_proximity: float = 4.0
    orientation_threshold: float = 0.5


class SpotYellowChairRamp(SpotBase):
    """Task getting Spot to move a box to a desired goal location."""

    def __init__(self, model_path: str = XML_PATH) -> None:
        super().__init__(model_path=model_path, use_legs=USE_LEGS)

        self.body_pose_idx = get_pos_indices(self.model, "base")
        self.object_pose_idx = get_pos_indices(self.model, ["yellow_chair_joint"])
        self.object_y_axis_idx = get_sensor_indices(self.model, "object_y_axis")
        self.end_effector_to_object_idx = get_sensor_indices(self.model, "sensor_arm_link_fngr")


    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpotYellowChairRampConfig,
        additional_info: dict[str, Any],
    ) -> np.ndarray:
        """Reward function for the Spot box moving task."""
        batch_size = states.shape[0]

        qpos = states[..., : self.model.nq]
        body_height = qpos[..., self.body_pose_idx[2]]
        body_pos = qpos[..., self.body_pose_idx[0:3]]
        object_pos = qpos[..., self.object_pose_idx[0:3]]

        object_y_axis = sensors[..., self.object_y_axis_idx]
        end_effector_to_object = sensors[..., self.end_effector_to_object_idx]

        # Check if any state in the rollout has spot fallen
        spot_fallen_reward = -config.fall_penalty * (body_height <= config.spot_fallen_threshold).any(axis=-1)

        # Compute l2 distance from object pos. to goal.
        goal_reward = -config.w_goal * np.linalg.norm(
            object_pos - np.array(config.goal_position)[None, None], axis=-1
        ).mean(-1)

        yellow_chair_orientation_reward = -config.w_orientation * np.abs(
            np.dot(object_y_axis, Z_AXIS) > config.orientation_threshold
        ).sum(axis=-1)

        # Compute l2 distance from torso pos. to object pos.
        torso_proximity_reward = config.w_torso_proximity * np.linalg.norm(body_pos - object_pos, axis=-1).mean(-1)

        # Compute l2 distance from torso pos. to object pos.
        gripper_proximity_reward = -config.w_gripper_proximity * np.linalg.norm(
            end_effector_to_object,
            axis=-1,
        ).mean(-1)

        # Compute a velocity penalty to prefer small velocity commands.
        controls_reward = -config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert yellow_chair_orientation_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return (
            spot_fallen_reward
            + goal_reward
            + yellow_chair_orientation_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + controls_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""
        radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        object_pos = np.array([radius * np.cos(theta), radius * np.cos(theta)]) + np.random.randn(2)
        reset_object_pose = np.array([*object_pos, 0.254, 1, 0, 0, 0])

        return np.array(
            [
                *np.random.randn(2),
                STANDING_HEIGHT,
                1,
                0,
                0,
                0,
                *LEGS_STANDING_POS,
                *self.reset_arm_pos,
                *reset_object_pose,
            ]
        )