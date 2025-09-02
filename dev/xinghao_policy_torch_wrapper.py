"""Torch wrapper around an ONNX policy with Spot-specific post-processing.

This module provides a torch.nn.Module that:
- accepts an observation vector (already assembled as in your C++ setObservation)
- runs ONNX inference
- applies post-processing to produce controls compatible with MuJoCo's data.ctrl

Post-processing mirrors the C++ logic:
- legs: control[:12] = default_joint_pos[:12] + orbit_to_mujoco_legs @ (0.2 * policy_output)
- arms: control[12:19] = observation[3+3+3+3 : 3+3+3+3+7]  # arm_command passthrough
- optional leg override from observation[3+3+3+3+7 : 3+3+3+3+7+12]

Note: This wrapper expects that the provided observation matches the ordering/layout
constructed in your C++ setObservation(). If you need Python-side observation
construction, you can add a helper around this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "onnxruntime is required for ONNXPolicyAsTorch. Please install onnxruntime."
    ) from exc


@dataclass
class PolicyTransforms:
    """Container for policy post-processing transforms.

    Attributes:
        default_joint_pos: Shape (19,) torch tensor. Default joint positions for all actuated joints
                           ordered as [legs 12, arms 7]. If None, zeros are used.
        orbit_to_mujoco_legs: Shape (12, 12) torch tensor. Mapping from Orbit leg outputs to MuJoCo leg joints.
                               If None, identity is used.
        scale_factor: Scalar applied to raw policy outputs for the legs (default 0.2).
    """

    default_joint_pos: Optional[torch.Tensor] = None
    orbit_to_mujoco_legs: Optional[torch.Tensor] = None
    scale_factor: float = 0.2

    def materialize(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if self.default_joint_pos is None:
            default_joint_pos = torch.zeros(19, dtype=torch.float32, device=device)
        else:
            default_joint_pos = self.default_joint_pos.to(device=device, dtype=torch.float32)

        if self.orbit_to_mujoco_legs is None:
            orbit_to_mujoco_legs = torch.eye(12, dtype=torch.float32, device=device)
        else:
            orbit_to_mujoco_legs = self.orbit_to_mujoco_legs.to(device=device, dtype=torch.float32)

        return default_joint_pos, orbit_to_mujoco_legs, float(self.scale_factor)


class ONNXPolicyAsTorch(nn.Module):
    """Wrap an ONNX policy inside a torch.nn.Module with Spot control post-processing.

    forward(observation) -> controls
      - observation: torch.Tensor of shape (obs_dim,) or (batch, obs_dim), float32/float64
      - returns controls: torch.Tensor of shape (19,) or (batch, 19)
    """

    def __init__(
        self,
        onnx_model_path: str,
        transforms: Optional[PolicyTransforms] = None,
        session_options: Optional[ort.SessionOptions] = None,
        providers: Optional[list[str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.onnx_model_path = onnx_model_path
        self.transforms = transforms or PolicyTransforms()
        self._device = device or torch.device("cpu")

        # Create ONNX Runtime session
        self._session_options = session_options or ort.SessionOptions()
        if providers is None:
            # Default CPU provider; user can pass CUDAExecutionProvider if available
            self._providers = ["CPUExecutionProvider"]
        else:
            self._providers = providers
        self._session = ort.InferenceSession(
            self.onnx_model_path, sess_options=self._session_options, providers=self._providers
        )

        # Cache I/O names and expected dims
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # Materialize transforms to the chosen device
        default_joint_pos, orbit_to_mujoco_legs, scale_factor = self.transforms.materialize(self._device)
        # Register as buffers so they move with .to(device) if needed
        self.register_buffer("default_joint_pos", default_joint_pos)
        self.register_buffer("orbit_to_mujoco_legs", orbit_to_mujoco_legs)
        self.scale_factor: float = scale_factor

        # Constants for observation slicing (as in C++)
        self._IDX_BASE_LINVEL = 0
        self._IDX_BASE_ANGVEL = self._IDX_BASE_LINVEL + 3
        self._IDX_GRAVITY = self._IDX_BASE_ANGVEL + 3
        self._IDX_TORSO_VEL_CMD = self._IDX_GRAVITY + 3
        self._IDX_ARM_CMD = self._IDX_TORSO_VEL_CMD + 3
        self._IDX_LEG_CMD = self._IDX_ARM_CMD + 7
        self._IDX_TORSO_POS_CMD = self._IDX_LEG_CMD + 12
        self._IDX_JOINT_POS = self._IDX_TORSO_POS_CMD + 3
        self._IDX_JOINT_VEL = self._IDX_JOINT_POS + 19
        self._IDX_POLICY_OUTPUT_IN_OBS = self._IDX_JOINT_VEL + 19

    @torch.no_grad()
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Run ONNX policy and apply post-processing to produce 19-d controls.

        Args:
            observation: shape (obs_dim,) or (batch, obs_dim)

        Returns:
            controls: torch tensor of shape (19,) or (batch, 19)
        """
        single_input = observation.dim() == 1
        if single_input:
            observation = observation.unsqueeze(0)

        # Ensure float32 on CPU for ORT
        obs_np = observation.detach().to(dtype=torch.float32, device=torch.device("cpu")).cpu().numpy()

        # ONNX inference
        ort_out = self._session.run([self._output_name], {self._input_name: obs_np})[0]
        # Convert back to torch on target device
        policy_output = torch.from_numpy(np.asarray(ort_out)).to(device=self._device, dtype=torch.float32)

        # Post-process legs: control[:12] = default + A @ (scale * policy_output)
        scaled_leg_output = self.scale_factor * policy_output  # (B, 12)
        # Matrix multiply per batch: (B,12) = (B,12) @ (12,12)^T or (12,12)
        # Use (12,12) right-multiply: (B,12) @ (12,12) -> (B,12)
        legs_mapped = scaled_leg_output @ self.orbit_to_mujoco_legs  # (B,12)
        legs_control = legs_mapped + self.default_joint_pos[:12].unsqueeze(0)

        # Arms passthrough from observation
        arm_cmd = observation[:, self._IDX_ARM_CMD : self._IDX_ARM_CMD + 7].to(device=self._device, dtype=torch.float32)

        # Assemble controls (B, 19)
        controls = torch.empty((observation.shape[0], 19), dtype=torch.float32, device=self._device)
        controls[:, :12] = legs_control
        controls[:, 12:19] = arm_cmd

        # Optional leg override from observation segment
        leg_joint_cmd = observation[:, self._IDX_LEG_CMD : self._IDX_LEG_CMD + 12].to(
            device=self._device, dtype=torch.float32
        )

        # Determine which leg (if any) to override per batch in the order FL, FR, HL, HR
        # Each is a 3-d segment; override if its norm > 0.
        if torch.any(leg_joint_cmd != 0):
            for b in range(observation.shape[0]):
                seg = leg_joint_cmd[b]
                if torch.norm(seg[0:3]) > 0:
                    controls[b, 0:3] = seg[0:3]
                elif torch.norm(seg[3:6]) > 0:
                    controls[b, 3:6] = seg[3:6]
                elif torch.norm(seg[6:9]) > 0:
                    controls[b, 6:9] = seg[6:9]
                elif torch.norm(seg[9:12]) > 0:
                    controls[b, 9:12] = seg[9:12]

        if single_input:
            return controls.squeeze(0)
        return controls


def _example_usage() -> None:  # pragma: no cover - illustrative
    """Minimal example showing how to use the wrapper with an observation array."""
    import os

    onnx_path = os.path.join(
        os.path.dirname(__file__), "..", "judo_cpp", "policy", "xinghao_policy_v1.onnx"
    )
    onnx_path = os.path.abspath(onnx_path)

    # Optional: provide real transforms if available; otherwise, identity/zeros are used
    transforms = PolicyTransforms(
        default_joint_pos=None,  # or torch.tensor([...], dtype=torch.float32)
        orbit_to_mujoco_legs=None,  # or torch.tensor([...], dtype=torch.float32).view(12, 12)
        scale_factor=0.2,
    )

    policy = ONNXPolicyAsTorch(onnx_model_path=onnx_path, transforms=transforms, device=torch.device("cpu"))

    # Dummy observation matching your C++ setObservation layout
    obs_dim = (
        3  # base linvel
        + 3  # base angvel
        + 3  # gravity
        + 3  # torso vel cmd
        + 7  # arm cmd
        + 12  # leg cmd
        + 3  # torso pos cmd
        + 19  # joint pos
        + 19  # joint vel
        + 12  # previous policy output
    )
    observation = torch.zeros(obs_dim, dtype=torch.float32)

    controls = policy(observation)  # shape (19,)
    print("Controls shape:", controls.shape)
    print("Controls:", controls)

    # If you have a MuJoCo data object:
    #   import mujoco
    #   data.ctrl[:] = controls.cpu().numpy()


if __name__ == "__main__":  # pragma: no cover - script mode
    _example_usage()


