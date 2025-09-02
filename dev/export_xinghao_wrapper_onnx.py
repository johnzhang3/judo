import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn


class PostProcessWrapper(nn.Module):
    """Pure-PyTorch post-processing wrapper matching C++ control logic.

    Inputs:
      - observation: (B, obs_dim)
      - policy_output: (B, 12)   # raw leg outputs from base ONNX policy

    Output:
      - controls: (B, 19)        # controls to write to data.ctrl

    The transform applies:
      legs_control = default[:12] + orbit_to_mujoco_legs @ (scale * policy_output)
      arms_control = observation[arm_slice]
      Optional leg override: first non-zero 3-d segment of leg_command replaces that segment
    """

    def __init__(
        self,
        obs_dim: int,
        scale: float = 0.2,
        default_joint_pos: Optional[torch.Tensor] = None,  # (19,)
        orbit_to_mujoco_legs_indices: Optional[torch.Tensor] = None,  # (12,) index mapping orbit->mujoco
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.scale = float(scale)
        self._device = device or torch.device("cpu")

        # Bake default_joint_pos (from C++): legs (12) in MuJoCo order, then arm (7)
        if default_joint_pos is None:
            default_joint_pos = torch.tensor([
                0.12, 0.5, -1.0,   -0.12, 0.5, -1.0,   0.12, 0.5, -1.0,   -0.12, 0.5, -1.0,
                0.0, -0.9, 1.8, 0.0, -0.9, 0.0, -1.54
            ], dtype=torch.float32)
        self.register_buffer("default_joint_pos", default_joint_pos.to(torch.float32))

        # Bake orbit->mujoco legs index permutation from C++
        # indices = [0,3,6,9,1,4,7,10,2,5,8,11]
        if orbit_to_mujoco_legs_indices is None:
            orbit_to_mujoco_legs_indices = torch.tensor([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], dtype=torch.long)
        self.register_buffer("orbit_to_mujoco_legs_indices", orbit_to_mujoco_legs_indices)

        # Observation slicing indices (must match C++ setObservation layout)
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

    def forward(self, observation: torch.Tensor, policy_output: torch.Tensor) -> torch.Tensor:
        # Ensure batch dimension exists
        assert observation.dim() == 2, "observation must be (B, obs_dim)"
        assert policy_output.dim() == 2 and policy_output.shape[1] == 12, "policy_output must be (B, 12)"

        # Legs post-processing
        scaled = self.scale * policy_output.to(torch.float32)  # (B,12)
        # Permute from orbit order -> mujoco order using index select
        legs_mapped = scaled.index_select(dim=1, index=self.orbit_to_mujoco_legs_indices)  # (B,12)
        legs_control = legs_mapped + self.default_joint_pos[:12].unsqueeze(0)

        # Arms passthrough from observation
        arm_cmd = observation[:, self._IDX_ARM_CMD : self._IDX_ARM_CMD + 7].to(torch.float32)

        # Assemble controls
        B = observation.shape[0]
        controls = torch.empty((B, 19), dtype=torch.float32)
        controls[:, :12] = legs_control
        controls[:, 12:19] = arm_cmd

        # Leg override with priority FL -> FR -> HL -> HR
        leg_cmd = observation[:, self._IDX_LEG_CMD : self._IDX_LEG_CMD + 12].to(torch.float32)
        fl_mask = (leg_cmd[:, 0:3].abs().sum(dim=1, keepdim=True) > 0)
        fr_mask = (leg_cmd[:, 3:6].abs().sum(dim=1, keepdim=True) > 0)
        hl_mask = (leg_cmd[:, 6:9].abs().sum(dim=1, keepdim=True) > 0)
        hr_mask = (leg_cmd[:, 9:12].abs().sum(dim=1, keepdim=True) > 0)

        # Apply in order, ensuring priority by masking out earlier selections
        controls[:, 0:3] = torch.where(fl_mask, leg_cmd[:, 0:3], controls[:, 0:3])
        controls[:, 3:6] = torch.where((~fl_mask) & fr_mask, leg_cmd[:, 3:6], controls[:, 3:6])
        controls[:, 6:9] = torch.where((~fl_mask) & (~fr_mask) & hl_mask, leg_cmd[:, 6:9], controls[:, 6:9])
        controls[:, 9:12] = torch.where((~fl_mask) & (~fr_mask) & (~hl_mask) & hr_mask, leg_cmd[:, 9:12], controls[:, 9:12])

        return controls


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the Xinghao policy post-processing wrapper to ONNX")
    parser.add_argument("--obs_dim", type=int, default=84, help="Observation dimension (batch included separately)")
    parser.add_argument("--scale", type=float, default=0.2, help="Scale factor applied to policy leg outputs")
    parser.add_argument("--default_joint_pos", type=str, default="", help="Optional path to .npy (19,) default joint positions")
    parser.add_argument(
        "--orbit_to_mujoco_legs", type=str, default="", help="Optional path to .npy (12,12) orbit->mujoco leg map"
    )
    parser.add_argument(
        "--out", type=str, default="judo_cpp/policy/xinghao_policy_wrapper.onnx", help="Output ONNX path"
    )
    args = parser.parse_args()

    # Load optional transforms
    djp = None
    if args.default_joint_pos:
        djp_np = np.load(args.default_joint_pos)
        assert djp_np.shape == (19,)
        djp = torch.from_numpy(djp_np.astype(np.float32))

    o2m_idx = None
    if args.orbit_to_mujoco_legs:
        # Allow either (12,) indices or a (12,12) permutation matrix
        o2m_np = np.load(args.orbit_to_mujoco_legs)
        if o2m_np.shape == (12,):
            o2m_idx = torch.from_numpy(o2m_np.astype(np.int64))
        elif o2m_np.shape == (12, 12):
            o2m_idx = torch.from_numpy(np.argmax(o2m_np, axis=1).astype(np.int64))
        else:
            raise ValueError("orbit_to_mujoco_legs must be (12,) indices or (12,12) permutation matrix")

    model = PostProcessWrapper(
        obs_dim=args.obs_dim,
        scale=args.scale,
        default_joint_pos=djp,
        orbit_to_mujoco_legs_indices=o2m_idx,
    ).eval()

    # Dummy batched inputs (batch dimension present)
    dummy_B = 2
    obs = torch.randn(dummy_B, args.obs_dim, dtype=torch.float32)
    pol = torch.randn(dummy_B, 12, dtype=torch.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (obs, pol),
        out_path.as_posix(),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["observation", "policy_output"],
        output_names=["controls"],
        dynamic_axes={
            "observation": {0: "batch"},
            "policy_output": {0: "batch"},
            "controls": {0: "batch"},
        },
    )

    print(f"Exported wrapper ONNX to {out_path}")


if __name__ == "__main__":
    main()


