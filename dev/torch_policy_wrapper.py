import torch
import torch.nn as nn
from onnx2torch import convert


class TorchXinghaoPolicyWrapper(nn.Module):
    def __init__(self, base_onnx_path: str) -> None:
        super().__init__()
        # Load base ONNX as a Torch nn.Module
        self.base = convert(base_onnx_path)

        # Indices and constants (registered as buffers so they export with ONNX)
        self.register_buffer(
            "mujoco_to_orbit_indices",
            torch.tensor([1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 0, 5, 10, 15, 16, 17, 18], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "orbit_to_mujoco_legs_indices",
            torch.tensor([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "default_joint_pos",
            torch.tensor(
                [
                    0.12,
                    0.5,
                    -1.0,
                    -0.12,
                    0.5,
                    -1.0,
                    0.12,
                    0.5,
                    -1.0,
                    -0.12,
                    0.5,
                    -1.0,
                    0.0,
                    -0.9,
                    1.8,
                    0.0,
                    -0.9,
                    0.0,
                    -1.54,
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )

        # Default MuJoCo index offsets (can be changed by user)
        self.base_qpos_start_idx: int = 0
        self.base_qvel_start_idx: int = 0
        self.leg_qpos_start_idx: int = 7
        self.leg_qvel_start_idx: int = 6

    @staticmethod
    def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
        # q: (..., 4) as (w, x, y, z)
        w = q[..., 0:1]
        xyz = -q[..., 1:4]
        return torch.cat([w, xyz], dim=-1)

    @staticmethod
    def rotate_vec_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # v: (..., 3), q: (..., 4) unit quaternion (w, x, y, z)
        # Using formula: v' = v + 2*w*(q_vec x v) + 2*(q_vec x (q_vec x v))
        w = q[..., 0:1]
        q_vec = q[..., 1:4]
        cross1 = torch.cross(q_vec, v, dim=-1)
        term1 = 2.0 * w * cross1
        cross2 = torch.cross(q_vec, cross1, dim=-1)
        term2 = 2.0 * cross2
        return v + term1 + term2

    def build_observation(self, qpos: torch.Tensor, qvel: torch.Tensor, command: torch.Tensor, prev_policy: torch.Tensor) -> torch.Tensor:
        # Slices
        base_quat = qpos[self.base_qpos_start_idx + 3 : self.base_qpos_start_idx + 7]
        inv_base_quat = self.quat_conjugate(base_quat)

        world_linvel = qvel[self.base_qvel_start_idx : self.base_qvel_start_idx + 3]
        base_linvel = self.rotate_vec_by_quat(world_linvel, inv_base_quat)

        base_angvel = qvel[self.base_qvel_start_idx + 3 : self.base_qvel_start_idx + 6]

        gravity = torch.tensor([0.0, 0.0, -1.0], dtype=qpos.dtype, device=qpos.device)
        projected_gravity = self.rotate_vec_by_quat(gravity, inv_base_quat)

        joint_pos_raw = qpos[self.leg_qpos_start_idx : self.leg_qpos_start_idx + 19]
        joint_vel_raw = qvel[self.leg_qvel_start_idx : self.leg_qvel_start_idx + 19]
        joint_pos_off = joint_pos_raw - self.default_joint_pos
        joint_pos = joint_pos_off.index_select(0, self.mujoco_to_orbit_indices)
        joint_vel = joint_vel_raw.index_select(0, self.mujoco_to_orbit_indices)

        torso_vel_cmd = command[0:3]
        arm_cmd = command[3:10]
        leg_cmd = command[10:22]
        torso_pos_cmd = command[22:25]

        obs_parts = [
            base_linvel,
            base_angvel,
            projected_gravity,
            torso_vel_cmd,
            arm_cmd,
            leg_cmd,
            torso_pos_cmd,
            joint_pos,
            joint_vel,
            prev_policy,
        ]
        observation = torch.cat(obs_parts, dim=0).to(dtype=torch.float32)
        return observation

    def forward(self, qpos: torch.Tensor, qvel: torch.Tensor, command: torch.Tensor, prev_policy: torch.Tensor) -> torch.Tensor:
        # Expect 1-D inputs; export will capture static shapes
        obs = self.build_observation(qpos, qvel, command, prev_policy)
        obs_batched = obs.unsqueeze(0)  # [1, 84]
        policy_out = self.base(obs_batched)  # [1, 12]

        # Post-process to 19 controls
        legs_orbit = 0.2 * policy_out  # [1, 12]
        legs_reordered = torch.index_select(legs_orbit, dim=1, index=self.orbit_to_mujoco_legs_indices)
        legs_with_offset = legs_reordered + self.default_joint_pos[:12].unsqueeze(0)
        arm_cmd = command[3:10].unsqueeze(0)  # [1, 7]
        control = torch.cat([legs_with_offset, arm_cmd], dim=1)  # [1, 19]
        return control


if __name__ == "__main__":
    # Quick sanity test
    base_path = "judo_cpp/policy/xinghao_policy_v1.onnx"
    model = TorchXinghaoPolicyWrapper(base_path)
    qpos = torch.zeros(26, dtype=torch.float32)
    qpos[3] = 1.0
    qvel = torch.zeros(25, dtype=torch.float32)
    command = torch.zeros(25, dtype=torch.float32)
    prev = torch.zeros(12, dtype=torch.float32)
    out = model(qpos, qvel, command, prev)
    print(out.shape)
    print(out)

    
