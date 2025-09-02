import numpy as np
import onnxruntime as ort
from typing import cast
import torch.nn as nn

class XinghaoPolicyWrapper(nn.Module):
    def __init__(self, onnx_path: str) -> None:
        super().__init__()
        self.session = ort.InferenceSession(onnx_path)
        self.input_meta = self.session.get_inputs()[0]
        self.output_meta = self.session.get_outputs()[0]
        self.input_name = self.input_meta.name
        self.output_name = self.output_meta.name
        self.input_shape = self.input_meta.shape
        # Static index/permutation helpers and defaults from C++ reference
        self.orbit_to_mujoco_legs_indices = np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], dtype=np.int64)
        self.mujoco_to_orbit_indices = np.array([1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 0, 5, 10, 15, 16, 17, 18], dtype=np.int64)
        self.default_joint_pos = np.array([
            0.12, 0.5, -1.0,
            -0.12, 0.5, -1.0,
            0.12, 0.5, -1.0,
            -0.12, 0.5, -1.0,
            0.0, -0.9, 1.8, 0.0, -0.9, 0.0, -1.54
        ], dtype=np.float64)
        self.prev_policy_output = np.zeros(12, dtype=np.float64)
        self.base_qpos_start_idx = 0
        self.base_qvel_start_idx = 0
        self.leg_qpos_start_idx = 7
        self.leg_qvel_start_idx = 6

    def _infer_from_observation(self, observation: np.ndarray) -> np.ndarray:
        obs = observation
        if obs.ndim == len(self.input_shape) - 1:
            obs = obs[np.newaxis, ...]
        output = self.session.run([self.output_name], {self.input_name: obs})
        return cast(np.ndarray, np.asarray(output[0]))

    # --------- Helpers for state->observation and postprocessing (from C++ logic) ---------
    @staticmethod
    def _quat_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=q1.dtype)

    @classmethod
    def _rotate_vec_by_quat(cls, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=q.dtype)
        q_conj = cls._quat_conjugate(q)
        r = cls._quat_multiply(cls._quat_multiply(q, v_quat), q_conj)
        return r[1:4]

    def build_observation_from_state(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        # Basic validation to avoid out-of-bounds slicing
        if qpos.ndim != 1 or qvel.ndim != 1:
            raise ValueError("qpos and qvel must be 1-D arrays")
        if qpos.size < self.base_qpos_start_idx + 7:
            raise ValueError("qpos too small for base pose (needs +7 from base_qpos_start_idx)")
        if qvel.size < self.base_qvel_start_idx + 6:
            raise ValueError("qvel too small for base velocity (needs +6 from base_qvel_start_idx)")
        if qpos.size < self.leg_qpos_start_idx + 19:
            raise ValueError("qpos too small for 19 joint positions starting at leg_qpos_start_idx")
        if qvel.size < self.leg_qvel_start_idx + 19:
            raise ValueError("qvel too small for 19 joint velocities starting at leg_qvel_start_idx")
        torso_vel_command = command[0:3]
        arm_command = command[3:10]
        leg_command = command[10:22]
        torso_pos_command = command[22:25]

        base_quat_start_index = self.base_qpos_start_idx + 3
        base_linvel_start_index = self.base_qvel_start_idx
        base_angvel_start_index = self.base_qvel_start_idx + 3

        base_quat = qpos[base_quat_start_index : base_quat_start_index + 4]
        inv_base_quat = self._quat_conjugate(base_quat)

        world_linvel = qvel[base_linvel_start_index : base_linvel_start_index + 3]
        base_linear_velocity = self._rotate_vec_by_quat(world_linvel, inv_base_quat)

        base_angular_velocity = qvel[base_angvel_start_index : base_angvel_start_index + 3]

        projected_gravity = self._rotate_vec_by_quat(np.array([0.0, 0.0, -1.0], dtype=qpos.dtype), inv_base_quat)

        joint_pos = qpos[self.leg_qpos_start_idx : self.leg_qpos_start_idx + 19] - self.default_joint_pos
        joint_vel = qvel[self.leg_qvel_start_idx : self.leg_qvel_start_idx + 19]
        joint_pos = joint_pos[self.mujoco_to_orbit_indices]
        joint_vel = joint_vel[self.mujoco_to_orbit_indices]

        parts = [
            base_linear_velocity,
            base_angular_velocity,
            projected_gravity,
            torso_vel_command,
            arm_command,
            leg_command,
            torso_pos_command,
            joint_pos,
            joint_vel,
            self.prev_policy_output.astype(qpos.dtype),
        ]
        observation = np.concatenate(parts, axis=0).astype(np.float32)
        return observation

    def set_state_indices_from_model(self, mj_model) -> None:
        try:
            import mujoco
        except Exception as exc:
            raise RuntimeError("mujoco must be installed to set state indices from model") from exc

        base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "spot/body")
        if base_id < 0:
            raise ValueError("Body 'spot/body' not found in model")
        base_joint_adr = int(mj_model.body_jntadr[base_id])
        self.base_qpos_start_idx = int(mj_model.jnt_qposadr[base_joint_adr])
        self.base_qvel_start_idx = int(mj_model.jnt_dofadr[base_joint_adr])

        first_leg_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "spot/front_left_hip")
        if first_leg_id < 0:
            raise ValueError("Body 'spot/front_left_hip' not found in model")
        leg_joint_adr = int(mj_model.body_jntadr[first_leg_id])
        self.leg_qpos_start_idx = int(mj_model.jnt_qposadr[leg_joint_adr])
        self.leg_qvel_start_idx = int(mj_model.jnt_dofadr[leg_joint_adr])

    def postprocess_policy_output_to_control(self, policy_out: np.ndarray, command: np.ndarray) -> np.ndarray:
        policy_out = np.asarray(policy_out).reshape(-1)[:12]
        self.prev_policy_output = policy_out.astype(np.float64, copy=False)
        control = np.zeros(19, dtype=np.float64)
        legs_orbit = 0.2 * policy_out.astype(np.float64)
        legs_mujoco = legs_orbit[self.orbit_to_mujoco_legs_indices]
        control[0:12] = self.default_joint_pos[0:12] + legs_mujoco
        arm_command = command[3:10].astype(np.float64)
        control[12:19] = arm_command
        leg_joint_command = command[10:22].astype(np.float64)
        if np.linalg.norm(leg_joint_command[0:3]) > 0:
            control[0:3] = leg_joint_command[0:3]
        elif np.linalg.norm(leg_joint_command[3:6]) > 0:
            control[3:6] = leg_joint_command[3:6]
        elif np.linalg.norm(leg_joint_command[6:9]) > 0:
            control[6:9] = leg_joint_command[6:9]
        elif np.linalg.norm(leg_joint_command[9:12]) > 0:
            control[9:12] = leg_joint_command[9:12]
        return control

    def forward(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        observation = self.build_observation_from_state(qpos, qvel, command)
        policy_out = self._infer_from_observation(observation)
        control = self.postprocess_policy_output_to_control(policy_out, command)
        return control

if __name__ == "__main__":
    model_path = "judo_cpp/policy/xinghao_policy_v1.onnx"
    policy = XinghaoPolicyWrapper(model_path)
    # Minimal demo with fabricated MuJoCo-like buffers and indices
    import mujoco
    from judo import MODEL_PATH
    spot_xml_path = str(MODEL_PATH / "xml/spot_locomotion.xml")
    spot_model = mujoco.MjModel.from_xml_path(spot_xml_path)
    spot_data = mujoco.MjData(spot_model)
    print(spot_data.qpos.shape)
    print(spot_data.qvel.shape)
    
    qpos = spot_data.qpos
    qvel = spot_data.qvel
    # Set unit quaternion at base (w=1) so rotations are well-defined
    qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    command = np.zeros(25, dtype=np.float64)

    ctrl_19 = policy(qpos, qvel, command)
    print(ctrl_19.shape)
    print(ctrl_19)
