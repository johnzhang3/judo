import mujoco
import numpy as np
import onnxruntime as ort
from judo import MODEL_PATH
from judo.tasks.spot.spot_constants import (
    ARM_STOWED_POS,
    STANDING_HEIGHT_CMD,
    STANDING_HEIGHT,
    LEGS_STANDING_POS,
)
import mujoco.viewer as viewer
onnx_path = "judo_cpp/policy/xinghao_policy_v1.onnx"

# Mappings and defaults (mirror C++)
ORBIT_TO_MJ_LEGS = np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], dtype=np.int32)
MJ_TO_ORBIT_19   = np.array([1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 0, 5, 10, 15, 16, 17, 18], dtype=np.int32)
DEFAULT_JOINT_POS = np.array([
    0.12, 0.5, -1.0,
    -0.12, 0.5, -1.0,
    0.12, 0.5, -1.0,
    -0.12, 0.5, -1.0,
    0.0, -0.9, 1.8,
    0.0, -0.9, 0.0, -1.54
], dtype=np.float64)

# Lazy global ONNX session
_onnx_sess = None
_onnx_in_name = None
_onnx_out_name = None


def _get_indices(model: mujoco.MjModel):
    # Prefer robust detection of the free joint (base)
    free_joint_idx = None
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            free_joint_idx = j
            break
    if free_joint_idx is None:
        # Fallback to joint 0
        free_joint_idx = 0
    base_qpos_start = int(model.jnt_qposadr[free_joint_idx])
    base_qvel_start = int(model.jnt_dofadr[free_joint_idx])

    # After the free joint come the actuated joints. For Spot, we assume 19 joints follow.
    leg_qpos_start = base_qpos_start + 7
    leg_qvel_start = base_qvel_start + 6
    return base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start


def _ensure_session():
    global _onnx_sess, _onnx_in_name, _onnx_out_name
    if _onnx_sess is None:
        _onnx_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        _onnx_in_name = _onnx_sess.get_inputs()[0].name
        _onnx_out_name = _onnx_sess.get_outputs()[0].name


def cmd_to_sim_ctrl(model: mujoco.MjModel,
                    data: mujoco.MjData,
                    prev_policy_output: np.ndarray,
                    cmd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build observation, run ONNX policy, map to 19-d control.

    Returns (ctrl[19], policy_out[12]).
    """
    _ensure_session()

    base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start = _get_indices(model)

    # Build observation (84)
    obs = np.zeros(84, dtype=np.float32)
    off = 0

    invq = np.zeros(4, dtype=np.float64)
    quat = np.array(data.qpos[base_qpos_start + 3: base_qpos_start + 7], dtype=np.float64, copy=True)
    mujoco.mju_negQuat(invq, quat)

    blin = np.zeros(3, dtype=np.float64)
    linvel = np.array(data.qvel[base_qvel_start: base_qvel_start + 3], dtype=np.float64, copy=True)
    mujoco.mju_rotVecQuat(blin, linvel, invq)
    obs[off:off + 3] = blin; off += 3

    obs[off:off + 3] = data.qvel[base_qvel_start + 3: base_qvel_start + 6]
    off += 3

    gvec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    mujoco.mju_rotVecQuat(gvec, gvec.copy(), invq)
    obs[off:off + 3] = gvec; off += 3

    # Command segments
    obs[off:off + 3] = cmd[0:3]; off += 3
    obs[off:off + 7] = cmd[3:10]; off += 7
    obs[off:off + 12] = cmd[10:22]; off += 12
    obs[off:off + 3] = cmd[22:25]; off += 3

    # Joint pos/vel in orbit order with default offset
    jpos_raw = data.qpos[leg_qpos_start: leg_qpos_start + 19] - DEFAULT_JOINT_POS
    jvel_raw = data.qvel[leg_qvel_start: leg_qvel_start + 19]
    jpos_orbit = jpos_raw[MJ_TO_ORBIT_19]
    jvel_orbit = jvel_raw[MJ_TO_ORBIT_19]
    obs[off:off + 19] = jpos_orbit; off += 19
    obs[off:off + 19] = jvel_orbit; off += 19

    # Previous policy output (12)
    if prev_policy_output is None or prev_policy_output.size != 12:
        prev_policy_output = np.zeros(12, dtype=np.float32)
    obs[off:off + 12] = prev_policy_output.astype(np.float32); off += 12

    # Run policy
    policy_out = _onnx_sess.run([_onnx_out_name], {_onnx_in_name: obs.reshape(1, -1)})[0].reshape(-1)
    policy_out = policy_out.astype(np.float32)

    # Map to control (19)
    ctrl = np.zeros(19, dtype=np.float64)
    legs_orbit = 0.2 * policy_out[:12].astype(np.float64)
    legs_mj = legs_orbit[ORBIT_TO_MJ_LEGS] + DEFAULT_JOINT_POS[:12]
    ctrl[:12] = legs_mj

    # Arm from command
    ctrl[12:19] = cmd[3:10]

    # Leg override
    leg_cmd = cmd[10:22]

    def _norm3(v):
        return np.abs(v[0]) + np.abs(v[1]) + np.abs(v[2])
    if _norm3(leg_cmd[0:3]) > 0.0:
        ctrl[0:3] = leg_cmd[0:3]
    elif _norm3(leg_cmd[3:6]) > 0.0:
        ctrl[3:6] = leg_cmd[3:6]
    elif _norm3(leg_cmd[6:9]) > 0.0:
        ctrl[6:9] = leg_cmd[6:9]
    elif _norm3(leg_cmd[9:12]) > 0.0:
        ctrl[9:12] = leg_cmd[9:12]

    return ctrl, policy_out


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH / "xml/spot_locomotion.xml"))
    data = mujoco.MjData(model)

    default_policy_command = np.array(
        [0, 0, 0] + list(ARM_STOWED_POS) + [0] * 12 + [0, 0, STANDING_HEIGHT_CMD],
        dtype=np.float64,
    )
    default_pose = np.array(
            [
                *np.random.randn(2),
                STANDING_HEIGHT,
                1,
                0,
                0,
                0,
                *LEGS_STANDING_POS,
                *ARM_STOWED_POS,
            ]
        )
    data.qpos[:] = default_pose
    data.qvel[:] = np.zeros_like(data.qvel)
    mujoco.mj_forward(model, data)
    prev_po = np.zeros(12, dtype=np.float32)

    # Optional viewer
    use_viewer = True

    T = 50
    if use_viewer:
        try:
            with viewer.launch_passive(model, data) as v:
                for _ in range(T):
                    ctrl, prev_po = cmd_to_sim_ctrl(model, data, prev_po, default_policy_command)
                    data.ctrl[:] = ctrl
                    mujoco.mj_step(model, data)
                    v.sync()
        except RuntimeError as e:
            # macOS requires running under mjpython for interactive viewer
            print(f"Viewer unavailable ({e}). Falling back to headless simulation.\n"
                  f"Tip: on macOS, run with 'mjpython dev/debug_spot_policy_sim.py' to enable the viewer.")
            for _ in range(T):
                ctrl, prev_po = cmd_to_sim_ctrl(model, data, prev_po, default_policy_command)
                data.ctrl[:] = ctrl
                print(prev_po)
                print(data.qpos[2])
                mujoco.mj_step(model, data)
    else:
        for _ in range(T):
            ctrl, prev_po = cmd_to_sim_ctrl(model, data, prev_po, default_policy_command)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            


if __name__ == "__main__":
    main()