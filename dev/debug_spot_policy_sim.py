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
from judo.tasks.spot.spot_constants import (
    STANDING_POS_RL,
    isaac_to_mujoco,
    mujoco_to_isaac,
)
import time
import os

import mediapy as media

onnx_path = "judo_cpp/policy/xinghao_policy_v1.onnx"

# Action scale parameter
ACTION_SCALE = 0.2

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
    gvec_rotated = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(gvec_rotated, gvec, invq)
    obs[off:off + 3] = gvec_rotated; off += 3

    # Command segments
    obs[off:off + 3] = cmd[0:3]; off += 3
    obs[off:off + 7] = cmd[3:10]; off += 7
    obs[off:off + 12] = cmd[10:22]; off += 12
    obs[off:off + 3] = cmd[22:25]; off += 3

    # Joint pos/vel with proper coordinate transformation
    jpos_raw = data.qpos[leg_qpos_start: leg_qpos_start + 19] - STANDING_POS_RL
    jvel_raw = data.qvel[leg_qvel_start: leg_qvel_start + 19]
    jpos_isaac = mujoco_to_isaac(jpos_raw)
    jvel_isaac = mujoco_to_isaac(jvel_raw)
    obs[off:off + 19] = jpos_isaac; off += 19
    obs[off:off + 19] = jvel_isaac; off += 19

    # Previous policy output (12)
    if prev_policy_output is None or prev_policy_output.size != 12:
        prev_policy_output = np.zeros(12, dtype=np.float32)
    obs[off:off + 12] = prev_policy_output.astype(np.float32); off += 12

    # Run policy
    policy_out = _onnx_sess.run([_onnx_out_name], {_onnx_in_name: obs.reshape(1, -1)})[0].reshape(-1)
    policy_out = policy_out.astype(np.float32)

    # Map to control (19)
    ctrl = np.zeros(19, dtype=np.float64)
    target_leg = isaac_to_mujoco(policy_out[:12]) * ACTION_SCALE + STANDING_POS_RL[:12]
    ctrl[:12] = target_leg

    # Arm from command
    ctrl[12:19] = cmd[3:10]

    # Leg override
    leg_cmd = cmd[10:22]

    # Leg override using element-wise non-zero check (matches reference)
    if np.any(leg_cmd[0:3] != 0):  # FL
        ctrl[0:3] = leg_cmd[0:3]
    elif np.any(leg_cmd[3:6] != 0):  # FR
        ctrl[3:6] = leg_cmd[3:6]
    elif np.any(leg_cmd[6:9] != 0):  # HL
        ctrl[6:9] = leg_cmd[6:9]
    elif np.any(leg_cmd[9:12] != 0):  # HR
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
                0, 0,
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

    # Headless render to video via mujoco.Renderer (+ fallback to imageio if mediapy unavailable)
    duration_seconds = 10.0
    target_fps = 30
    steps_per_frame = max(1, int(round(1.0 / (model.opt.timestep * target_fps))))
    total_frames = int(duration_seconds * target_fps)

    # Use model's configured offscreen framebuffer size to avoid allocation errors
    try:
        vis_global = model.vis.global_
    except AttributeError:
        vis_global = getattr(model.vis, 'global', None)
    offw = int(getattr(vis_global, 'offwidth', 0) or 0) if vis_global is not None else 0
    offh = int(getattr(vis_global, 'offheight', 0) or 0) if vis_global is not None else 0
    fb_w = offw if offw > 0 else 640
    fb_h = offh if offh > 0 else 480
    # Note: mujoco.Renderer expects (height, width)
    renderer = mujoco.Renderer(model, fb_h, fb_w)
    frames = []

    for _ in range(total_frames):
        # Run several sim steps per frame to hit target_fps
        for _ in range(steps_per_frame):
            ctrl, prev_po = cmd_to_sim_ctrl(model, data, prev_po, default_policy_command)
            # Use the policy output instead of hardcoded positions
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

        renderer.update_scene(data)
        rgb = renderer.render()
        frames.append(rgb)


    renderer.close()

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "spot_policy_debug.mp4")
    if media is not None:
        media.write_video(out_path, frames, fps=target_fps)
        print(f"Saved video to {out_path} ({len(frames)} frames @ {target_fps} fps)")
    else:
        try:
            import imageio.v2 as imageio
            imageio.mimsave(out_path, frames, fps=target_fps)
            print(f"Saved video to {out_path} via imageio ({len(frames)} frames @ {target_fps} fps)")
        except Exception as e:
            print("Unable to save video: install mediapy or imageio.")
            raise e
            


if __name__ == "__main__":
    main()