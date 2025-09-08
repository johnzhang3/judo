import argparse
from typing import Tuple, cast
import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Xinghao wrapped ONNX inference")
    parser.add_argument("--model", type=str, default="judo_cpp/policy/xinghao_policy_wrapped_torch.onnx")
    args = parser.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    assert input_names == ["qpos", "qvel", "command", "prev_policy"], f"Unexpected inputs: {input_names}"
    output_names = [o.name for o in sess.get_outputs()]
    assert output_names == ["control", "policy_out"], f"Unexpected outputs: {output_names}"

    qpos = np.zeros((26,), dtype=np.float32)
    qpos[3] = 1.0  # w of unit quaternion
    qvel = np.zeros((25,), dtype=np.float32)
    command = np.zeros((25,), dtype=np.float32)
    prev_policy = np.zeros((12,), dtype=np.float32)

    control, policy_out = cast(Tuple[np.ndarray, np.ndarray], sess.run(output_names, {
        "qpos": qpos,
        "qvel": qvel,
        "command": command,
        "prev_policy": prev_policy,
    }))

    print("Control shape:", control.shape)
    print("Policy out shape:", policy_out.shape)
    print("Control[0, :10]:", control[0, :10])
    print("Policy out[0, :10]:", policy_out[0, :10])


if __name__ == "__main__":
    main()


