import argparse
import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Xinghao wrapped ONNX inference")
    parser.add_argument("--model", type=str, default="judo_cpp/policy/xinghao_policy_wrapped.onnx")
    args = parser.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in sess.get_inputs()}
    assert list(inputs.keys()) == ["qpos", "qvel", "command", "prev_policy"], f"Unexpected inputs: {list(inputs.keys())}"

    qpos = np.zeros((26,), dtype=np.float32)
    qpos[3] = 1.0  # w of unit quaternion
    qvel = np.zeros((25,), dtype=np.float32)
    command = np.zeros((25,), dtype=np.float32)
    prev_policy = np.zeros((12,), dtype=np.float32)

    output = sess.run(None, {
        "qpos": qpos,
        "qvel": qvel,
        "command": command,
        "prev_policy": prev_policy,
    })[0]

    print("Output shape:", output.shape)
    print("First row preview:", output[0, :10])


if __name__ == "__main__":
    main()


