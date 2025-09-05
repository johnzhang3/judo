import argparse
import torch
from torch.onnx import export

from dev.torch_policy_wrapper import TorchXinghaoPolicyWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Torch-based Xinghao wrapper to ONNX")
    parser.add_argument("--base", type=str, default="judo_cpp/policy/xinghao_policy_v1.onnx")
    parser.add_argument("--out", type=str, default="judo_cpp/policy/xinghao_policy_wrapped_torch.onnx")
    args = parser.parse_args()

    model = TorchXinghaoPolicyWrapper(args.base)
    model.eval()

    # Example static inputs (match sizes): qpos(26), qvel(25), command(25), prev_policy(12)
    qpos = torch.zeros(26, dtype=torch.float32)
    qpos[3] = 1.0
    qvel = torch.zeros(25, dtype=torch.float32)
    command = torch.zeros(25, dtype=torch.float32)
    prev = torch.zeros(12, dtype=torch.float32)

    export(
        model,
        (qpos, qvel, command, prev),
        args.out,
        input_names=["qpos", "qvel", "command", "prev_policy"],
        output_names=["control", "policy_out"],
        opset_version=17,
        dynamic_axes=None,
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()


