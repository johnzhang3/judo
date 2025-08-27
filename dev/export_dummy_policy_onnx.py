# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import argparse
from pathlib import Path

import torch
from torch import nn


def build_dummy_policy(input_dim: int, action_dim: int) -> nn.Module:
    class ZeroPolicy(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            # minimal MLP stub; we ignore inputs and return zeros, but keep a head for shape clarity
            self.net = nn.Sequential(
                nn.Linear(in_dim, max(4, min(64, in_dim))),
                nn.ReLU(),
                nn.Linear(max(4, min(64, in_dim)), out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, input_dim]
            B = x.shape[0]
            return torch.zeros(B, action_dim, dtype=x.dtype, device=x.device)

    return ZeroPolicy(input_dim, action_dim).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a dummy zero-action policy to ONNX")
    parser.add_argument("--state_dim", type=int, required=True)
    parser.add_argument("--action_dim", type=int, required=True)
    parser.add_argument("--state_history", type=int, default=10)
    parser.add_argument("--action_history", type=int, default=5)
    parser.add_argument("--additional_input_dim", type=int, default=0)
    parser.add_argument("--out", type=str, default="dev/dummy_policy.onnx")
    args = parser.parse_args()

    input_dim = args.state_dim * args.state_history + args.action_dim * args.action_history + args.additional_input_dim

    model = build_dummy_policy(input_dim, args.action_dim)

    # Dummy input with dynamic batch dimension
    dummy_batch = 2
    dummy_input = torch.randn(dummy_batch, input_dim, dtype=torch.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        out_path.as_posix(),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["actions"],
        dynamic_axes={"input": {0: "batch"}, "actions": {0: "batch"}},
    )

    print(f"Exported dummy zero-action policy ONNX to {out_path}")


if __name__ == "__main__":
    main()
