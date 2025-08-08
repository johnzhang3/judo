#!/usr/bin/env python3
# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Create a simple identity network and save it as ONNX for testing."""

import numpy as np
import torch
from torch import nn


class IdentityNetwork(nn.Module):
    """A simple identity network that passes input through unchanged."""

    def __init__(self, input_size: int) -> None:
        """Initialize the identity network.

        Args:
            input_size: Size of the input layer.
        """
        super().__init__()
        self.input_size = input_size
        # Create a linear layer with identity weights
        self.linear = nn.Linear(input_size, input_size, bias=False)
        # Set weights to identity matrix
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(input_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the identity network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor (same as input for identity network).
        """
        return self.linear(x)


def create_identity_onnx(input_size: int = 10, output_path: str = "identity_network.onnx") -> str:
    """Create and save an identity network as ONNX."""
    # Create the network
    model = IdentityNetwork(input_size)
    model.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(1, input_size)

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Identity network saved to {output_path}")

    # Test the ONNX model
    import onnxruntime as ort

    session = ort.InferenceSession(output_path)

    # Test with random input
    test_input = np.random.randn(3, input_size).astype(np.float32)
    result = session.run(None, {"input": test_input})

    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {result[0].shape}")
    print(f"Identity test passed: {np.allclose(test_input, result[0])}")

    return output_path


if __name__ == "__main__":
    # Create identity network for typical state size (position + velocity)
    # For a simple system, let's use 6 DOF (nq=3, nv=3)
    state_size = 6
    onnx_path = create_identity_onnx(state_size, "identity_network.onnx")
    print(f"ONNX model created at: {onnx_path}")
