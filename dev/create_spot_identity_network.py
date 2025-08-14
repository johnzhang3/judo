# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import mujoco
import numpy as np
import torch
from torch import nn

from judo import MODEL_PATH


def create_spot_identity_network() -> tuple[str, int]:
    """Create an identity network for the Spot locomotion model."""
    # Load the Spot model to get the state dimensions
    XML_PATH = str(MODEL_PATH / "xml/spot_locomotion.xml")
    model = mujoco.MjModel.from_xml_path(XML_PATH)

    nq = model.nq  # generalized coordinates
    nv = model.nv  # generalized velocities
    nstate = nq + nv  # total state dimension

    print("Spot model dimensions:")
    print(f"  nq (positions): {nq}")
    print(f"  nv (velocities): {nv}")
    print(f"  nstate (total): {nstate}")

    class IdentityNetwork(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super(IdentityNetwork, self).__init__()
            self.input_dim = input_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Perfect identity: output = input
            return x

    # Create the identity network
    net = IdentityNetwork(nstate)

    # Test with dummy input
    dummy_input = torch.randn(1, nstate)
    output = net(dummy_input)

    print("\nNetwork test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Identity check: {torch.allclose(dummy_input, output)}")

    # Export to ONNX
    onnx_path = "identity_network_spot.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"\nONNX model exported to: {onnx_path}")

    # Verify the ONNX model
    import onnxruntime as ort

    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)

    # Get input/output info
    input_info = ort_session.get_inputs()[0]
    output_info = ort_session.get_outputs()[0]

    print("\nONNX model verification:")
    print(f"  Input name: {input_info.name}")
    print(f"  Input shape: {input_info.shape}")
    print(f"  Output name: {output_info.name}")
    print(f"  Output shape: {output_info.shape}")

    # Test ONNX inference
    test_input = np.random.randn(1, nstate).astype(np.float32)
    onnx_output = ort_session.run(None, {input_info.name: test_input})[0]

    print("\nONNX inference test:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {onnx_output.shape}")
    print(f"  Identity check: {np.allclose(test_input, onnx_output, atol=1e-6)}")
    print(f"  Max difference: {np.max(np.abs(test_input - onnx_output))}")

    return onnx_path, nstate


if __name__ == "__main__":
    onnx_path, nstate = create_spot_identity_network()
    print(f"\n✅ Successfully created Spot identity network: {onnx_path}")
    print(f"✅ State dimension: {nstate}")
