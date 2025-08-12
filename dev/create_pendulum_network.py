# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from create_identity_network import create_identity_onnx

# Create identity network for pendulum (nq=1, nv=1, so nstate=2)
create_identity_onnx(2, "identity_network_pendulum.onnx")
print("Created pendulum identity network!")
