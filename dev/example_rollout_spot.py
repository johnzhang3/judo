import numpy as np
import mujoco

from judo import MODEL_PATH
from judo_cpp import persistent_onnx_policy_rollout
from copy import deepcopy

def main() -> None:
    # Load Spot locomotion model
    spot_xml = str((MODEL_PATH / "xml/spot_locomotion.xml").resolve())
    model = mujoco.MjModel.from_xml_path(spot_xml)

    # Create B=1 data
    data = mujoco.MjData(model)
    # copy the model and data 16 times
    models = [deepcopy(model) for _ in range(16)]
    datas = [deepcopy(data) for _ in range(16)]

    # Initial state x0: (B, nq+nv)
    nq, nv = model.nq, model.nv
    x0 = np.zeros((16, nq + nv), dtype=np.double)
    # Set base quaternion to identity (w=1)
    x0[:, 3] = 1.0

    # Horizon and model path
    horizon = 10
    onnx_model_path = "judo_cpp/policy/xinghao_policy_wrapped_torch.onnx"

    # Commands: provide zeros broadcast, or shape (B, horizon*25)
    commands = np.zeros((16, horizon, 25), dtype=np.double)

    # Run rollout
    states, sensors = persistent_onnx_policy_rollout(models, datas, x0, horizon, onnx_model_path, commands)

    print("states:", states.shape)
    print("sensors:", sensors.shape)

    print("states:", states)

if __name__ == "__main__":
    main()


