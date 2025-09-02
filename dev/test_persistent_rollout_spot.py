import numpy as np
import mujoco

from judo import MODEL_PATH
from judo_cpp import persistent_onnx_policy_rollout


def main() -> None:
    # Load Spot locomotion model
    spot_xml = str((MODEL_PATH / "xml/spot_locomotion.xml").resolve())
    model = mujoco.MjModel.from_xml_path(spot_xml)

    # Create B=1 data
    data = mujoco.MjData(model)
    models = [model]
    datas = [data]

    # Initial state x0: (B, nq+nv)
    nq, nv = model.nq, model.nv
    x0 = np.zeros((1, nq + nv), dtype=np.double)
    # Set base quaternion to identity (w=1)
    x0[0, 3] = 1.0

    # Horizon and model path
    horizon = 10
    onnx_model_path = "judo_cpp/policy/xinghao_policy_wrapped_torch.onnx"

    # Commands: provide zeros broadcast, or shape (B, horizon*25)
    commands = np.zeros((16, horizon, 19), dtype=np.double)

    # Run rollout
    states, sensors = persistent_onnx_policy_rollout(models, datas, x0, horizon, onnx_model_path, commands)

    print("states:", states.shape)
    print("sensors:", sensors.shape)

    print("states:", states)

if __name__ == "__main__":
    main()


