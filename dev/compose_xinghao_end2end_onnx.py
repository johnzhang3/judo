import argparse
from pathlib import Path

import onnx
from onnx import compose
from onnx import version_converter
import onnxruntime as ort
import numpy as np


def replace_node_input_names(graph: onnx.GraphProto, old: str, new: str) -> None:
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new


def remove_graph_input(graph: onnx.GraphProto, name: str) -> None:
    inputs = list(graph.input)
    keep = [vi for vi in inputs if vi.name != name]
    del graph.input[:]  # clear
    graph.input.extend(keep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose base Xinghao ONNX with wrapper to make end-to-end model")
    parser.add_argument("--base", type=str, default="judo_cpp/policy/xinghao_policy_v1.onnx", help="Base policy ONNX path")
    parser.add_argument(
        "--wrapper", type=str, default="judo_cpp/policy/xinghao_policy_wrapper.onnx", help="Wrapper ONNX path"
    )
    parser.add_argument(
        "--out", type=str, default="judo_cpp/policy/xinghao_policy_end2end.onnx", help="Output end-to-end ONNX path"
    )
    args = parser.parse_args()

    base = onnx.load(args.base)
    wrapper = onnx.load(args.wrapper)

    # Align opset and IR versions
    target_opset = 17
    try:
        base = version_converter.convert_version(base, target_opset)
    except Exception:
        pass
    try:
        wrapper = version_converter.convert_version(wrapper, target_opset)
    except Exception:
        pass
    # Set consistent IR version (use higher of the two)
    common_ir = max(getattr(base, "ir_version", 7), getattr(wrapper, "ir_version", 7))
    base.ir_version = common_ir
    wrapper.ir_version = common_ir
    # Ensure opset imports contain ai.onnx target_opset
    def ensure_opset(model: onnx.ModelProto, domain: str, version: int) -> None:
        found = False
        for imp in model.opset_import:
            if imp.domain == domain:
                imp.version = version
                found = True
                break
        if not found:
            model.opset_import.extend([onnx.helper.make_operatorsetid(domain, version)])

    ensure_opset(base, "", target_opset)
    ensure_opset(wrapper, "", target_opset)

    # Identify IO names
    base_in = base.graph.input[0].name
    base_out = base.graph.output[0].name

    # Expect wrapper inputs: observation, policy_output; output: controls
    wrap_obs = wrapper.graph.input[0].name
    wrap_pol_in = wrapper.graph.input[1].name
    wrap_out = wrapper.graph.output[0].name

    # Merge models by wiring base policy output -> wrapper policy_output
    merged = compose.merge_models(
        base,
        wrapper,
        io_map=[(base_out, wrap_pol_in)],
        prefix1="base_",
        prefix2="wrap_",
    )

    # After prefixing, update names accordingly
    base_in_pref = f"base_{base_in}"
    base_out_pref = f"base_{base_out}"
    wrap_obs_pref = f"wrap_{wrap_obs}"
    wrap_out_pref = f"wrap_{wrap_out}"

    # Route wrapper's observation to base's input tensor to have a single external input
    replace_node_input_names(merged.graph, wrap_obs_pref, base_in_pref)
    remove_graph_input(merged.graph, wrap_obs_pref)

    # Expose only the wrapper controls as graph output
    del merged.graph.output[:]
    merged.graph.output.extend([
        onnx.helper.make_tensor_value_info(wrap_out_pref, onnx.TensorProto.FLOAT, None)
    ])

    # Force dynamic batch on the single input and output
    # Input
    if merged.graph.input:
        vi = merged.graph.input[0]
        t = vi.type.tensor_type
        if t.shape.dim and len(t.shape.dim) >= 1:
            t.shape.dim[0].dim_param = "batch"
            t.shape.dim[0].ClearField("dim_value")
    # Output
    if merged.graph.output:
        vo = merged.graph.output[0]
        t = vo.type.tensor_type
        if t.shape.dim and len(t.shape.dim) >= 1:
            t.shape.dim[0].dim_param = "batch"
            t.shape.dim[0].ClearField("dim_value")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(merged, out_path.as_posix())
    print(f"Saved end-to-end ONNX to {out_path}")

    # Quick runtime check: single input 'base_<base_in>' should exist
    sess = ort.InferenceSession(out_path.as_posix(), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print("Runtime input name:", input_name)
    obs_shape = sess.get_inputs()[0].shape
    obs_dim = obs_shape[1] if isinstance(obs_shape[1], int) else 84
    obs = np.random.randn(3, obs_dim).astype(np.float32)
    controls = sess.run(None, {input_name: obs})[0]
    print("controls shape:", controls.shape)


if __name__ == "__main__":
    main()


