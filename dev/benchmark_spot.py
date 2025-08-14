# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import os
import time
from copy import deepcopy
from typing import Any

import mujoco
import numpy as np
from mujoco.rollout import Rollout

from judo import MODEL_PATH

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Will create simple ONNX identity network.")

try:
    import importlib.util

    judo_cpp_spec = importlib.util.find_spec("judo_cpp")
    JUDO_CPP_AVAILABLE = judo_cpp_spec is not None
except ImportError:
    JUDO_CPP_AVAILABLE = False
    print("Warning: judo_cpp not available. ONNX interleaved rollouts will be skipped.")

XML_PATH = str(MODEL_PATH / "xml/spot_locomotion.xml")

if TORCH_AVAILABLE:

    class IdentityNetwork(nn.Module):
        """A simple identity network that passes input through unchanged."""

        def __init__(self, input_size: int) -> None:
            """Initialize the identity network."""
            super().__init__()
            self.input_size = input_size
            # Create a linear layer with identity weights
            self.linear = nn.Linear(input_size, input_size, bias=False)
            # Set weights to identity matrix
            with torch.no_grad():
                self.linear.weight.copy_(torch.eye(input_size))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through the identity network."""
            return self.linear(x)


def create_spot_identity_network(model: mujoco.MjModel) -> str:
    """Create an identity network for the Spot locomotion model and save as ONNX."""
    state_size = model.nq + model.nv  # position + velocity
    output_path = "identity_network_spot.onnx"

    print(f"Creating identity network for Spot (state_size={state_size})...")

    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot create ONNX identity network.")
        print("Please install PyTorch or use existing identity network.")
        return output_path

    # Create the network
    network = IdentityNetwork(state_size)
    network.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(1, state_size)

    # Export to ONNX
    torch.onnx.export(
        network,
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
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)

        # Test with random input
        test_input = np.random.randn(3, state_size).astype(np.float32)
        result = session.run(None, {"input": test_input})

        print(f"Test input shape: {test_input.shape}")
        print(f"Test output shape: {result[0].shape}")
        print(f"Identity test passed: {np.allclose(test_input, result[0])}")
    except ImportError:
        print("onnxruntime not available for testing, but ONNX file created.")

    return output_path


def benchmark_native_rollout(
    model: Any,  # noqa: ANN401
    x0: np.ndarray,
    controls: np.ndarray,
    num_threads: int,
    num_trials: int,
    simple: bool,
) -> dict[str, Any]:
    """Benchmark native MuJoCo rollout."""
    batch_size = num_threads

    # Create model/data pairs for threaded rollout
    models = [deepcopy(model) for _ in range(batch_size)]
    datas = [mujoco.MjData(m) for m in models]

    # Setup MuJoCo rollout directly
    rollout_obj = Rollout(nthread=num_threads)

    # Prepare initial state with time (MuJoCo rollout expects time as first element)
    x0_batched = np.tile(x0, (batch_size, 1))
    full_states = np.concatenate([time.time() * np.ones((batch_size, 1)), x0_batched], axis=-1)
    # Warmup
    if not simple:
        print("  Warming up...")
    for _ in range(3):
        rollout_obj.rollout(models, datas, full_states, controls)

    # Benchmark
    if not simple:
        print("  Running benchmark...")
    times = []

    for trial in range(num_trials):
        start_time = time.time()
        states_raw, sensors = rollout_obj.rollout(models, datas, full_states, controls)
        end_time = time.time()

        # Remove time from states (first column)
        states = np.array(states_raw)[..., 1:]
        sensors = np.array(sensors)

        elapsed = end_time - start_time
        times.append(elapsed)

        if trial == 0:  # Verify output shapes on first trial
            if not simple:
                print(f"  Output shapes: states={states.shape}, sensors={sensors.shape}")

    # Clean up rollout object
    rollout_obj.close()

    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = batch_size * controls.shape[1] / mean_time

    result = {
        "type": "native",
        "mean": mean_time,
        "std": std_time,
        "throughput": throughput,
        "batch_size": batch_size,
        "num_threads": num_threads,
    }

    print(f"  Mean time: {mean_time:.4f} ± {std_time:.4f}s")
    print(f"  Throughput: {throughput:.1f} steps/second")

    return result


def benchmark_cpp_rollout(
    model: Any,  # noqa: ANN401
    x0: np.ndarray,
    controls: np.ndarray,
    num_threads: int,
    trials: int = 5,
) -> dict[str, Any]:
    """Benchmark pure C++ rollout without any Python or ONNX overhead."""
    # Import here to avoid dependency issues if not available
    try:
        import judo_cpp
    except ImportError:
        return {"error": True, "msg": "judo_cpp not available"}

    import mujoco
    import numpy as np

    # Create models and data for each thread
    models = [deepcopy(model) for _ in range(num_threads)]
    datas = [mujoco.MjData(m) for m in models]

    # Create test data
    x0_batched = np.tile(x0, (num_threads, 1))  # Shape: (num_threads, nstate)

    times = []
    for _ in range(trials):
        start_time = time.time()

        # Run pure C++ rollout with batched initial states
        states, sensors, inputs = judo_cpp.pure_cpp_rollout(models, datas, x0_batched, controls)

        end_time = time.time()
        times.append(end_time - start_time)
        states = np.array(states)
        print(f"C++ Rollout time: {times[-1]:.4f} seconds")

    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = (num_threads * controls.shape[1]) / mean_time

    return {"mean": mean_time, "std": std_time, "throughput": throughput, "times": times, "error": False}


def benchmark_onnx_rollout(
    model_path: str, num_threads: int, num_timesteps: int, trials: int = 5, inference_frequency: int = 1
) -> dict[str, Any]:
    """Benchmark ONNX interleaved rollout."""
    # Import here to avoid dependency issues if not available
    try:
        import judo_cpp
    except ImportError:
        return {"error": True, "msg": "judo_cpp not available"}

    import mujoco
    import numpy as np

    # Load the model once
    model = mujoco.MjModel.from_xml_path(model_path)
    nq, nv, nu = model.nq, model.nv, model.nu
    nstate = nq + nv

    # Create models and data for each thread
    models = [model] * num_threads
    datas = [mujoco.MjData(model) for _ in range(num_threads)]

    # Create test data
    x0_single = np.zeros(nstate)
    x0_batched = np.tile(x0_single, (num_threads, 1))  # Shape: (num_threads, nstate)
    controls = np.random.randn(num_threads, num_timesteps, nu) * 0.1

    # ONNX model path for Spot
    onnx_path = "identity_network_spot.onnx"
    if not os.path.exists(onnx_path):
        return {"error": True, "msg": f"ONNX model not found: {onnx_path}"}

    times = []
    for _ in range(trials):
        start_time = time.time()

        # Run ONNX interleaved rollout with batched initial states
        states, sensors, inputs, inferences = judo_cpp.onnx_interleave_rollout(
            models, datas, x0_batched, controls, onnx_path, inference_frequency
        )

        end_time = time.time()
        times.append(end_time - start_time)

    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = (num_threads * num_timesteps) / mean_time

    return {"mean": mean_time, "std": std_time, "throughput": throughput, "times": times, "error": False}


def print_comparison_summary(
    all_results: dict[str, Any], thread_counts: list[int], inference_frequencies: list[int], compare_onnx: bool
) -> None:
    """Print a comprehensive comparison summary."""
    print(f"\n{'=' * 80}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'=' * 80}")

    # Header
    if compare_onnx:
        print(f"{'Threads':<8} {'Type':<15} {'Freq':<6} {'Time (ms)':<12} {'Throughput':<15} {'vs Native':<10}")
        print(f"{'-' * 80}")
    else:
        print(f"{'Threads':<8} {'Type':<15} {'Time (ms)':<12} {'Throughput':<15}")
        print(f"{'-' * 50}")

    for num_threads in thread_counts:
        native_key = f"native_{num_threads}"
        cpp_key = f"cpp_{num_threads}"
        native_result = all_results.get(native_key)
        cpp_result = all_results.get(cpp_key)

        if native_result:
            # Print native result
            if compare_onnx:
                print(
                    f"{num_threads:<8} {'Native (Py)':<15} {'-':<6} "
                    f"{native_result['mean'] * 1000:.2f} ± {native_result['std'] * 1000:.2f} "
                    f"{native_result['throughput']:.1f} steps/s   {'baseline':<10}"
                )
            else:
                print(
                    f"{num_threads:<8} {'Native (Py)':<15} "
                    f"{native_result['mean'] * 1000:.2f} ± {native_result['std'] * 1000:.2f} "
                    f"{native_result['throughput']:.1f} steps/s"
                )

        if cpp_result and not cpp_result.get("error", False):
            # Print C++ result with comparison to native
            if native_result:
                overhead = ((cpp_result["mean"] - native_result["mean"]) / native_result["mean"]) * 100
                comparison = f"{overhead:+.1f}%"
            else:
                comparison = "N/A"

            if compare_onnx:
                print(
                    f"{'':<8} {'Pure C++':<15} {'-':<6} "
                    f"{cpp_result['mean'] * 1000:.2f} ± {cpp_result['std'] * 1000:.2f} "
                    f"{cpp_result['throughput']:.1f} steps/s   {comparison:<10}"
                )
            else:
                print(
                    f"{num_threads:<8} {'Pure C++':<15} "
                    f"{cpp_result['mean'] * 1000:.2f} ± {cpp_result['std'] * 1000:.2f} "
                    f"{cpp_result['throughput']:.1f} steps/s"
                )
        elif cpp_result and cpp_result.get("error", False):
            if compare_onnx:
                print(f"{'':<8} {'Pure C++':<15} {'-':<6} {'ERROR':<12} {'N/A':<15} {'N/A':<10}")
            else:
                print(f"{num_threads:<8} {'Pure C++':<15} {'ERROR':<12} {'N/A':<15}")

        if compare_onnx and native_result:
            # Print ONNX results
            for freq in inference_frequencies:
                onnx_key = f"onnx_{num_threads}_freq{freq}"
                onnx_result = all_results.get(onnx_key)

                if onnx_result and not onnx_result.get("error", False):
                    overhead = ((onnx_result["mean"] - native_result["mean"]) / native_result["mean"]) * 100
                    print(
                        f"{'':<8} {'ONNX':<15} {freq:<6} "
                        f"{onnx_result['mean'] * 1000:.2f} ± {onnx_result['std'] * 1000:.2f} "
                        f"{onnx_result['throughput']:.1f} steps/s   {overhead:+.1f}%"
                    )
                elif onnx_result and onnx_result.get("error", False):
                    print(f"{'':<8} {'ONNX':<15} {freq:<6} {'ERROR':<12} {'N/A':<15} {'N/A':<10}")

        if num_threads != thread_counts[-1]:
            print()  # Add spacing between thread counts


def benchmark_threaded_rollout(simple: bool = False, compare_onnx: bool = True) -> None:
    """Benchmark MuJoCo threaded rollout vs ONNX interleaved rollout on spot_locomotion model.

    Args:
        simple: If True, run a quick benchmark with single thread count.
                If False, run comprehensive benchmark across multiple thread counts.
        compare_onnx: If True, also benchmark ONNX interleaved rollouts for comparison.
    """
    print("=" * 80)
    print("Benchmarking MuJoCo Threaded Rollout vs ONNX Interleaved - Spot Locomotion")
    print("=" * 80)

    # Load model
    model_path = XML_PATH
    model = mujoco.MjModel.from_xml_path(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Model dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}, nsensordata={model.nsensordata}")

    # Create or check for identity network
    onnx_path = "identity_network_spot.onnx"
    if compare_onnx and JUDO_CPP_AVAILABLE:
        if not os.path.exists(onnx_path):
            print("\nCreating identity network for Spot...")
            onnx_path = create_spot_identity_network(model)
        else:
            print(f"\nUsing existing identity network: {onnx_path}")
    elif compare_onnx and not JUDO_CPP_AVAILABLE:
        print("\nWarning: judo_cpp not available. Skipping ONNX comparison.")
        compare_onnx = False

    # Test parameters
    if simple:
        thread_counts = [4]  # Just test 4 threads for quick benchmark
        num_trials = 5
        inference_frequencies = [1, 5] if compare_onnx else []
    else:
        thread_counts = [64]  # Comprehensive test (reduced for comparison)
        num_trials = 10
        inference_frequencies = [1, 2, 5, 10] if compare_onnx else []

    time_steps = 100

    # Create initial state (neutral/home position)
    x0 = np.zeros(model.nq + model.nv)
    # Set qpos to home position (mujoco resets to this by default)
    x0[: model.nq] = model.qpos0

    print("\nBenchmark parameters:")
    print(f"  Mode: {'Simple' if simple else 'Comprehensive'}")
    print(f"  Time steps: {time_steps}")
    print(f"  Trials per config: {num_trials}")
    print(f"  Thread counts: {thread_counts}")
    print(f"  ONNX comparison: {compare_onnx}")
    if compare_onnx:
        print(f"  Inference frequencies: {inference_frequencies}")

    all_results = {}

    for num_threads in thread_counts:
        print(f"\n{'=' * 60}")
        print(f"Testing with {num_threads} threads")
        print(f"{'=' * 60}")

        batch_size = num_threads  # batch_size must equal num_threads for mujoco rollout

        # Create random control inputs
        controls = np.random.randn(batch_size, time_steps, model.nu) * 0.1

        # 1. Native MuJoCo Rollout Benchmark
        print("\n--- Native MuJoCo Rollout ---")
        native_results = benchmark_native_rollout(model, x0, controls, num_threads, num_trials, simple)
        all_results[f"native_{num_threads}"] = native_results

        # 2. Pure C++ Rollout Benchmark
        if JUDO_CPP_AVAILABLE:
            print("\n--- Pure C++ Rollout ---")
            cpp_results = benchmark_cpp_rollout(model, x0, controls, num_threads, num_trials)
            all_results[f"cpp_{num_threads}"] = cpp_results

        # 3. ONNX Interleaved Rollout Benchmark
        # if compare_onnx:
        #     for inference_freq in inference_frequencies:
        #         print(f"\n--- ONNX Interleaved (freq={inference_freq}) ---")
        #         onnx_results = benchmark_onnx_rollout(
        #             model_path, num_threads, time_steps, num_trials, inference_freq
        #         )
        #         all_results[f"onnx_{num_threads}_freq{inference_freq}"] = onnx_results

    # Print comparison summary
    print_comparison_summary(all_results, thread_counts, inference_frequencies if compare_onnx else [], compare_onnx)

    print(f"\n{'=' * 80}")
    print("Benchmark Complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    import sys

    # Check command line arguments
    simple_mode = "--simple" in sys.argv or "-s" in sys.argv
    no_onnx = "--no-onnx" in sys.argv

    benchmark_threaded_rollout(simple=simple_mode, compare_onnx=not no_onnx)
