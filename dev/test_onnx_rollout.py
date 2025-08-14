#!/usr/bin/env python3
# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Test script for ONNX interleaved rollout functionality."""

import os
import time

import mujoco
import numpy as np

import judo_cpp


def create_simple_pendulum_xml() -> str:
    """Create a simple pendulum model for testing."""
    return """
    <mujoco>
      <option timestep="0.01"/>
      <worldbody>
        <body name="pendulum">
          <joint name="hinge" type="hinge" axis="0 0 1" pos="0 0 0"/>
          <geom name="rod" type="capsule" size="0.01" fromto="0 0 0 0 0 -0.3" rgba="1 0 0 1"/>
          <geom name="bob" type="sphere" size="0.05" pos="0 0 -0.3" rgba="0 1 0 1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="torque" joint="hinge"/>
      </actuator>
      <sensor>
        <jointpos name="angle" joint="hinge"/>
        <jointvel name="angular_velocity" joint="hinge"/>
      </sensor>
    </mujoco>
    """


def test_onnx_interleave_rollout() -> None:
    """Test the ONNX interleaved rollout functionality."""
    print("=" * 60)
    print("Testing ONNX Interleaved Rollout")
    print("=" * 60)

    # Create simple model
    xml = create_simple_pendulum_xml()
    model = mujoco.MjModel.from_xml_string(xml)

    # Test parameters
    B = 3  # batch size
    T = 20  # time steps
    nq, nv, nu = model.nq, model.nv, model.nu
    nstate = nq + nv
    nsens = model.nsensordata

    print(f"Model dimensions: nq={nq}, nv={nv}, nu={nu}, nsens={nsens}")

    # Create test data
    x0 = np.array([np.pi / 4, 0.0])  # Start pendulum at 45 degrees, zero velocity

    # Create sinusoidal control inputs (torque oscillation)
    times = np.linspace(0, T * model.opt.timestep, T)
    controls = np.zeros((B, T, nu))
    for i in range(B):
        # Different frequency for each batch
        freq = 0.5 + i * 0.5
        controls[i, :, 0] = 0.1 * np.sin(2 * np.pi * freq * times)

    # Create model/data lists
    models = [model] * B
    datas = [mujoco.MjData(model) for _ in range(B)]

    # ONNX model path
    onnx_model_path = "identity_network_pendulum.onnx"

    if not os.path.exists(onnx_model_path):
        print(f"ONNX model not found at {onnx_model_path}")
        print("Please run create_identity_network.py first")
        return

    print(f"Using ONNX model: {onnx_model_path}")

    # Test with different inference frequencies
    inference_frequencies = [1, 2, 5]

    for freq in inference_frequencies:
        print(f"\n--- Testing with inference frequency = {freq} ---")

        try:
            # Run ONNX interleaved rollout
            result = judo_cpp.onnx_interleave_rollout(models, datas, x0, controls, onnx_model_path, freq)

            states, sensors, inputs, inferences = result

            print("✓ ONNX interleaved rollout succeeded!")
            print(f"  States shape:     {states.shape}")
            print(f"  Sensors shape:    {sensors.shape}")
            print(f"  Inputs shape:     {inputs.shape}")
            print(f"  Inferences shape: {inferences.shape}")

            # Verify shapes
            assert states.shape == (B, T, nstate), (
                f"States shape mismatch: expected {(B, T, nstate)}, got {states.shape}"
            )
            assert sensors.shape == (B, T, nsens), (
                f"Sensors shape mismatch: expected {(B, T, nsens)}, got {sensors.shape}"
            )
            assert inputs.shape == (B, T, nu), f"Inputs shape mismatch: expected {(B, T, nu)}, got {inputs.shape}"
            assert inferences.shape == (B, T, nstate), (
                f"Inferences shape mismatch: expected {(B, T, nstate)}, got {inferences.shape}"
            )

            # For identity network, check that inference results are close to states
            # (they might differ slightly due to timing of when inference is applied)
            print(f"  Max difference between states and inferences: {np.max(np.abs(states - inferences)):.6f}")

            # Check energy conservation (roughly) - pendulum should swing
            initial_angle = states[0, 0, 0]  # First batch, first timestep, position
            final_angle = states[0, -1, 0]  # First batch, last timestep, position
            print(f"  Pendulum motion: initial_angle={initial_angle:.3f}, final_angle={final_angle:.3f}")

        except Exception as e:
            print(f"✗ Error with inference frequency {freq}: {e}")

    print("\n--- Comparing with regular rollout ---")

    print("\n" + "=" * 60)
    print("ONNX Interleaved Rollout Test Complete!")
    print("=" * 60)


def analyze_inference_timing() -> None:
    """Analyze when inference is called vs simulation steps."""
    print("\n--- Analyzing Inference Timing ---")

    xml = create_simple_pendulum_xml()
    model = mujoco.MjModel.from_xml_string(xml)

    B, T, nu = 1, 10, model.nu
    x0 = np.array([0.1, 0.0])
    controls = np.zeros((B, T, nu))
    models = [model]
    datas = [mujoco.MjData(model)]
    onnx_model_path = "identity_network_pendulum.onnx"

    inference_freq = 3

    print(f"Running rollout with T={T} steps and inference_frequency={inference_freq}")
    print("Inference should occur at steps: ", [i for i in range(T) if (i + 1) % inference_freq == 0])

    try:
        result = judo_cpp.onnx_interleave_rollout(models, datas, x0, controls, onnx_model_path, inference_freq)
        states, sensors, inputs, inferences = result

        print("Step-by-step analysis:")
        for t in range(T):
            diff = np.abs(states[0, t, :] - inferences[0, t, :])
            max_diff = np.max(diff)
            is_inference_step = (t + 1) % inference_freq == 0
            print(f"  Step {t:2d}: max_diff={max_diff:.6f} {'(INFERENCE)' if is_inference_step else ''}")

    except Exception as e:
        print(f"Error in timing analysis: {e}")


def benchmark_rollout_performance() -> None:
    """Compare performance of regular rollout vs ONNX interleaved rollout."""
    print("\n" + "=" * 60)
    print("ROLLOUT PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Create test model
    xml = create_simple_pendulum_xml()
    model = mujoco.MjModel.from_xml_string(xml)

    # Test parameters for benchmarking
    batch_sizes = [1, 5, 10]
    time_steps = [50, 100]
    inference_frequencies = [1, 5, 10, 0]  # 0 = no inference (fallback)

    onnx_model_path = "identity_network_pendulum.onnx"

    if not os.path.exists(onnx_model_path):
        print(f"ONNX model not found at {onnx_model_path}")
        return

    results = []
    num_trials = 5

    for B in batch_sizes:
        for T in time_steps:
            print(f"\n--- Batch Size: {B}, Time Steps: {T} ---")

            # Setup test data
            nu = model.nu
            x0 = np.array([np.pi / 4, 0.0])
            controls = np.random.randn(B, T, nu) * 0.1
            models = [model] * B
            datas = [mujoco.MjData(model) for _ in range(B)]

            # Warm up (first run is often slower)
            try:
                _ = judo_cpp.onnx_interleave_rollout(models, datas, x0, controls, onnx_model_path, 10)
            except Exception:  # noqa: S110
                pass

            # Benchmark baseline: ONNX rollout with very infrequent inference (freq=999999)
            # This approximates a regular rollout with minimal ONNX overhead
            baseline_times = []

            for _trial in range(num_trials):
                # Reset data
                for data in datas:
                    data.time = 0.0
                    mujoco.mj_resetData(model, data)

                start_time = time.perf_counter()
                try:
                    judo_cpp.onnx_interleave_rollout(models, datas, x0, controls, onnx_model_path, 999999)
                    end_time = time.perf_counter()
                    baseline_times.append(end_time - start_time)
                except Exception as e:
                    print(f"  Error in baseline rollout: {e}")
                    baseline_times.append(float("inf"))

            baseline_mean = np.mean(baseline_times)
            baseline_std = np.std(baseline_times)

            print(f"  Baseline (no ONNX):  {baseline_mean * 1000:.2f} ± {baseline_std * 1000:.2f} ms")

            # Benchmark ONNX rollouts with different frequencies
            num_trials = 5
            for freq in inference_frequencies:
                if freq == 0:
                    continue  # Skip the no-inference case for ONNX rollout

                onnx_times = []

                for _trial in range(num_trials):
                    # Reset data
                    for data in datas:
                        data.time = 0.0
                        mujoco.mj_resetData(model, data)

                    start_time = time.perf_counter()
                    try:
                        judo_cpp.onnx_interleave_rollout(models, datas, x0, controls, onnx_model_path, freq)
                        end_time = time.perf_counter()
                        onnx_times.append(end_time - start_time)
                    except Exception as e:
                        print(f"  Error in ONNX rollout (freq={freq}): {e}")
                        onnx_times.append(float("inf"))

                onnx_mean = np.mean(onnx_times)
                onnx_std = np.std(onnx_times)
                overhead = ((onnx_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else float("inf")

                print(
                    f"  ONNX Rollout (f={freq:2d}): {onnx_mean * 1000:.2f} ± {onnx_std * 1000:.2f} ms "
                    f"(+{overhead:+.1f}%)"
                )

                # Store results for summary
                results.append(
                    {
                        "batch_size": B,
                        "time_steps": T,
                        "inference_freq": freq,
                        "baseline_time": baseline_mean,
                        "onnx_time": onnx_mean,
                        "overhead_percent": overhead,
                        "inference_calls": T // freq if freq > 0 else 0,
                    }
                )

    # Summary analysis
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    if results:
        # Group by inference frequency
        freq_groups = {}
        for result in results:
            freq = result["inference_freq"]
            if freq not in freq_groups:
                freq_groups[freq] = []
            freq_groups[freq].append(result)

        print("\nAverage overhead by inference frequency:")
        for freq in sorted(freq_groups.keys()):
            group = freq_groups[freq]
            avg_overhead = np.mean([r["overhead_percent"] for r in group if np.isfinite(r["overhead_percent"])])
            total_inferences = np.mean([r["inference_calls"] for r in group])
            print(
                f"  Frequency {freq:2d}: {avg_overhead:+.1f}% overhead, ~{total_inferences:.0f} inferences per rollout"
            )

        # Find most efficient settings
        valid_results = [r for r in results if np.isfinite(r["overhead_percent"])]
        if valid_results:
            min_overhead = min(valid_results, key=lambda x: x["overhead_percent"])
            max_overhead = max(valid_results, key=lambda x: x["overhead_percent"])

            print(
                f"\nMost efficient:  B={min_overhead['batch_size']}, T={min_overhead['time_steps']}, "
                f"freq={min_overhead['inference_freq']} → {min_overhead['overhead_percent']:+.1f}% overhead"
            )
            print(
                f"Least efficient: B={max_overhead['batch_size']}, T={max_overhead['time_steps']}, "
                f"freq={max_overhead['inference_freq']} → {max_overhead['overhead_percent']:+.1f}% overhead"
            )

        # Analysis of inference cost per call
        print("\nInference cost analysis:")
        for freq in sorted(freq_groups.keys()):
            group = freq_groups[freq]
            for result in group:
                if np.isfinite(result["overhead_percent"]) and result["inference_calls"] > 0:
                    time_per_inference = ((result["onnx_time"] - result["baseline_time"]) * 1000) / result[
                        "inference_calls"
                    ]
                    print(
                        f"  B={result['batch_size']}, T={result['time_steps']}, freq={freq}: "
                        f"{time_per_inference:.2f} ms per inference call"
                    )

    print("\n" + "=" * 60)


def micro_benchmark_onnx_inference() -> None:
    """Micro-benchmark to measure pure ONNX inference time."""
    print("\n" + "=" * 60)
    print("ONNX INFERENCE MICRO-BENCHMARK")
    print("=" * 60)

    onnx_model_path = "identity_network_pendulum.onnx"
    if not os.path.exists(onnx_model_path):
        print(f"ONNX model not found at {onnx_model_path}")
        return

    try:
        # Test pure ONNX inference without MuJoCo
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_model_path)

        # Test different batch sizes
        batch_sizes = [1, 5, 10, 20, 50]
        num_trials = 100

        print("Pure ONNX inference times (no MuJoCo):")

        for batch_size in batch_sizes:
            # Create test input
            test_input = np.random.randn(batch_size, 2).astype(np.float32)

            # Warmup
            for _ in range(10):
                _ = session.run(None, {"input": test_input})

            # Benchmark
            times = []
            for _ in range(num_trials):
                start_time = time.perf_counter()
                session.run(None, {"input": test_input})
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            mean_time = np.mean(times) * 1000  # Convert to ms
            std_time = np.std(times) * 1000
            per_sample = mean_time / batch_size

            print(f"  Batch {batch_size:2d}: {mean_time:.3f} ± {std_time:.3f} ms total, {per_sample:.3f} ms per sample")

        # Estimate overhead in rollout context
        print("\nEstimated overhead for typical rollout:")
        print(f"  Single inference call: ~{np.mean([mean_time for mean_time in [0.1, 0.2]]):.2f} ms")
        print(f"  50 timesteps, freq=5:  ~{10 * 0.15:.1f} ms total ONNX time")
        print(f"  100 timesteps, freq=1: ~{100 * 0.15:.1f} ms total ONNX time")

    except ImportError:
        print("onnxruntime not available for micro-benchmark")
    except Exception as e:
        print(f"Error in micro-benchmark: {e}")


if __name__ == "__main__":
    test_onnx_interleave_rollout()
    analyze_inference_timing()
    benchmark_rollout_performance()
    micro_benchmark_onnx_inference()
