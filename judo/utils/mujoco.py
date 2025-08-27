# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from copy import deepcopy
from typing import Literal, Optional

import numpy as np
from mujoco import MjData, MjModel
from mujoco.rollout import Rollout

try:
    import judo_cpp

    JUDO_CPP_AVAILABLE = True
except ImportError:
    judo_cpp = None  # type: ignore
    JUDO_CPP_AVAILABLE = False


def make_model_data_pairs(model: MjModel, num_pairs: int) -> list[tuple[MjModel, MjData]]:
    """Create model/data pairs for mujoco threaded rollout."""
    models = [deepcopy(model) for _ in range(num_pairs)]
    datas = [MjData(m) for m in models]
    model_data_pairs = list(zip(models, datas, strict=True))
    return model_data_pairs


class RolloutBackend:
    """The backend for conducting multithreaded rollouts."""

    def __init__(
        self,
        num_threads: int,
        backend: Literal[
            "mujoco",
            "cpp",
            "cpp_persistent",
            "onnx",
            "onnx_persistent",
            "onnx_policy",
            "onnx_policy_persistent",
        ],
    ) -> None:
        """Initialize the backend with a number of threads.

        Args:
            num_threads: Number of threads to use for rollouts
            backend: The backend to use for rollouts:
                - "mujoco": Use MuJoCo's threaded rollout API
                - "cpp": Use pure C++ rollout implementation
                - "cpp_persistent": Use C++ rollout with persistent thread pool
                - "onnx": Use C++ rollout with ONNX inference interleaving
                - "onnx_persistent": Use C++ rollout with ONNX inference and persistent thread pool
        """
        self.backend = backend
        self.num_threads = num_threads
        self.onnx_model_path: Optional[str] = None
        self.inference_frequency: int = 1
        print(f"Using backend: {self.backend}")
        if self.backend == "mujoco":
            self.setup_mujoco_backend(num_threads)
        elif self.backend in [
            "cpp",
            "cpp_persistent",
            "onnx",
            "onnx_persistent",
            "onnx_policy",
            "onnx_policy_persistent",
        ]:
            self.setup_cpp_backend()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def setup_mujoco_backend(self, num_threads: int) -> None:
        """Setup the mujoco backend."""
        self.rollout_obj = Rollout(nthread=num_threads)
        self.rollout_func = lambda m, d, x0, u: self.rollout_obj.rollout(m, d, x0, u)

    def setup_cpp_backend(self) -> None:
        """Setup the C++ backend."""
        if not JUDO_CPP_AVAILABLE or judo_cpp is None:
            raise ImportError("judo_cpp module is not available. Please compile the C++ extension.")
        print(f"Using C++ backend: {self.backend}")
        if self.backend == "cpp":
            self.rollout_func = judo_cpp.pure_cpp_rollout
        elif self.backend == "cpp_persistent":
            self.rollout_func = judo_cpp.persistent_cpp_rollout
        elif self.backend == "onnx":
            self.rollout_func = judo_cpp.onnx_interleave_rollout
        elif self.backend == "onnx_persistent":
            self.rollout_func = judo_cpp.persistent_onnx_interleave_rollout
        elif self.backend in ("onnx_policy", "onnx_policy_persistent"):
            # Policy backends are exercised via policy_rollout(); a direct rollout() call is unsupported.
            self.rollout_func = None  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown C++ backend: {self.backend}")

    def set_onnx_config(self, model_path: str, inference_frequency: int = 1) -> None:
        """Set ONNX configuration for ONNX-enabled backends.

        Args:
            model_path: Path to the ONNX model file
            inference_frequency: Run inference every N steps (default: 1, every step)
        """
        if self.backend not in ["onnx", "onnx_persistent"]:
            raise ValueError(
                f"ONNX configuration is only valid for 'onnx' and 'onnx_persistent' backends, got '{self.backend}'"
            )

        self.onnx_model_path = model_path
        self.inference_frequency = inference_frequency

    def set_onnx_policy_config(
        self,
        model_path: str,
        state_history_length: int = 10,
        action_history_length: int = 5,
        inference_frequency: int = 1,
        additional_input_dims: Optional[dict[str, int]] = None,
    ) -> None:
        """Set ONNX policy configuration for policy-driven rollouts.

        Args:
            model_path: Path to the ONNX model file
            state_history_length: Number of previous states to track per thread
            action_history_length: Number of previous actions to track per thread
            inference_frequency: Run inference every N steps
            additional_input_dims: Dict of additional input names to their dimensions
                                 e.g., {'goal': 3, 'command': 6}
        """
        if self.backend not in [
            "onnx",
            "onnx_persistent",
            "onnx_policy",
            "onnx_policy_persistent",
        ]:
            raise ValueError(f"ONNX policy config only valid for ONNX backends, got '{self.backend}'")

        self.policy_config = {
            "model_path": model_path,
            "state_history_length": state_history_length,
            "action_history_length": action_history_length,
            "inference_frequency": inference_frequency,
            "additional_input_dims": additional_input_dims or {},
            "num_threads": self.num_threads,  # Pass thread count for initialization
        }

    def rollout(
        self,
        model_data_pairs: list[tuple[MjModel, MjData]],
        x0: np.ndarray,
        controls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conduct a rollout depending on the backend."""
        # unpack models into a list of models and data
        ms, ds = zip(*model_data_pairs, strict=True)
        ms = list(ms)
        ds = list(ds)

        # getting shapes
        nq = ms[0].nq
        nv = ms[0].nv
        nu = ms[0].nu

        # prepare states based on backend
        if self.backend == "mujoco":
            # the state passed into mujoco's rollout function includes the time
            # shape = (num_rollouts, num_states + 1)
            x0_batched = np.tile(x0, (len(ms), 1))
            full_states = np.concatenate([time.time() * np.ones((len(ms), 1)), x0_batched], axis=-1)
            assert full_states.shape[-1] == nq + nv + 1
            assert full_states.ndim == 2
            assert controls.ndim == 3
            assert controls.shape[-1] == nu
            assert controls.shape[0] == full_states.shape[0]

            # rollout
            _states, _out_sensors = self.rollout_func(ms, ds, full_states, controls)
            out_states = np.array(_states)[..., 1:]  # remove time from state
            out_sensors = np.array(_out_sensors)

        elif self.backend in ["cpp", "cpp_persistent"]:
            # C++ backends expect x0 without time
            x0_batched = np.tile(x0, (len(ms), 1))
            assert x0_batched.shape[-1] == nq + nv
            assert x0_batched.ndim == 2
            assert controls.ndim == 3
            assert controls.shape[-1] == nu
            assert controls.shape[0] == x0_batched.shape[0]

            # rollout returns (states, sensors, inputs)
            result = self.rollout_func(ms, ds, x0_batched, controls)
            out_states, out_sensors = np.array(result[0]), np.array(result[1])

        elif self.backend in ["onnx", "onnx_persistent"]:
            # ONNX backends expect x0 without time and require ONNX model path
            if self.onnx_model_path is None:
                raise ValueError("ONNX model path must be set using set_onnx_config() for ONNX backends")

            x0_batched = np.tile(x0, (len(ms), 1))
            assert x0_batched.shape[-1] == nq + nv
            assert x0_batched.ndim == 2
            assert controls.ndim == 3
            assert controls.shape[-1] == nu
            assert controls.shape[0] == x0_batched.shape[0]

            # rollout returns (states, sensors, inputs, inferences)
            try:
                result = self.rollout_func(ms, ds, x0_batched, controls, self.onnx_model_path, self.inference_frequency)
                out_states, out_sensors = np.array(result[0]), np.array(result[1])
            except Exception as e:
                if "Got invalid dimensions" in str(e) or "incompatible function arguments" in str(e):
                    expected_dim = nq + nv
                    raise ValueError(
                        f"ONNX model issue:\n"
                        f"  Current simulation state size: {expected_dim}\n"
                        f"  ONNX model: {self.onnx_model_path}\n"
                        f"  Original error: {e}\n\n"
                        f"Solutions:\n"
                        f"  1. Use a different backend: rollout_backend: 'cpp_persistent'\n"
                        f"  2. Use an ONNX model that matches this task's state dimensions\n"
                        f"  3. Test with a different task that matches the model dimensions"
                    ) from e
                else:
                    raise

        elif self.backend in ["onnx_policy", "onnx_policy_persistent"]:
            raise ValueError("rollout() is not supported for onnx_policy backends. Use policy_rollout() instead.")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        return out_states, out_sensors

    def policy_rollout(
        self,
        model_data_pairs: list[tuple[MjModel, MjData]],
        x0: np.ndarray,
        horizon: int,
        additional_inputs: Optional[dict[str, np.ndarray]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Conduct a policy-driven rollout where ONNX model generates actions.

        Args:
            model_data_pairs: List of (model, data) pairs for parallel rollouts
            x0: Initial state (nq + nv,) - will be tiled across rollouts
            horizon: Number of simulation steps
            additional_inputs: Dict of additional inputs like goals/commands
                             Each value should be shape (num_rollouts, dim) or (dim,) to broadcast

        Returns:
            states: (num_rollouts, horizon+1, state_dim)
            actions: (num_rollouts, horizon, action_dim) - actions generated by policy
            sensors: (num_rollouts, horizon, sensor_dim)
        """
        if self.backend not in [
            "onnx",
            "onnx_persistent",
            "onnx_policy",
            "onnx_policy_persistent",
        ]:
            raise ValueError("Policy rollout only available for ONNX-based backends")

        if not hasattr(self, "policy_config") or self.policy_config is None:
            raise ValueError("Must call set_onnx_policy_config() before policy_rollout()")

        ms, ds = zip(*model_data_pairs, strict=True)
        x0_batched = np.tile(x0, (len(ms), 1))

        # Prepare additional inputs
        processed_additional_inputs = np.array([])  # Default empty array
        if additional_inputs:
            # For now, assume single additional input. Could be extended for multiple inputs
            input_list = []
            for name, data in additional_inputs.items():
                data_array = np.array(data)
                if data_array.ndim == 1:
                    # Broadcast to all rollouts
                    data_array = np.tile(data_array, (len(ms), 1))
                elif data_array.shape[0] != len(ms):
                    raise ValueError(
                        f"Additional input '{name}' has wrong batch size: {data_array.shape[0]}, expected {len(ms)}"
                    )
                input_list.append(data_array)

            # Concatenate all additional inputs
            if input_list:
                processed_additional_inputs = np.concatenate(input_list, axis=1)

        # Call C++ policy rollout function
        if self.backend in ("onnx", "onnx_policy"):
            rollout_func = judo_cpp.onnx_policy_rollout
        else:  # persistent variants
            rollout_func = judo_cpp.persistent_onnx_policy_rollout

        result = rollout_func(
            list(ms),
            list(ds),
            x0_batched,
            horizon,
            self.policy_config["model_path"],
            self.policy_config["state_history_length"],
            self.policy_config["action_history_length"],
            self.policy_config["inference_frequency"],
            processed_additional_inputs,
        )

        return result[0], result[1], result[2]  # states, actions, sensors

    def update(self, num_threads: int) -> None:
        """Update the backend with a new number of threads."""
        self.num_threads = num_threads

        if self.backend == "mujoco":
            self.rollout_obj.close()
            self.setup_mujoco_backend(num_threads)
        elif self.backend in ["cpp", "cpp_persistent", "onnx", "onnx_persistent"]:
            # For C++ backends, we just need to update the internal thread count
            # The actual thread management is handled by the C++ implementation
            # For persistent backends, the thread pool may need to be recreated
            if self.backend in ["cpp_persistent", "onnx_persistent"]:
                # Shutdown existing thread pool if using persistent backends
                if JUDO_CPP_AVAILABLE and judo_cpp is not None:
                    judo_cpp.shutdown_thread_pool()
            # No need to recreate the rollout function as it doesn't depend on thread count
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def shutdown(self) -> None:
        """Shutdown the backend and clean up resources."""
        if self.backend == "mujoco":
            if hasattr(self, "rollout_obj"):
                self.rollout_obj.close()
        elif self.backend in ["cpp_persistent", "onnx_persistent"]:
            # Shutdown persistent thread pool for C++ backends
            if JUDO_CPP_AVAILABLE and judo_cpp is not None:
                judo_cpp.shutdown_thread_pool()
