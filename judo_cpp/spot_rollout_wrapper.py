#!/usr/bin/env python3
"""
Python wrapper for SpotRollout that mimics mujoco.rollout.Rollout API.

This module provides a clean Python interface that matches the mujoco.rollout API
while using the optimized C++ SpotRollout implementation.
"""

from typing import Optional, Union, Tuple, List, Any
import numpy as np
try:
    from . import _judo_cpp
except ImportError:
    import _judo_cpp


class SpotRollout:
    """
    Spot Rollout object containing a thread pool for parallel rollouts.

    This class mimics the mujoco.rollout.Rollout API but uses Spot-specific
    ONNX policy inference for control generation.

    Examples:
        # Single-threaded usage
        with SpotRollout(nthread=0) as rollout:
            states, sensors = rollout.rollout(models, data, x0, controls)

        # Multi-threaded usage
        rollout = SpotRollout(nthread=4)
        try:
            states, sensors = rollout.rollout(models, data, x0, controls)
        finally:
            rollout.close()
    """

    def __init__(self, *, nthread: Optional[int] = None):
        """
        Construct a SpotRollout object containing a thread pool for parallel rollouts.

        Args:
            nthread: Number of threads in pool. If None, uses hardware concurrency.
                    If 0, this runs single-threaded with no thread pool overhead.
        """
        if nthread is None:
            import os
            nthread = os.cpu_count() or 4

        self._nthread = nthread

        # Create appropriate C++ rollout object based on thread count
        if nthread == 0:
            self._rollout = _judo_cpp.SpotRollout(nthread)
        elif nthread == 2:
            self._rollout = _judo_cpp.SpotRollout2T(nthread)
        elif nthread == 4:
            self._rollout = _judo_cpp.SpotRollout4T(nthread)
        elif nthread == 8:
            self._rollout = _judo_cpp.SpotRollout8T(nthread)
        else:
            # For other thread counts, use single-threaded for now
            # Could be extended to support dynamic thread counts
            self._rollout = _judo_cpp.SpotRollout(nthread)

    def __enter__(self) -> 'SpotRollout':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close the rollout object and cleanup resources."""
        if hasattr(self, '_rollout') and self._rollout:
            self._rollout.close()
            self._rollout = None

    def rollout(
        self,
        model: Union[Any, List[Any]],
        data: Union[Any, List[Any]],
        initial_state: Union[np.ndarray, List[float]],
        control: Union[np.ndarray, List[float], None] = None,
        *,
        skip_checks: bool = False,
        nstep: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Roll out open-loop trajectories from initial states with Spot ONNX policy.

        This method matches the mujoco.rollout.Rollout.rollout() API signature
        but uses Spot-specific control generation via ONNX policy inference.

        Args:
            model: An instance or sequence of MjModel with the same size signature.
            data: Associated mjData instance or sequence of instances.
            initial_state: Array of initial states from which to roll out trajectories.
                          Shape: ([nbatch or 1] x nstate)
            control: Command sequences for the Spot policy.
                    Shape: ([nbatch or 1] x [nstep or 1] x 25)
                    Each control vector contains the 25-dim policy command.
            skip_checks: Whether to skip internal shape and type checks.
            nstep: Number of steps in rollouts (inferred from control if unspecified).

        Returns:
            states: State output array, shape (nbatch x nstep x nstate).
            sensordata: Sensor data output array, shape (nbatch x nstep x nsensordata).

        Raises:
            RuntimeError: Rollout requested after thread pool shutdown.
            ValueError: Bad shapes or sizes.
        """
        if not hasattr(self, '_rollout') or not self._rollout:
            raise RuntimeError("Rollout requested after object was closed")

        if control is None:
            raise ValueError("Control commands are required for Spot rollout")

        # Convert inputs to lists if single instances
        if not isinstance(model, (list, tuple)):
            model = [model]
        if not isinstance(data, (list, tuple)):
            data = [data]

        # Convert to numpy arrays
        initial_state = np.asarray(initial_state, dtype=np.float64)
        control = np.asarray(control, dtype=np.float64)

        # Ensure proper shapes
        if initial_state.ndim == 1:
            initial_state = initial_state[np.newaxis, :]  # (1, nstate)

        if control.ndim == 2:
            control = control[np.newaxis, :, :]  # (1, nstep, ncontrol)

        # Basic shape validation
        nbatch = len(model)
        if len(data) != nbatch:
            raise ValueError(f"Model and data counts must match: {len(model)} vs {len(data)}")

        if initial_state.shape[0] not in (1, nbatch):
            raise ValueError(f"Initial state batch size must be 1 or {nbatch}, got {initial_state.shape[0]}")

        if control.shape[0] not in (1, nbatch):
            raise ValueError(f"Control batch size must be 1 or {nbatch}, got {control.shape[0]}")

        # Tile inputs if needed
        if initial_state.shape[0] == 1 and nbatch > 1:
            initial_state = np.tile(initial_state, (nbatch, 1))

        if control.shape[0] == 1 and nbatch > 1:
            control = np.tile(control, (nbatch, 1, 1))

        # Call C++ implementation
        return self._rollout.rollout(model, data, initial_state, control)

    @property
    def nthread(self) -> int:
        """Number of threads in the thread pool."""
        return self._nthread


# Factory function for backward compatibility
def create_spot_rollout(nthread: Optional[int] = None) -> SpotRollout:
    """
    Create a SpotRollout object.

    Args:
        nthread: Number of threads. If None, uses hardware concurrency.

    Returns:
        SpotRollout object ready for use.
    """
    return SpotRollout(nthread=nthread)


# Legacy function for backward compatibility with existing code
def rollout_spot(models, data, x0, controls):
    """
    Legacy function for backward compatibility.

    This function provides the same interface as the old rollout_spot function
    but uses the new SpotRollout class internally.
    """
    with SpotRollout(nthread=0) as rollout:
        return rollout.rollout(models, data, x0, controls)


__all__ = ['SpotRollout', 'create_spot_rollout', 'rollout_spot']