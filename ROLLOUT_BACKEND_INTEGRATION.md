# Rollout Backend Integration - Complete Implementation

## Summary

✅ **Successfully integrated C++ rollout backends into the judo app!**

The integration includes:
- 5 rollout backend types: `mujoco`, `cpp`, `cpp_persistent`, `onnx`, `onnx_persistent`
- Dynamic backend switching during runtime
- Configuration system with task-specific overrides
- Complete compatibility with the pixi dev environment

## Available Backends

Based on the current pixi dev environment, all backends are available:

| Backend | Description | Status |
|---------|-------------|--------|
| `mujoco` | Default MuJoCo threaded rollouts | ✅ Working |
| `cpp` | Pure C++ rollout implementation | ✅ Working |
| `cpp_persistent` | C++ rollout with persistent thread pool | ✅ Working (Recommended) |
| `onnx` | C++ rollout with ONNX inference | ✅ Working* |
| `onnx_persistent` | C++ + ONNX + persistent thread pool | ✅ Working* |

*ONNX backends require model configuration via `set_onnx_config()`

## How to Use Different Backends

### 1. Command Line Overrides

```bash
# Use C++ persistent backend for spot_door_box task
pixi run -e dev judo +controller_config_overrides.spot_door_box.rollout_backend=cpp_persistent

# Use C++ backend for cartpole task
pixi run -e dev judo +controller_config_overrides.cartpole.rollout_backend=cpp

# Use MuJoCo backend (default)
pixi run -e dev judo +controller_config_overrides.cartpole.rollout_backend=mujoco
```

### 2. Configuration Files

```bash
# Use the test configuration file
pixi run -e dev judo --config-name=backends_test
```

The `backends_test.yaml` config demonstrates different backends for different tasks.

### 3. Direct Configuration Overrides

```bash
# Override specific controller config
pixi run -e dev judo controller_config_overrides.spot_door_box.rollout_backend=cpp_persistent
```

## Configuration Architecture

### 1. ControllerConfig Enhancement
- Added `rollout_backend` field to `ControllerConfig` dataclass
- Default value: `"mujoco"`
- Type: `Literal["mujoco", "cpp", "cpp_persistent", "onnx", "onnx_persistent"] | None`

### 2. Controller Class Updates
- Added `controller_cfg` property with setter
- Automatic backend switching when configuration changes
- Proper cleanup of old backends

### 3. RolloutBackend Class
- Unified interface for all backend types
- Backend-specific setup methods
- ONNX configuration support

## File Modifications

### Core Files Modified:
1. **`judo/utils/mujoco.py`** - Enhanced RolloutBackend class
2. **`judo/controller/controller.py`** - Added configuration integration
3. **`judo/configs/backends_test.yaml`** - Example configuration

### Test Files Created:
1. **`dev/test_complete_integration.py`** - Comprehensive integration test
2. **`dev/debug_backend_switching.py`** - Backend switching debug script

## Validation Results

All integration tests passed:

```
Backend Tests:
  ✓ mujoco: PASSED
  ✓ cpp_persistent: PASSED
  ✓ cpp: PASSED
  ✓ onnx: PASSED
  ✓ onnx_persistent: PASSED

Dynamic Switching: ✓ PASSED
Configuration System: ✓ PASSED

OVERALL: 7/7 tests passed
```

## Performance Characteristics

From test results in pixi dev environment:
- **mujoco**: ~0.0004s (fastest, baseline)
- **cpp_persistent**: ~0.0004s (fastest, recommended for production)
- **cpp**: ~0.02s (slower due to thread pool recreation)
- **onnx/onnx_persistent**: ~0.005-0.05s (depends on model complexity)

**Recommendation**: Use `cpp_persistent` for best performance.

## Technical Implementation Details

### Backend Switching Logic
The Controller uses a property setter to detect backend changes:

```python
@controller_cfg.setter
def controller_cfg(self, new_config: ControllerConfig) -> None:
    current_backend_type = getattr(self.rollout_backend, 'backend', None) if hasattr(self, 'rollout_backend') else None
    self._controller_cfg = new_config

    if current_backend_type != new_config.rollout_backend:
        old_rollout_backend = getattr(self, 'rollout_backend', None)
        if old_rollout_backend:
            old_rollout_backend.shutdown()
        self.rollout_backend = RolloutBackend(num_threads=self.optimizer_cfg.num_rollouts, backend=new_config.rollout_backend)
```

### Configuration Override System
Task-specific overrides work through the existing OverridableConfig pattern:

```python
# Register override
set_config_overrides("task_name", ControllerConfig, {"rollout_backend": "cpp_persistent"})

# Apply override
config = ControllerConfig()
config.set_override("task_name")  # Now uses cpp_persistent backend
```

## Next Steps

The integration is complete and working. You can now:

1. **Use different backends** via command line or config files
2. **Add new backend types** by extending the RolloutBackend class
3. **Configure ONNX models** for neural network-enhanced rollouts
4. **Benchmark performance** across different tasks and backends

## Testing

Run the comprehensive test to verify everything works:

```bash
pixi run -e dev python dev/test_complete_integration.py
```

This test validates all backends, dynamic switching, and configuration systems.
