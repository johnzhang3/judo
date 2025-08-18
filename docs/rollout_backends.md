# Rollout Backend Configuration Guide

This guide explains how to configure and use different rollout backends in the judo app.

## Available Backends

The following rollout backends are now available:

1. **`"mujoco"`** (default) - Original MuJoCo threaded rollout API
2. **`"cpp"`** - Pure C++ rollout implementation
3. **`"cpp_persistent"`** - C++ rollout with persistent thread pool
4. **`"onnx"`** - C++ rollout with ONNX inference interleaving
5. **`"onnx_persistent"`** - C++ rollout with ONNX inference and persistent thread pool

## How Backend Switching Works

The judo app uses a dynamic backend switching system:

1. **Startup**: The controller initializes with the default backend (typically "mujoco")
2. **Configuration Application**: When task-specific overrides are applied, the system detects backend changes
3. **Automatic Switching**: The controller recreates the rollout backend automatically when needed
4. **Runtime Updates**: Backend changes are applied immediately without restarting the app

You'll see console messages like:
```
Using backend: mujoco        # Initial default
Using backend: cpp_persistent  # After override is applied
Using C++ backend: cpp_persistent
```

## Configuration Methods

### Method 1: Task-Specific Configuration (Recommended)

Create or modify a configuration file to set backends for specific tasks:

```yaml
# example_configs/my_config.yaml
defaults:
  - judo_default

# Task-specific backend configuration
controller_config_overrides:
  spot_door_box:
    rollout_backend: "cpp_persistent"
    horizon: 1.0

  cartpole:
    rollout_backend: "cpp"

  cylinder_push:
    rollout_backend: "mujoco"  # Keep default for simple tasks
```

Then run with your config:
```bash
pixi run judo --config-name=my_config
```

### Method 2: Global Default Configuration

Modify the main configuration file:

```yaml
# judo/configs/judo_default.yaml
controller_config_overrides:
  # Set backend for all tasks (unless overridden)
  _global_:
    rollout_backend: "cpp_persistent"
```

### Method 3: Programmatic Configuration

When creating controllers programmatically:

```python
from judo.controller.controller import ControllerConfig, Controller

# Configure via ControllerConfig (recommended approach)
config = ControllerConfig()
config.rollout_backend = "cpp_persistent"

controller = Controller(
    controller_config=config,
    task=task,
    task_config=task_config,
    optimizer=optimizer,
    optimizer_config=optimizer_config,
)
```

## Backend Selection Guidelines

### When to use each backend:

- **`"mujoco"`**: Good baseline, reliable and well-tested
- **`"cpp"`**: Better performance for larger problems, some overhead for small problems
- **`"cpp_persistent"`**: Best for repeated rollouts, stable performance
- **`"onnx"`**: When you need neural network inference during simulation
- **`"onnx_persistent"`**: Best performance for repeated ONNX-enabled rollouts

### Performance characteristics (approximate):

| Backend | Small Problems (<1000 sims) | Large Problems (>5000 sims) | Memory Usage | Startup Cost |
|---------|----------------------------|------------------------------|--------------|--------------|
| mujoco | ⭐⭐⭐ | ⭐⭐⭐ | Low | Low |
| cpp | ⭐⭐ | ⭐⭐⭐⭐ | Low | Medium |
| cpp_persistent | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium | High |
| onnx | ⭐⭐ | ⭐⭐⭐ | Medium | Medium |
| onnx_persistent | ⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | High |

## Prerequisites

### For C++ backends (`cpp`, `cpp_persistent`):
```bash
# Make sure judo_cpp is compiled
cd judo_cpp
pixi run build
```

### For ONNX backends (`onnx`, `onnx_persistent`):
```bash
# Ensure ONNX Runtime is available and models exist
# Note: ONNX backends require additional configuration for model paths
# This feature may need extension to be fully usable from config files
```

## Example Configurations

### High-Performance Setup
```yaml
# For maximum performance on large problems
controller_config_overrides:
  spot_door_box:
    rollout_backend: "cpp_persistent"
    horizon: 2.0
  spot_locomotion:
    rollout_backend: "cpp_persistent"
    horizon: 1.5
```

### Conservative Setup
```yaml
# Stick with reliable defaults but use C++ for complex tasks
controller_config_overrides:
  spot_door_box:
    rollout_backend: "cpp"
  # All others use default "mujoco"
```

### Development/Testing Setup
```yaml
# Use faster backends for development
controller_config_overrides:
  _global_:  # Apply to all tasks
    rollout_backend: "cpp_persistent"
    horizon: 0.5  # Shorter horizons for faster iteration
```

## Troubleshooting

### Common Issues:

1. **"judo_cpp module not available"**
   - Solution: Compile the C++ extension: `cd judo_cpp && pixi run build`

2. **Performance worse with C++ backends**
   - For small problems, overhead may outweigh benefits
   - Try increasing `num_rollouts` in optimizer config
   - Use `cpp_persistent` for repeated operations

3. **ONNX backends not working**
   - Ensure ONNX model files exist
   - Check ONNX Runtime installation
   - ONNX backends may require additional configuration

### Verification:

Test your configuration:
```bash
pixi run python dev/test_backend_config.py
```

## Advanced Usage

### Dynamic Backend Switching

For applications that need different backends based on problem size:

```python
def choose_backend(num_rollouts: int, horizon_steps: int) -> str:
    total_sims = num_rollouts * horizon_steps
    if total_sims < 1000:
        return "mujoco"
    elif total_sims < 10000:
        return "cpp"
    else:
        return "cpp_persistent"

# Update controller config dynamically
config.rollout_backend = choose_backend(32, 100)
```

### Monitoring Performance

Add timing to your rollouts:

```python
import time

start_time = time.time()
states, sensors = controller.rollout_backend.rollout(pairs, x0, controls)
rollout_time = time.time() - start_time

print(f"Rollout with {controller.rollout_backend.backend} took {rollout_time:.4f}s")
```
