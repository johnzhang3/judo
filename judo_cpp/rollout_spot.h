#pragma once

#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>
#include <vector>

namespace py = pybind11;

// Spot-specific rollout that runs ONNX policy inference at each step.
py::tuple RolloutSpot(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>&        data,
    const py::array_t<double>&         x0,
    const py::array_t<double>&         controls
);

// Single-step simulation that first runs the ONNX policy to produce controls.
// Returns the new policy output for next iteration.
py::array_t<float> SimSpot(
    const mjModel* model,
    mjData*        data,
    const py::array_t<double>& x0,
    const py::array_t<double>& controls,
    const py::array_t<float>& prev_policy
);


