#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>

#include "onnx_rollout.h"

namespace py = pybind11;

static std::vector<const mjModel*> getModelVector(const py::list& python_models) {
    std::vector<const mjModel*> model_vector;
    model_vector.reserve(python_models.size());
    for (auto&& item : python_models) {
        auto ptr = item.attr("_address").cast<std::uintptr_t>();
        model_vector.push_back(reinterpret_cast<const mjModel*>(ptr));
    }
    return model_vector;
}

static std::vector<mjData*> getDataVector(const py::list& python_data) {
    std::vector<mjData*> data_vector;
    data_vector.reserve(python_data.size());
    for (auto&& item : python_data) {
        auto ptr = item.attr("_address").cast<std::uintptr_t>();
        data_vector.push_back(reinterpret_cast<mjData*>(ptr));
    }
    return data_vector;
}

PYBIND11_MODULE(_judo_cpp, m) {
    // ONNX interleaved rollout
    m.def("onnx_interleave_rollout",
          [](const py::list& models,
             const py::list& data,
             const py::array_t<double>& x0,
             const py::array_t<double>& controls,
             const std::string& onnx_model_path,
             int inference_frequency = 1)
          {
              // turn Python lists into vectors of mjModel*/mjData*
              auto models_cpp = getModelVector(models);
              auto data_cpp   = getDataVector(data);

              // call into your C++ implementation
              return ONNXInterleaveRollout(models_cpp, data_cpp, x0, controls,
                                         onnx_model_path, inference_frequency);
          },
          py::arg("models"),
          py::arg("data"),
          py::arg("x0"),
          py::arg("controls"),
          py::arg("onnx_model_path"),
          py::arg("inference_frequency") = 1,
          R"doc(
Run parallel MuJoCo rollouts with interleaved ONNX inference.

Args:
    models:              length-B list of mujoco._structs.MjModel
    data:                length-B list of mujoco._structs.MjData
    x0:                  1D array of length nq+nv, initial [qpos;qvel]
    controls:            3D array of shape (B, T, nu), open-loop controls
    onnx_model_path:     path to ONNX model file
    inference_frequency: run inference every N steps (default: 1)

Returns:
    tuple of four np.ndarray:
      states     -> shape (B, T, nq+nv) - MuJoCo simulation states
      sensors    -> shape (B, T, nsensordata) - sensor data
      inputs     -> shape (B, T, nu) - applied controls
      inferences -> shape (B, T, nq+nv) - ONNX inference results
)doc");
}
