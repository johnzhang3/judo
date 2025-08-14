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
    x0:                  2D array of shape (B, nq+nv), batched initial [qpos;qvel]
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

    // Pure C++ rollout without ONNX
    m.def("pure_cpp_rollout",
          [](const py::list& models,
             const py::list& data,
             const py::array_t<double>& x0,
             const py::array_t<double>& controls)
          {
              // turn Python lists into vectors of mjModel*/mjData*
              auto models_cpp = getModelVector(models);
              auto data_cpp   = getDataVector(data);

              // call into your C++ implementation
              return PureCppRollout(models_cpp, data_cpp, x0, controls);
          },
          py::arg("models"),
          py::arg("data"),
          py::arg("x0"),
          py::arg("controls"),
          R"doc(
Run pure C++ parallel MuJoCo rollouts without any ONNX inference.

Args:
    models:    length-B list of mujoco._structs.MjModel
    data:      length-B list of mujoco._structs.MjData
    x0:        2D array of shape (B, nq+nv), batched initial [qpos;qvel]
    controls:  3D array of shape (B, T, nu), open-loop controls

Returns:
    tuple of three np.ndarray:
      states  -> shape (B, T, nq+nv) - MuJoCo simulation states
      sensors -> shape (B, T, nsensordata) - sensor data
      inputs  -> shape (B, T, nu) - applied controls
)doc");

    // Persistent thread pool C++ rollout
    m.def("persistent_cpp_rollout",
          [](const py::list& models,
             const py::list& data,
             const py::array_t<double>& x0,
             const py::array_t<double>& controls)
          {
              // turn Python lists into vectors of mjModel*/mjData*
              auto models_cpp = getModelVector(models);
              auto data_cpp   = getDataVector(data);

              // call into your C++ implementation with persistent thread pool
              return PersistentCppRollout(models_cpp, data_cpp, x0, controls);
          },
          py::arg("models"),
          py::arg("data"),
          py::arg("x0"),
          py::arg("controls"),
          R"doc(
Run pure C++ parallel MuJoCo rollouts using a persistent thread pool for stable performance.

This version maintains a persistent thread pool between calls, providing more stable
performance similar to Python's mujoco.rollout.Rollout.

Args:
    models:    length-B list of mujoco._structs.MjModel
    data:      length-B list of mujoco._structs.MjData
    x0:        2D array of shape (B, nq+nv), batched initial [qpos;qvel]
    controls:  3D array of shape (B, T, nu), open-loop controls

Returns:
    tuple of three np.ndarray:
      states  -> shape (B, T, nq+nv) - MuJoCo simulation states
      sensors -> shape (B, T, nsensordata) - sensor data
      inputs  -> shape (B, T, nu) - applied controls
)doc");

    // Function to shutdown persistent thread pool
    m.def("shutdown_thread_pool",
          []() {
              ThreadPoolManager::instance().shutdown();
          },
          R"doc(
Shutdown the persistent thread pool.

Call this function to clean up the persistent thread pool when done with rollouts.
The pool will be automatically recreated on the next call to persistent_cpp_rollout.
)doc");

    // Persistent thread pool ONNX interleaved rollout
    m.def("persistent_onnx_interleave_rollout",
          [](const py::list& models,
             const py::list& data,
             const py::array_t<double>& x0,
             const py::array_t<double>& controls,
             const std::string& onnx_model_path,
             int inference_frequency)
          {
              // turn Python lists into vectors of mjModel*/mjData*
              auto models_cpp = getModelVector(models);
              auto data_cpp   = getDataVector(data);

              // call into your C++ implementation with persistent thread pool
              return PersistentONNXInterleaveRollout(models_cpp, data_cpp, x0, controls, onnx_model_path, inference_frequency);
          },
          py::arg("models"),
          py::arg("data"),
          py::arg("x0"),
          py::arg("controls"),
          py::arg("onnx_model_path"),
          py::arg("inference_frequency") = 1,
          R"doc(
Run parallel MuJoCo rollouts with ONNX inference interleaving using a persistent thread pool.

This version maintains a persistent thread pool between calls, providing more stable
performance. ONNX inference is performed at specified intervals during the rollout.

Args:
    models:              length-B list of mujoco._structs.MjModel
    data:                length-B list of mujoco._structs.MjData
    x0:                  2D array of shape (B, nq+nv), batched initial [qpos;qvel]
    controls:            3D array of shape (B, T, nu), open-loop controls
    onnx_model_path:     Path to the ONNX model file
    inference_frequency: Run inference every N steps (default: 1, every step)

Returns:
    tuple of four np.ndarray:
      states     -> shape (B, T, nq+nv) - MuJoCo simulation states
      sensors    -> shape (B, T, nsensordata) - sensor data
      inputs     -> shape (B, T, nu) - applied controls
      inferences -> shape (B, T, nq+nv) - ONNX inference results
)doc");
}
