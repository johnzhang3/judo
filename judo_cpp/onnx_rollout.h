#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>
#include <vector>
#include <string>
#include <memory>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

namespace py = pybind11;

// Utility function to create numpy arrays from C++ vectors
py::array_t<double> make_array(std::vector<double>& buf, int B, int T, int D);

class ONNXInference {
public:
    ONNXInference(const std::string& model_path);
    ~ONNXInference();

    // Run inference on a batch of inputs
    std::vector<float> run_inference(const std::vector<float>& input,
                                   const std::vector<int64_t>& input_shape);

    // Get input/output dimensions
    std::vector<int64_t> get_input_shape() const;
    std::vector<int64_t> get_output_shape() const;

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;

    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
};

py::tuple ONNXInterleaveRollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>&        data,
    const py::array_t<double>&         x0,
    const py::array_t<double>&         controls,
    const std::string&                 onnx_model_path,
    int                                inference_frequency = 1
);
