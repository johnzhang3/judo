#pragma once

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>

class ONNXWrappedPolicy {
public:
    explicit ONNXWrappedPolicy(const std::string& model_path);

    std::vector<float> run(const double* qpos_ptr, int nq,
                           const double* qvel_ptr, int nv,
                           const float* command_ptr, int ncmd,
                           const float* prev_ptr, int nprev);

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_str_;
    std::vector<const char*> input_names_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> output_names_;
};


