#include "onnx_interface_wrapped.h"

#include <stdexcept>

ONNXWrappedPolicy::ONNXWrappedPolicy(const std::string& model_path) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXWrappedPolicy");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(1);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);

        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session_->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_str_.push_back(std::string(input_name.get()));
            input_names_.push_back(input_names_str_.back().c_str());
        }

        size_t num_output_nodes = session_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_str_.push_back(std::string(output_name.get()));
            output_names_.push_back(output_names_str_.back().c_str());
        }
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("Failed to load ONNX wrapped policy: " + std::string(e.what()));
    }
}

std::vector<float> ONNXWrappedPolicy::run(const double* qpos_ptr, int nq,
                                          const double* qvel_ptr, int nv,
                                          const float* command_ptr, int ncmd,
                                          const float* prev_ptr, int nprev) {
    const int expected_qpos = 26;
    const int expected_qvel = 25;
    const int expected_cmd  = 25;
    const int expected_prev = 12;
    if (nq < expected_qpos || nv < expected_qvel || ncmd < expected_cmd || nprev < expected_prev) {
        throw std::runtime_error("ONNXWrappedPolicy: input sizes are smaller than expected");
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> qpos_f(expected_qpos);
    std::vector<int64_t> qpos_shape = {1, expected_qpos};
    for (int i = 0; i < expected_qpos; ++i) qpos_f[i] = static_cast<float>(qpos_ptr[i]);
    Ort::Value qpos_tensor = Ort::Value::CreateTensor<float>(memory_info, qpos_f.data(), qpos_f.size(), qpos_shape.data(), qpos_shape.size());

    std::vector<float> qvel_f(expected_qvel);
    std::vector<int64_t> qvel_shape = {1, expected_qvel};
    for (int i = 0; i < expected_qvel; ++i) qvel_f[i] = static_cast<float>(qvel_ptr[i]);
    Ort::Value qvel_tensor = Ort::Value::CreateTensor<float>(memory_info, qvel_f.data(), qvel_f.size(), qvel_shape.data(), qvel_shape.size());

    std::vector<float> cmd_f(expected_cmd);
    std::vector<int64_t> cmd_shape = {1, expected_cmd};
    for (int i = 0; i < expected_cmd; ++i) cmd_f[i] = command_ptr[i];
    Ort::Value cmd_tensor = Ort::Value::CreateTensor<float>(memory_info, cmd_f.data(), cmd_f.size(), cmd_shape.data(), cmd_shape.size());

    std::vector<float> prev_f(expected_prev);
    std::vector<int64_t> prev_shape = {1, expected_prev};
    for (int i = 0; i < expected_prev; ++i) prev_f[i] = prev_ptr[i];
    Ort::Value prev_tensor = Ort::Value::CreateTensor<float>(memory_info, prev_f.data(), prev_f.size(), prev_shape.data(), prev_shape.size());

    auto find_index = [&](const char* name) -> int {
        for (size_t i = 0; i < input_names_.size(); ++i) if (input_names_[i] == std::string(name)) return static_cast<int>(i);
        return -1;
    };
    std::vector<Ort::Value> ordered_inputs(input_names_.size());
    int iqpos = find_index("qpos");
    int iqvel = find_index("qvel");
    int icmd  = find_index("command");
    int iprev = find_index("prev_policy");
    if (iqpos < 0 || iqvel < 0 || icmd < 0 || iprev < 0) {
        throw std::runtime_error("ONNXWrappedPolicy: expected inputs qpos,qvel,command,prev_policy not found");
    }
    ordered_inputs[iqpos] = std::move(qpos_tensor);
    ordered_inputs[iqvel] = std::move(qvel_tensor);
    ordered_inputs[icmd]  = std::move(cmd_tensor);
    ordered_inputs[iprev] = std::move(prev_tensor);

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        ordered_inputs.data(),
        ordered_inputs.size(),
        output_names_.data(),
        output_names_.size());

    float* out_ptr = output_tensors[0].GetTensorMutableData<float>();
    auto out_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t out_count = out_info.GetElementCount();
    return std::vector<float>(out_ptr, out_ptr + out_count);
}


