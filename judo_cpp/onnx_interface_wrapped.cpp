#include "onnx_interface_wrapped.h"

#include <stdexcept>
#include <iostream>

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
        }

        size_t num_output_nodes = session_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_str_.push_back(std::string(output_name.get()));
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
    std::vector<int64_t> qpos_shape = {expected_qpos};
    for (int i = 0; i < expected_qpos; ++i) qpos_f[i] = static_cast<float>(qpos_ptr[i]);
    Ort::Value qpos_tensor = Ort::Value::CreateTensor<float>(memory_info, qpos_f.data(), qpos_f.size(), qpos_shape.data(), qpos_shape.size());

    std::vector<float> qvel_f(expected_qvel);
    std::vector<int64_t> qvel_shape = {expected_qvel};
    for (int i = 0; i < expected_qvel; ++i) qvel_f[i] = static_cast<float>(qvel_ptr[i]);
    Ort::Value qvel_tensor = Ort::Value::CreateTensor<float>(memory_info, qvel_f.data(), qvel_f.size(), qvel_shape.data(), qvel_shape.size());

    std::vector<float> cmd_f(expected_cmd);
    std::vector<int64_t> cmd_shape = {expected_cmd};
    for (int i = 0; i < expected_cmd; ++i) cmd_f[i] = command_ptr[i];
    Ort::Value cmd_tensor = Ort::Value::CreateTensor<float>(memory_info, cmd_f.data(), cmd_f.size(), cmd_shape.data(), cmd_shape.size());

    std::vector<float> prev_f(expected_prev);
    std::vector<int64_t> prev_shape = {expected_prev};
    for (int i = 0; i < expected_prev; ++i) prev_f[i] = prev_ptr[i];
    Ort::Value prev_tensor = Ort::Value::CreateTensor<float>(memory_info, prev_f.data(), prev_f.size(), prev_shape.data(), prev_shape.size());

    auto find_index_contains = [&](const char* needle) -> int {
        for (size_t i = 0; i < input_names_str_.size(); ++i) {
            if (input_names_str_[i].find(needle) != std::string::npos) return static_cast<int>(i);
        }
        return -1;
    };
    int iqpos = find_index_contains("qpos");
    int iqvel = find_index_contains("qvel");
    int icmd  = find_index_contains("command");
    if (icmd < 0) icmd = find_index_contains("cmd");
    int iprev = find_index_contains("prev_policy");
    if (iprev < 0) iprev = find_index_contains("prev");

    std::vector<const char*> feed_names;
    std::vector<Ort::Value> feed_values;
    feed_names.reserve(4);
    feed_values.reserve(4);

    bool have_all = (iqpos >= 0 && iqvel >= 0 && icmd >= 0 && iprev >= 0);
    if (have_all) {
        feed_names.push_back(input_names_str_[iqpos].c_str());
        feed_values.push_back(std::move(qpos_tensor));
        feed_names.push_back(input_names_str_[iqvel].c_str());
        feed_values.push_back(std::move(qvel_tensor));
        feed_names.push_back(input_names_str_[icmd].c_str());
        feed_values.push_back(std::move(cmd_tensor));
        feed_names.push_back(input_names_str_[iprev].c_str());
        feed_values.push_back(std::move(prev_tensor));
    } else {
        // Fallback to positional first four inputs
        if (input_names_str_.size() < 4) {
            throw std::runtime_error("ONNXWrappedPolicy: model exposes fewer than 4 inputs");
        }
        feed_names.push_back(input_names_str_[0].c_str());
        feed_values.push_back(std::move(qpos_tensor));
        feed_names.push_back(input_names_str_[1].c_str());
        feed_values.push_back(std::move(qvel_tensor));
        feed_names.push_back(input_names_str_[2].c_str());
        feed_values.push_back(std::move(cmd_tensor));
        feed_names.push_back(input_names_str_[3].c_str());
        feed_values.push_back(std::move(prev_tensor));
    }

    std::vector<const char*> output_names_cstr;
    output_names_cstr.reserve(output_names_str_.size());
    for (const auto& s : output_names_str_) output_names_cstr.push_back(s.c_str());

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        feed_names.data(),
        feed_values.data(),
        feed_values.size(),
        output_names_cstr.data(),
        output_names_cstr.size());

    float* out_ptr = output_tensors[0].GetTensorMutableData<float>();
    auto out_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t out_count = out_info.GetElementCount();
    return std::vector<float>(out_ptr, out_ptr + out_count);
}


