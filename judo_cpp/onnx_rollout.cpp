#include "onnx_rollout.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>
#include <vector>
#include <iostream>
#include <omp.h>

namespace py = pybind11;

py::array_t<double> make_array(std::vector<double>& buf,
                              int B, int T, int D) {
    std::vector<ssize_t> shape   = { B, T, D };
    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(sizeof(double) * T * D),
        static_cast<ssize_t>(sizeof(double) *     D),
        static_cast<ssize_t>(sizeof(double))
    };

    // Move buf onto the heap so we can own it in the capsule:
    auto heap_buf = new std::vector<double>(std::move(buf));
    // Create a capsule that will delete the vector when the array is gone:
    py::capsule free_when_done(heap_buf, [](void *p) {
        delete reinterpret_cast<std::vector<double>*>(p);
    });

    // Build the array pointing into heap_buf->data() and owning the capsule:
    return py::array_t<double>(
        shape, strides,
        heap_buf->data(),
        free_when_done
    );
}

ONNXInference::ONNXInference(const std::string& model_path) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(1);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);

        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;

        // Input info
        size_t num_input_nodes = session_->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_str_.push_back(std::string(input_name.get()));
            input_names_.push_back(input_names_str_.back().c_str());

            auto input_type_info = session_->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            input_shape_ = input_tensor_info.GetShape();
        }

        // Output info
        size_t num_output_nodes = session_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_str_.push_back(std::string(output_name.get()));
            output_names_.push_back(output_names_str_.back().c_str());

            auto output_type_info = session_->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            output_shape_ = output_tensor_info.GetShape();
        }

        std::cout << "ONNX model loaded successfully" << std::endl;
        std::cout << "Input shape: [";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            std::cout << input_shape_[i];
            if (i < input_shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

    } catch (const Ort::Exception& e) {
        throw std::runtime_error("Failed to load ONNX model: " + std::string(e.what()));
    }
}

ONNXInference::~ONNXInference() = default;

std::vector<float> ONNXInference::run_inference(const std::vector<float>& input,
                                               const std::vector<int64_t>& input_shape) {
    try {
        // Create input tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input.data()),
            input.size(),
            input_shape.data(),
            input_shape.size()
        );

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            1,
            output_names_.data(),
            1
        );

        // Extract output
        float* float_array = output_tensors[0].GetTensorMutableData<float>();
        auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        size_t output_count = output_info.GetElementCount();

        return std::vector<float>(float_array, float_array + output_count);

    } catch (const Ort::Exception& e) {
        throw std::runtime_error("ONNX inference failed: " + std::string(e.what()));
    }
}

std::vector<int64_t> ONNXInference::get_input_shape() const {
    return input_shape_;
}

std::vector<int64_t> ONNXInference::get_output_shape() const {
    return output_shape_;
}

py::tuple ONNXInterleaveRollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>&        data,
    const py::array_t<double>&         x0,
    const py::array_t<double>&         controls,
    const std::string&                 onnx_model_path,
    int                                inference_frequency
) {
    int B = (int)models.size();
    if (B == 0 || B != (int)data.size()) {
        throw std::runtime_error("models/data must have same non-zero length");
    }

    // dims from first model
    const mjModel* m0 = models[0];
    int nq     = m0->nq;
    int nv     = m0->nv;
    int nu     = m0->nu;
    int nsens  = m0->nsensordata;
    int nstate = nq + nv;

    // x0: 1D of length nstate
    if (x0.ndim() != 1 || x0.shape(0) != nstate) {
        throw std::runtime_error("x0 must be a 1D array of length nq+nv");
    }
    const double* x0_ptr = x0.data();

    // controls: 3D (B, T, nu)
    if (controls.ndim() != 3
    || controls.shape(0) != B
    || controls.shape(2) != nu) {
        throw std::runtime_error(
        "controls must be a 3D array of shape (B, T, nu)");
    }
    int T = (int)controls.shape(1);
    const double* ctrl_ptr = controls.data();

    py::print("ONNXInterleaveRollout called");

    // Load ONNX model (one per thread to avoid conflicts)
    std::vector<std::unique_ptr<ONNXInference>> onnx_models(B);

    // allocate outputs
    std::vector<double> states_buf(B * T * nstate);
    std::vector<double> inputs_buf(B * T *     nu);
    std::vector<double> sens_buf  (B * T *  nsens);
    std::vector<double> inference_buf(B * T * nstate);  // Store inference results

    {
        py::gil_scoped_release release;

        #pragma omp parallel for
        for (int i = 0; i < B; i++) {
            try {
                // Create ONNX model for this thread
                onnx_models[i] = std::make_unique<ONNXInference>(onnx_model_path);

                mjData* d = data[i];
                // set initial qpos+qvel
                d->time = 0.0;
                mj_setState(models[i], d, x0_ptr, mjSTATE_QPOS | mjSTATE_QVEL);
                mj_forward(models[i], d);
                mju_zero(d->qacc_warmstart, m0->nv);

                double* st_ptr = &states_buf[i * T * nstate];
                double* in_ptr = &inputs_buf[i * T *     nu];
                double* se_ptr = &sens_buf  [i * T *  nsens];
                double* inf_ptr = &inference_buf[i * T * nstate];
                const double* ctrl_i = ctrl_ptr + i * (T * nu);

                for (int t = 0; t < T; t++) {
                    // apply this instance's t-th control
                    for (int j = 0; j < nu; j++) {
                        d->ctrl[j] = ctrl_i[t*nu + j];
                    }

                    mj_step(models[i], d);

                    // record qpos/qvel before potential ONNX modification
                    std::vector<double> current_state(nstate);
                    for (int j = 0; j < nq;  j++) current_state[j] = d->qpos[j];
                    for (int j = 0; j < nv;  j++) current_state[nq+j] = d->qvel[j];

                    // Store original state
                    for (int j = 0; j < nstate; j++) {
                        st_ptr[t*nstate + j] = current_state[j];
                    }

                    // Run ONNX inference if it's time
                    if (inference_frequency > 0 && (t + 1) % inference_frequency == 0) {
                        // Convert to float for ONNX
                        std::vector<float> state_float(nstate);
                        for (int j = 0; j < nstate; j++) {
                            state_float[j] = static_cast<float>(current_state[j]);
                        }

                        // Run inference (assuming batch size 1)
                        std::vector<int64_t> input_shape = {1, nstate};
                        std::vector<float> onnx_output = onnx_models[i]->run_inference(state_float, input_shape);

                        // Store inference result
                        for (int j = 0; j < nstate && j < (int)onnx_output.size(); j++) {
                            inf_ptr[t*nstate + j] = static_cast<double>(onnx_output[j]);
                        }

                        // Optionally apply ONNX output back to MuJoCo state
                        // For identity network, this should be the same, but for other networks
                        // you might want to modify the state here
                        //
                        // Example (commented out for safety):
                        // for (int j = 0; j < nq && j < (int)onnx_output.size(); j++) {
                        //     d->qpos[j] = static_cast<double>(onnx_output[j]);
                        // }
                        // for (int j = 0; j < nv && (j + nq) < (int)onnx_output.size(); j++) {
                        //     d->qvel[j] = static_cast<double>(onnx_output[j + nq]);
                        // }
                        // mj_forward(models[i], d);  // Update derived quantities
                    } else {
                        // No inference this step, just copy original state
                        for (int j = 0; j < nstate; j++) {
                            inf_ptr[t*nstate + j] = current_state[j];
                        }
                    }

                    // record sensors
                    for (int j = 0; j < nsens; j++) se_ptr[t*nsens + j] = d->sensordata[j];
                    // record applied control
                    for (int j = 0; j < nu;   j++) in_ptr[t*nu   + j] = d->ctrl[j];
                }
            } catch (const std::exception& e) {
                // Handle exceptions in parallel region
                std::cerr << "Error in rollout thread " << i << ": " << e.what() << std::endl;
                // Fill with zeros or original values as fallback
                double* st_ptr = &states_buf[i * T * nstate];
                double* in_ptr = &inputs_buf[i * T *     nu];
                double* se_ptr = &sens_buf  [i * T *  nsens];
                double* inf_ptr = &inference_buf[i * T * nstate];

                for (int t = 0; t < T; t++) {
                    for (int j = 0; j < nstate; j++) {
                        st_ptr[t*nstate + j] = 0.0;
                        inf_ptr[t*nstate + j] = 0.0;
                    }
                    for (int j = 0; j < nsens; j++) se_ptr[t*nsens + j] = 0.0;
                    for (int j = 0; j < nu;   j++) in_ptr[t*nu   + j] = 0.0;
                }
            }
        }
    }

    auto states_arr = make_array(states_buf, B, T, nstate);
    auto sens_arr   = make_array(sens_buf,   B, T,  nsens);
    auto inputs_arr = make_array(inputs_buf, B, T,     nu);
    auto inference_arr = make_array(inference_buf, B, T, nstate);

    return py::make_tuple(states_arr, sens_arr, inputs_arr, inference_arr);
}
