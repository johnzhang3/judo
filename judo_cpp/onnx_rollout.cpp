#include "onnx_rollout.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>
#include <vector>
#include <iostream>
#include <omp.h>

namespace py = pybind11;

// ========== Thread Pool Implementation ==========

PersistentThreadPool::PersistentThreadPool(int num_threads)
    : num_threads_(num_threads), stop_(false), active_workers_(0), total_tasks_(0), completed_tasks_(0) {
    threads_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads_.emplace_back(&PersistentThreadPool::worker_thread, this);
    }
}

PersistentThreadPool::~PersistentThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void PersistentThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty()) {
                return;
            }

            if (!tasks_.empty()) {
                task = std::move(tasks_.front());
                tasks_.pop();
                active_workers_++;
            }
        }

        if (task) {
            task();

            std::unique_lock<std::mutex> lock(queue_mutex_);
            active_workers_--;
            completed_tasks_++;
            if (completed_tasks_ == total_tasks_) {
                finished_.notify_one();
            }
        }
    }
}

void PersistentThreadPool::execute_parallel(std::function<void(int)> func, int total_work) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        total_tasks_ = total_work;
        completed_tasks_ = 0;

        for (int i = 0; i < total_work; ++i) {
            tasks_.emplace([func, i] { func(i); });
        }
    }

    condition_.notify_all();

    // Wait for all tasks to complete
    std::unique_lock<std::mutex> lock(queue_mutex_);
    finished_.wait(lock, [this] { return completed_tasks_ == total_tasks_; });
}

// Thread Pool Manager Implementation
ThreadPoolManager& ThreadPoolManager::instance() {
    static ThreadPoolManager instance;
    return instance;
}

PersistentThreadPool* ThreadPoolManager::get_pool(int num_threads) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    if (!current_pool_ || current_num_threads_ != num_threads) {
        current_pool_.reset();  // Destroy old pool
        current_pool_ = std::make_unique<PersistentThreadPool>(num_threads);
        current_num_threads_ = num_threads;
    }

    return current_pool_.get();
}

void ThreadPoolManager::shutdown() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    current_pool_.reset();
    current_num_threads_ = 0;
}

// ========== End Thread Pool Implementation ==========

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



py::tuple ONNXPolicyRollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>&        data,
    const py::array_t<double>&         x0,
    int                                horizon,
    const std::string&                 onnx_model_path,
    int                                state_history_length,
    int                                action_history_length,
    int                                inference_frequency,
    const py::array_t<double>&         additional_inputs
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

    // x0: 2D of shape (B, nstate) for batched initial states
    if (x0.ndim() != 2 || x0.shape(0) != B || x0.shape(1) != nstate) {
        throw std::runtime_error("x0 must be a 2D array of shape (B, nq+nv)");
    }
    const double* x0_ptr = x0.data();

    // Parse additional inputs
    std::vector<double> additional_input_data;
    int additional_input_dim = 0;
    if (additional_inputs.size() > 0) {
        if (additional_inputs.ndim() == 2 && additional_inputs.shape(0) == B) {
            additional_input_dim = additional_inputs.shape(1);
            additional_input_data.resize(B * additional_input_dim);
            std::memcpy(additional_input_data.data(), additional_inputs.data(),
                       B * additional_input_dim * sizeof(double));
        } else if (additional_inputs.ndim() == 1) {
            // Broadcast to all rollouts
            additional_input_dim = additional_inputs.shape(0);
            additional_input_data.resize(B * additional_input_dim);
            for (int b = 0; b < B; ++b) {
                std::memcpy(&additional_input_data[b * additional_input_dim],
                           additional_inputs.data(), additional_input_dim * sizeof(double));
            }
        } else {
            throw std::runtime_error("additional_inputs must be 1D or 2D array");
        }
    }

    // allocate outputs
    std::vector<double> states_buf(B * (horizon + 1) * nstate);  // +1 for initial state
    std::vector<double> actions_buf(B * horizon * nu);
    std::vector<double> sens_buf(B * horizon * nsens);

    // Create thread-local ONNX models and states
    std::vector<std::unique_ptr<ONNXInference>> onnx_models(B);
    std::vector<PolicyThreadState> thread_states;
    thread_states.reserve(B);

    {
        py::gil_scoped_release release;

        // Initialize thread states
        for (int i = 0; i < B; i++) {
            thread_states.emplace_back(state_history_length, action_history_length, nstate, nu);
            thread_states[i].initialize_onnx_model(onnx_model_path);
            thread_states[i].inference_frequency = inference_frequency;
        }

        #pragma omp parallel for
        for (int i = 0; i < B; i++) {
            try {
                mjData* d = data[i];
                PolicyThreadState& thread_state = thread_states[i];

                // set initial qpos+qvel for this batch
                d->time = 0.0;
                const double* x0_i = x0_ptr + i * nstate;
                mj_setState(models[i], d, x0_i, mjSTATE_QPOS | mjSTATE_QVEL);
                mj_forward(models[i], d);
                mju_zero(d->qacc_warmstart, m0->nv);

                double* st_ptr = &states_buf[i * (horizon + 1) * nstate];
                double* act_ptr = &actions_buf[i * horizon * nu];
                double* se_ptr = &sens_buf[i * horizon * nsens];

                // Store initial state
                for (int j = 0; j < nq; j++) st_ptr[j] = d->qpos[j];
                for (int j = 0; j < nv; j++) st_ptr[nq + j] = d->qvel[j];

                // Add initial state to history
                std::vector<double> current_state(nstate);
                for (int j = 0; j < nq; j++) current_state[j] = d->qpos[j];
                for (int j = 0; j < nv; j++) current_state[nq + j] = d->qvel[j];
                thread_state.state_history.push(current_state);

                for (int t = 0; t < horizon; t++) {
                    std::vector<double> action(nu, 0.0);  // Default zero action

                    // Always go through ONNX inference (additional inputs are part of the ONNX input vector)
                    if (inference_frequency > 0 && (t + 1) % inference_frequency == 0) {
                        std::vector<double> addl_inputs;
                        if (additional_input_dim == horizon * nu) {
                            // Use only the next candidate action (nu) from the provided plan for this timestep
                            const int base = (i * additional_input_dim) + (t * nu);
                            addl_inputs.assign(
                                &additional_input_data[base],
                                &additional_input_data[base + nu]
                            );
                        } else if (additional_input_dim > 0) {
                            addl_inputs.assign(
                                &additional_input_data[i * additional_input_dim],
                                &additional_input_data[(i + 1) * additional_input_dim]
                            );
                        }

                        std::vector<float> onnx_input = thread_state.prepare_onnx_input(addl_inputs);
                        std::vector<int64_t> input_shape = {1, (int64_t)onnx_input.size()};
                        std::vector<float> onnx_output = thread_state.onnx_model->run_inference(onnx_input, input_shape);
                        for (int j = 0; j < nu && j < (int)onnx_output.size(); j++) {
                            action[j] = static_cast<double>(onnx_output[j]);
                        }
                    }

                    // Apply action to simulation
                    for (int j = 0; j < nu; j++) {
                        d->ctrl[j] = action[j];
                    }

                    // Step simulation
                    mj_step(models[i], d);

                    // Record new state
                    for (int j = 0; j < nq; j++) current_state[j] = d->qpos[j];
                    for (int j = 0; j < nv; j++) current_state[nq + j] = d->qvel[j];

                    // Store in output buffers
                    for (int j = 0; j < nstate; j++) {
                        st_ptr[(t + 1) * nstate + j] = current_state[j];
                    }
                    for (int j = 0; j < nu; j++) {
                        act_ptr[t * nu + j] = action[j];
                    }
                    for (int j = 0; j < nsens; j++) {
                        se_ptr[t * nsens + j] = d->sensordata[j];
                    }

                    // Update histories for next iteration
                    thread_state.state_history.push(current_state);
                    thread_state.action_history.push(action);
                }

            } catch (const std::exception& e) {
                std::cerr << "Error in policy rollout thread " << i << ": " << e.what() << std::endl;
                // Fill with zeros as fallback
                double* st_ptr = &states_buf[i * (horizon + 1) * nstate];
                double* act_ptr = &actions_buf[i * horizon * nu];
                double* se_ptr = &sens_buf[i * horizon * nsens];

                for (int t = 0; t <= horizon; t++) {
                    for (int j = 0; j < nstate; j++) {
                        st_ptr[t * nstate + j] = 0.0;
                    }
                }
                for (int t = 0; t < horizon; t++) {
                    for (int j = 0; j < nu; j++) act_ptr[t * nu + j] = 0.0;
                    for (int j = 0; j < nsens; j++) se_ptr[t * nsens + j] = 0.0;
                }
            }
        }
    }

    auto states_arr = make_array(states_buf, B, horizon + 1, nstate);
    auto actions_arr = make_array(actions_buf, B, horizon, nu);
    auto sens_arr = make_array(sens_buf, B, horizon, nsens);

    return py::make_tuple(states_arr, actions_arr, sens_arr);
}

py::tuple PersistentONNXPolicyRollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>&        data,
    const py::array_t<double>&         x0,
    int                                horizon,
    const std::string&                 onnx_model_path,
    int                                state_history_length,
    int                                action_history_length,
    int                                inference_frequency,
    const py::array_t<double>&         additional_inputs
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

    // x0: 2D of shape (B, nstate) for batched initial states
    if (x0.ndim() != 2 || x0.shape(0) != B || x0.shape(1) != nstate) {
        throw std::runtime_error("x0 must be a 2D array of shape (B, nq+nv)");
    }
    const double* x0_ptr = x0.data();

    // Parse additional inputs
    std::vector<double> additional_input_data;
    int additional_input_dim = 0;
    if (additional_inputs.size() > 0) {
        if (additional_inputs.ndim() == 2 && additional_inputs.shape(0) == B) {
            additional_input_dim = additional_inputs.shape(1);
            additional_input_data.resize(B * additional_input_dim);
            std::memcpy(additional_input_data.data(), additional_inputs.data(),
                       B * additional_input_dim * sizeof(double));
        } else if (additional_inputs.ndim() == 1) {
            // Broadcast to all rollouts
            additional_input_dim = additional_inputs.shape(0);
            additional_input_data.resize(B * additional_input_dim);
            for (int b = 0; b < B; ++b) {
                std::memcpy(&additional_input_data[b * additional_input_dim],
                           additional_inputs.data(), additional_input_dim * sizeof(double));
            }
        } else {
            throw std::runtime_error("additional_inputs must be 1D or 2D array");
        }
    }

    // allocate outputs
    std::vector<double> states_buf(B * (horizon + 1) * nstate);
    std::vector<double> actions_buf(B * horizon * nu);
    std::vector<double> sens_buf(B * horizon * nsens);

    {
        py::gil_scoped_release release;

        // Use persistent thread pool
        PersistentThreadPool* pool = ThreadPoolManager::instance().get_pool(B);

        pool->execute_parallel([&](int i) {
            try {
                // Get thread-local state
                PolicyThreadState* thread_state = PolicyThreadStateManager::instance()
                    .get_thread_state(nstate, nu, state_history_length, action_history_length);

                thread_state->reset();
                thread_state->initialize_onnx_model(onnx_model_path);
                thread_state->inference_frequency = inference_frequency;

                mjData* d = data[i];

                // set initial qpos+qvel for this batch
                d->time = 0.0;
                const double* x0_i = x0_ptr + i * nstate;
                mj_setState(models[i], d, x0_i, mjSTATE_QPOS | mjSTATE_QVEL);
                mj_forward(models[i], d);
                mju_zero(d->qacc_warmstart, m0->nv);

                double* st_ptr = &states_buf[i * (horizon + 1) * nstate];
                double* act_ptr = &actions_buf[i * horizon * nu];
                double* se_ptr = &sens_buf[i * horizon * nsens];

                // Store initial state
                for (int j = 0; j < nq; j++) st_ptr[j] = d->qpos[j];
                for (int j = 0; j < nv; j++) st_ptr[nq + j] = d->qvel[j];

                // Add initial state to history
                std::vector<double> current_state(nstate);
                for (int j = 0; j < nq; j++) current_state[j] = d->qpos[j];
                for (int j = 0; j < nv; j++) current_state[nq + j] = d->qvel[j];
                thread_state->state_history.push(current_state);

                for (int t = 0; t < horizon; t++) {
                    std::vector<double> action(nu, 0.0);  // Default zero action

                    // Always run ONNX inference (no pass-through in C++)
                    if (inference_frequency > 0 && (t + 1) % inference_frequency == 0) {
                        std::vector<double> addl_inputs;
                        if (additional_input_dim == horizon * nu) {
                            const int base = (i * additional_input_dim) + (t * nu);
                            addl_inputs.assign(
                                &additional_input_data[base],
                                &additional_input_data[base + nu]
                            );
                        } else if (additional_input_dim > 0) {
                            addl_inputs.assign(
                                &additional_input_data[i * additional_input_dim],
                                &additional_input_data[(i + 1) * additional_input_dim]
                            );
                        }
                        std::vector<float> onnx_input = thread_state->prepare_onnx_input(addl_inputs);
                        std::vector<int64_t> input_shape = {1, (int64_t)onnx_input.size()};
                        std::vector<float> onnx_output = thread_state->onnx_model->run_inference(onnx_input, input_shape);
                        for (int j = 0; j < nu && j < (int)onnx_output.size(); j++) {
                            action[j] = static_cast<double>(onnx_output[j]);
                        }
                    }

                    // Apply action to simulation
                    for (int j = 0; j < nu; j++) {
                        d->ctrl[j] = action[j];
                    }

                    // Step simulation
                    mj_step(models[i], d);

                    // Record new state
                    for (int j = 0; j < nq; j++) current_state[j] = d->qpos[j];
                    for (int j = 0; j < nv; j++) current_state[nq + j] = d->qvel[j];

                    // Store in output buffers
                    for (int j = 0; j < nstate; j++) {
                        st_ptr[(t + 1) * nstate + j] = current_state[j];
                    }
                    for (int j = 0; j < nu; j++) {
                        act_ptr[t * nu + j] = action[j];
                    }
                    for (int j = 0; j < nsens; j++) {
                        se_ptr[t * nsens + j] = d->sensordata[j];
                    }

                    // Update histories for next iteration
                    thread_state->state_history.push(current_state);
                    thread_state->action_history.push(action);
                }

            } catch (const std::exception& e) {
                std::cerr << "Error in persistent policy rollout thread " << i << ": " << e.what() << std::endl;
                // Fill with zeros as fallback
                double* st_ptr = &states_buf[i * (horizon + 1) * nstate];
                double* act_ptr = &actions_buf[i * horizon * nu];
                double* se_ptr = &sens_buf[i * horizon * nsens];

                for (int t = 0; t <= horizon; t++) {
                    for (int j = 0; j < nstate; j++) {
                        st_ptr[t * nstate + j] = 0.0;
                    }
                }
                for (int t = 0; t < horizon; t++) {
                    for (int j = 0; j < nu; j++) act_ptr[t * nu + j] = 0.0;
                    for (int j = 0; j < nsens; j++) se_ptr[t * nsens + j] = 0.0;
                }
            }
        }, B);
    }

    auto states_arr = make_array(states_buf, B, horizon + 1, nstate);
    auto actions_arr = make_array(actions_buf, B, horizon, nu);
    auto sens_arr = make_array(sens_buf, B, horizon, nsens);

    return py::make_tuple(states_arr, actions_arr, sens_arr);
}
