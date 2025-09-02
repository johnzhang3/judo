#include "onnx_rollout.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>
#include <vector>
#include <iostream>
#include <omp.h>
#include <string>
#include "onnx_interface_wrapped.h"

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

// Removed local ONNXWrappedPolicy (now in onnx_interface_wrapped.{h,cpp})

py::array_t<float> ONNXWrappedPolicyStep(
    const mjModel* model,
    mjData*        data,
    const std::string& onnx_model_path,
    const py::array_t<float>& command,
    const py::array_t<float>& prev_policy
) {
    if (command.ndim() != 1 || command.shape(0) < 25) {
        throw std::runtime_error("command must be 1-D with length >= 25");
    }
    if (prev_policy.ndim() != 1 || prev_policy.shape(0) < 12) {
        throw std::runtime_error("prev_policy must be 1-D with length >= 12");
    }

    const double* qpos_ptr = data->qpos;
    const double* qvel_ptr = data->qvel;
    const float* cmd_ptr = command.data();
    const float* prev_ptr = prev_policy.data();

    static std::unique_ptr<ONNXWrappedPolicy> policy;
    if (!policy) {
        policy = std::make_unique<ONNXWrappedPolicy>(onnx_model_path);
    }

    std::vector<float> ctrl = policy->run(qpos_ptr, model->nq, qvel_ptr, model->nv, cmd_ptr, (int)command.shape(0), prev_ptr, (int)prev_policy.shape(0));
    return py::array_t<float>(ctrl.size(), ctrl.data());
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
    const py::array_t<double>&         commands
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

    // Parse commands (per-step length-25 vectors)
    std::vector<double> commands_buf;
    bool has_commands = false;
    if (commands.size() > 0) {
        has_commands = true;
        if (commands.ndim() == 2) {
            if (commands.shape(0) != B || commands.shape(1) != horizon * 25) {
                throw std::runtime_error("commands must have shape (B, horizon*25)");
            }
            commands_buf.resize(B * horizon * 25);
            std::memcpy(commands_buf.data(), commands.data(), B * horizon * 25 * sizeof(double));
        } else if (commands.ndim() == 1) {
            if (commands.shape(0) != 25) {
                throw std::runtime_error("commands 1D must have length 25");
            }
            commands_buf.resize(B * horizon * 25);
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < horizon; ++t) {
                    std::memcpy(&commands_buf[(b * horizon + t) * 25], commands.data(), 25 * sizeof(double));
                }
            }
        } else {
            throw std::runtime_error("commands must be 1D or 2D");
        }
    }

    // allocate outputs
    std::vector<double> states_buf(B * (horizon + 1) * nstate);  // +1 for initial state
    std::vector<double> actions_buf(B * horizon * nu);
    std::vector<double> sens_buf(B * horizon * nsens);

    {
        py::gil_scoped_release release;

        #pragma omp parallel for
        for (int i = 0; i < B; i++) {
            try {
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
                std::vector<double> current_state(nstate);
                for (int j = 0; j < nq; j++) current_state[j] = d->qpos[j];
                for (int j = 0; j < nv; j++) current_state[nq + j] = d->qvel[j];

                // Create policy instance per batch item
                ONNXWrappedPolicy policy(onnx_model_path);
                std::vector<float> prev(12, 0.0f);

                for (int t = 0; t < horizon; t++) {
                    std::vector<float> prev_out = prev;  // currently zeros
                    float cmd_step[25] = {0};
                    if (has_commands) {
                        const int base = (i * horizon + t) * 25;
                        for (int k = 0; k < 25; ++k) cmd_step[k] = static_cast<float>(commands_buf[base + k]);
                    }
                    std::vector<float> ctrl = policy.run(d->qpos, nq, d->qvel, nv, cmd_step, 25, prev_out.data(), 12);
                    for (int j = 0; j < nu && j < (int)ctrl.size(); j++) d->ctrl[j] = ctrl[j];

                    // Step simulation
                    mj_step(models[i], d);

                    // Record new state
                    for (int j = 0; j < nq; j++) current_state[j] = d->qpos[j];
                    for (int j = 0; j < nv; j++) current_state[nq + j] = d->qvel[j];

                    // Store in output buffers
                    for (int j = 0; j < nstate; j++) {
                        st_ptr[(t + 1) * nstate + j] = current_state[j];
                    }
                    for (int j = 0; j < nu && j < (int)ctrl.size(); j++) act_ptr[t * nu + j] = ctrl[j];
                    for (int j = 0; j < nsens; j++) {
                        se_ptr[t * nsens + j] = d->sensordata[j];
                    }
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
    const py::array_t<double>&         commands
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

    // Parse commands
    std::vector<double> commands_buf;
    bool has_commands = false;
    if (commands.size() > 0) {
        has_commands = true;
        if (commands.ndim() == 2) {
            if (commands.shape(0) != B || commands.shape(1) != horizon * 25) {
                throw std::runtime_error("commands must have shape (B, horizon*25)");
            }
            commands_buf.resize(B * horizon * 25);
            std::memcpy(commands_buf.data(), commands.data(), B * horizon * 25 * sizeof(double));
        } else if (commands.ndim() == 1) {
            if (commands.shape(0) != 25) {
                throw std::runtime_error("commands 1D must have length 25");
            }
            commands_buf.resize(B * horizon * 25);
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < horizon; ++t) {
                    std::memcpy(&commands_buf[(b * horizon + t) * 25], commands.data(), 25 * sizeof(double));
                }
            }
        } else {
            throw std::runtime_error("commands must be 1D or 2D");
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
                // Create per-thread policy
                ONNXWrappedPolicy policy(onnx_model_path);
                std::vector<float> prev(12, 0.0f);
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

                std::vector<double> current_state(nstate);
                for (int j = 0; j < nq; j++) current_state[j] = d->qpos[j];
                for (int j = 0; j < nv; j++) current_state[nq + j] = d->qvel[j];

                for (int t = 0; t < horizon; t++) {
                    std::vector<double> action(nu, 0.0);  // Default zero action

                    float cmd_step[25] = {0};
                    if (has_commands) {
                        const int base = (i * horizon + t) * 25;
                        for (int k = 0; k < 25; ++k) cmd_step[k] = static_cast<float>(commands_buf[base + k]);
                    }
                    std::vector<float> ctrl = policy.run(d->qpos, nq, d->qvel, nv, cmd_step, 25, prev.data(), 12);
                    for (int j = 0; j < nu && j < (int)ctrl.size(); ++j) action[j] = ctrl[j];

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

                    // No histories to maintain
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
