#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <deque>
#include <unordered_map>

namespace py = pybind11;

// Circular buffer template for efficient history management
template<typename T>
class CircularBuffer {
public:
    CircularBuffer(size_t capacity) : capacity_(capacity), data_(capacity) {}

    void push(const T& item) {
        data_[head_] = item;
        head_ = (head_ + 1) % capacity_;
        if (size_ < capacity_) {
            size_++;
        }
    }

    void clear() {
        head_ = 0;
        size_ = 0;
    }

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }
    bool full() const { return size_ == capacity_; }

    // Get item at index (0 = oldest, size()-1 = newest)
    const T& operator[](size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        size_t actual_index = (head_ - size_ + index + capacity_) % capacity_;
        return data_[actual_index];
    }

    // Get flattened vector of all data in chronological order (oldest to newest)
    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(size_);
        for (size_t i = 0; i < size_; ++i) {
            result.push_back((*this)[i]);
        }
        return result;
    }

private:
    size_t capacity_;
    size_t head_ = 0;
    size_t size_ = 0;
    std::vector<T> data_;
};

// Forward declaration
class ONNXInference;

// Thread-local state for policy rollouts
struct PolicyThreadState {
    // ONNX inference components
    std::unique_ptr<ONNXInference> onnx_model;

    // History buffers
    CircularBuffer<std::vector<double>> state_history;
    CircularBuffer<std::vector<double>> action_history;

    // Pre-allocated buffers for efficiency
    std::vector<float> onnx_input_buffer;
    std::vector<float> onnx_output_buffer;
    std::vector<double> combined_input;  // For concatenating state/action histories

    // Configuration
    int state_history_length;
    int action_history_length;
    int inference_frequency;
    int state_dim;
    int action_dim;

    PolicyThreadState(int state_hist_len, int action_hist_len, int state_d, int action_d)
        : state_history(state_hist_len), action_history(action_hist_len),
          state_history_length(state_hist_len), action_history_length(action_hist_len),
          state_dim(state_d), action_dim(action_d), inference_frequency(1) {

        // Pre-allocate buffers
        int max_input_size = state_hist_len * state_d + action_hist_len * action_d;
        onnx_input_buffer.reserve(max_input_size);
        combined_input.reserve(max_input_size);
    }

    void initialize_onnx_model(const std::string& model_path) {
        onnx_model = std::make_unique<ONNXInference>(model_path);
    }

    void reset() {
        state_history.clear();
        action_history.clear();
        onnx_input_buffer.clear();
        onnx_output_buffer.clear();
        combined_input.clear();
    }

    // Prepare ONNX input by concatenating state and action histories
    std::vector<float> prepare_onnx_input(const std::vector<double>& additional_inputs = {}) {
        combined_input.clear();

        // Add state history (chronological order)
        auto state_hist = state_history.to_vector();
        for (const auto& state : state_hist) {
            combined_input.insert(combined_input.end(), state.begin(), state.end());
        }

        // Pad with zeros if we don't have enough state history
        size_t states_needed = state_history_length - state_history.size();
        for (size_t i = 0; i < states_needed; ++i) {
            combined_input.insert(combined_input.end(), state_dim, 0.0);
        }

        // Add action history (chronological order)
        auto action_hist = action_history.to_vector();
        for (const auto& action : action_hist) {
            combined_input.insert(combined_input.end(), action.begin(), action.end());
        }

        // Pad with zeros if we don't have enough action history
        size_t actions_needed = action_history_length - action_history.size();
        for (size_t i = 0; i < actions_needed; ++i) {
            combined_input.insert(combined_input.end(), action_dim, 0.0);
        }

        // Add any additional inputs (goals, commands, etc.)
        combined_input.insert(combined_input.end(), additional_inputs.begin(), additional_inputs.end());

        // Convert to float for ONNX
        onnx_input_buffer.clear();
        onnx_input_buffer.reserve(combined_input.size());
        for (double val : combined_input) {
            onnx_input_buffer.push_back(static_cast<float>(val));
        }

        return onnx_input_buffer;
    }
};

// Thread-local storage manager for policy states
class PolicyThreadStateManager {
public:
    static PolicyThreadStateManager& instance() {
        static PolicyThreadStateManager instance;
        return instance;
    }

    // Get or create thread-local state for current thread
    PolicyThreadState* get_thread_state(int state_dim, int action_dim,
                                       int state_hist_len, int action_hist_len) {
        std::thread::id thread_id = std::this_thread::get_id();
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = thread_states_.find(thread_id);
        if (it == thread_states_.end()) {
            // Create new state for this thread
            auto state = std::make_unique<PolicyThreadState>(
                state_hist_len, action_hist_len, state_dim, action_dim);
            PolicyThreadState* state_ptr = state.get();
            thread_states_[thread_id] = std::move(state);
            return state_ptr;
        } else {
            // Return existing state, but update dimensions if needed
            PolicyThreadState* state = it->second.get();
            if (state->state_dim != state_dim || state->action_dim != action_dim ||
                state->state_history_length != state_hist_len ||
                state->action_history_length != action_hist_len) {
                // Recreate with new dimensions
                auto new_state = std::make_unique<PolicyThreadState>(
                    state_hist_len, action_hist_len, state_dim, action_dim);
                PolicyThreadState* new_state_ptr = new_state.get();
                it->second = std::move(new_state);
                return new_state_ptr;
            }
            return state;
        }
    }

    // Clean up thread states (call when shutting down)
    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex_);
        thread_states_.clear();
    }

    // Reset all thread states
    void reset_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pair : thread_states_) {
            pair.second->reset();
        }
    }

private:
    std::mutex mutex_;
    std::unordered_map<std::thread::id, std::unique_ptr<PolicyThreadState>> thread_states_;
};

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

// Thread pool for persistent thread management
class PersistentThreadPool {
public:
    PersistentThreadPool(int num_threads);
    ~PersistentThreadPool();

    // Execute a function across the thread pool
    void execute_parallel(std::function<void(int)> func, int total_work);

    int get_num_threads() const { return num_threads_; }

private:
    int num_threads_;
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable finished_;
    bool stop_;
    int active_workers_;
    int total_tasks_;
    int completed_tasks_;

    void worker_thread();
};

// Global persistent thread pool manager
class ThreadPoolManager {
public:
    static ThreadPoolManager& instance();
    PersistentThreadPool* get_pool(int num_threads);
    void shutdown();

private:
    std::mutex pool_mutex_;
    std::unique_ptr<PersistentThreadPool> current_pool_;
    int current_num_threads_ = 0;
};

py::tuple PureCppRollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>&        data,
    const py::array_t<double>&         x0,
    const py::array_t<double>&         controls
);

py::tuple PersistentCppRollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>&        data,
    const py::array_t<double>&         x0,
    const py::array_t<double>&         controls
);

py::tuple PersistentONNXInterleaveRollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>&        data,
    const py::array_t<double>&         x0,
    const py::array_t<double>&         controls,
    const std::string&                 onnx_model_path,
    int                                inference_frequency = 1
);

// New policy-driven rollout functions
py::tuple ONNXPolicyRollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>&        data,
    const py::array_t<double>&         x0,
    int                                horizon,
    const std::string&                 onnx_model_path,
    int                                state_history_length = 10,
    int                                action_history_length = 5,
    int                                inference_frequency = 1,
    const py::array_t<double>&         additional_inputs = py::array_t<double>()
);

py::tuple PersistentONNXPolicyRollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>&        data,
    const py::array_t<double>&         x0,
    int                                horizon,
    const std::string&                 onnx_model_path,
    int                                state_history_length = 10,
    int                                action_history_length = 5,
    int                                inference_frequency = 1,
    const py::array_t<double>&         additional_inputs = py::array_t<double>()
);
