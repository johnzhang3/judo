#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <mujoco/mujoco.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>

namespace py = pybind11;

// Forward declarations
class OnnxPolicy;

/**
 * Template-based thread pool for compile-time optimization
 *
 * @tparam NumThreads Number of threads (0 = single-threaded, >0 = multi-threaded)
 */
template<int NumThreads = 0>
class SpotThreadPool {
public:
    explicit SpotThreadPool();
    ~SpotThreadPool();

    // Execute function across all threads
    void execute_parallel(std::function<void(int)> func, int total_work);

    // Get effective thread count
    constexpr int get_num_threads() const { return NumThreads; }

private:
    // Compile-time branch: single-threaded specialization
    void execute_single_threaded(std::function<void(int)> func, int total_work);

    // Compile-time branch: multi-threaded implementation
    void execute_multi_threaded(std::function<void(int)> func, int total_work);

    // Multi-threaded members (only instantiated when NumThreads > 0)
    std::conditional_t<(NumThreads > 0), std::vector<std::thread>, std::nullptr_t> threads_;
    std::conditional_t<(NumThreads > 0), std::queue<std::function<void()>>, std::nullptr_t> tasks_;
    std::conditional_t<(NumThreads > 0), std::mutex, std::nullptr_t> queue_mutex_;
    std::conditional_t<(NumThreads > 0), std::condition_variable, std::nullptr_t> condition_;
    std::conditional_t<(NumThreads > 0), std::condition_variable, std::nullptr_t> finished_;
    std::conditional_t<(NumThreads > 0), std::atomic<bool>, std::nullptr_t> stop_;
    std::conditional_t<(NumThreads > 0), std::atomic<int>, std::nullptr_t> active_workers_;
    std::conditional_t<(NumThreads > 0), std::atomic<int>, std::nullptr_t> total_tasks_;
    std::conditional_t<(NumThreads > 0), std::atomic<int>, std::nullptr_t> completed_tasks_;

    void worker_thread(); // Only used when NumThreads > 0
};

/**
 * Spot Rollout class mimicking mujoco.rollout.Rollout API
 *
 * Template parameters allow compile-time optimization of thread management:
 * - SpotRollout<0>: Single-threaded, no thread pool overhead
 * - SpotRollout<N>: N-threaded with optimized thread pool
 */
template<int NumThreads = 0>
class SpotRollout {
public:
    /**
     * Constructor
     * @param nthread Number of threads (ignored for template specialization, but kept for API compatibility)
     */
    explicit SpotRollout(int nthread = NumThreads);

    /**
     * Destructor - cleans up resources
     */
    ~SpotRollout();

    /**
     * Close the rollout object and cleanup resources
     * Matches mujoco.rollout.Rollout.close() API
     */
    void close();

    /**
     * Perform rollout with Spot-specific ONNX policy
     * Matches mujoco.rollout.Rollout.rollout() API signature
     *
     * @param models Vector of MuJoCo models
     * @param data Vector of MuJoCo data objects
     * @param initial_state Initial state array (nbatch x nstate)
     * @param controls Control commands array (nbatch x nsteps x 25)
     * @return Tuple of (states, sensordata) arrays
     */
    py::tuple rollout(
        const std::vector<const mjModel*>& models,
        const std::vector<mjData*>& data,
        const py::array_t<double>& initial_state,
        const py::array_t<double>& controls
    );

    /**
     * Context manager support - enter
     */
    SpotRollout<NumThreads>* __enter__();

    /**
     * Context manager support - exit
     */
    void __exit__(py::object exc_type, py::object exc_val, py::object exc_tb);

    /**
     * Get the number of threads
     */
    constexpr int get_num_threads() const { return NumThreads; }

private:
    SpotThreadPool<NumThreads> thread_pool_;
    std::shared_ptr<Ort::Session> onnx_session_;
    std::unique_ptr<OnnxPolicy> policy_;
    bool closed_ = false;

    // Helper methods for rollout computation
    void compute_indices(const mjModel* m, int& base_qpos_start, int& base_qvel_start,
                        int& leg_qpos_start, int& leg_qvel_start);

    void build_observation(const mjModel* m, mjData* d, const double* command_ptr,
                          const std::vector<float>& prev_policy,
                          int base_qpos_start, int base_qvel_start,
                          int leg_qpos_start, int leg_qvel_start,
                          std::vector<float>& obs_out);

    void compute_control_from_policy(const float* policy_out, const double* command_ptr,
                                   std::vector<double>& ctrl_out);

    // Initialize ONNX session and policy
    void initialize_policy();
};

// Type aliases for common configurations
using SpotRolloutST = SpotRollout<0>;          // Single-threaded
using SpotRollout2T = SpotRollout<2>;          // 2 threads
using SpotRollout4T = SpotRollout<4>;          // 4 threads
using SpotRollout8T = SpotRollout<8>;          // 8 threads
using SpotRolloutMT = SpotRollout<-1>;         // Dynamic thread count (hardware_concurrency)

// Factory function to create rollout object with runtime thread count
std::unique_ptr<py::object> create_spot_rollout(int nthread);

// Utility function to create numpy arrays from C++ vectors
py::array_t<double> make_array_owned(std::vector<double>& buf, int B, int T, int D);

// Legacy function for backwards compatibility
py::tuple RolloutSpot(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>& data,
    const py::array_t<double>& x0,
    const py::array_t<double>& controls
);

// Single-step simulation function
py::array_t<float> SimSpot(
    const mjModel* model,
    mjData* data,
    const py::array_t<double>& x0,
    const py::array_t<double>& controls,
    const py::array_t<float>& prev_policy
);