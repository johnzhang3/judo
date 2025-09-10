#include "rollout_spot.h"
#include "rollout.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <mujoco/mujoco.h>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <array>
#include <unordered_map>
#include <mutex>

namespace py = pybind11;


// Helper to turn std::vector<double> into py::array with ownership
static py::array_t<double> make_array_owned(std::vector<double>& buf, int B, int T, int D) {
    std::vector<ssize_t> shape   = { B, T, D };
    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(sizeof(double) * T * D),
        static_cast<ssize_t>(sizeof(double) *     D),
        static_cast<ssize_t>(sizeof(double))
    };
    auto heap_buf = new std::vector<double>(std::move(buf));
    py::capsule free_when_done(heap_buf, [](void *p) { delete reinterpret_cast<std::vector<double>*>(p); });
    return py::array_t<double>(shape, strides, heap_buf->data(), free_when_done);
}

// Shared ONNX Session allocator (similar to OnnxInterface::allocateOrtSession)
static std::shared_ptr<Ort::Session> allocate_shared_session(const std::string& onnx_path) {
    static Ort::Env env;
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    return std::make_shared<Ort::Session>(env, onnx_path.c_str(), opts);
}

// Lightweight Policy wrapper following OnnxInterface structure
class OnnxPolicy {
public:
    explicit OnnxPolicy(const std::shared_ptr<Ort::Session>& session)
        : session_(session), memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
        Ort::Allocator allocator(*session_, memory_info_);
        input_name_  = session_->GetInputNameAllocated(0, allocator).get();
        output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
        input_shape_  = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        output_shape_ = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        input_size_ = static_cast<int>(input_shape_[1]);
        output_size_ = static_cast<int>(output_shape_[1]);
    }

    std::vector<float> run(const std::vector<float>& observation) {
        if ((int)observation.size() != input_size_) {
            throw std::runtime_error("Observation size does not match ONNX input dimension");
        }
        std::array<int64_t, 2> ishape = { 1, static_cast<int64_t>(observation.size()) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float*>(observation.data()), observation.size(), ishape.data(), 2);
        const char* in_names[1] = { input_name_.c_str() };
        const char* out_names[1] = { output_name_.c_str() };
        auto outputs = session_->Run(run_options_, in_names, &input_tensor, 1, out_names, 1);
        auto& out = outputs[0];
        float* ptr = out.GetTensorMutableData<float>();
        auto info = out.GetTensorTypeAndShapeInfo();
        size_t n = info.GetElementCount();
        return std::vector<float>(ptr, ptr + n);
    }

    int input_size() const { return input_size_; }
    int output_size() const { return output_size_; }

private:
    std::shared_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    Ort::RunOptions run_options_;
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    int input_size_ = 0;
    int output_size_ = 0;
};

// Run ONNX policy path; fixed relative path
static std::string get_policy_path() {
    return std::string("judo_cpp/policy/xinghao_policy_v1.onnx");
}

// Constants matching Python implementation
static const double ACTION_SCALE = 0.2;
static const double STANDING_POS_RL[19] = {
    0.12, 0.5, -1.0,  -0.12, 0.5, -1.0,  0.12, 0.5, -1.0,  -0.12, 0.5, -1.0,
    0.0, -0.9, 1.8,  0.0, -0.9, 0.0, -1.54
};

// Isaac <-> MuJoCo conversion indices
static const int ISAAC_TO_MUJOCO_INDICES_12[12] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
static const int ISAAC_TO_MUJOCO_INDICES_19[19] = {1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 0, 5, 10, 15, 16, 17, 18};
static const int MUJOCO_TO_ISAAC_INDICES_12[12] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
static const int MUJOCO_TO_ISAAC_INDICES_19[19] = {12, 0, 3, 6, 9, 13, 1, 4, 7, 10, 14, 2, 5, 8, 11, 15, 16, 17, 18};

// Helper functions for coordinate transformations
static void isaac_to_mujoco_12(const double* isaac, double* mujoco) {
    for (int i = 0; i < 12; i++) {
        mujoco[i] = isaac[ISAAC_TO_MUJOCO_INDICES_12[i]];
    }
}

static void mujoco_to_isaac_19(const double* mujoco, double* isaac) {
    for (int i = 0; i < 19; i++) {
        isaac[i] = mujoco[MUJOCO_TO_ISAAC_INDICES_19[i]];
    }
}

static inline void compute_indices(const mjModel* m, int& base_qpos_start, int& base_qvel_start, int& leg_qpos_start, int& leg_qvel_start) {
    // Prefer robust detection of the free joint (base)
    int free_joint_idx = -1;
    for (int j = 0; j < m->njnt; j++) {
        if (m->jnt_type[j] == mjJNT_FREE) {
            free_joint_idx = j;
            break;
        }
    }
    if (free_joint_idx == -1) {
        // Fallback to joint 0
        free_joint_idx = 0;
    }
    base_qpos_start = m->jnt_qposadr[free_joint_idx];
    base_qvel_start = m->jnt_dofadr[free_joint_idx];
    
    // After the free joint come the actuated joints. For Spot, we assume 19 joints follow.
    leg_qpos_start = base_qpos_start + 7;
    leg_qvel_start = base_qvel_start + 6;
}

static inline void build_observation(const mjModel* m, mjData* d, const double* command_ptr,
                                     const std::vector<float>& prev_policy,
                                     int base_qpos_start, int base_qvel_start, int leg_qpos_start, int leg_qvel_start,
                                     std::vector<float>& obs_out) {
    obs_out.resize(84);
    int off = 0;
    
    // Get quaternion (w,x,y,z format in MuJoCo)
    double quat[4];
    for (int i = 0; i < 4; i++) {
        quat[i] = d->qpos[base_qpos_start + 3 + i];
    }
    
    // Compute inverse quaternion for body-frame transformations
    double invq[4];
    mju_negQuat(invq, quat);
    
    // Base linear velocity in body frame (inverse rotation)
    double blin[3];
    mju_rotVecQuat(blin, d->qvel + base_qvel_start, invq);
    for (int i=0;i<3;i++) obs_out[off++] = static_cast<float>(blin[i]);
    
    // Base angular velocity (already in body frame)
    for (int i=0;i<3;i++) obs_out[off++] = static_cast<float>(d->qvel[base_qvel_start + 3 + i]);
    
    // Projected gravity in body frame (inverse rotation)
    double gvec[3] = {0.0, 0.0, -1.0};
    double gvec_rotated[3];
    mju_rotVecQuat(gvec_rotated, gvec, invq);
    for (int i=0;i<3;i++) obs_out[off++] = static_cast<float>(gvec_rotated[i]);
    for (int i=0;i<3;i++) obs_out[off++] = static_cast<float>(command_ptr[0 + i]);
    for (int i=0;i<7;i++) obs_out[off++] = static_cast<float>(command_ptr[3 + i]);
    for (int i=0;i<12;i++) obs_out[off++] = static_cast<float>(command_ptr[10 + i]);
    for (int i=0;i<3;i++) obs_out[off++] = static_cast<float>(command_ptr[22 + i]);
    double jpos_raw[19];
    double jvel_raw[19];
    for (int i=0;i<19;i++) {
        jpos_raw[i] = d->qpos[leg_qpos_start + i] - STANDING_POS_RL[i];
        jvel_raw[i] = d->qvel[leg_qvel_start + i];
    }
    double jpos_isaac[19];
    double jvel_isaac[19];
    mujoco_to_isaac_19(jpos_raw, jpos_isaac);
    mujoco_to_isaac_19(jvel_raw, jvel_isaac);
    for (int i=0;i<19;i++) obs_out[off++] = static_cast<float>(jpos_isaac[i]);
    for (int i=0;i<19;i++) obs_out[off++] = static_cast<float>(jvel_isaac[i]);
    for (int i=0;i<12;i++) obs_out[off++] = (i < (int)prev_policy.size() ? prev_policy[i] : 0.0f);
}

static inline void compute_control_from_policy(const float* policy_out, const double* command_ptr, std::vector<double>& ctrl_out) {
    ctrl_out.resize(19);
    double action_isaac[12];
    for (int i=0;i<12;i++) action_isaac[i] = static_cast<double>(policy_out[i]);
    double target_leg_isaac[12];
    for (int i=0;i<12;i++) target_leg_isaac[i] = action_isaac[i] * ACTION_SCALE;
    double target_leg_mj[12];
    isaac_to_mujoco_12(target_leg_isaac, target_leg_mj);
    for (int i=0;i<12;i++) target_leg_mj[i] += STANDING_POS_RL[i];
    for (int i=0;i<12;i++) ctrl_out[i] = target_leg_mj[i];
    for (int i=0;i<7;i++) ctrl_out[12 + i] = static_cast<double>(command_ptr[3 + i]);
    const double* leg_cmd = command_ptr + 10;
    auto has_nonzero = [](const double* v){ return v[0] != 0.0 || v[1] != 0.0 || v[2] != 0.0; };
    if (has_nonzero(leg_cmd + 0))      { for (int i=0;i<3;i++) ctrl_out[0 + i] = leg_cmd[0 + i]; }
    else if (has_nonzero(leg_cmd + 3)) { for (int i=0;i<3;i++) ctrl_out[3 + i] = leg_cmd[3 + i]; }
    else if (has_nonzero(leg_cmd + 6)) { for (int i=0;i<3;i++) ctrl_out[6 + i] = leg_cmd[6 + i]; }
    else if (has_nonzero(leg_cmd + 9)) { for (int i=0;i<3;i++) ctrl_out[9 + i] = leg_cmd[9 + i]; }
}

py::tuple RolloutSpot(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>&        data,
    const py::array_t<double>&         x0,
    const py::array_t<double>&         controls
) {
    int B = (int)models.size();
    if (B == 0 || B != (int)data.size()) {
        throw std::runtime_error("models/data must have same non-zero length");
    }

    int horizon = (int)controls.shape(1);  // controls: command over time (B, T, 25)

    const mjModel* m0 = models[0];
    int nq     = m0->nq;
    int nv     = m0->nv;
    int nu     = m0->nu;
    int nsens  = m0->nsensordata;
    int nstate = nq + nv;

    if (x0.ndim() != 2 || x0.shape(0) != B || x0.shape(1) != nstate) {
        throw std::runtime_error("x0 must be a 2D array of shape (B, nq+nv)");
    }

    std::vector<double> states_buf(B * (horizon + 1) * nstate);
    std::vector<double> sens_buf(B * horizon * nsens);

    // Shared ONNX Session and policy wrapper (thread-safe Run)
    auto session = allocate_shared_session(get_policy_path());
    OnnxPolicy policy(session);

    auto controls_unchecked = controls.unchecked<3>();
    const double* x0_ptr = x0.data();

    // For each batch element, maintain prev_policy (size 12) as float32
    std::vector<std::vector<float>> prev_policy(B, std::vector<float>(12, 0.0f));

    // Parallelize per-environment using persistent thread pool
    int num_threads = std::min<int>(B, std::max(1, (int)std::thread::hardware_concurrency()));
    PersistentThreadPool* pool = ThreadPoolManager::instance().get_pool(num_threads);

    {
        py::gil_scoped_release release;
        pool->execute_parallel([&](int i){
            mjData* d = data[i];
            const mjModel* m = models[i];

            d->time = 0.0;
            const double* x0_i = x0_ptr + i * nstate;
            mj_setState(m, d, x0_i, mjSTATE_QPOS | mjSTATE_QVEL);
            mj_forward(m, d);
            mju_zero(d->qacc_warmstart, m->nv);

            double* st_ptr = &states_buf[i * (horizon + 1) * nstate];
            double* se_ptr = &sens_buf[i * horizon * nsens];

            for (int j = 0; j < nq; j++) st_ptr[j] = d->qpos[j];
            for (int j = 0; j < nv; j++) st_ptr[nq + j] = d->qvel[j];
            std::vector<double> current_state(nstate);
            for (int j = 0; j < nq; j++) current_state[j] = d->qpos[j];
            for (int j = 0; j < nv; j++) current_state[nq + j] = d->qvel[j];

            int base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start;
            compute_indices(m, base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start);

            for (int t = 0; t < horizon; t++) {
                std::vector<float> obs;
                double cmd_buf[25];
                for (int j=0;j<25;j++) cmd_buf[j] = static_cast<double>(controls_unchecked(i, t, j));
                build_observation(m, d, cmd_buf, prev_policy[i], base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start, obs);

                auto policy_out_vec = policy.run(obs);

                std::vector<double> ctrl;
                compute_control_from_policy(policy_out_vec.data(), cmd_buf, ctrl);
                if ((int)ctrl.size() != nu) {
                    throw std::runtime_error("Computed control size does not match model nu");
                }
                for (int j = 0; j < nu; j++) d->ctrl[j] = ctrl[j];

                mj_step(m, d);

                prev_policy[i] = std::move(policy_out_vec);

                for (int j = 0; j < nq; j++) current_state[j] = d->qpos[j];
                for (int j = 0; j < nv; j++) current_state[nq + j] = d->qvel[j];
                for (int j = 0; j < nstate; j++) st_ptr[(t + 1) * nstate + j] = current_state[j];
                for (int j = 0; j < nsens; j++) se_ptr[t * nsens + j] = d->sensordata[j];
            }
        }, B);
    }

    auto states_arr = make_array_owned(states_buf, B, horizon + 1, nstate);
    auto sens_arr   = make_array_owned(sens_buf,   B, horizon,     nsens);
    return py::make_tuple(states_arr, sens_arr);
}

py::array_t<float> SimSpot(
    const mjModel* model,
    mjData*        data,
    const py::array_t<double>& x0,
    const py::array_t<double>& controls,
    const py::array_t<float>& prev_policy
) {
    int nq = model->nq;
    int nv = model->nv;
    int nu = model->nu;

    if (x0.ndim() != 1 || x0.shape(0) != nq + nv) {
        throw std::runtime_error("x0 must be a 1D array of shape (nq+nv)");
    }
    if (controls.ndim() != 1) {
        throw std::runtime_error("controls must be a 1D array of command");
    }
    if (prev_policy.ndim() != 1 || prev_policy.shape(0) != 12) {
        throw std::runtime_error("prev_policy must be a 1D array of shape (12)");
    }

    data->time = 0.0;
    const double* x0_ptr = x0.data();
    mj_setState(model, data, x0_ptr, mjSTATE_QPOS | mjSTATE_QVEL);
    mj_forward(model, data);
    mju_zero(data->qacc_warmstart, nv);

    int base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start;
    compute_indices(model, base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start);

    int cmd_dim = (int)controls.shape(0);
    auto ctrl_unchecked = controls.unchecked<1>();
    double cmd_buf[25];
    for (int j = 0; j < cmd_dim && j < 25; j++) cmd_buf[j] = static_cast<double>(ctrl_unchecked(j));
    for (int j = cmd_dim; j < 25; j++) cmd_buf[j] = 0.0;

    std::vector<float> prev(prev_policy.data(), prev_policy.data() + 12);

    std::vector<float> obs;
    build_observation(model, data, cmd_buf, prev, base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start, obs);

    auto session = allocate_shared_session(get_policy_path());
    OnnxPolicy policy(session);
    auto policy_out_vec = policy.run(obs);
    std::vector<double> ctrl;
    compute_control_from_policy(policy_out_vec.data(), cmd_buf, ctrl);
    if ((int)ctrl.size() != nu) {
        throw std::runtime_error("Computed control size does not match model nu");
    }
    for (int j = 0; j < nu; j++) data->ctrl[j] = ctrl[j];
    mj_step(model, data);

    // Return new policy output as numpy array
    auto result = py::array_t<float>(12);
    auto result_ptr = result.mutable_unchecked<1>();
    for (int i = 0; i < 12; i++) {
        result_ptr(i) = policy_out_vec[i];
    }
    return result;
}
