#include "rollout_spot.h"
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

// Persist prev_policy per-simulator (MjData*) across SimSpot calls
static std::unordered_map<const mjData*, std::vector<float>> g_prev_policy_map;
static std::mutex g_prev_policy_mu;

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

// Spot policy wrapper for ONNXRuntime
struct SpotPolicy {
    Ort::Env env;
    Ort::Session session;
    Ort::SessionOptions opts;
    Ort::AllocatorWithDefaultOptions allocator;

    // cache input/output names
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    SpotPolicy(const std::string& onnx_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "spot-policy"),
          opts(),
          session(nullptr)
    {
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        session = Ort::Session(env, onnx_path.c_str(), opts);

        size_t n_in = session.GetInputCount();
        size_t n_out = session.GetOutputCount();
        input_names.reserve(n_in);
        output_names.reserve(n_out);
        for (size_t i = 0; i < n_in; ++i) {
            char* name = session.GetInputNameAllocated(i, allocator).release();
            input_names.push_back(name);
        }
        for (size_t i = 0; i < n_out; ++i) {
            char* name = session.GetOutputNameAllocated(i, allocator).release();
            output_names.push_back(name);
        }
    }

    ~SpotPolicy() {
        // Free names allocated via allocator.release()
        for (auto* p : input_names) allocator.Free(const_cast<char*>(p));
        for (auto* p : output_names) allocator.Free(const_cast<char*>(p));
    }

    // Run policy: returns pair (control[1,19], policy_out[1,12])
    std::pair<std::vector<float>, std::vector<float>> run(
        const std::vector<float>& qpos,
        const std::vector<float>& qvel,
        const std::vector<float>& command,
        const std::vector<float>& prev_policy
    ) {
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        std::array<int64_t, 1> shape1 = { static_cast<int64_t>(qpos.size()) };
        std::array<int64_t, 1> shape2 = { static_cast<int64_t>(qvel.size()) };
        std::array<int64_t, 1> shape3 = { static_cast<int64_t>(command.size()) };
        std::array<int64_t, 1> shape4 = { static_cast<int64_t>(prev_policy.size()) };

        Ort::Value t_qpos = Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(qpos.data()), qpos.size(), shape1.data(), 1);
        Ort::Value t_qvel = Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(qvel.data()), qvel.size(), shape2.data(), 1);
        Ort::Value t_cmd  = Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(command.data()), command.size(), shape3.data(), 1);
        Ort::Value t_prev = Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(prev_policy.data()), prev_policy.size(), shape4.data(), 1);

        std::array<Ort::Value,4> inputs = { std::move(t_qpos), std::move(t_qvel), std::move(t_cmd), std::move(t_prev) };

        auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(), output_names.data(), output_names.size());
        if (outputs.size() != 2) {
            throw std::runtime_error("Expected 2 outputs from ONNX policy: control and policy_out");
        }

        // Extract outputs as float vectors
        auto& control_val = outputs[0];
        auto& policy_val  = outputs[1];

        float* control_ptr = control_val.GetTensorMutableData<float>();
        float* policy_ptr  = policy_val.GetTensorMutableData<float>();

        auto control_info = control_val.GetTensorTypeAndShapeInfo();
        auto policy_info  = policy_val.GetTensorTypeAndShapeInfo();
        size_t control_elems = control_info.GetElementCount();
        size_t policy_elems  = policy_info.GetElementCount();

        std::vector<float> control(control_ptr, control_ptr + control_elems);
        std::vector<float> policy(policy_ptr,   policy_ptr  + policy_elems);
        return {std::move(control), std::move(policy)};
    }
};

// Run ONNX policy path; for now fixed relative path as in Python example
static std::string get_policy_path() {
    return std::string("judo_cpp/policy/xinghao_policy_wrapped_torch.onnx");
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

    // allocate outputs
    std::vector<double> states_buf(B * (horizon + 1) * nstate);
    std::vector<double> sens_buf(B * horizon * nsens);

    // policy
    SpotPolicy policy(get_policy_path());

    auto controls_unchecked = controls.unchecked<3>();
    const double* x0_ptr = x0.data();

    // For each batch element, maintain prev_policy (size 12) as float32
    std::vector<std::vector<float>> prev_policy(B, std::vector<float>(12, 0.0f));

    for (int i = 0; i < B; i++) {
        mjData* d = data[i];
        const mjModel* m = models[i];

        // set initial qpos+qvel
        d->time = 0.0;
        const double* x0_i = x0_ptr + i * nstate;
        mj_setState(m, d, x0_i, mjSTATE_QPOS | mjSTATE_QVEL);
        mj_forward(m, d);
        mju_zero(d->qacc_warmstart, m->nv);

        double* st_ptr = &states_buf[i * (horizon + 1) * nstate];
        double* se_ptr = &sens_buf[i * horizon * nsens];

        // store initial state
        for (int j = 0; j < nq; j++) st_ptr[j] = d->qpos[j];
        for (int j = 0; j < nv; j++) st_ptr[nq + j] = d->qvel[j];

        std::vector<double> current_state(nstate);
        for (int j = 0; j < nq; j++) current_state[j] = d->qpos[j];
        for (int j = 0; j < nv; j++) current_state[nq + j] = d->qvel[j];

        for (int t = 0; t < horizon; t++) {
            // Build policy inputs
            std::vector<float> qpos_in(nq);
            std::vector<float> qvel_in(nv);
            for (int j = 0; j < nq; j++) qpos_in[j] = static_cast<float>(d->qpos[j]);
            for (int j = 0; j < nv; j++) qvel_in[j] = static_cast<float>(d->qvel[j]);

            // command is controls_unchecked(i, t, :) assumed len 25, float32 cast
            int cmd_dim = (int)controls.shape(2);
            std::vector<float> cmd_in(cmd_dim);
            for (int j = 0; j < cmd_dim; j++) cmd_in[j] = static_cast<float>(controls_unchecked(i, t, j));

            auto [control_vec, policy_out_vec] = policy.run(qpos_in, qvel_in, cmd_in, prev_policy[i]);

            // Apply first part as controls to MuJoCo (nu expected 19)
            if ((int)control_vec.size() != nu) {
                throw std::runtime_error("ONNX control output size does not match model nu");
            }
            for (int j = 0; j < nu; j++) d->ctrl[j] = static_cast<double>(control_vec[j]);

            // Step simulation
            mj_step(m, d);

            // Update prev_policy for next step
            prev_policy[i] = std::move(policy_out_vec);

            // Record new state and sensors
            for (int j = 0; j < nq; j++) current_state[j] = d->qpos[j];
            for (int j = 0; j < nv; j++) current_state[nq + j] = d->qvel[j];
            for (int j = 0; j < nstate; j++) st_ptr[(t + 1) * nstate + j] = current_state[j];
            for (int j = 0; j < nsens; j++) se_ptr[t * nsens + j] = d->sensordata[j];
        }
    }

    auto states_arr = make_array_owned(states_buf, B, horizon + 1, nstate);
    auto sens_arr   = make_array_owned(sens_buf,   B, horizon,     nsens);
    return py::make_tuple(states_arr, sens_arr);
}

void SimSpot(
    const mjModel* model,
    mjData*        data,
    const py::array_t<double>& x0,
    const py::array_t<double>& controls
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

    // set initial qpos+qvel
    data->time = 0.0;
    const double* x0_ptr = x0.data();
    mj_setState(model, data, x0_ptr, mjSTATE_QPOS | mjSTATE_QVEL);
    mj_forward(model, data);
    mju_zero(data->qacc_warmstart, nv);

    // Build one-step policy inputs
    std::vector<float> qpos_in(nq);
    std::vector<float> qvel_in(nv);
    for (int j = 0; j < nq; j++) qpos_in[j] = static_cast<float>(data->qpos[j]);
    for (int j = 0; j < nv; j++) qvel_in[j] = static_cast<float>(data->qvel[j]);

    int cmd_dim = (int)controls.shape(0);
    auto ctrl_unchecked = controls.unchecked<1>();
    std::vector<float> cmd_in(cmd_dim);
    for (int j = 0; j < cmd_dim; j++) cmd_in[j] = static_cast<float>(ctrl_unchecked(j));

    // Fetch previous policy output for this simulator instance
    std::vector<float> prev;
    {
        std::lock_guard<std::mutex> lock(g_prev_policy_mu);
        auto it = g_prev_policy_map.find(data);
        if (it != g_prev_policy_map.end()) {
            prev = it->second;
        } else {
            prev.assign(12, 0.0f);
        }
    }
    SpotPolicy policy(get_policy_path());
    auto [control_vec, policy_out_vec] = policy.run(qpos_in, qvel_in, cmd_in, prev);
    if ((int)control_vec.size() != nu) {
        throw std::runtime_error("ONNX control output size does not match model nu");
    }
    for (int j = 0; j < nu; j++) data->ctrl[j] = static_cast<double>(control_vec[j]);
    mj_step(model, data);

    // Store policy_out for next call
    {
        std::lock_guard<std::mutex> lock(g_prev_policy_mu);
        g_prev_policy_map[data] = policy_out_vec;  // copy
    }
}


