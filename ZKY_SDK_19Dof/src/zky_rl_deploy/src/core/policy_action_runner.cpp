#include "zky_rl_deploy/core/policy_action_runner.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

#if ZKY_RL_DEPLOY_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace zky_rl_deploy {

class PolicyActionRunner::Impl {
 public:
#if ZKY_RL_DEPLOY_HAS_ONNXRUNTIME
  Impl(const std::string& onnx_path, const PolicyActionIoContract& io_contract)
      : io_contract_(io_contract) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = std::make_unique<Ort::Session>(GetOrtEnv(), onnx_path.c_str(), session_options);
  }

  std::vector<double> Run(const std::vector<double>& observation,
                          float time_step_scalar) const {
    if (observation.size() != io_contract_.obs_input_dim) {
      throw std::invalid_argument("observation size does not match policy io contract");
    }

    std::vector<float> observation_f32(observation.size(), 0.0F);
    std::transform(observation.begin(),
                   observation.end(),
                   observation_f32.begin(),
                   [](double value) { return static_cast<float>(value); });
    std::array<float, 1> time_step = {time_step_scalar};

    const std::array<std::int64_t, 2> obs_shape = {
        1, static_cast<std::int64_t>(io_contract_.obs_input_dim)};
    const std::array<std::int64_t, 2> time_step_shape = {
        1, static_cast<std::int64_t>(io_contract_.time_step_input_dim)};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value obs_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                            observation_f32.data(),
                                                            observation_f32.size(),
                                                            obs_shape.data(),
                                                            obs_shape.size());
    Ort::Value time_step_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                  time_step.data(),
                                                                  time_step.size(),
                                                                  time_step_shape.data(),
                                                                  time_step_shape.size());

    std::array<const char*, 2> input_names = {
        io_contract_.obs_input_name.c_str(), io_contract_.time_step_input_name.c_str()};
    std::array<const char*, 1> output_names = {io_contract_.action_output_name.c_str()};
    std::array<Ort::Value, 2> input_tensors = {std::move(obs_tensor), std::move(time_step_tensor)};

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                        input_names.data(),
                                        input_tensors.data(),
                                        input_tensors.size(),
                                        output_names.data(),
                                        output_names.size());
    if (output_tensors.size() != 1U) {
      throw std::runtime_error("policy inference did not return exactly one actions tensor");
    }

    const auto tensor_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
    if (tensor_shape.size() != 2U || tensor_shape[0] != 1LL ||
        tensor_shape[1] != static_cast<std::int64_t>(io_contract_.action_output_dim)) {
      throw std::runtime_error("policy actions tensor shape does not match io_contract");
    }

    const float* action_data = output_tensors.front().GetTensorData<float>();
    std::vector<double> action(io_contract_.action_output_dim, 0.0);
    for (std::size_t index = 0; index < action.size(); ++index) {
      action[index] = static_cast<double>(action_data[index]);
    }
    return action;
  }

 private:
  static Ort::Env& GetOrtEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "zky_rl_deploy_policy_action_runner");
    return env;
  }

  PolicyActionIoContract io_contract_;
  std::unique_ptr<Ort::Session> session_;
#else
  Impl(const std::string& onnx_path, const PolicyActionIoContract& io_contract) {
    (void)onnx_path;
    (void)io_contract;
  }

  std::vector<double> Run(const std::vector<double>& observation,
                          float time_step_scalar) const {
    (void)observation;
    (void)time_step_scalar;
    throw std::runtime_error("ONNX Runtime support is disabled");
  }
#endif
};

PolicyActionRunner::PolicyActionRunner(const std::string& onnx_path,
                                       PolicyActionIoContract io_contract)
    : io_contract_(std::move(io_contract)) {
#if ZKY_RL_DEPLOY_HAS_ONNXRUNTIME
  try {
    impl_ = std::make_unique<Impl>(onnx_path, io_contract_);
    available_ = true;
  } catch (const std::exception& exception) {
    blockers_.push_back(std::string("failed to create ONNX Runtime session: ") + exception.what());
    available_ = false;
  }
#else
  (void)onnx_path;
  blockers_.push_back(
      "ONNX Runtime support is disabled. Action inference can only be logged after the dependency is restored.");
  available_ = false;
#endif
}

PolicyActionRunner::~PolicyActionRunner() = default;
PolicyActionRunner::PolicyActionRunner(PolicyActionRunner&&) noexcept = default;
PolicyActionRunner& PolicyActionRunner::operator=(PolicyActionRunner&&) noexcept = default;

std::vector<double> PolicyActionRunner::Run(const std::vector<double>& observation,
                                            float time_step_scalar) const {
  if (!available_ || !impl_) {
    throw std::runtime_error("policy action runner is not available");
  }
  return impl_->Run(observation, time_step_scalar);
}

}  // namespace zky_rl_deploy
