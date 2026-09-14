#pragma once

#include <memory>
#include <string>
#include <vector>

namespace zky_rl_deploy {

struct PolicyActionIoContract {
  std::string obs_input_name{"obs"};
  std::size_t obs_input_dim{69U};
  std::string time_step_input_name{"time_step"};
  std::size_t time_step_input_dim{1U};
  std::string action_output_name{"actions"};
  std::size_t action_output_dim{12U};
  double time_step_unit_s{0.02};
};

// 该 runner 只负责在 dry_run/shadow 阶段读取真实 ONNX action，供日志和离线复盘对齐使用。
// 调用它不会自动放开任何硬件输出；候选命令是否生效仍由上层 FSM、SafetyGate 和 stage 联锁决定。
class PolicyActionRunner {
 public:
  PolicyActionRunner(const std::string& onnx_path, PolicyActionIoContract io_contract);
  ~PolicyActionRunner();

  PolicyActionRunner(const PolicyActionRunner&) = delete;
  PolicyActionRunner& operator=(const PolicyActionRunner&) = delete;
  PolicyActionRunner(PolicyActionRunner&&) noexcept;
  PolicyActionRunner& operator=(PolicyActionRunner&&) noexcept;

  bool available() const { return available_; }
  const std::vector<std::string>& blockers() const { return blockers_; }

  std::vector<double> Run(const std::vector<double>& observation,
                          float time_step_scalar) const;

 private:
  class Impl;

  PolicyActionIoContract io_contract_;
  std::vector<std::string> blockers_;
  bool available_{false};
  std::unique_ptr<Impl> impl_;
};

}  // namespace zky_rl_deploy
