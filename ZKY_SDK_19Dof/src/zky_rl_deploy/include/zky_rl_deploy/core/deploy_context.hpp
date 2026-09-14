#pragma once

#include <string>

namespace zky_rl_deploy {

// 该文件定义部署骨架阶段共享的最小上下文。
// 这里先只保留参数命名空间和 dry_run/shadow 锁，避免在没有 checklist 的情况下误放开硬件输出。
struct DeployContext {
  std::string parameter_namespace;
  std::string config_namespace;
  std::string config_root;
  bool dry_run_only{true};
  bool shadow_mode{true};
  bool enable_motor_output{false};
  bool publish_shadow{true};
};

// 统一生成启动摘要，便于节点日志和后续测试都能复用同一份安全状态描述。
std::string BuildStartupSummary(const DeployContext& context);

}  // namespace zky_rl_deploy
