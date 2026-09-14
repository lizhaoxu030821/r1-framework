#include "zky_rl_deploy/core/deploy_context.hpp"

#include <sstream>

namespace zky_rl_deploy {

std::string BuildStartupSummary(const DeployContext& context) {
  std::ostringstream stream;
  stream << "parameter_namespace=" << context.parameter_namespace
         << ", config_namespace=" << context.config_namespace
         << ", config_root=" << context.config_root
         << ", dry_run_only=" << (context.dry_run_only ? "true" : "false")
         << ", shadow_mode=" << (context.shadow_mode ? "true" : "false")
         << ", enable_motor_output=" << (context.enable_motor_output ? "true" : "false")
         << ", publish_shadow=" << (context.publish_shadow ? "true" : "false");
  return stream.str();
}

}  // namespace zky_rl_deploy
