#include <gtest/gtest.h>

#include <string>

#include "zky_rl_deploy/core/deploy_context.hpp"

namespace zky_rl_deploy {
namespace {

TEST(DeployContextTest, StartupSummaryKeepsDryRunMarkers) {
  // 该测试只验证骨架默认摘要仍然带有 dry_run/shadow 标记，
  // 防止后续有人在未完成 checklist 前把默认行为改成可直通硬件输出。
  DeployContext context;
  context.parameter_namespace = "/zky_deploy_node";
  context.config_namespace = "/zky_rl_deploy";
  context.config_root = "/tmp/zky_config";

  const std::string summary = BuildStartupSummary(context);

  EXPECT_NE(summary.find("dry_run_only=true"), std::string::npos);
  EXPECT_NE(summary.find("shadow_mode=true"), std::string::npos);
  EXPECT_NE(summary.find("enable_motor_output=false"), std::string::npos);
  EXPECT_NE(summary.find("publish_shadow=true"), std::string::npos);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
