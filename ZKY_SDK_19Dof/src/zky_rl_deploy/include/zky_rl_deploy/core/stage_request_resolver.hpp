#pragma once

#include <string>

#include "zky_rl_deploy/core/stage_manager.hpp"

namespace zky_rl_deploy {

// 把“最终想去的高阶 stage”解析成“这一轮应该申请的下一阶 stage”。
// 这样 launch 仍可直接声明目标 stage4/stage7，但节点会按 StageManager 的单级联锁逐步推进，
// 不会因为 stage0 -> stage4 的直跳而在第一轮就被永久卡住。
std::string ResolveNextRequestedStageName(const StageManager& stage_manager,
                                          const std::string& desired_stage_name);

}  // namespace zky_rl_deploy
