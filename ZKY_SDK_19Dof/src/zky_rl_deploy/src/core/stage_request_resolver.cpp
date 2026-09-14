#include "zky_rl_deploy/core/stage_request_resolver.hpp"

#include <cstddef>

namespace zky_rl_deploy {

std::string ResolveNextRequestedStageName(const StageManager& stage_manager,
                                          const std::string& desired_stage_name) {
  const auto& ordered_stages = stage_manager.ordered_stages();
  std::size_t current_index = ordered_stages.size();
  std::size_t desired_index = ordered_stages.size();

  for (std::size_t index = 0; index < ordered_stages.size(); ++index) {
    if (ordered_stages[index].name == stage_manager.current_stage().name) {
      current_index = index;
    }
    if (ordered_stages[index].name == desired_stage_name) {
      desired_index = index;
    }
  }

  if (current_index >= ordered_stages.size() || desired_index >= ordered_stages.size()) {
    return desired_stage_name;
  }
  if (desired_index <= current_index + 1U) {
    return desired_stage_name;
  }
  return ordered_stages[current_index + 1U].name;
}

}  // namespace zky_rl_deploy
