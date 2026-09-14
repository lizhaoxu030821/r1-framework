#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "zky_rl_deploy/core/motion_clip.hpp"

namespace zky_rl_deploy {

// 该 provider 只服务于纯软件 dry_run/shadow 数据链：
// 它从 JSON 副本读取 motion clip 帧和参考 pelvis 姿态，避免在当前阶段把核心逻辑绑死到硬件或 ROS topic。
class JsonMotionFrameProvider final : public MotionFrameProvider {
 public:
  explicit JsonMotionFrameProvider(const std::string& json_path);

  std::size_t FrameCount() const override;
  MotionFrame GetFrame(std::size_t frame_index) const override;

  const std::array<double, 4>& ReferencePelvisOrientationWxyz(std::size_t frame_index) const;
  const std::vector<std::string>& warnings() const { return warnings_; }
  const std::string& source_path() const { return source_path_; }

 private:
  std::string source_path_;
  std::vector<MotionFrame> frames_;
  std::vector<std::array<double, 4>> reference_pelvis_orientation_wxyz_;
  std::vector<std::string> warnings_;
};

}  // namespace zky_rl_deploy
