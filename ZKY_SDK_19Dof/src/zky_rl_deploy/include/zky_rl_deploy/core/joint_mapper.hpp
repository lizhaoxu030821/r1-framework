#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace zky_rl_deploy {

// 该模块负责把策略、硬件、NPZ 等不同关节顺序统一到“按关节名显式映射”的方式。
// 这里刻意不允许直接依赖数组位置，因为 ONNX、NPZ 和 hardware_order 的排列并不一致，
// 任何“第 i 个元素默认代表同一个关节”的假设都会把左右腿或踝关节命令悄悄发错位置。
class JointMapper {
 public:
  using JointName = std::string;
  using JointOrder = std::vector<JointName>;
  using JointValues = std::vector<double>;
  using SignConvention = std::unordered_map<JointName, int>;
  using SignVerification = std::unordered_map<JointName, bool>;

  JointMapper(JointOrder policy_order,
              JointOrder hardware_order,
              JointOrder npz_order,
              SignConvention sign_convention,
              SignVerification sign_verified = {});

  // 策略输出最终要发到 hardware_order，因此这里在重排同时应用 sign_convention，
  // 保证“策略坐标中的正方向”会被转换成底层硬件真正需要的正负号。
  JointValues PolicyToHardwareCommand(const JointValues& policy_values) const;

  // 电机反馈或 joint_states 返回的是 hardware_order 坐标；进入观测前必须重排到 policy_order，
  // 同时应用 sign_convention，把反馈方向统一回训练/URDF 使用的策略坐标系。
  JointValues HardwareToPolicyFeedback(const JointValues& hardware_values) const;

  // NPZ 参考动作本来就在策略语义坐标下，只需要按关节名做顺序重排，
  // 不能对它再次应用硬件 sign_convention，否则会把训练 reference 错误翻转。
  JointValues NpzToPolicyReference(const JointValues& npz_values) const;

  // Stage 3 之前，sign_convention 还没有逐关节实机验证完成。
  // 因此功能态进入条件必须查询这里；只要有任一关节未验证，就不能进入 ACTIVE_FUNCTION。
  bool AllSignsVerifiedForActiveFunction() const;
  std::vector<JointName> GetUnverifiedSignJoints() const;
  void ThrowIfSignsUnverifiedForActiveFunction() const;

  const JointOrder& policy_order() const { return policy_order_; }
  const JointOrder& hardware_order() const { return hardware_order_; }
  const JointOrder& npz_order() const { return npz_order_; }

 private:
  JointValues ReorderValues(const JointValues& source_values,
                            std::size_t expected_size,
                            const std::vector<std::size_t>& target_from_source_indices,
                            const std::vector<int>* target_signs,
                            const std::string& source_name,
                            const std::string& target_name) const;

  static std::vector<std::size_t> BuildTargetFromSourceIndices(
      const JointOrder& source_order,
      const JointOrder& target_order,
      const std::string& source_name,
      const std::string& target_name);

  JointOrder policy_order_;
  JointOrder hardware_order_;
  JointOrder npz_order_;
  std::vector<std::size_t> hardware_from_policy_indices_;
  std::vector<std::size_t> policy_from_hardware_indices_;
  std::vector<std::size_t> policy_from_npz_indices_;
  std::vector<int> policy_signs_;
  std::vector<int> hardware_signs_;
  std::vector<bool> policy_sign_verified_;
};

}  // namespace zky_rl_deploy
