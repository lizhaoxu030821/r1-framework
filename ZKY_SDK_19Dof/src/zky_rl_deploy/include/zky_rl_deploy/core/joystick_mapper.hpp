#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace zky_rl_deploy {

enum class FunctionRequest {
  kBeyondMimic,
  kReservedFunction1,
  kReservedFunction2,
  kReservedFunction3,
};

struct JoystickState {
  bool start_pressed{false};
  bool back_pressed{false};
  bool home_pressed{false};
  bool rb_pressed{false};
  bool a_pressed{false};
  bool x_pressed{false};
  bool b_pressed{false};
  bool y_pressed{false};
};

struct JoystickMapperConfig {};

struct JoystickEvents {
  bool request_zero_output_estop{false};
  bool request_passive{false};
  bool request_stand_init{false};
  bool rb_held{false};
  bool rb_pressed{false};
  bool rb_released{false};
  std::optional<FunctionRequest> function_request;
  std::vector<std::string> warnings;

  std::string Summary() const;
};

class JoystickMapper {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  explicit JoystickMapper(JoystickMapperConfig config = {});

  // 控制逻辑只消费语义事件，不直接依赖物理按钮编号。
  // START 映射为“开始重新使能链路”，BACK 映射为“回到初始位姿”，HOME 映射为急停清零。
  // zero_output_estop_latched=true 时，只允许识别 HOME 和 START；
  // BACK/RB+A/X/B/Y 会被显式屏蔽，避免急停锁存中被普通按键误解锁。
  JoystickEvents Update(TimePoint now,
                        const JoystickState& state,
                        bool zero_output_estop_latched);

  void Reset();

  const JoystickMapperConfig& config() const { return config_; }

  static const char* ToString(FunctionRequest request);

 private:
  static void ValidateConfig(const JoystickMapperConfig& config);
  static void ValidateState(const JoystickState& state);

  JoystickMapperConfig config_;
  bool has_previous_state_{false};
  JoystickState previous_state_{};
};

}  // namespace zky_rl_deploy
