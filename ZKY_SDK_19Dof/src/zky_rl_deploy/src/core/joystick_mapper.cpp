#include "zky_rl_deploy/core/joystick_mapper.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace zky_rl_deploy {
namespace {

bool RisingEdge(bool current, bool previous, bool has_previous_state) {
  return has_previous_state ? (current && !previous) : false;
}

bool FallingEdge(bool current, bool previous, bool has_previous_state) {
  return has_previous_state ? (!current && previous) : false;
}

void AppendFunctionRequestIfNeeded(bool rb_held,
                                   bool a_rising,
                                   bool x_rising,
                                   bool b_rising,
                                   bool y_rising,
                                   JoystickEvents* events) {
  if (!rb_held) {
    return;
  }

  const int rising_count = static_cast<int>(a_rising) + static_cast<int>(x_rising) +
                           static_cast<int>(b_rising) + static_cast<int>(y_rising);
  if (rising_count > 1) {
    events->warnings.push_back(
        "multiple RB+button function requests rose in the same sample; using the first priority match.");
  }

  if (a_rising) {
    events->function_request = FunctionRequest::kBeyondMimic;
  } else if (x_rising) {
    events->function_request = FunctionRequest::kReservedFunction1;
  } else if (b_rising) {
    events->function_request = FunctionRequest::kReservedFunction2;
  } else if (y_rising) {
    events->function_request = FunctionRequest::kReservedFunction3;
  }
}

std::string JoinWarnings(const std::vector<std::string>& warnings) {
  if (warnings.empty()) {
    return "none";
  }

  std::ostringstream stream;
  for (std::size_t index = 0; index < warnings.size(); ++index) {
    if (index != 0U) {
      stream << "; ";
    }
    stream << warnings[index];
  }
  return stream.str();
}

}  // namespace

JoystickMapper::JoystickMapper(JoystickMapperConfig config) : config_(std::move(config)) {
  ValidateConfig(config_);
}

JoystickEvents JoystickMapper::Update(TimePoint now,
                                      const JoystickState& state,
                                      bool zero_output_estop_latched) {
  (void)now;
  ValidateState(state);

  const bool start_rising =
      RisingEdge(state.start_pressed, previous_state_.start_pressed, has_previous_state_);
  const bool back_rising =
      RisingEdge(state.back_pressed, previous_state_.back_pressed, has_previous_state_);
  const bool a_rising = RisingEdge(state.a_pressed, previous_state_.a_pressed, has_previous_state_);
  const bool x_rising = RisingEdge(state.x_pressed, previous_state_.x_pressed, has_previous_state_);
  const bool b_rising = RisingEdge(state.b_pressed, previous_state_.b_pressed, has_previous_state_);
  const bool y_rising = RisingEdge(state.y_pressed, previous_state_.y_pressed, has_previous_state_);

  JoystickEvents events;
  events.rb_held = state.rb_pressed;
  events.rb_pressed =
      RisingEdge(state.rb_pressed, previous_state_.rb_pressed, has_previous_state_);
  events.rb_released =
      FallingEdge(state.rb_pressed, previous_state_.rb_pressed, has_previous_state_);

  if (state.home_pressed) {
    events.request_zero_output_estop = true;
    has_previous_state_ = true;
    previous_state_ = state;
    return events;
  }

  if (zero_output_estop_latched) {
    if (start_rising) {
      // 急停恢复改成与用户一致的单键 START 语义：
      // HOME 负责立即清零并进入急停锁存，START 负责在确认后重新请求使能。
      // 因此急停锁存内继续屏蔽 BACK 和功能组合，只放行 START 这一条恢复路径。
      events.request_passive = true;
    }

    has_previous_state_ = true;
    previous_state_ = state;
    return events;
  }

  if (start_rising) {
    events.request_passive = true;
  }
  if (back_rising) {
    events.request_stand_init = true;
  }

  AppendFunctionRequestIfNeeded(
      state.rb_pressed, a_rising, x_rising, b_rising, y_rising, &events);

  has_previous_state_ = true;
  previous_state_ = state;
  return events;
}

void JoystickMapper::Reset() {
  has_previous_state_ = false;
  previous_state_ = JoystickState{};
}

const char* JoystickMapper::ToString(FunctionRequest request) {
  switch (request) {
    case FunctionRequest::kBeyondMimic:
      return "beyond_mimic";
    case FunctionRequest::kReservedFunction1:
      return "reserved_function_1";
    case FunctionRequest::kReservedFunction2:
      return "reserved_function_2";
    case FunctionRequest::kReservedFunction3:
      return "reserved_function_3";
  }

  return "unknown";
}

std::string JoystickEvents::Summary() const {
  std::ostringstream stream;
  stream << "estop=" << (request_zero_output_estop ? "true" : "false")
         << ", start_enable=" << (request_passive ? "true" : "false")
         << ", reset_initial_position=" << (request_stand_init ? "true" : "false")
         << ", rb_held=" << (rb_held ? "true" : "false")
         << ", rb_pressed=" << (rb_pressed ? "true" : "false")
         << ", rb_released=" << (rb_released ? "true" : "false")
         << ", function_request="
         << (function_request.has_value() ? JoystickMapper::ToString(*function_request) : "none")
         << ", warnings=" << JoinWarnings(warnings);
  return stream.str();
}

void JoystickMapper::ValidateConfig(const JoystickMapperConfig& config) {
  (void)config;
}

void JoystickMapper::ValidateState(const JoystickState& state) {
  (void)state;
}

}  // namespace zky_rl_deploy
