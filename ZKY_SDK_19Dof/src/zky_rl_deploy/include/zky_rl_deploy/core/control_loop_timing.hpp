#pragma once

#include <chrono>
#include <cstddef>

namespace zky_rl_deploy {

// 这个辅助类把 fixed-period 控制循环的 dt/jitter/overrun 统计单独抽出来，
// 避免在主循环里把“正常 sleep_until 唤醒延迟”和“真正错过下一周期截止时间”混为一谈。
class ControlLoopTimingTracker {
 public:
  using Clock = std::chrono::steady_clock;
  using Duration = Clock::duration;
  using TimePoint = Clock::time_point;

  struct LoopMetrics {
    double dt_ms{0.0};
    double jitter_ms{0.0};
    std::size_t consecutive_overruns{0U};
  };

  explicit ControlLoopTimingTracker(Duration nominal_period);
  ControlLoopTimingTracker(Duration nominal_period, TimePoint initial_loop_start);

  const TimePoint& next_loop_start() const { return next_loop_start_; }

  LoopMetrics BeginIteration(TimePoint actual_loop_start);
  std::size_t FinishIteration(TimePoint loop_finish_time);

 private:
  static double DurationToMilliseconds(Duration duration);

  Duration nominal_period_;
  TimePoint previous_loop_start_;
  TimePoint scheduled_loop_start_;
  TimePoint next_loop_start_;
  std::size_t consecutive_overruns_{0U};
  bool iteration_started_{false};
};

}  // namespace zky_rl_deploy
