#include "zky_rl_deploy/core/control_loop_timing.hpp"

#include <cmath>
#include <stdexcept>

namespace zky_rl_deploy {

ControlLoopTimingTracker::ControlLoopTimingTracker(Duration nominal_period)
    : ControlLoopTimingTracker(nominal_period, Clock::now()) {}

ControlLoopTimingTracker::ControlLoopTimingTracker(Duration nominal_period,
                                                   TimePoint initial_loop_start)
    : nominal_period_(nominal_period),
      previous_loop_start_(initial_loop_start),
      scheduled_loop_start_(initial_loop_start),
      next_loop_start_(initial_loop_start + nominal_period) {
  if (nominal_period_ <= Duration::zero()) {
    throw std::invalid_argument("nominal_period must be > 0");
  }
}

ControlLoopTimingTracker::LoopMetrics ControlLoopTimingTracker::BeginIteration(
    TimePoint actual_loop_start) {
  scheduled_loop_start_ = next_loop_start_;
  next_loop_start_ += nominal_period_;

  const Duration dt = actual_loop_start - previous_loop_start_;
  previous_loop_start_ = actual_loop_start;
  iteration_started_ = true;

  LoopMetrics metrics;
  metrics.dt_ms = DurationToMilliseconds(dt);
  metrics.jitter_ms =
      std::abs(metrics.dt_ms - DurationToMilliseconds(nominal_period_));
  // 这里返回的是“进入本轮之前已经累计的 deadline miss 次数”。
  // 当前这一轮是否又错过了下一周期截止时间，只能在本轮末尾调用 FinishIteration 后才知道。
  metrics.consecutive_overruns = consecutive_overruns_;
  return metrics;
}

std::size_t ControlLoopTimingTracker::FinishIteration(TimePoint loop_finish_time) {
  if (!iteration_started_) {
    throw std::logic_error("FinishIteration called before BeginIteration");
  }

  // 只有当本轮执行结束时已经晚于下一次计划启动时刻，才算真正 overrun。
  // 单纯 sleep_until 醒来比计划晚几百微秒或几毫秒，属于 jitter，不应直接累计成 overrun。
  if (loop_finish_time > next_loop_start_) {
    ++consecutive_overruns_;
  } else {
    consecutive_overruns_ = 0U;
  }

  iteration_started_ = false;
  return consecutive_overruns_;
}

double ControlLoopTimingTracker::DurationToMilliseconds(Duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

}  // namespace zky_rl_deploy
