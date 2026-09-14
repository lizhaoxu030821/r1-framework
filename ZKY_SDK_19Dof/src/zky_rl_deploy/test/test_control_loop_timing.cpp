#include <gtest/gtest.h>

#include <chrono>

#include "zky_rl_deploy/core/control_loop_timing.hpp"

namespace zky_rl_deploy {
namespace {

using Clock = ControlLoopTimingTracker::Clock;
using Duration = ControlLoopTimingTracker::Duration;

Clock::time_point MillisecondsSinceEpoch(int value_ms) {
  return Clock::time_point(std::chrono::milliseconds(value_ms));
}

TEST(ControlLoopTimingTrackerTest, KeepsNominalLoopAsJitterOnlyWithoutOverrun) {
  ControlLoopTimingTracker tracker(Duration(std::chrono::milliseconds(20)),
                                   MillisecondsSinceEpoch(0));

  const auto first_metrics = tracker.BeginIteration(MillisecondsSinceEpoch(20));
  EXPECT_DOUBLE_EQ(first_metrics.dt_ms, 20.0);
  EXPECT_DOUBLE_EQ(first_metrics.jitter_ms, 0.0);
  EXPECT_EQ(first_metrics.consecutive_overruns, 0U);
  EXPECT_EQ(tracker.FinishIteration(MillisecondsSinceEpoch(35)), 0U);

  const auto second_metrics = tracker.BeginIteration(MillisecondsSinceEpoch(40));
  EXPECT_DOUBLE_EQ(second_metrics.dt_ms, 20.0);
  EXPECT_DOUBLE_EQ(second_metrics.jitter_ms, 0.0);
  EXPECT_EQ(second_metrics.consecutive_overruns, 0U);
}

TEST(ControlLoopTimingTrackerTest, ReportsPreviousDeadlineMissOnNextIteration) {
  ControlLoopTimingTracker tracker(Duration(std::chrono::milliseconds(20)),
                                   MillisecondsSinceEpoch(0));

  tracker.BeginIteration(MillisecondsSinceEpoch(20));
  EXPECT_EQ(tracker.FinishIteration(MillisecondsSinceEpoch(45)), 1U);

  const auto overrun_metrics = tracker.BeginIteration(MillisecondsSinceEpoch(45));
  EXPECT_DOUBLE_EQ(overrun_metrics.dt_ms, 25.0);
  EXPECT_DOUBLE_EQ(overrun_metrics.jitter_ms, 5.0);
  EXPECT_EQ(overrun_metrics.consecutive_overruns, 1U);

  EXPECT_EQ(tracker.FinishIteration(MillisecondsSinceEpoch(58)), 0U);

  const auto catch_up_metrics = tracker.BeginIteration(MillisecondsSinceEpoch(60));
  EXPECT_DOUBLE_EQ(catch_up_metrics.dt_ms, 15.0);
  EXPECT_DOUBLE_EQ(catch_up_metrics.jitter_ms, 5.0);
  EXPECT_EQ(catch_up_metrics.consecutive_overruns, 0U);
}

TEST(ControlLoopTimingTrackerTest, CountsConsecutiveDeadlineMisses) {
  ControlLoopTimingTracker tracker(Duration(std::chrono::milliseconds(20)),
                                   MillisecondsSinceEpoch(0));

  tracker.BeginIteration(MillisecondsSinceEpoch(20));
  EXPECT_EQ(tracker.FinishIteration(MillisecondsSinceEpoch(45)), 1U);

  const auto second_metrics = tracker.BeginIteration(MillisecondsSinceEpoch(45));
  EXPECT_EQ(second_metrics.consecutive_overruns, 1U);
  EXPECT_EQ(tracker.FinishIteration(MillisecondsSinceEpoch(70)), 2U);

  const auto third_metrics = tracker.BeginIteration(MillisecondsSinceEpoch(70));
  EXPECT_EQ(third_metrics.consecutive_overruns, 2U);
  EXPECT_EQ(tracker.FinishIteration(MillisecondsSinceEpoch(95)), 3U);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
