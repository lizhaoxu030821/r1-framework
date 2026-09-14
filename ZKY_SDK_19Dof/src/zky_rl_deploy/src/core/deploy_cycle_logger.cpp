#include "zky_rl_deploy/core/deploy_cycle_logger.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace zky_rl_deploy {
namespace {

std::string EscapeCsv(const std::string& value) {
  bool needs_quotes = false;
  for (const char ch : value) {
    if (ch == ',' || ch == '"' || ch == '\n') {
      needs_quotes = true;
      break;
    }
  }

  if (!needs_quotes) {
    return value;
  }

  std::string escaped;
  escaped.reserve(value.size() + 2U);
  escaped.push_back('"');
  for (const char ch : value) {
    if (ch == '"') {
      escaped.push_back('"');
    }
    escaped.push_back(ch);
  }
  escaped.push_back('"');
  return escaped;
}

std::string BuildTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm local_time = *std::localtime(&now_time);

  std::ostringstream stream;
  stream << std::put_time(&local_time, "%Y%m%d_%H%M%S");
  return stream.str();
}

void AppendScalar(std::ostringstream* line, bool first, const std::string& value) {
  if (!first) {
    *line << ",";
  }
  *line << EscapeCsv(value);
}

template <typename T>
void AppendNumeric(std::ostringstream* line, bool first, const T& value) {
  if (!first) {
    *line << ",";
  }
  *line << value;
}

void ValidateSize(const std::vector<double>& values,
                  std::size_t expected_size,
                  const char* label) {
  if (values.size() != expected_size) {
    std::ostringstream stream;
    stream << label << " size mismatch: expected " << expected_size << ", got "
           << values.size();
    throw std::invalid_argument(stream.str());
  }
}

void ValidateSize(const std::vector<int>& values,
                  std::size_t expected_size,
                  const char* label) {
  if (values.size() != expected_size) {
    std::ostringstream stream;
    stream << label << " size mismatch: expected " << expected_size << ", got "
           << values.size();
    throw std::invalid_argument(stream.str());
  }
}

}  // namespace

DeployCycleCsvLogger::DeployCycleCsvLogger(const std::string& workspace_root)
    : file_path_(BuildOutputPath(workspace_root)) {
  std::filesystem::create_directories(file_path_.parent_path());
  stream_.open(file_path_);
  if (!stream_.is_open()) {
    throw std::runtime_error("failed to open deploy cycle csv log: " + file_path_.string());
  }
  WriteHeader(&stream_);
}

void DeployCycleCsvLogger::Append(const DeployCycleLogRecord& record) {
  ValidateRecord(record);

  std::ostringstream line;
  bool first = true;
  const auto append_string = [&](const std::string& value) {
    AppendScalar(&line, first, value);
    first = false;
  };
  const auto append_bool = [&](bool value) {
    AppendNumeric(&line, first, (value ? 1 : 0));
    first = false;
  };
  const auto append_double = [&](double value) {
    AppendNumeric(&line, first, value);
    first = false;
  };
  const auto append_size = [&](std::size_t value) {
    AppendNumeric(&line, first, value);
    first = false;
  };
  const auto append_int = [&](int value) {
    AppendNumeric(&line, first, value);
    first = false;
  };

  append_double(record.ros_time_s);
  append_size(record.loop_index);
  append_string(record.fsm_state);
  append_string(record.fsm_request_trace);
  append_string(record.current_policy_trace);
  append_string(record.stage_name);
  append_string(record.candidate_source);

  append_bool(record.joy_received);
  append_bool(record.joy_semantic_mapping_ready);
  append_bool(record.joy_start_pressed);
  append_bool(record.joy_back_pressed);
  append_bool(record.joy_home_pressed);
  append_bool(record.joy_rb_pressed);
  append_bool(record.joy_a_pressed);
  append_bool(record.joy_b_pressed);
  append_bool(record.joy_x_pressed);
  append_bool(record.joy_y_pressed);
  append_string(record.joy_summary);

  append_bool(record.beyond_mimic_action_valid);
  append_string(record.beyond_mimic_action_source);
  append_double(record.inference_time_ms);
  for (double value : record.beyond_mimic_action) {
    append_double(value);
  }

  for (double value : record.target_joint_position) {
    append_double(value);
  }

  append_string(record.limited_motor_params_route);
  append_bool(record.publish_motor_params);
  append_bool(record.publish_shadow_motor_params);
  for (double value : record.limited_motor_params) {
    append_double(value);
  }
  for (int value : record.joint_modes) {
    append_int(value);
  }

  append_bool(record.joint_feedback_received);
  for (double value : record.joint_feedback_position) {
    append_double(value);
  }
  for (double value : record.joint_feedback_velocity) {
    append_double(value);
  }

  append_bool(record.t265_odom_received);
  append_bool(record.t265_imu_received);
  for (double value : record.t265_orientation_xyzw) {
    append_double(value);
  }
  for (double value : record.t265_base_ang_vel_body) {
    append_double(value);
  }
  for (double value : record.t265_odom_position) {
    append_double(value);
  }
  append_double(record.joint_state_age_ms);
  append_double(record.t265_odom_age_ms);
  append_double(record.t265_imu_age_ms);
  append_double(record.joint_state_t265_odom_dt_ms);
  append_double(record.joint_state_t265_imu_dt_ms);
  append_double(record.t265_odom_t265_imu_dt_ms);

  append_double(record.dt_ms);
  append_double(record.jitter_ms);
  append_size(record.overrun_count);
  append_size(record.stale_frame_count);
  append_int(record.stale_hold_time_ms);
  append_size(record.max_frame_jump_observed);
  append_size(record.shadow_cycles_without_fault);
  append_string(record.safety_trigger_reason);

  stream_ << line.str() << "\n";
  stream_.flush();
  if (!stream_) {
    throw std::runtime_error("failed to append deploy cycle csv row");
  }
}

std::filesystem::path DeployCycleCsvLogger::BuildOutputPath(const std::string& workspace_root) {
  return std::filesystem::path(workspace_root) / "record_data" / "zky_deploy_record" /
         ("zky_deploy_cycle_" + BuildTimestamp() + ".csv");
}

void DeployCycleCsvLogger::WriteHeader(std::ofstream* stream) {
  *stream << "ros_time_s,loop_index"
          << ",fsm_state,fsm_request_trace,current_policy_trace,stage_name,candidate_source"
          << ",joy_received,joy_semantic_mapping_ready"
          << ",joy_start,joy_back,joy_home,joy_rb,joy_a,joy_b,joy_x,joy_y,joy_summary"
          << ",beyond_mimic_action_valid,beyond_mimic_action_source,inference_time_ms";
  for (std::size_t joint_index = 0; joint_index < kMotorCommandJointCount; ++joint_index) {
    *stream << ",action_j" << std::setw(2) << std::setfill('0') << (joint_index + 1U);
  }
  for (std::size_t joint_index = 0; joint_index < kMotorCommandJointCount; ++joint_index) {
    *stream << ",target_joint_pos_j" << std::setw(2) << std::setfill('0') << (joint_index + 1U);
  }
  *stream << ",limited_motor_params_route,publish_motor_params,publish_shadow_motor_params";
  for (std::size_t joint_index = 0; joint_index < kMotorCommandJointCount; ++joint_index) {
    const std::size_t joint_number = joint_index + 1U;
    *stream << ",limited_motor_param_j" << std::setw(2) << std::setfill('0') << joint_number
            << "_kp"
            << ",limited_motor_param_j" << std::setw(2) << std::setfill('0') << joint_number
            << "_kd"
            << ",limited_motor_param_j" << std::setw(2) << std::setfill('0') << joint_number
            << "_pos"
            << ",limited_motor_param_j" << std::setw(2) << std::setfill('0') << joint_number
            << "_vel"
            << ",limited_motor_param_j" << std::setw(2) << std::setfill('0') << joint_number
            << "_tau"
            << ",limited_motor_param_j" << std::setw(2) << std::setfill('0') << joint_number
            << "_mode";
  }
  for (std::size_t joint_index = 0; joint_index < kMotorCommandJointCount; ++joint_index) {
    *stream << ",limited_mode_j" << std::setw(2) << std::setfill('0') << (joint_index + 1U);
  }
  *stream << ",joint_feedback_received";
  for (std::size_t joint_index = 0; joint_index < kMotorCommandJointCount; ++joint_index) {
    *stream << ",joint_feedback_pos_j" << std::setw(2) << std::setfill('0') << (joint_index + 1U);
  }
  for (std::size_t joint_index = 0; joint_index < kMotorCommandJointCount; ++joint_index) {
    *stream << ",joint_feedback_vel_j" << std::setw(2) << std::setfill('0') << (joint_index + 1U);
  }
  *stream << ",t265_odom_received,t265_imu_received"
          << ",t265_orientation_x,t265_orientation_y,t265_orientation_z,t265_orientation_w"
          << ",t265_base_ang_vel_x,t265_base_ang_vel_y,t265_base_ang_vel_z"
          << ",t265_odom_x,t265_odom_y,t265_odom_z"
          << ",joint_state_age_ms,t265_odom_age_ms,t265_imu_age_ms"
          << ",joint_state_t265_odom_dt_ms,joint_state_t265_imu_dt_ms,t265_odom_t265_imu_dt_ms"
          << ",dt_ms,jitter_ms,overrun_count,stale_frame_count,stale_hold_time_ms"
          << ",max_frame_jump_observed,shadow_cycles_without_fault,safety_trigger_reason\n";
}

void DeployCycleCsvLogger::ValidateRecord(const DeployCycleLogRecord& record) {
  ValidateSize(record.beyond_mimic_action,
               kMotorCommandJointCount,
               "beyond_mimic_action");
  ValidateSize(record.target_joint_position,
               kMotorCommandJointCount,
               "target_joint_position");
  ValidateSize(record.limited_motor_params,
               kMotorParamsLength,
               "limited_motor_params");
  ValidateSize(record.joint_modes,
               kMotorCommandJointCount,
               "joint_modes");
  ValidateSize(record.joint_feedback_position,
               kMotorCommandJointCount,
               "joint_feedback_position");
  ValidateSize(record.joint_feedback_velocity,
               kMotorCommandJointCount,
               "joint_feedback_velocity");
}

}  // namespace zky_rl_deploy
