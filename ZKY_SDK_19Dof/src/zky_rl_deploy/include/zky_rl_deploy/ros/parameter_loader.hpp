#pragma once

#include <ros/node_handle.h>

#include "zky_rl_deploy/core/deploy_context.hpp"
#include "zky_rl_deploy/ros/ros_adapter.hpp"

namespace zky_rl_deploy {
namespace ros_adapter {

// 该文件负责从 ROS 私有参数命名空间读取部署骨架所需的最小配置。
// 目前只允许读取安全相关开关和命名空间信息，不在这里接入任何真实电机输出参数。
class ParameterLoader {
 public:
  DeployContext Load(const ::ros::NodeHandle& private_nh) const;
  RuntimeConfig LoadRuntimeConfig(const ::ros::NodeHandle& private_nh) const;
};

}  // namespace ros_adapter
}  // namespace zky_rl_deploy
