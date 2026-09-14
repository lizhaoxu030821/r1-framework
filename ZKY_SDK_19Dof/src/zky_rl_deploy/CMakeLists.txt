cmake_minimum_required(VERSION 3.0.2)
project(zky_rl_deploy)

# 第一阶段先与现有工作区统一使用 C++17，降低后续和 motor_control 联调时的编译差异。
add_compile_options(-std=c++17 -Wall -Wextra -Wpedantic)

option(ZKY_RL_DEPLOY_ENABLE_ONNXRUNTIME
  "Enable ONNX Runtime metadata loading support for PolicyRuntime." ON)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

find_path(ZKY_RL_DEPLOY_YAML_CPP_INCLUDE_DIR
  NAMES yaml-cpp/yaml.h
  HINTS
    /usr/include
    /usr/local/include
)
find_library(ZKY_RL_DEPLOY_YAML_CPP_LIBRARY
  NAMES yaml-cpp
  HINTS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
)

if(NOT ZKY_RL_DEPLOY_YAML_CPP_INCLUDE_DIR OR NOT ZKY_RL_DEPLOY_YAML_CPP_LIBRARY)
  message(FATAL_ERROR "zky_rl_deploy: yaml-cpp is required for StageManager.")
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES zky_rl_deploy_core
  CATKIN_DEPENDS nav_msgs roscpp rospy sensor_msgs std_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

set(ZKY_RL_DEPLOY_HAS_ONNXRUNTIME FALSE)
if(ZKY_RL_DEPLOY_ENABLE_ONNXRUNTIME)
  find_path(ZKY_RL_DEPLOY_ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
      /opt/onnxruntime-linux-x64-1.16.3/include
      /usr/include
      /usr/local/include
  )
  find_library(ZKY_RL_DEPLOY_ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS
      /opt/onnxruntime-linux-x64-1.16.3/lib
      /usr/lib
      /usr/local/lib
  )

  if(ZKY_RL_DEPLOY_ONNXRUNTIME_INCLUDE_DIR AND ZKY_RL_DEPLOY_ONNXRUNTIME_LIBRARY)
    set(ZKY_RL_DEPLOY_HAS_ONNXRUNTIME TRUE)
    message(STATUS "zky_rl_deploy: ONNX Runtime enabled.")
  else()
    message(WARNING
      "zky_rl_deploy: ONNX Runtime was requested but headers/libraries were not found. "
      "PolicyRuntime will build with metadata-loading stubs only.")
  endif()
else()
  message(STATUS "zky_rl_deploy: ONNX Runtime support disabled by option.")
endif()

# 核心库当前只承载与 ROS topic 解耦的部署上下文，后续单元测试优先直接链接这里，
# 避免骨架阶段就把测试路径和真实硬件适配层绑死。
add_library(zky_rl_deploy_core
  src/core/active_output_gate.cpp
  src/core/deploy_cycle_logger.cpp
  src/core/control_loop_timing.cpp
  src/core/deploy_context.cpp
  src/core/fsm.cpp
  src/core/json_motion_frame_provider.cpp
  src/core/joint_mapper.cpp
  src/core/joystick_mapper.cpp
  src/core/motion_clip.cpp
  src/core/obs_builder.cpp
  src/core/policy_action_runner.cpp
  src/core/policy_runtime.cpp
  src/core/safety_gate.cpp
  src/core/sensor_sync.cpp
  src/core/stage_manager.cpp
  src/core/stage_request_resolver.cpp
  src/core/stand_trajectory.cpp
  src/core/stored_npz_archive.cpp
  src/core/t265_extrinsics.cpp
)

target_compile_definitions(zky_rl_deploy_core
  PUBLIC ZKY_RL_DEPLOY_HAS_ONNXRUNTIME=$<BOOL:${ZKY_RL_DEPLOY_HAS_ONNXRUNTIME}>
)

target_include_directories(zky_rl_deploy_core SYSTEM PRIVATE
  ${ZKY_RL_DEPLOY_YAML_CPP_INCLUDE_DIR}
)

if(ZKY_RL_DEPLOY_HAS_ONNXRUNTIME)
  target_include_directories(zky_rl_deploy_core SYSTEM PRIVATE
    ${ZKY_RL_DEPLOY_ONNXRUNTIME_INCLUDE_DIR}
  )
target_link_libraries(zky_rl_deploy_core PRIVATE
    ${ZKY_RL_DEPLOY_ONNXRUNTIME_LIBRARY}
  )
  get_filename_component(ZKY_RL_DEPLOY_ONNXRUNTIME_LIBRARY_DIR
    ${ZKY_RL_DEPLOY_ONNXRUNTIME_LIBRARY} DIRECTORY)
  set_target_properties(zky_rl_deploy_core PROPERTIES
    BUILD_RPATH "${ZKY_RL_DEPLOY_ONNXRUNTIME_LIBRARY_DIR}"
    INSTALL_RPATH "${ZKY_RL_DEPLOY_ONNXRUNTIME_LIBRARY_DIR}"
  )
endif()

target_link_libraries(zky_rl_deploy_core PRIVATE
  ${ZKY_RL_DEPLOY_YAML_CPP_LIBRARY}
)

# 主节点负责把 dry_run/shadow/active hardware 三条输出路径串起来，
# 但真实 /motor_params 是否发布仍必须经过 StageManager、SensorSync 和 SafetyGate 的联合门控。
add_executable(zky_deploy_node
  src/zky_deploy_node.cpp
  src/ros/parameter_loader.cpp
  src/ros/ros_adapter.cpp
)

add_executable(zky_stage0_dry_run
  src/stage0_dry_run.cpp
)

target_compile_definitions(zky_deploy_node PRIVATE
  ZKY_RL_DEPLOY_PACKAGE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_definitions(zky_stage0_dry_run PRIVATE
  ZKY_RL_DEPLOY_PACKAGE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(zky_deploy_node SYSTEM PRIVATE
  ${ZKY_RL_DEPLOY_YAML_CPP_INCLUDE_DIR}
)

target_include_directories(zky_stage0_dry_run SYSTEM PRIVATE
  ${ZKY_RL_DEPLOY_YAML_CPP_INCLUDE_DIR}
)

if(ZKY_RL_DEPLOY_HAS_ONNXRUNTIME)
  target_include_directories(zky_stage0_dry_run SYSTEM PRIVATE
    ${ZKY_RL_DEPLOY_ONNXRUNTIME_INCLUDE_DIR}
  )
endif()

target_link_libraries(zky_deploy_node
  zky_rl_deploy_core
  ${ZKY_RL_DEPLOY_YAML_CPP_LIBRARY}
  ${catkin_LIBRARIES}
)

target_link_libraries(zky_stage0_dry_run
  zky_rl_deploy_core
  ${ZKY_RL_DEPLOY_YAML_CPP_LIBRARY}
)

if(ZKY_RL_DEPLOY_HAS_ONNXRUNTIME)
  target_link_libraries(zky_stage0_dry_run
    ${ZKY_RL_DEPLOY_ONNXRUNTIME_LIBRARY}
  )
endif()

# 先预留最小 smoke test 接口，方便后续逐步补齐关节顺序和观测构造等单元测试。
if(CATKIN_ENABLE_TESTING)
  # 统一封装测试目标接线，保证所有核心单元测试都只链接纯核心库，不触碰硬件输出链路。
  function(zky_add_core_gtest target_name source_file)
    catkin_add_gtest(${target_name}
      ${source_file}
    )

    if(TARGET ${target_name})
      target_compile_definitions(${target_name} PRIVATE
        ZKY_RL_DEPLOY_PACKAGE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
      )
      target_link_libraries(${target_name}
        zky_rl_deploy_core
        ${catkin_LIBRARIES}
      )
    endif()
  endfunction()

  zky_add_core_gtest(zky_rl_deploy_smoke_test test/test_smoke.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_control_loop_timing test/test_control_loop_timing.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_joint_order test/test_joint_order.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_npz_order test/test_npz_order.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_joint_sign test/test_joint_sign.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_onnx_io test/test_onnx_io.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_motion_clip test/test_motion_clip.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_obs_builder test/test_obs_builder.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_sensor_sync test/test_sensor_sync.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_t265_extrinsics test/test_t265_extrinsics.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_fsm_transitions test/test_fsm_transitions.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_active_output_gate test/test_active_output_gate.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_safety_gate test/test_safety_gate.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_stage0_assets test/test_stage0_assets.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_deploy_cycle_logger test/test_deploy_cycle_logger.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_stage_manager test/test_stage_manager.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_stage_request_resolver test/test_stage_request_resolver.cpp)
  zky_add_core_gtest(zky_rl_deploy_test_stand_trajectory test/test_stand_trajectory.cpp)
endif()

install(TARGETS zky_rl_deploy_core zky_deploy_node zky_stage0_dry_run
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/zky_set_stage
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# 配置和 launch 先一并安装，确保后续 dry_run/shadow 启动方式可以被稳定复用。
install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
