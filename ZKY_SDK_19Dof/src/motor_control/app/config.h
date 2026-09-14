/*
 * @Description:
 * @Author: kx zhang
 * @Date: 2022-09-20 09:21:54
 * @LastEditTime: 2022-11-13 17:00:10
 */


#pragma once

#include <inttypes.h>
#include <string.h>

// 19 自由度硬件拓扑常量：
// - 从站0：左腿 6 电机（CAN1: id1~3，CAN2: id4~6）
// - 从站1：右腿 6 电机（CAN1: id1~3，CAN2: id4~6）
// - 从站2：上肢/腰部 7 电机（CAN1: id1~3，CAN2: id4~7）
// 这些宏同时被 C 与 C++ 文件包含，所有数组长度必须从这里取值，避免
// 旧版 12 电机硬编码和新版 19 电机控制链路混用导致越界。
#define ZKY_SLAVE_NUMBER 3
#define ZKY_LEG_MOTORS_PER_SLAVE 6
#define ZKY_UPPER_MOTORS_ON_SLAVE 7
#define ZKY_MAX_MOTORS_PER_SLAVE 7
#define ZKY_TOTAL_MOTORS 19

//********************************************//
//***********EtherCAT Message*****************//
//********************************************//

#pragma pack(push, 1)

struct Motor_Msg
{
  uint32_t id;
  uint8_t rtr;
  uint8_t dlc;
  uint8_t data[8];
};

typedef struct
{
  uint8_t motor_num;
  uint8_t can_ide;
  // 旧版每个 EtherCAT 从站只传 6 个 CAN 电机帧；19 DOF 新上肢/腰部
  // 从站需要 7 个帧槽位。底层收发代码会按实际 PDO 字节数限长拷贝，
  // 因此旧 6 帧腿部从站不会被写越界，新 7 帧从站也能覆盖 body_joint。
  struct Motor_Msg motor[ZKY_MAX_MOTORS_PER_SLAVE];

} EtherCAT_Msg;

#pragma pack(pop)
