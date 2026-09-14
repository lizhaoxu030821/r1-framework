extern "C" {
#include "ethercat.h"  
#include "motor_control.h"
#include "transmit.h"
}
#include <iostream>
#include "queue.h"
#include <sys/time.h>
#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstring>

#define EC_TIMEOUTM

spsc_queue<EtherCAT_Msg_ptr, capacity<10>> messages[SLAVE_NUMBER];
std::atomic<bool> running{ false };
std::thread runThread;

YKSMotorData motorDate_recv[TOTAL_MOTOR_NUMBER];
YKSIMUData imuData_recv;

char IOmap[4096];
OSAL_THREAD_HANDLE checkThread;
int expectedWKC;
boolean needlf;
volatile int wkc;
boolean inOP;
uint8 currentgroup = 0;
uint64_t num;
bool isConfig[SLAVE_NUMBER]{ false };

#define EC_TIMEOUTMON 500

void EtherCAT_Data_Get();

void EtherCAT_Command_Set();
void EtherCAT_Get_State();
void EtherCAT_Send_Command(YKSMotorData* data);
void EtherCAT_Send_Command_Position(YKSMotorData* data);
void EtherCAT_Send_Command_Speed(float* data);
void Revert_State(YKSMotorData* motor_data);

struct MotorRoute
{
  uint8_t slave;
  uint8_t channel;
  uint16_t motor_id;
};

// 19 DOF 下发路由表：
// 全局索引 0~11 保持原 12 下肢顺序；12~18 为左臂、右臂和腰部。
// channel 表示从站 PDO 内 CAN 帧槽位：腿部从站 channel1~3 对应 CAN1，
// channel4~6 对应 CAN2；上肢/腰部从站 channel1~3 对应从站2 CAN1，
// channel4~7 对应从站2 CAN2。
static constexpr std::array<MotorRoute, TOTAL_MOTOR_NUMBER> kMotorRoutes = {{
    {0, 1, 1},  // 0  left_hip_yaw
    {0, 2, 2},  // 1  left_hip_roll
    {0, 3, 3},  // 2  left_hip_pitch
    {0, 4, 4},  // 3  left_knee
    {0, 5, 5},  // 4  left_ankle_pitch -> motor5
    {0, 6, 6},  // 5  left_ankle_roll  -> motor6
    {1, 1, 1},  // 6  right_hip_yaw
    {1, 2, 2},  // 7  right_hip_roll
    {1, 3, 3},  // 8  right_hip_pitch
    {1, 4, 4},  // 9  right_knee
    {1, 5, 5},  // 10 right_ankle_pitch -> motor5
    {1, 6, 6},  // 11 right_ankle_roll  -> motor6
    {2, 1, 1},  // 12 left_shoulder_pitch
    {2, 2, 2},  // 13 left_shoulder_roll
    {2, 3, 3},  // 14 left_elbow
    {2, 4, 4},  // 15 right_shoulder_pitch
    {2, 5, 5},  // 16 right_shoulder_roll
    {2, 6, 6},  // 17 right_elbow
    {2, 7, 7},  // 18 body_joint
}};

static int motorCountForSlave(int slave)
{
  if (slave == 0 || slave == 1)
    return ZKY_LEG_MOTORS_PER_SLAVE;
  if (slave == 2)
    return ZKY_UPPER_MOTORS_ON_SLAVE;
  return 0;
}

static int feedbackOffsetForSlave(int slave)
{
  if (slave == 0)
    return 0;
  if (slave == 1)
    return ZKY_LEG_MOTORS_PER_SLAVE;
  if (slave == 2)
    return ZKY_LEG_MOTORS_PER_SLAVE * 2;
  return -1;
}

static int configuredSlaveCount()
{
  return std::max(0, std::min(ec_slavecount, SLAVE_NUMBER));
}

static size_t processDataBytesForSlave(int slave, bool input)
{
  if (slave < 0 || slave >= ec_slavecount)
    return 0U;

  const auto& ec_slave_ref = ec_slave[slave + 1];
  int bytes = input ? ec_slave_ref.Ibytes : ec_slave_ref.Obytes;
  const int bits = input ? ec_slave_ref.Ibits : ec_slave_ref.Obits;
  if (bytes == 0 && bits > 0)
    bytes = 1;
  if (bytes <= 0)
    return 0U;
  return std::min(static_cast<size_t>(bytes), sizeof(EtherCAT_Msg));
}

static size_t frameCapacityForSlave(int slave, bool input)
{
  const size_t bytes = processDataBytesForSlave(slave, input);
  if (bytes <= 2U)
    return 0U;
  return std::min(static_cast<size_t>(ZKY_MAX_MOTORS_PER_SLAVE),
                  (bytes - 2U) / sizeof(Motor_Msg));
}

static void readSlaveMessage(int slave, EtherCAT_Msg* dst)
{
  memset(dst, 0, sizeof(EtherCAT_Msg));
  if (slave < 0 || slave >= ec_slavecount)
    return;
  const size_t bytes = processDataBytesForSlave(slave, true);
  if (ec_slave[slave + 1].inputs && bytes > 0U)
    memcpy(dst, ec_slave[slave + 1].inputs, bytes);
}

static void writeSlaveMessage(int slave, const EtherCAT_Msg* src)
{
  if (slave < 0 || slave >= ec_slavecount)
    return;
  const size_t bytes = processDataBytesForSlave(slave, false);
  if (ec_slave[slave + 1].outputs && bytes > 0U)
    memcpy(ec_slave[slave + 1].outputs, src, bytes);
}

static void warnIfPdoCapacityMismatch()
{
  for (int slave = 0; slave < configuredSlaveCount(); ++slave)
  {
    const int required = motorCountForSlave(slave);
    const size_t out_frames = frameCapacityForSlave(slave, false);
    const size_t in_frames = frameCapacityForSlave(slave, true);
    printf("[EtherCAT Init] Slave %d PDO frame capacity: OUT %zu, IN %zu, required motors %d\n",
           slave, out_frames, in_frames, required);
    if (required > 0 &&
        (out_frames < static_cast<size_t>(required) ||
         in_frames < static_cast<size_t>(required)))
    {
      printf("[EtherCAT Warning] Slave %d PDO capacity is smaller than 19DOF route requirement. "
             "Please update slave firmware/ESI if upper-body motor frames are missing.\n",
             slave);
    }
  }
}

static void degraded_handler()
{
  printf("[EtherCAT Error] Logging error...\n");
  time_t current_time = time(NULL);
  char* time_str = ctime(&current_time);
  printf("ESTOP. EtherCAT became degraded at %s.\n", time_str);
  printf("[EtherCAT Error] Stopping RT process.\n");
}

static int run_ethercat(const char* ifname)
{
  int i;
  int oloop, iloop, chk;
  needlf = FALSE;
  inOP = FALSE;

  num = 1;

  /* initialise SOEM, bind socket to ifname */
  if (ec_init(ifname))
  {
    printf("[EtherCAT Init] Initialization on device %s succeeded.\n", ifname);
    /* find and auto-config slaves */

    if (ec_config_init(FALSE) > 0)
    {
      printf("[EtherCAT Init] %d slaves found and configured.\n", ec_slavecount);
      if (ec_slavecount < SLAVE_NUMBER)
      {
        printf("[RT EtherCAT] Warning: Expected %d slaves, found %d.\n", SLAVE_NUMBER, ec_slavecount);
      }

      for (int slave_idx = 0; slave_idx < ec_slavecount; slave_idx++)
        ec_slave[slave_idx + 1].CoEdetails &= ~ECT_COEDET_SDOCA;

      ec_config_map(&IOmap);
      ec_configdc();

      printf("[EtherCAT Init] Mapped slaves.\n");
      /* wait for all slaves to reach SAFE_OP state */
      ec_statecheck(0, EC_STATE_SAFE_OP, EC_TIMEOUTSTATE * SLAVE_NUMBER);

      for (int slave_idx = 0; slave_idx < ec_slavecount; slave_idx++)
      {
        printf("[SLAVE %d]\n", slave_idx);
        printf("  IN  %d bytes, %d bits\n", ec_slave[slave_idx].Ibytes, ec_slave[slave_idx].Ibits);
        printf("  OUT %d bytes, %d bits\n", ec_slave[slave_idx].Obytes, ec_slave[slave_idx].Obits);
        printf("\n");
      }

      oloop = ec_slave[0].Obytes;
      if ((oloop == 0) && (ec_slave[0].Obits > 0))
        oloop = 1;
      if (oloop > 8)
        oloop = 8;
      iloop = ec_slave[0].Ibytes;
      if ((iloop == 0) && (ec_slave[0].Ibits > 0))
        iloop = 1;
      if (iloop > 8)
        iloop = 8;

      printf("[EtherCAT Init] segments : %d : %d %d %d %d\n", ec_group[0].nsegments, ec_group[0].IOsegment[0],
             ec_group[0].IOsegment[1], ec_group[0].IOsegment[2], ec_group[0].IOsegment[3]);

      warnIfPdoCapacityMismatch();

      printf("[EtherCAT Init] Requesting operational state for all slaves...\n");
      expectedWKC = (ec_group[0].outputsWKC * 2) + ec_group[0].inputsWKC;
      printf("[EtherCAT Init] Calculated workcounter %d\n", expectedWKC);
      ec_slave[0].state = EC_STATE_OPERATIONAL;
      /* send one valid process data to make outputs in slaves happy*/
      ec_send_processdata();
      ec_receive_processdata(EC_TIMEOUTRET);
      /* request OP state for all slaves */
      ec_writestate(0);
      chk = 40;
      /* wait for all slaves to reach OP state */
      do
      {
        ec_send_processdata();
        ec_receive_processdata(EC_TIMEOUTRET);
        ec_statecheck(0, EC_STATE_OPERATIONAL, 50000);
      } while (chk-- && (ec_slave[0].state != EC_STATE_OPERATIONAL));

      if (ec_slave[0].state == EC_STATE_OPERATIONAL)
      {
        printf("[EtherCAT Init] Operational state reached for all slaves.\n");
        inOP = TRUE;
        return 1;
      }
      else
      {
        printf("[EtherCAT Error] Not all slaves reached operational state.\n");
        ec_readstate();
        for (i = 1; i <= ec_slavecount; i++)
        {
          if (ec_slave[i].state != EC_STATE_OPERATIONAL)
          {
            printf("[EtherCAT Error] Slave %d State=0x%2.2x StatusCode=0x%4.4x : %s\n", i, ec_slave[i].state,
                   ec_slave[i].ALstatuscode, ec_ALstatuscode2string(ec_slave[i].ALstatuscode));
          }
        }
      }
    }
    else
    {
      printf("[EtherCAT Error] No slaves found!\n");
    }
  }
  else
  {
    printf("[EtherCAT Error] No socket connection on %s - are you running run.sh?\n", ifname);
  }
  return 0;
}

static int err_count = 0;
static int err_iteration_count = 0;
/**@brief EtherCAT errors are measured over this period of loop iterations */
#define K_ETHERCAT_ERR_PERIOD 100

/**@brief Maximum number of etherCAT errors before a fault per period of loop iterations */
#define K_ETHERCAT_ERR_MAX 20

static OSAL_THREAD_FUNC ecatcheck(void* ptr)
{
  (void)ptr;
  int slave = 0;
  while (1)
  {
    // count errors
    if (err_iteration_count > K_ETHERCAT_ERR_PERIOD)
    {
      err_iteration_count = 0;
      err_count = 0;
    }

    if (err_count > K_ETHERCAT_ERR_MAX)
    {
      // possibly shut down
      printf("[EtherCAT Error] EtherCAT connection degraded.\n");
      printf("[Simulink-Linux] Shutting down....\n");
      degraded_handler();
      break;
    }
    err_iteration_count++;

    if (inOP && ((wkc < expectedWKC) || ec_group[currentgroup].docheckstate))
    {
      if (needlf)
      {
        needlf = FALSE;
        printf("\n");
      }
      /* one ore more slaves are not responding */
      ec_group[currentgroup].docheckstate = FALSE;
      ec_readstate();
      for (slave = 1; slave <= ec_slavecount; slave++)
      {
        if ((ec_slave[slave].group == currentgroup) && (ec_slave[slave].state != EC_STATE_OPERATIONAL))
        {
          ec_group[currentgroup].docheckstate = TRUE;
          if (ec_slave[slave].state == (EC_STATE_SAFE_OP + EC_STATE_ERROR))
          {
            printf("[EtherCAT Error] Slave %d is in SAFE_OP + ERROR, attempting ack.\n", slave);
            ec_slave[slave].state = (EC_STATE_SAFE_OP + EC_STATE_ACK);
            ec_writestate(slave);
            err_count++;
          }
          else if (ec_slave[slave].state == EC_STATE_SAFE_OP)
          {
            printf("[EtherCAT Error] Slave %d is in SAFE_OP, change to OPERATIONAL.\n", slave);
            ec_slave[slave].state = EC_STATE_OPERATIONAL;
            ec_writestate(slave);
            err_count++;
          }
          else if (ec_slave[slave].state > 0)
          {
            if (ec_reconfig_slave(slave, EC_TIMEOUTMON))
            {
              ec_slave[slave].islost = FALSE;
              printf("[EtherCAT Status] Slave %d reconfigured\n", slave);
            }
          }
          else if (!ec_slave[slave].islost)
          {
            /* re-check state */
            ec_statecheck(slave, EC_STATE_OPERATIONAL, EC_TIMEOUTRET);
            if (!ec_slave[slave].state)
            {
              ec_slave[slave].islost = TRUE;
              printf("[EtherCAT Error] Slave %d lost\n", slave);
              err_count++;
            }
          }
        }
        if (ec_slave[slave].islost)
        {
          if (!ec_slave[slave].state)
          {
            if (ec_recover_slave(slave, EC_TIMEOUTMON))
            {
              ec_slave[slave].islost = FALSE;
              printf("[EtherCAT Status] Slave %d recovered\n", slave);
            }
          }
          else
          {
            ec_slave[slave].islost = FALSE;
            printf("[EtherCAT Status] Slave %d found\n", slave);
          }
        }
      }
      if (!ec_group[currentgroup].docheckstate)
        printf("[EtherCAT Status] All slaves resumed OPERATIONAL.\n");
    }
    osal_usleep(50000);
  }
}

int EtherCAT_Init(char* ifname)
{
  int i;
  int rc;
  printf("[EtherCAT] Initializing EtherCAT\n");
  osal_thread_create((void*)&checkThread, 128000, (void*)&ecatcheck, (void*)&ctime);
  for (i = 1; i < 10; i++)
  {
    printf("[EtherCAT] Attempting to start EtherCAT, try %d of 10.\n", i);
    rc = run_ethercat(ifname);
    if (rc)
      break;
    osal_usleep(1000000);
  }
  if (rc)
    printf("[EtherCAT] EtherCAT successfully initialized on attempt %d \n", i);
  else
  {
    printf("[EtherCAT Error] Failed to initialize EtherCAT after 10 tries. \n");
  }
  return ec_slavecount;
}

static int wkc_err_count = 0;
static int wkc_err_iteration_count = 0;

EtherCAT_Msg Rx_Message[SLAVE_NUMBER];
EtherCAT_Msg Tx_Message[SLAVE_NUMBER];

/**
 * @brief EtherCAT实时通信核心函数 - 执行一个完整的通信周期
 * 
 * 功能说明：
 * 1. 发送控制指令到所有电机从站（位置、速度、力矩、PD增益等）
 * 2. 接收电机反馈数据（实际位置、速度、力矩）
 * 3. 监控通信质量，检测丢包和错误
 * 4. 错误管理：每100次循环内若超过20次错误则触发安全停机
 * 
 * 使用方式：
 * - 在实时控制循环中周期性调用（典型频率：1kHz）
 * - 必须先调用 EtherCAT_Init() 完成初始化
 * - 通常在独立线程中循环执行（见 runImpl() 函数）
 * 
 * @note 此函数为阻塞式调用，执行时间约为1ms
 * @warning 不要在此函数中执行耗时操作，否则会影响实时性
 */
void EtherCAT_Run()
{
  // 错误计数周期性重置（每100次循环）
  if (wkc_err_iteration_count > K_ETHERCAT_ERR_PERIOD)
  {
    wkc_err_count = 0;
    wkc_err_iteration_count = 0;
  }
  
  // 错误计数超限检查，触发安全停机
  if (wkc_err_count > K_ETHERCAT_ERR_MAX)
  {
    printf("[EtherCAT Error] Error count too high!\n");
    degraded_handler();
  }
  
  // ===== 发送阶段 =====
  // 准备发送数据（从消息队列获取控制指令）
  EtherCAT_Command_Set();
  // 通过EtherCAT总线发送控制指令到所有从站
  ec_send_processdata();
  
  // ===== 接收阶段 =====
  // 接收从站反馈数据，wkc为工作计数器（Working Counter）
  wkc = ec_receive_processdata(EC_TIMEOUTRET);
  // 解析接收到的电机状态数据并存储到 motorDate_recv[] 数组
  EtherCAT_Data_Get();
  
  // ===== 通信质量检测 =====
  // 检测丢包：wkc < expectedWKC 表示有从站未正确响应
  if (wkc < expectedWKC)
  {
    printf("\x1b[31m[EtherCAT Error] Dropped packet (Bad WKC!)\x1b[0m\n");
    wkc_err_count++;  // 错误计数递增
  }
  else
  {
    needlf = TRUE;  // 通信正常标志
  }
  wkc_err_iteration_count++;  // 循环计数递增
}

void EtherCAT_Data_Get()
{
  for (int slave = 0; slave < configuredSlaveCount(); ++slave)
  {
    readSlaveMessage(slave, &Rx_Message[slave]);
    RV_can_data_repack(&Rx_Message[slave], comm_ack, slave);
  }
}
// void Revert_State(YKSMotorData* motor_data)
// {
//   YKSMotorData tmp;

//   for (int i = 3; i < 6; i++)
//   {
//     tmp = motor_data[i];
//     motor_data[i] = motor_data[i + 6];  // 0-3
//     motor_data[i + 6] = tmp;
//   }
// }

// *****************************
// *****************************
// *****************************
/*函数功能：获取EtherCAT总线上从设备的状态信息*/
void EtherCAT_Get_State()
{
  wkc = ec_receive_processdata(EC_TIMEOUTRET);
  static int debug_counter = 0;
  // bool should_print = (debug_counter++ % 100 == 0); // 每100次打印一次
  bool should_print = 0; // 控制打印开关，设为0关闭打印
  for (int slave = 0; slave < configuredSlaveCount(); ++slave)  {
    // 在每个 slave 循环开始时，只清空该从站实际负责的反馈段。
    // 从站2负责 7 个上肢/腰部电机，不能再按旧版固定 6 个清零。
    const int motor_offset = feedbackOffsetForSlave(slave);
    const int motor_count = motorCountForSlave(slave);
    if (motor_offset >= 0 && motor_count > 0) {
      memset(rv_motor_msg + motor_offset, 0, sizeof(OD_Motor_Msg) * motor_count);
    }

    readSlaveMessage(slave, &Rx_Message[slave]);
    
    if (motor_offset >= 0 && motor_count > 0)
    {
      if (should_print) {
        printf("[DEBUG] Before RV_can_data_repack - Slave %d:\n", slave);
        for (int motor_index = 0; motor_index < motor_count; motor_index++) {
          int idx = motor_offset + motor_index;
          printf("  rv_motor_msg[%d]: pos=%.3f, vel=%.3f, tau=%.3f\n", 
                 idx, rv_motor_msg[idx].angle_actual_rad, 
                 rv_motor_msg[idx].speed_actual_rad, 
                 rv_motor_msg[idx].current_actual_float);
        }
      }
      
      // 解析电机数据到rv_motor_msg缓冲区
      RV_can_data_repack(&Rx_Message[slave], comm_ack, slave);
      
      if (should_print) {
        printf("[DEBUG] After RV_can_data_repack - Slave %d:\n", slave);
        for (int motor_index = 0; motor_index < motor_count; motor_index++) {
          int idx = motor_offset + motor_index;
          printf("  rv_motor_msg[%d]: pos=%.3f, vel=%.3f, tau=%.3f\n", 
                 idx, rv_motor_msg[idx].angle_actual_rad, 
                 rv_motor_msg[idx].speed_actual_rad, 
                 rv_motor_msg[idx].current_actual_float);
        }
      }
      
      // 立即复制到对应的 motorDate_recv 位置，避免被下一个 slave 覆盖。
      for (int motor_index = 0; motor_index < motor_count; motor_index++)
      {
        int rv_idx = motor_offset + motor_index;
        motorDate_recv[rv_idx].pos_ = rv_motor_msg[rv_idx].angle_actual_rad;
        motorDate_recv[rv_idx].vel_ = rv_motor_msg[rv_idx].speed_actual_rad;
        motorDate_recv[rv_idx].tau_ = rv_motor_msg[rv_idx].current_actual_float;
      }
      
      if (should_print) {
        printf("[DEBUG] After copying to motorDate_recv - Slave %d:\n", slave);
        for (int motor_index = 0; motor_index < motor_count; motor_index++) {
          int idx = motor_offset + motor_index;
          printf("  motorDate_recv[%d]: pos=%.3f, vel=%.3f, tau=%.3f\n", 
                 idx, motorDate_recv[idx].pos_, 
                 motorDate_recv[idx].vel_, 
                 motorDate_recv[idx].tau_);
        }
      }
    }
  }
  
  if (should_print) {
    printf("\n[DEBUG] ===== Final motorDate_recv state =====\n");
    for (int slave = 0; slave < SLAVE_NUMBER; ++slave) {
      const int motor_offset = feedbackOffsetForSlave(slave);
      const int motor_count = motorCountForSlave(slave);
      if (motor_offset < 0 || motor_count <= 0) {
        continue;
      }
      printf("[DEBUG] Slave %d motors (%d-%d):", slave, motor_offset, motor_offset + motor_count - 1);
      for (int motor_index = 0; motor_index < motor_count; ++motor_index) {
        printf(" %.3f", motorDate_recv[motor_offset + motor_index].pos_);
      }
      printf("\n");
    }
    printf("\n");
  }
  
  if (debug_counter++ % 1000 == 0) {
    printf("\n[DEBUG] ===== Final motorDate_recv state =====\n");
    for (int slave = 0; slave < SLAVE_NUMBER; ++slave) {
      const int motor_offset = feedbackOffsetForSlave(slave);
      const int motor_count = motorCountForSlave(slave);
      if (motor_offset < 0 || motor_count <= 0) {
        continue;
      }
      printf("[DEBUG] Slave %d motors (%d-%d):", slave, motor_offset, motor_offset + motor_count - 1);
      for (int motor_index = 0; motor_index < motor_count; ++motor_index) {
        printf(" %.3f", motorDate_recv[motor_offset + motor_index].pos_);
      }
      printf("\n");
    }
    printf("\n");
  }
  //  check for dropped packet
  if (wkc < expectedWKC)
  {
    printf("\x1b[31m[EtherCAT Error] Dropped packet (Bad WKC!)\x1b[0m\n");
    wkc_err_count++;
  }
  else
  {
    needlf = TRUE;
  }
  wkc_err_iteration_count++;
}

void EtherCAT_Send_Command(YKSMotorData* mot_data)
{
  if (wkc_err_iteration_count > K_ETHERCAT_ERR_PERIOD)
  {
    wkc_err_count = 0;
    wkc_err_iteration_count = 0;
  }
  if (wkc_err_count > K_ETHERCAT_ERR_MAX)
  {
    printf("[EtherCAT Error] Error count too high!\n");
    degraded_handler();
  }
  
  // 每次直接控制下发都重新构造 3 个从站的 PDO，避免第7槽或旧命令残留。
  memset(Tx_Message, 0, sizeof(Tx_Message));
  for (int slave = 0; slave < SLAVE_NUMBER; ++slave)
  {
    Tx_Message[slave].motor_num = static_cast<uint8_t>(motorCountForSlave(slave));
  }

  for (size_t index = 0; index < kMotorRoutes.size(); ++index)
  {
    const MotorRoute& route = kMotorRoutes[index];
    if (route.slave >= SLAVE_NUMBER) {
      continue;
    }
    send_motor_ctrl_cmd(&Tx_Message[route.slave],
                        route.channel,
                        route.motor_id,
                        mot_data[index].kp_,
                        mot_data[index].kd_,
                        mot_data[index].pos_des_,
                        mot_data[index].vel_des_,
                        mot_data[index].ff_);
  }

  for (int slave = 0; slave < configuredSlaveCount(); ++slave)
  {
    writeSlaveMessage(slave, &Tx_Message[slave]);
  }
  ec_send_processdata();
}

void EtherCAT_Command_Set()
{
  for (int slave = 0; slave < configuredSlaveCount(); ++slave)
  {
    EtherCAT_Msg_ptr msg;
    if (messages[slave].pop(msg))
    {
      memcpy(&Tx_Message[slave], msg.get(), sizeof(EtherCAT_Msg));
      isConfig[slave] = true;
    }
    writeSlaveMessage(slave, &Tx_Message[slave]);
  }
}

void runImpl()
{
  while (running)
  {
    EtherCAT_Run();
  }
}

void startRun()
{
  running = true;
  runThread = std::thread(runImpl);
}
