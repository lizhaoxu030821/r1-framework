import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path


def get_latest_file(directory, pattern):
    """获取指定目录下匹配模式的最新文件"""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        raise FileNotFoundError(f"在 {directory} 中未找到匹配 {pattern} 的文件")
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


# ========== 配置区域 ==========
# 方式1: 自动读取最新文件（默认）
# 设置 data_file_path = None 将自动读取 data_directory 中最新的 policy_data_*.csv 文件
data_file_path = None

# 方式2: 手动指定文件路径
# 如需指定特定文件，请取消注释下面这行并填写完整路径
# data_file_path = '/home/tib/code/bipedal_deploy_ZKY/record_data/policy_data_record/policy_data_20251119_212210.csv'

# 自动检测项目根目录和数据目录
script_dir = Path(__file__).parent
project_root = script_dir.parent
data_directory = str(project_root / "record_data" / "policy_data_record")
file_pattern = 'policy_data_*.csv'
# ============================

# 确定要读取的文件
if data_file_path is None:
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"数据目录不存在: {data_directory}\n请先运行测试程序生成数据")
    data_file_path = get_latest_file(data_directory, file_pattern)
    print(f"自动选择最新文件: {data_file_path}")
else:
    print(f"使用指定文件: {data_file_path}")

# 读取CSV数据文件
df = pd.read_csv(data_file_path)

# # 转换时间列为数字
# df['Time'] = pd.to_numeric(df['Time'])

# # 将时间从0开始
# # df['Time'] -= df['Time'].iloc[0]
# df['Time'] = df['Time'].to_numpy() - df['Time'].iloc[0]
# 转换时间列为数字类型并处理为NumPy数组 
# time_array = df['Time'].to_numpy() 
# start_time_value = time_array[0] 
# df['Time'] = time_array - start_time_value
# 定义时间范围
start_time = 0.0
# end_time = 1000
end_time = df['Time'].max() 

# 筛选数据
filtered_data = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]

# 指定要绘制的列
columns_to_plot = ['counter']
# columns_to_plot = ['sin_phase', 'cos_phase']
# columns_to_plot = ['cmd_x', 'cmd_y', 'cmd_yaw']
# columns_to_plot = ['qj_1', 'dqj_1']
# columns_to_plot = ['last_action_0']
# columns_to_plot = ['ang_vel_x', 'ang_vel_y', 'ang_vel_z']
# columns_to_plot = ['euler_roll', 'euler_pitch', 'euler_yaw']
# columns_to_plot = ['action_0', 'action_1', 'action_2', 'action_3']
# columns_to_plot = ['action_4', 'action_5', 'action_6', 'action_7']
# columns_to_plot = ['target_q_1', 'target_q_4', 'target_q_5', 'target_q_6']
# columns_to_plot = ['target_q_4', 'target_q_5', 'target_q_6', 'target_q_7']
# columns_to_plot = ['target_tau_0', 'target_tau_1', 'target_tau_2', 'target_tau_3']
# columns_to_plot = ['target_tau_4', 'target_tau_5', 'target_tau_6', 'target_tau_7']

# 将数据转换为 NumPy 数组以确保兼容性
time_data = filtered_data['Time'].to_numpy()

# 绘制数据
plt.figure(figsize=(12, 8))
colors = plt.cm.get_cmap('tab20', 20)

# for i, column in enumerate(columns_to_plot):
#     plt.plot(filtered_data['Time'], filtered_data[column], label=column, color=colors(i))
for i, column in enumerate(columns_to_plot):
    column_data = filtered_data[column].to_numpy() # 转换列数据为 NumPy 数组
    plt.plot(time_data, column_data, label=column, color=colors(2*i+2), marker='x', markersize=4,linewidth=1.5)
# 设置图形细节
plt.title('Policy Data Plot')
plt.xlabel('Time (s)')
plt.ylabel('Values')
plt.legend(loc='upper right')
plt.grid()
plt.show()