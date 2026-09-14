import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def get_latest_file(directory, pattern):
    """获取指定目录下匹配模式的最新文件"""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        raise FileNotFoundError(f"在 {directory} 中未找到匹配 {pattern} 的文件")
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


# ========== 配置区域 ==========
# 方式1: 自动读取最新文件（默认）
# 设置 data_file_path = None 将自动读取 data_directory 中最新的 motor_data_*.csv 文件
data_file_path = None

# 方式2: 手动指定文件路径
# 如需指定特定文件，请取消注释下面这行并填写完整路径
# data_file_path = '/home/tib/code/bipedal_deploy_ZKY/record_data/motor_data_record/motor_data_20260126_145149.csv'

# 默认数据目录（当 data_file_path = None 时使用）
data_directory = '/home/tib/code/bipedal_deploy_ZKY/record_data/motor_data_record'
file_pattern = 'motor_data_*.csv'
# ============================

# 确定要读取的文件
if data_file_path is None:
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
# columns_to_plot = [ 'M6_vel', 'M6_vel_filter', 'M6_vel_target']
# columns_to_plot = ['M1_tau','M2_tau','M3_tau','M4_tau','M5_tau','M6_tau'] # 根据需要修改
# columns_to_plot = ['M7_tau','M5_tau','M6_tau','M8_tau']
# columns_to_plot = ['pos_est_z', 'odom_z']
# columns_to_plot = ['M8_pos', 'M8_pos_target']
# columns_to_plot = ['AnkleAngle_l', 'AnkleAngle_r']
# columns_to_plot = ['contact_l', 'contact_r']
columns_to_plot = ['M3_pos', 'M3_pos_target']
# columns_to_plot = ['M1_tau_target', 'M2_tau_target','M3_tau_target','M4_tau_target','M5_tau_target','M6_tau_target','M7_tau_target','M8_tau_target']
# columns_to_plot = ['M1_pos_target', 'M2_pos_target','M3_pos_target','M4_pos_target','M5_pos_target','M6_pos_target','M7_pos_target','M8_pos_target']


# columns_to_plot = ['M1_tau', 'M5_tau']

# columns_to_plot = [ 'odom_z']
# columns_to_plot = ['raw_laser_h', 'filtered_laser_h']
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
plt.title('Motor Data Plot')
plt.xlabel('Time (s)')
plt.ylabel('Values')
plt.legend(loc='upper right')
plt.grid()
plt.show()