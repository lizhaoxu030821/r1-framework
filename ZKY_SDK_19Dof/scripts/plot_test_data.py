#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真测试数据可视化工具

用于分析deploy_real.py记录的CSV数据，生成各种图表帮助理解机器人行为。

使用方法:
    python3 scripts/plot_test_data.py

功能:
- 自动查找最新的CSV文件
- 绘制关节位置、速度、动作曲线
- 显示速度命令和IMU数据
- 分析策略切换情况
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体支持（可选）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


def find_latest_csv(data_dir):
    """查找最新的CSV文件"""
    csv_files = glob.glob(os.path.join(data_dir, "policy_data_*.csv"))
    if not csv_files:
        return None
    # 按修改时间排序，返回最新的
    latest_file = max(csv_files, key=os.path.getmtime)
    return latest_file


def plot_joint_positions(df, num_joints=12):
    """绘制关节位置"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle('关节位置 (Joint Positions)', fontsize=16, fontweight='bold')
    
    for i in range(min(num_joints, 12)):
        ax = axes[i // 4, i % 4]
        col_name = f'q_{i}'
        if col_name in df.columns:
            ax.plot(df['Time'].values, df[col_name].values, label=f'Joint {i}', linewidth=1.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position (rad)')
            ax.set_title(f'Joint {i}')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    return fig


def plot_joint_velocities(df, num_joints=12):
    """绘制关节速度"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle('关节速度 (Joint Velocities)', fontsize=16, fontweight='bold')
    
    for i in range(min(num_joints, 12)):
        ax = axes[i // 4, i % 4]
        col_name = f'dq_{i}'
        if col_name in df.columns:
            ax.plot(df['Time'].values, df[col_name].values, label=f'Joint {i}', linewidth=1.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Velocity (rad/s)')
            ax.set_title(f'Joint {i}')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    return fig


def plot_actions(df, num_joints=12):
    """绘制策略输出动作"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle('策略输出动作 (Policy Actions)', fontsize=16, fontweight='bold')
    
    for i in range(min(num_joints, 12)):
        ax = axes[i // 4, i % 4]
        col_name = f'action_{i}'
        if col_name in df.columns:
            ax.plot(df['Time'].values, df[col_name].values, label=f'Action {i}', linewidth=1.5, color='orange')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Action')
            ax.set_title(f'Action {i}')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    return fig


def plot_velocity_commands(df):
    """绘制速度命令"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('速度命令 (Velocity Commands)', fontsize=16, fontweight='bold')
    
    if 'vel_cmd_x' in df.columns:
        ax.plot(df['Time'].values, df['vel_cmd_x'].values, label='Vx (前后)', linewidth=2)
    if 'vel_cmd_y' in df.columns:
        ax.plot(df['Time'].values, df['vel_cmd_y'].values, label='Vy (左右)', linewidth=2)
    if 'vel_cmd_yaw' in df.columns:
        ax.plot(df['Time'].values, df['vel_cmd_yaw'].values, label='Vyaw (旋转)', linewidth=2)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Velocity Command', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    plt.tight_layout()
    return fig


def plot_imu_data(df):
    """绘制IMU数据（重力方向和角速度）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('IMU数据 (IMU Data)', fontsize=16, fontweight='bold')
    
    # 重力方向
    if all(col in df.columns for col in ['gravity_ori_x', 'gravity_ori_y', 'gravity_ori_z']):
        ax1.plot(df['Time'].values, df['gravity_ori_x'].values, label='Gravity X', linewidth=1.5)
        ax1.plot(df['Time'].values, df['gravity_ori_y'].values, label='Gravity Y', linewidth=1.5)
        ax1.plot(df['Time'].values, df['gravity_ori_z'].values, label='Gravity Z', linewidth=1.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Gravity Orientation')
        ax1.set_title('重力方向投影')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # 角速度
    if all(col in df.columns for col in ['ang_vel_x', 'ang_vel_y', 'ang_vel_z']):
        ax2.plot(df['Time'].values, df['ang_vel_x'].values, label='Angular Vel X', linewidth=1.5)
        ax2.plot(df['Time'].values, df['ang_vel_y'].values, label='Angular Vel Y', linewidth=1.5)
        ax2.plot(df['Time'].values, df['ang_vel_z'].values, label='Angular Vel Z', linewidth=1.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.set_title('角速度')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    return fig


def plot_pd_gains(df, num_joints=12):
    """绘制PD增益（kp和kd）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('PD增益 (PD Gains)', fontsize=16, fontweight='bold')
    
    # KP
    for i in range(min(num_joints, 12)):
        col_name = f'kp_{i}'
        if col_name in df.columns:
            ax1.plot(df['Time'].values, df[col_name].values, label=f'KP {i}', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('KP')
    ax1.set_title('位置增益 (KP)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=4, fontsize=8)
    
    # KD
    for i in range(min(num_joints, 12)):
        col_name = f'kd_{i}'
        if col_name in df.columns:
            ax2.plot(df['Time'].values, df[col_name].values, label=f'KD {i}', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('KD')
    ax2.set_title('速度增益 (KD)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=4, fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_policy_timeline(df):
    """绘制策略切换时间线"""
    if 'current_policy' not in df.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle('策略切换时间线 (Policy Timeline)', fontsize=16, fontweight='bold')
    
    # 获取所有唯一的策略名称
    policies = df['current_policy'].unique()
    policy_colors = plt.cm.tab10(np.linspace(0, 1, len(policies)))
    policy_to_num = {p: i for i, p in enumerate(policies)}
    
    # 为每个时间点分配策略编号
    policy_nums = df['current_policy'].map(policy_to_num).values
    
    # 绘制时间线
    ax.scatter(df['Time'].values, policy_nums, c=[policy_colors[n] for n in policy_nums], 
               s=10, alpha=0.6)
    
    # 设置y轴刻度为策略名称
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels(policies)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Current Policy', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def print_statistics(df):
    """打印统计信息"""
    print("\n" + "="*60)
    print("数据统计摘要")
    print("="*60)
    
    print(f"\n总时长: {df['Time'].iloc[-1]:.2f} 秒")
    print(f"数据点数: {len(df)}")
    print(f"平均采样率: {len(df)/df['Time'].iloc[-1]:.1f} Hz")
    
    if 'current_policy' in df.columns:
        print("\n策略模式统计:")
        policy_counts = df['current_policy'].value_counts()
        for policy, count in policy_counts.items():
            duration = count / (len(df) / df['Time'].iloc[-1])
            print(f"  {policy}: {count} 步 ({duration:.2f}秒, {100*count/len(df):.1f}%)")
    
    # 关节位置范围
    print("\n关节位置范围 (rad):")
    q_cols = [col for col in df.columns if col.startswith('q_')]
    for col in q_cols[:6]:  # 只显示前6个
        joint_num = col.split('_')[1]
        print(f"  Joint {joint_num}: [{df[col].min():.3f}, {df[col].max():.3f}]")
    
    # 速度命令统计
    if 'vel_cmd_x' in df.columns:
        print("\n速度命令范围:")
        print(f"  Vx: [{df['vel_cmd_x'].min():.3f}, {df['vel_cmd_x'].max():.3f}]")
        print(f"  Vy: [{df['vel_cmd_y'].min():.3f}, {df['vel_cmd_y'].max():.3f}]")
        print(f"  Vyaw: [{df['vel_cmd_yaw'].min():.3f}, {df['vel_cmd_yaw'].max():.3f}]")
    
    print("\n" + "="*60 + "\n")


def main():
    """主函数"""
    print("="*60)
    print("机器人仿真测试数据可视化工具")
    print("="*60)
    
    # 查找数据目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "record_data" / "policy_data_record"
    
    if not data_dir.exists():
        print(f"\n错误: 数据目录不存在: {data_dir}")
        print("请先运行测试程序生成数据")
        return
    
    # 查找最新的CSV文件
    csv_file = find_latest_csv(str(data_dir))
    if not csv_file:
        print(f"\n错误: 在 {data_dir} 中未找到CSV文件")
        print("请先运行测试程序生成数据")
        return
    
    print(f"\n找到数据文件: {os.path.basename(csv_file)}")
    print(f"文件大小: {os.path.getsize(csv_file)/1024:.1f} KB")
    
    # 读取数据
    print("\n正在读取数据...")
    try:
        df = pd.read_csv(csv_file)
        print(f"成功读取 {len(df)} 行数据")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 打印统计信息
    print_statistics(df)
    
    # 检测关节数量
    q_cols = [col for col in df.columns if col.startswith('q_')]
    num_joints = len(q_cols)
    print(f"检测到 {num_joints} 个关节")
    
    # 生成图表
    print("\n正在生成图表...")
    figures = []
    
    print("  - 关节位置图")
    figures.append(plot_joint_positions(df, num_joints))
    
    print("  - 关节速度图")
    figures.append(plot_joint_velocities(df, num_joints))
    
    print("  - 策略动作图")
    figures.append(plot_actions(df, num_joints))
    
    print("  - 速度命令图")
    figures.append(plot_velocity_commands(df))
    
    print("  - IMU数据图")
    figures.append(plot_imu_data(df))
    
    print("  - PD增益图")
    figures.append(plot_pd_gains(df, num_joints))
    
    if 'current_policy' in df.columns:
        print("  - 策略时间线图")
        fig = plot_policy_timeline(df)
        if fig:
            figures.append(fig)
    
    print(f"\n共生成 {len(figures)} 个图表")
    
    # 保存图表
    output_dir = data_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n正在保存图表到: {output_dir}")
    for i, fig in enumerate(figures):
        if fig is not None:
            output_file = output_dir / f"plot_{i+1}_{fig._suptitle.get_text().split('(')[0].strip()}.png"
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"  保存: {output_file.name}")
    
    print("\n" + "="*60)
    print("所有图表已生成并保存")
    print(f"查看图表: ls {output_dir}")
    print("="*60)
    
    # 显示图表
    print("\n显示图表窗口（关闭窗口继续）...")
    plt.show()


if __name__ == "__main__":
    main()

