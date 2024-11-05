import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datapath = '/home/heron/Desktop/loop_output/exp'

v1_r11_gt = datapath + '/v1_dld_rs_ref11_gt.txt'
v1_r11_loop = datapath + '/v1_dld_rs_ref11_loop.txt'
v1_r11_vinsloop = datapath + '/v1_r11_vinsloop.txt'
v1_r11_vins = datapath + '/v1_r11_vins.txt'

# 读取数据
gt_data = pd.read_csv(v1_r11_gt, sep=" ", header=None)
loop_data = pd.read_csv(v1_r11_loop, sep=" ", header=None)
vinsloop_data = pd.read_csv(v1_r11_vinsloop, sep=" ", header=None)
vins_data = pd.read_csv(v1_r11_vins, sep=" ", header=None)

# 提取xy坐标
gt_xy = gt_data
loop_xy = loop_data
vinsloop_xy = vinsloop_data
vins_xy = vins_data

# 设置全局字体大小
plt.rcParams.update({'font.size': 30})

# 绘制轨迹图
plt.figure(figsize=(12, 10))
plt.plot(gt_xy[1], gt_xy[2], label='LIO', linestyle='-.', linewidth=2.0, color='blue')
plt.plot(loop_xy[1], loop_xy[2], label='Ours', linestyle='-', linewidth=2.0, color='purple')
plt.plot(vinsloop_xy[1], vinsloop_xy[2], label='VINS-Fusion+Map', linestyle='-', linewidth=2.0, color='green')
plt.plot(vins_xy[1], vins_xy[2], label='VINS-Fusion', linestyle='-', linewidth=2.0, color='red')

# 添加图例和标签
plt.legend()
plt.xlabel('X(m)')
plt.ylabel('Y(m)')
plt.title('Trajectory Comparison')
plt.grid(True)

# 显示图像
plt.show()
plt.savefig('/home/heron/Desktop/loop_output/exp/trajectory.png')