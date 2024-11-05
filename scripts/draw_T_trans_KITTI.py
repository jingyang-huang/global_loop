import numpy as np
from numpy import *
import os
import yaml
import cv2

# import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


IMU_2_V =np.mat([[9.999976e-01, 7.553071e-04, -2.035826e-03,-8.086759e-01],
                   [-7.854027e-04, 9.998898e-01, -1.482298e-02, 3.195559e-01],
                   [2.024406e-03, 1.482454e-02, 9.998881e-01,-7.997231e-01],
                   [0,0,0,1]])
 
V_2_C = np.mat([[7.967514e-03, -9.999679e-01, -8.462264e-04, -1.377769e-02],
                    [-2.771053e-03, 8.241710e-04, -9.999958e-01, -5.542117e-02],
                    [9.999644e-01, 7.969825e-03, -2.764397e-03, -2.918589e-01],
                    [0,0,0,1]])
 
  
 
C_2_C1 = np.mat([[9.993440e-01, 1.814887e-02, -3.134011e-02, -5.370000e-01],
                    [-1.842595e-02, 9.997935e-01, -8.575221e-03, 5.964270e-03],
                    [3.117801e-02, 9.147067e-03, 9.994720e-01, -1.274584e-02],
                    [0,0,0,1]])
 
 
C_2_C1_ = np.mat([[1, 0, 0, 0.537150653267924],
                   [0, 1, 0, 0,],
                   [0, 0, 1, 0,],
                   [0, 0, 0, 1]])

IMU_2_C0 = V_2_C*IMU_2_V
IMU_2_C0 = IMU_2_C0.I
 
#####################################
 
IMU_2_C1 = C_2_C1_.I*V_2_C*IMU_2_V
IMU_2_C1 = IMU_2_C1.I

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制坐标系的函数


def draw_frame(T, ax, label):
    # 原点
    origin = np.array([0, 0, 0, 1])
    # 基本向量
    x_axis = np.array([0.1, 0, 0, 1])
    y_axis = np.array([0, 0.1, 0, 1])
    z_axis = np.array([0, 0, 0.1, 1])

    # 变换到全局坐标系
    origin_transformed = T @ origin
    x_axis_transformed = T @ x_axis
    y_axis_transformed = T @ y_axis
    z_axis_transformed = T @ z_axis

    # 绘制坐标轴
    ax.plot([origin_transformed[0], x_axis_transformed[0]], [origin_transformed[1],
            x_axis_transformed[1]], [origin_transformed[2], x_axis_transformed[2]], 'r')
    ax.plot([origin_transformed[0], y_axis_transformed[0]], [origin_transformed[1],
            y_axis_transformed[1]], [origin_transformed[2], y_axis_transformed[2]], 'g')
    ax.plot([origin_transformed[0], z_axis_transformed[0]], [origin_transformed[1],
            z_axis_transformed[1]], [origin_transformed[2], z_axis_transformed[2]], 'b')
    ax.text(origin_transformed[0], origin_transformed[1],
            origin_transformed[2], label)


print("Tb_2_c0:")
print(mat(IMU_2_C0))
 
print("Tb_2_c1:")
print(mat(IMU_2_C1))

# cam0_T_body write to yaml
new_config = cv2.FileStorage("./kitti.yaml", cv2.FILE_STORAGE_WRITE)
new_config.write("body_T_cam0", mat(IMU_2_C0))
new_config.write("body_T_cam1", mat(IMU_2_C1))
new_config.write("body_T_lidar", mat(IMU_2_V))

# 绘制cam0和cam1坐标系
draw_frame(np.eye(4), ax, 'Global')
draw_frame(mat(IMU_2_C0), ax, 'cam0')
draw_frame(mat(IMU_2_C1), ax, 'cam1')
draw_frame(mat(IMU_2_V), ax, 'body_T_lidar')


# 设置图形属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Coordinate Systems')
plt.show()
