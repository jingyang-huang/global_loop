import numpy as np
import os
import yaml
import cv2

# import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义变换矩阵
cam0_T_body = np.array([[0.0, 0.0, 1.0, 0.05],
                        [-1.0, 0.0, 0.0, 0.025],
                        [0.0, -1.0, 0.0, -0.020],
                        [0.0, 0.0, 0.0, 1]])

cam1_T_cam0 = np.array([[1.0, 0.0, 0.0, 0.049916],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1]])

cam1_T_cam0_worldRef = np.array([[1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, -0.049916],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1]])


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


# 计算T
body_T_cam0 = np.linalg.inv(cam0_T_body).reshape(4, 4)
cam0_T_cam1 = np.linalg.inv(cam1_T_cam0).reshape(4, 4)
body_T_cam1 = body_T_cam0.dot(cam0_T_cam1).reshape(
    4, 4)  # body_T_cam0 一定不能弄错了 body_T_cam0 * cam0_T_cam1
print("body_T_cam0: \n", body_T_cam0)
print("body_T_cam1: \n", body_T_cam1)

# cam0_T_body write to yaml
new_config = cv2.FileStorage("./vins.yaml", cv2.FILE_STORAGE_WRITE)
new_config.write("body_T_cam0", body_T_cam0)
new_config.write("body_T_cam1", body_T_cam1)

cam1_T_body = (cam0_T_body @ cam1_T_cam0).reshape(4,
                                                  4)  # 坐标系变了，是在cam0的坐标系下，所以右乘
cam1_T_body2 = cam1_T_cam0_worldRef @ cam0_T_body  # 这个是对的，相对的位移是在世界坐标系下，所以左乘
new_config.write("body_T_cam0", cam0_T_body)
new_config.write("body_T_cam1", cam1_T_body)

# 绘制cam0和cam1坐标系
draw_frame(np.eye(4), ax, 'Global')
draw_frame(body_T_cam0, ax, 'cam0')
draw_frame(body_T_cam1, ax, 'cam1')
draw_frame(cam0_T_body, ax, 'cam0_T_body')
draw_frame(cam1_T_body, ax, 'cam1_T_body')
draw_frame(cam1_T_body2, ax, 'cam1_T_body2')


# 设置图形属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Coordinate Systems')
plt.show()
