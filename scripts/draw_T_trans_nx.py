import numpy as np
import os
import yaml
import cv2

# import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# T_b_c0
body_T_cam0 = np.array([[0.04370391, 0.07758816, 0.99602713, 0.00464338],
                        [-0.99880591, 0.02518219, 0.04186421, 0.00371389],
                        [-0.02183398, -0.99666741, 0.07859607, 0.00137396],
                        [0.0, 0.0, 0.0, 1.0]])

body_T_cam1 = np.array([[0.04311599, 0.07642786, 0.99614246, 0.00672846],
                        [-0.99883441, 0.02495301, 0.04131802, -0.04635738],
                        [-0.02169891, -0.99676283, 0.07741465, 0.00059569],
                        [0.0, 0.0, 0.0, 1.0]])

lidar_T_body = np.array([[0.9659258, 0.0000000, -0.258819, 0.0027],
                         [0.0000000, 1.0000000, 0.0000000, 0.0029],
                         [0.258819, 0.0000000, 0.9659258, -0.083410],
                         [0.0, 0.0, 0.0, 1.0]])

# lidar_T_cam0 = np.array([[-0.301027, 0.0646489, 0.951422, 0.932394],
#                          [-0.9536, -0.0146204, -0.300722, -0.0286036],
#                          [-0.00553123, -0.997801, 0.0660503, 0.518685],
#                          [0, 0, 0, 1]])

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
    # pt_sensor = pt_o * T_o_sensor
    print("origin_transformed: ", origin_transformed)
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

# normalize R


def normalize_R(T):
        # 假设R是你的旋转矩阵
        R = T[:3, :3]  # 提取旋转矩阵，假设你的变换矩阵是4x4的
        # 计算范数
        # 计算每一列的范数
        norms = np.linalg.norm(R, axis=0)
        # 防止除以零
        norms[norms == 0] = 1
        # 归一化每一列
        R_normalized = R / norms
        # 确保正交性
        R_ortho = np.dot(R_normalized.T, R_normalized)
        I = np.eye(3)  # 单位矩阵
        if not np.allclose(R_ortho, I):
                raise ValueError("归一化后的矩阵不是正交的，请检查输入矩阵是否正确。")
        T[:3, :3] = R_normalized
        return T

body_T_cam0 = normalize_R(body_T_cam0)
body_T_cam1 = normalize_R(body_T_cam1)
print("body_T_cam0: ", body_T_cam0)
print("body_T_cam1: ", body_T_cam1)
# 计算T
body_T_lidar  = np.linalg.inv(lidar_T_body)
# cam0_T_lidar_calc = (body_T_cam0 @ body_T_lidar).reshape(4, 4)
# body_T_cam0 write to yaml
new_config = cv2.FileStorage("./vins.yaml", cv2.FILE_STORAGE_WRITE)
# new_config.write("body_T_cam0", body_T_cam0)
# new_config.write("body_T_cam1", body_T_cam1)

new_config.write("body_T_cam0", body_T_cam0)
new_config.write("body_T_cam1", body_T_cam1)

# 绘制cam0和cam1坐标系
draw_frame(np.eye(4), ax, 'Global')
draw_frame(body_T_cam0, ax, 'cam0')
draw_frame(body_T_cam1, ax, 'cam1')
# draw_frame(body_T_cam0_calc, ax, 'body_T_cam0_calc')
# draw_frame(body_T_cam0, ax, 'body_T_cam0')
# draw_frame(body_T_cam1, ax, 'body_T_cam1')
# draw_frame(body_T_cam12, ax, 'body_T_cam12')


# 设置图形属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Coordinate Systems')
plt.show()
