import numpy as np
import os
import yaml
import cv2

# import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# T_b_c0
body_T_cam0 = np.array([[0., 0., 1., 7.1000000000000000e-02],
                        [-1., 0., 0., 0.0],
                        [0., -1., 0., -1.5000000000000000e-02],
                        [0., 0., 0., 1.]])

body_T_cam1 = np.array([[0., 0., -1., -9.6000000000000000e-02],
                        [1., 0., 0., 0.],
                        [0., -1., 0., -2.2000000000000000e-02],
                        [0., 0., 0., 1.]])
body_T_cam2 = np.array([[1., 0., 0., 0.],
                        [0., 0., 1., 7.2000000000000000e-02],
                        [0., -1., 0., -1.7000000000000000e-02],
                        [0., 0., 0., 1.]])
body_T_cam3 = np.array([[-1., 0., 0., 0.],
                        [0., 0., -1., -7.2000000000000000e-02],
                        [0., -1., 0., -1.7000000000000000e-02],
                        [0., 0., 0., 1.]])

lidar_T_body = np.array([[0.9659258, 0.0000000, -0.258819, 0.00],
                         [0.0000000, 1.0000000, 0.0000000, 0.00],
                         [0.258819, 0.0000000, 0.9659258, -0.05],
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

# 以四元数形式打印


def print_trans_quat(T):
    trans = T[:3, 3]
    quat = np.zeros(4)
    quat[0] = np.sqrt(1 + T[0, 0] + T[1, 1] + T[2, 2]) / 2
    quat[1] = (T[2, 1] - T[1, 2]) / (4 * quat[0])
    quat[2] = (T[0, 2] - T[2, 0]) / (4 * quat[0])
    quat[3] = (T[1, 0] - T[0, 1]) / (4 * quat[0])
    print("trans", trans)
    print("quat", quat)
#     return quat

# 给定绕xyz的角度，返回旋转矩阵


def euler2rot(x, y, z):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    return T


def angle2rad(angle):
    return angle / 180 * np.pi


# 计算T
body_T_lidar = np.linalg.inv(lidar_T_body)
lidar_T_cam0_calc = (lidar_T_body @ body_T_cam0).reshape(4, 4)
cam0_T_lidar_calc = np.linalg.inv(lidar_T_cam0_calc)
# cam0_T_lidar_calc = (body_T_cam0 @ body_T_lidar).reshape(4, 4)

lidar_T_cam2_calc = (lidar_T_body @ body_T_cam2).reshape(4, 4)
cam2_T_lidar_calc = np.linalg.inv(lidar_T_cam2_calc)
lidar_T_cam3_calc = (lidar_T_body @ body_T_cam3).reshape(4, 4)
cam3_T_lidar_calc = np.linalg.inv(lidar_T_cam3_calc)
print("lidar_T_cam2_calc: \n", lidar_T_cam2_calc)
print("lidar_T_cam3_calc: \n", lidar_T_cam3_calc)
print("cam2_T_lidar_calc: \n", cam2_T_lidar_calc)
print("cam3_T_lidar_calc: \n", cam3_T_lidar_calc)
print_trans_quat(lidar_T_cam2_calc)
print_trans_quat(lidar_T_cam3_calc)


# 手动添加旋转
# rot2 = euler2rot(angle2rad(0), angle2rad(0), angle2rad(-5)) #正 顺时针 负 逆时针
# body_T_cam2 =  body_T_cam2 @rot2

# rot3 = euler2rot(angle2rad(0), angle2rad(0), angle2rad(10)) #正 顺时针 负 逆时针
# body_T_cam3 =  body_T_cam3 @rot3

# body_T_cam0 write to yaml
new_config = cv2.FileStorage("vins.yaml", cv2.FILE_STORAGE_WRITE)
new_config.write("body_T_cam2", body_T_cam2)
new_config.write("body_T_cam3", body_T_cam3)

new_config.write("lidar_T_cam2_calc", lidar_T_cam2_calc)
new_config.write("cam2_T_lidar_calc", cam2_T_lidar_calc)
new_config.write("lidar_T_cam3_calc", lidar_T_cam3_calc)
new_config.write("cam3_T_lidar_calc", cam3_T_lidar_calc)

# 绘制cam0和cam1坐标系
draw_frame(np.eye(4), ax, 'Global')
draw_frame(body_T_cam0, ax, 'cam0')
draw_frame(body_T_cam1, ax, 'cam1')
draw_frame(body_T_cam2, ax, 'cam2')
draw_frame(body_T_cam3, ax, 'cam3')
draw_frame(body_T_lidar, ax, 'body_T_lidar')
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
