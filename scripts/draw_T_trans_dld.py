import numpy as np
import os
import yaml
import cv2

# import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# T_b_c0
body_T_cam0 = np.array([[0.0, 0.0, 1.0, 0.093],
                        [-1.0, 0.0, 0.0, 0.025],
                        [0.0, -1.0, 0.0, 0.034],
                        [0.0, 0.0, 0.0, 1]])

body_T_cam1 = np.array([[0.0, 0.0, 1.0, 0.093],
                        [-1.0, 0.0, 0.0, -0.025],
                        [0.0, -1.0, 0.0, 0.034],
                        [0.0, 0.0, 0.0, 1]])
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
    print("origin_transformed: ", origin_transformed) # pt_sensor = pt_o * T_o_sensor
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
body_T_lidar  = np.linalg.inv(lidar_T_body)
lidar_T_cam0_calc = (lidar_T_body @ body_T_cam0).reshape(4, 4)
cam0_T_lidar_calc = np.linalg.inv(lidar_T_cam0_calc)
# cam0_T_lidar_calc = (body_T_cam0 @ body_T_lidar).reshape(4, 4)
print("lidar_T_cam0_calc: \n", lidar_T_cam0_calc)
# body_T_cam0 write to yaml
new_config = cv2.FileStorage("./vins.yaml", cv2.FILE_STORAGE_WRITE)
# new_config.write("body_T_cam0", body_T_cam0)
# new_config.write("body_T_cam1", body_T_cam1)

new_config.write("lidar_T_cam0_calc", lidar_T_cam0_calc)
new_config.write("cam0_T_lidar_calc", cam0_T_lidar_calc)

# 绘制cam0和cam1坐标系
draw_frame(np.eye(4), ax, 'Global')
draw_frame(body_T_cam0, ax, 'cam0')
draw_frame(body_T_cam1, ax, 'cam1')
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
