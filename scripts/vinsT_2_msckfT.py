import numpy as np
import os
import yaml
import cv2

# Load the yaml config file
vins_config = "/home/heron/SJTU/Codes/VIO/VINS_ws/src/VINS-Fusion/config/nx_2/realsense_stereo_imu_config.yaml"

config = cv2.FileStorage(vins_config, cv2.FILE_STORAGE_READ)
body_T_cam0 = config.getNode("body_T_cam0").mat()
body_T_cam1 = config.getNode("body_T_cam1").mat()
#逗号分隔的输出
print("body_T_cam0: \n", body_T_cam0)
print("body_T_cam1: \n", body_T_cam1)
# To get msckf T
# 求逆后输出
cam0_T_body = np.linalg.inv(body_T_cam0).reshape(4, 4)
cam1_T_body = np.linalg.inv(body_T_cam1).reshape(4, 4)
#cam0_T_body write to yaml
new_config = cv2.FileStorage("./msckf.yaml", cv2.FILE_STORAGE_WRITE)
new_config.write("T_cam_imu", cam0_T_body)
new_config.write("T_cam_imu", cam1_T_body)


# T_imu_cam1 = T_cam0_cam1 * T_imu_cam0;  T_cam0_cam1 = T_imu_cam1 * T_imu_cam0_inv
T_cam0_cam1 = cam1_T_body.dot(body_T_cam0).reshape(4, 4) #body_T_cam0 一定不能弄错了
new_config.write("T_cn_cnm1", T_cam0_cam1)