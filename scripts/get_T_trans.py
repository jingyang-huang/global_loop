import numpy as np
import os
import yaml
import cv2

# Load the matrix
# vins_config = "/home/heron/SJTU/Codes/VIO/VINS_ws/src/VINS-Fusion/config/nx_2/realsense_stereo_imu_config.yaml"

# config = cv2.FileStorage(vins_config, cv2.FILE_STORAGE_READ)
# body_T_cam0
# 
            #   [0.0,0.0,-1.0,0.0, 
            #    -1.0,0.0,0.0,0.2, 
            #     0.0,1.0,0.0,0.6, 
            #     0.0,0.0,0.0,1]
#body_T_cam1 
            #   [0.0,0.0,-1.0,0.0, 
            #   -1.0,0.0,0.0,0.0, 
            #     0.0,1.0,0.0,0.30, 
            #     0.0,0.0,0.0,1]

# infra1_to_fcu:    [ 0,          0,          1,     0.050,
#                    -1,          0,          0,     0.025,
#                     0,         -1,          0,    -0.020,
#                     0,          0,          0,         1 ]



# infra2_to_infra1: [ 1,          0,          0,  0.049916,
#                     0,          1,          0,         0,
#                     0,          0,          1,         0,
#                     0,          0,          0,         1 ]



cam0_T_body = np.array([[0.0,0.0,1.0,0.05],
                        [-1.0,0.0,0.0,0.025],
                        [0.0,-1.0,0.0,-0.020],
                        [0.0,0.0,0.0,1]])

cam1_T_cam0 = np.array([[1.0,0.0,0.0,0.049916],
                        [0.0,1.0,0.0,0.0],
                        [0.0,0.0,1.0,0.0],
                        [0.0,0.0,0.0,1]])

# body_T_cam1 = np.array([[0.0,0.0,-1.0,0.0],
#                         [-1.0,0.0,0.0,0.0],
#                         [0.0,1.0,0.0,0.30],
#                         [0.0,0.0,0.0,1]])



#逗号分隔的输出
print("body_T_cam0: \n", cam0_T_body)
print("cam0_T_cam1: \n", cam1_T_cam0)
# To get msckf T
# 求逆后输出
body_T_cam0 = np.linalg.inv(cam0_T_body).reshape(4, 4)
cam0_T_cam1 = np.linalg.inv(cam1_T_cam0).reshape(4, 4)
body_T_cam1 = body_T_cam0.dot(cam0_T_cam1).reshape(4, 4) #body_T_cam0 一定不能弄错了 body_T_cam0 * cam0_T_cam1
print("body_T_cam0: \n", body_T_cam0)
print("body_T_cam1: \n", body_T_cam1)

#cam0_T_body write to yaml
new_config = cv2.FileStorage("./vins.yaml", cv2.FILE_STORAGE_WRITE)
new_config.write("body_T_cam0", body_T_cam0)
new_config.write("body_T_cam1", body_T_cam1)

cam1_T_body = cam1_T_cam0.dot(cam0_T_body).reshape(4, 4) #body_T_cam0 一定不能弄错了
new_config.write("cam0_T_body", cam0_T_body)
new_config.write("cam1_T_body", cam1_T_body)


# T_imu_cam1 = T_cam0_cam1 * T_imu_cam0;  T_cam0_cam1 = T_imu_cam1 * T_imu_cam0_inv
# T_cam0_cam1 = cam1_T_body.dot(body_T_cam0).reshape(4, 4) #body_T_cam0 一定不能弄错了
# new_config.write("T_cn_cnm1", T_cam0_cam1)