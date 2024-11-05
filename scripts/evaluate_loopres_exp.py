datapath= '/home/heron/Desktop/loop_output'
v1_dld_rs_ref11_reloT = datapath + '/v1_dld_rs_ref11_reloT.txt'
v2_dld_rs_ref11_reloT = datapath + '/v2_dld_rs_ref11_reloT.txt'
v3_dld_rs_ref11_reloT = datapath + '/v3_dld_rs_ref11_reloT.txt'
v4_dld_rs_ref11_reloT = datapath + '/v4_dld_rs_ref11_reloT.txt'
v0_nx_rs_ref12_reloT = datapath + '/v0_nx_rs_ref12_reloT.txt'
dld_rs_ref11_mappT = datapath + '/dld_omni_ref11_mappT.txt'
v1_r11_vinsreloT = datapath + '/v1_r11_vinsreloT.txt'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from evaluate_smooth_traj import plot_trajectory_and_deltas

# 读取数据
loopPdFrame = pd.read_csv(v1_dld_rs_ref11_reloT, sep=" ", header=None)
# loopPath = pd.read_csv(v1_nx_rs_ryjy3_loop, sep=" ", header=None)
# vioPath = pd.read_csv(v1_nx_rs_ryjy3_vio, sep=" ", header=None)

loopPdFrame_mapp = pd.read_csv(dld_rs_ref11_mappT, sep=" ", header=None)
# 提取前 8列作为lio frame并保存
ref_lioFrame = loopPdFrame_mapp[[0, 1, 2, 3, 4, 5, 6, 7]]


# 处理回环结果
# 创建新的DataFrame，每个回环结果一个
loopPdFrame_0 = loopPdFrame[loopPdFrame[1] == 0]
loopPdFrame_1 = loopPdFrame[loopPdFrame[1] == 1]
loopPdFrame_2 = loopPdFrame[loopPdFrame[1] == 2]
print("all loop: ", loopPdFrame.shape[0])
print("right loop: ", loopPdFrame_1.shape[0])
print("false loop: ", loopPdFrame_2.shape[0])
print("pnp pass rate: ", loopPdFrame_1.shape[0]/(loopPdFrame_1.shape[0]+loopPdFrame_2.shape[0]))
# 绘制图形
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
# 轨迹
# 创建一个颜色映射
cmap = plt.get_cmap('cool')
# 创建一个归一化对象
norm = plt.Normalize(loopPdFrame[0].min(), loopPdFrame[0].max())
# 绘制轨迹
# plt.scatter(loopPdFrame_1[16], loopPdFrame_1[15], c=loopPdFrame_1[0], cmap=cmap, norm=norm, linestyle='-',linewidths=2.0, marker='.',label='pnp')
# plt.plot(loopPdFrame[3], loopPdFrame[2],'-.', color ='royalblue',label='lio', linewidth=1.0)

# 绘制3d轨迹
ax.scatter3D(loopPdFrame_1[16], loopPdFrame_1[17], loopPdFrame_1[18], c=loopPdFrame_1[0], cmap=cmap, norm=norm, linestyle='-',linewidths=2.0, marker='.',label='pnp')
ax.plot3D(loopPdFrame[2], loopPdFrame[3], loopPdFrame[4],'-.', color ='royalblue',label='lio', linewidth=1.0)
ax.plot3D(loopPdFrame[23], loopPdFrame[24], loopPdFrame[25],'--', color ='orange',label='loop', linewidth=1.0)
ax.plot3D(loopPdFrame[9], loopPdFrame[10], loopPdFrame[11],'--', color ='green',label='vio', linewidth=1.0)
ax.plot3D(ref_lioFrame[1], ref_lioFrame[2], ref_lioFrame[3],'-.', color ='black',label='ref', linewidth=1.0)

# ax.plot3D(loopPath[2], loopPath[1], loopPath[3],'-.', color ='orange',label='loop', linewidth=1.0)
# ax.plot3D(vioPath[2], vioPath[1], vioPath[3],'-.', color ='green',label='vio', linewidth=1.0)

# plt.plot(loopPdFrame_1[16], loopPdFrame_1[15],'.', color ='orange',label='pnp')

# 添加颜色条
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical')
cbar.set_label('Time')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Position')
ax.legend()

axes = fig.gca()
axes.set_xlim(- 20,130)
axes.set_ylim(-70,40)
axes.set_zlim(-1,5)


#统计Recall和Precision
PR_list = []

limit_radius=1.0
TP = 0
drifts_list = {0.5: 0, 1.0: 0, 1.5: 0, 2.0: 0, 2.5: 0, 3.0: 0, 3.5: 0, 4.0: 0, 4.5: 0, 5.0: 0, 5.5: 0, 6.0: 0, 6.5: 0, 7.0: 0, 7.5: 0, 8.0: 0, 8.5: 0, 9.0: 0, 9.5: 0, 10.0: 0}

groundtruth_num = 0
#获得loopPdFrame_1的每一行
for i in range(loopPdFrame_1.shape[0]):
    loopResOutput = loopPdFrame_1.iloc[i,1]
    #获得lio位姿
    lio_t = np.array([loopPdFrame_1.iloc[i,2],loopPdFrame_1.iloc[i,3],loopPdFrame_1.iloc[i,4]])
    lio_q = np.array([loopPdFrame_1.iloc[i,5],loopPdFrame_1.iloc[i,6],loopPdFrame_1.iloc[i,7],loopPdFrame_1.iloc[i,8]])
    # 获得PnP的位姿
    pnp_t = np.array([loopPdFrame_1.iloc[i,16],loopPdFrame_1.iloc[i,17],loopPdFrame_1.iloc[i,18]])
    pnp_q = np.array([loopPdFrame_1.iloc[i,19],loopPdFrame_1.iloc[i,20],loopPdFrame.iloc[i,21],loopPdFrame.iloc[i,22]])
    #计算lio中每一个点到loopPdFrame_1的距离
    vec_t= np.sqrt((lio_t-pnp_t)**2)
    distance = np.linalg.norm(vec_t)
    yaw = 2*np.arccos(np.abs(np.dot(lio_q,pnp_q)))
    
    if(loopResOutput == 1): #算法认为是对的
        Predition = 1       #Predition=1影响Precision
    else:
        continue
        
    #如果有一个点的距离小于limit_radius，则认为这个点在loopPdFrame_1的范围内
    if (distance)<=limit_radius:
        groundtruth = 1            #groundtruth=1影响Recall
        groundtruth_num = groundtruth_num + 1
    else:
        groundtruth = 0
        
        # 计算TP
    if groundtruth == 1 and Predition == 1:
        TP = TP +1
        
    for limit in sorted(drifts_list.keys(), reverse=False):
        if distance < limit:
            drifts_list[limit] += 1
            break
    
    
        
Prediction_num = loopPdFrame_1.shape[0]
Total_num = loopPdFrame.shape[0]
Recall = Prediction_num / (Total_num)
Precision = TP / (Prediction_num)
print("Recall: ", Recall)
print("Precision: ", Precision)
#还有一些更大的被过滤掉了，所以数量小于right loop num
print("drifts sum: ", np.sum(list(drifts_list.values())))

#绘制直方图
draw_list = [500,500,10,10,10,10,10]
plt.figure(figsize=(10, 10))
plt.bar(drifts_list.keys(), drifts_list.values(), width=0.5)
plt.xlabel('Drift Distance')
plt.ylabel('Number')
plt.xticks(np.arange(0, 10, 0.5))
plt.title('Distance Histogram')


#计算和lio的APE
x_diff = loopPdFrame[2] - loopPdFrame[23]
y_diff = loopPdFrame[3] - loopPdFrame[24]
z_diff = loopPdFrame[4] - loopPdFrame[25]
t_diff_norm = 0.25*np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)/3
APE = np.mean(t_diff_norm)
# plot
plt.figure(figsize=(10, 10))
plt.plot(t_diff_norm, label='APE')
#plot APE as a line
plt.axhline(y=APE, color='r', linestyle='--', label='mean APE')
plt.xlabel('Count')
plt.ylabel('APE')
plt.title('APE')
plt.legend()



plt.show()

