datapath= '/home/heron/Desktop/loop_output/loop_res'
ref0_gt=datapath + '/ref0_gt.txt'
ref1_gt=datapath + '/ref1_gt.txt'
v0_r0_gt=datapath + '/v0_r0_gt.txt'
v1_r0_gt=datapath + '/v1_r0_gt.txt'
v2_r0_gt=datapath + '/v2_r0_gt.txt'
v3_r0_gt=datapath + '/v3_r0_gt.txt'
v0_r1_gt=datapath + '/v0_r1_gt.txt'

v0_r0_loop=datapath + '/v0_r0_loop.txt'
v1_r0_loop=datapath + '/v1_r0_loop.txt'
v2_r0_loop=datapath + '/v2_r0_loop.txt'
v3_r0_loop=datapath + '/v3_r0_loop.txt'
v0_r1_loop=datapath + '/v0_r1_loop.txt'

viov0_r0=datapath + '/v0_r0_vio.txt'
viov1_r0=datapath + '/v1_r0_vio.txt'
viov2_r0=datapath + '/v2_r0_vio.txt'
viov3_r0=datapath + '/v3_r0_vio.txt'
viov0_r1=datapath + '/v0_r1_vio.txt'

v3_r0_loop_abl=datapath + '/v3_r0_loop_abl.txt'

v0_r0_reloT=datapath + '/v0_r0_reloT.txt'
v1_r0_reloT=datapath + '/v1_r0_reloT.txt'
v2_r0_reloT=datapath + '/v2_r0_reloT.txt'
v3_r0_reloT=datapath + '/v3_r0_reloT.txt'
v0_r1_reloT=datapath + '/v0_r1_reloT.txt'

v4_rot_r0_loop=datapath + '/v4_rot_r0_loop.txt'
viov4_rot_r0=datapath + '/v4_rot_r0_vio.txt'
v4_rot_r0_reloT=datapath + '/v4_rot_r0_reloT.txt'


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 读取数据
loopPdFrame = pd.read_csv(v4_rot_r0_reloT, sep=" ", header=None)

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
plt.figure(figsize=(10, 10))
# 轨迹
# 创建一个颜色映射
cmap = plt.get_cmap('viridis')
# 创建一个归一化对象
norm = plt.Normalize(loopPdFrame[0].min(), loopPdFrame[0].max())
# 绘制轨迹
plt.plot(loopPdFrame[3], loopPdFrame[2],'-.', color ='royalblue',label='lio')
# plt.plot(loopPdFrame_1[16], loopPdFrame_1[15],'.', color ='orange',label='pnp')
plt.scatter(loopPdFrame_1[16], loopPdFrame_1[15], c=loopPdFrame_1[0], cmap=cmap, norm=norm, linestyle='-',linewidths=1.0, marker='.',label='pnp')

# 添加颜色条
cbar = plt.colorbar()
cbar.set_label('Time')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Position')
plt.legend()

axes = plt.gca()
axes.set_xlim(-150,10)
axes.set_ylim(-75,75)

#统计Recall和Precision
PR_list = []

limit_radius=1.0
TP = 0
drifts_list = {0.5: 0, 1.0: 0, 1.5: 0, 2.0: 0, 2.5: 0, 3.0: 0, 3.5: 0, 4.0: 0, 4.5: 0, 5.0: 0, 5.5: 0, 6.0: 0, 6.5: 0, 7.0: 0, 7.5: 0, 8.0: 0, 8.5: 0, 9.0: 0, 9.5: 0, 10.0: 0}


groundtruth_num = 0
#获得loopPdFrame_1的每一行
for i in range(loopPdFrame.shape[0]):
    loopResOutput = loopPdFrame.iloc[i,1]
    #获得lio位姿
    lio_t = np.array([loopPdFrame.iloc[i,2],loopPdFrame.iloc[i,3],loopPdFrame.iloc[i,4]])
    lio_q = np.array([loopPdFrame.iloc[i,5],loopPdFrame.iloc[i,6],loopPdFrame.iloc[i,7],loopPdFrame.iloc[i,8]])
    # 获得PnP的位姿
    pnp_t = np.array([loopPdFrame.iloc[i,15],loopPdFrame.iloc[i,16],loopPdFrame.iloc[i,17]])
    pnp_q = np.array([loopPdFrame.iloc[i,18],loopPdFrame.iloc[i,19],loopPdFrame.iloc[i,20],loopPdFrame.iloc[i,21]])
    #计算lio中每一个点到loopPdFrame_1的距离
    vec_t= np.sqrt((lio_t-pnp_t)**2)
    distance = np.linalg.norm(vec_t)
    yaw = 2*np.arccos(np.abs(np.dot(lio_q,pnp_q)))
    
    if(loopResOutput == 1): #算法认为是对的
        Predition = 1       #Predition=1影响Precision
    else:
        continue
        
    #如果有一个点的距离小于limit_radius，则认为这个点在loopPdFrame_1的范围内
    if (distance)<limit_radius:
        groundtruth = 1            #groundtruth=1影响Recall
        groundtruth_num = groundtruth_num + 1
    else:
        groundtruth = 0
    
    for limit in sorted(drifts_list.keys(), reverse=False):
        if distance < limit:
            drifts_list[limit] += 1
            break
    
    
    # 计算TP
    if groundtruth == 1 and Predition == 1:
        TP = TP +1
        
Prediction_num = loopPdFrame_1.shape[0]
Total_num = loopPdFrame.shape[0]
Recall = TP / (Total_num)
Precision = TP / (Prediction_num)
print("Recall: ", Recall)
print("Precision: ", Precision)


#绘制直方图
draw_list = [500,500,10,10,10,10,10]
plt.figure(figsize=(10, 10))
plt.bar(drifts_list.keys(), drifts_list.values(), width=0.5)
plt.xlabel('Drift Distance')
plt.ylabel('Number')
plt.xticks(np.arange(0, 10, 0.5))
plt.title('Distance Histogram')


# 绘制图形
plt.figure(figsize=(10, 10))
# 轨迹
plt.plot(loopPdFrame[3], loopPdFrame[2],'-.', color ='royalblue',label='lio')
# 发生回环的位置
plt.plot(loopPdFrame_1[3], loopPdFrame_1[2], 'g*', label='right loop')  # 星型标记
plt.plot(loopPdFrame_2[3], loopPdFrame_2[2], 'rx', label='false loop')  # x型标记


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Position')
plt.legend()


# 设置x和y轴的刻度范围
axes = plt.gca()
axes.set_ylim(-75,75)
axes.set_xlim(-150,10)

plt.figure(figsize=(10, 10))
ref = pd.read_csv(ref0_gt, sep=" ", header=None)
loop = pd.read_csv(v4_rot_r0_loop, sep=" ", header=None)
vio = pd.read_csv(viov4_rot_r0, sep=" ", header=None)
plt.plot(ref[2], ref[1], '--', color ='black',label='ref')
plt.plot(loop[2], loop[1],'-', color ='orange',label='loop fusion')
plt.plot(vio[2], vio[1],'-',color ='green', label='vio')
plt.plot(loopPdFrame[3], loopPdFrame[2],'-.', color ='royalblue',label='lio')
plt.legend()
axes = plt.gca()
axes.set_ylim(-75,75)
axes.set_xlim(-150,10)

#绘制Matched Yaw Over Time
plt.figure(figsize=(10, 10))
mean_yaw = np.mean(loopPdFrame_1[25])
plt.plot(loopPdFrame_1[0],loopPdFrame_1[25], '-',label='yaw')
plt.axhline(y=mean_yaw, color='r', linestyle='--',label='mean yaw')
plt.xlabel('Time')
plt.ylabel('Yaw')
plt.title('Matched Yaw Over Time')
plt.legend()


plt.show()

