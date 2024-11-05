datapath= '/home/heron/Desktop/loop_output/loop_res'
ref0_gt=datapath + '/ref0_gt.txt'
ref1_gt=datapath + '/ref1_gt.txt'
omni_ref0_gt=datapath + '/omni_ref0_gt.txt'


v0_r0_gt=datapath + '/v0_r0_gt.txt'
v1_r0_gt=datapath + '/v1_r0_gt.txt'
v2_r0_gt=datapath + '/v2_r0_gt.txt'
v3_r0_gt=datapath + '/v3_r0_gt.txt'
v0_r1_gt=datapath + '/v0_r1_gt.txt'
v0_omni_r0_gt=datapath + '/v0_omni_r0_gt.txt'

v0_r0_loop=datapath + '/v0_r0_loop.txt'
v1_r0_loop=datapath + '/v1_r0_loop.txt'
v2_r0_loop=datapath + '/v2_r0_loop.txt'
v3_r0_loop=datapath + '/v3_r0_loop.txt'
v0_r1_loop=datapath + '/v0_r1_loop.txt'
v0_omni_r0_loop=datapath + '/v0_omni_r0_loop.txt'
v0_dld_rs_refyjy1_loop=datapath + '/v0_dld_rs_refyjy1_loop.txt'

v0_r0_vio=datapath + '/v0_r0_vio.txt'
v1_r0_vio=datapath + '/v1_r0_vio.txt'
v2_r0_vio=datapath + '/v2_r0_vio.txt'
v3_r0_vio=datapath + '/v3_r0_vio.txt'
v0_r1_vio=datapath + '/v0_r1_vio.txt'
v0_omni_r0_vio=datapath + '/v0_omni_r0_vio.txt'
v0_dld_rs_refyjy1_vio=datapath + '/v0_dld_rs_refyjy1_vio.txt'

v3_r0_loop_abl=datapath + '/v3_r0_loop_abl.txt'
v0_r1_loop_abl=datapath + '/v0_r1_loop_abl.txt'

v0_r0_reloT=datapath + '/v0_r0_reloT.txt'
v1_r0_reloT=datapath + '/v1_r0_reloT.txt'
v2_r0_reloT=datapath + '/v2_r0_reloT.txt'
v3_r0_reloT=datapath + '/v3_r0_reloT.txt'
v0_r1_reloT=datapath + '/v0_r1_reloT.txt'
v0_omni_r0_reloT=datapath + '/v0_omni_r0_reloT.txt'
v0_dld_rs_refyjy1_reloT = datapath + '/v0_dld_rs_refyjy1_reloT.txt'

v3_r0_reloT_abl=datapath + '/v3_r0_reloT_abl.txt'
v0_r1_reloT_abl=datapath + '/v0_r1_reloT_abl.txt'

v3_r0_reloT_comp=datapath + '/v3_r0_reloT_comp.txt'
v0_r1_reloT_comp=datapath + '/v0_r1_reloT_comp.txt'


import pandas as pd
import matplotlib.pyplot as plt

# fout_reloTime << cur_kf->time_stamp << " ";
#     fout_reloTime.precision(9);
#     fout_reloTime << loop_res << " "
#     << lio_t(0) <<  " " << lio_t(1) <<  " " << lio_t(2) <<  " "
#     << lio_q.x() << " " << lio_q.y() << " " << lio_q.z() << " " << lio_q.w() << " "
#     << vio_t_cur(0) <<  " " << vio_t_cur(1) <<  " " << vio_t_cur(2) <<  " "
#     << vio_t_cur(0) <<  " " << vio_t_cur(1) <<  " " << vio_t_cur(2) <<  " "
#     << PnP_T(0) << " " << PnP_T(1) << " " << PnP_T(2) << " "
#     << PnP_q.x() << " " << PnP_q.y() << " " << PnP_q.z() << " " << PnP_q.w() << " "
#     << relative_t(0) <<  " " << relative_t(1) <<  " " << relative_t(2) <<  " " << relative_yaw <<  " "
#     << final_matched_num << " " <<  cur_kf->index << " "  
#     << t_detectLoop << " "  << t_match << " "  << t_PnPRANSAC << " " << endl;
#     fout_reloTime.close();

# 读取数据
ref = pd.read_csv(omni_ref0_gt, sep=" ", header=None)
loop = pd.read_csv(v0_omni_r0_loop, sep=" ", header=None)
vio = pd.read_csv(v0_omni_r0_vio, sep=" ", header=None)
lio = pd.read_csv(v0_omni_r0_gt, sep=" ", header=None)
# abl = pd.read_csv(v0_r0_loop_abl, sep=" ", header=None)
# loopRes = pd.read_csv(v3_r0_reloT, sep=" ", header=None)

# # 处理回环结果
# # 创建新的DataFrame，每个回环结果一个
# loopRes_0 = loopRes[loopRes[1] == 0][[2, 3, 4]]
# loopRes_1 = loopRes[loopRes[1] == 1][[2, 3, 4]]
# loopRes_2 = loopRes[loopRes[1] == 2][[2, 3, 4]]
# print("right loop: ", loopRes_1.shape)
# print("false loop: ", loopRes_2.shape)

# 绘制图形
plt.figure(figsize=(10, 10))
# 轨迹
plt.plot(ref[2], ref[1], '--', color ='black',label='ref')
plt.plot(loop[2], loop[1],'-', color ='orange',label='loop fusion')
plt.plot(vio[2], vio[1],'-',color ='green', label='vio')
plt.plot(lio[2], lio[1],'-.', color ='royalblue',label='lio')
# plt.plot(abl[2], abl[1],'-', color ='purple',label='ablation(vision mapping)')
# 发生回环的位置
# plt.plot(loopRes_1[3], loopRes_1[2], 'g*', label='right loop')  # 星型标记
# plt.plot(loopRes_2[3], loopRes_2[2], 'rx', label='false loop')  # x型标记


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Position')
plt.legend()


# 设置x和y轴的刻度范围
axes = plt.gca()
axes.set_ylim(-75,75)
axes.set_xlim(-150,10)
plt.show()

