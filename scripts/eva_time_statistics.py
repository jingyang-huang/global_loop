import matplotlib.pyplot as plt
import pandas as pd
datapath = '/home/heron/Desktop/loop_output/loop_res'
v1_dld_rs_ref11_reloT = datapath + '/v1_dld_rs_ref11_reloT.txt'
v0_dld_rs_ref11_reloT = datapath + '/v0_dld_rs_ref11_reloT.txt'
v3_dld_rs_ref11_reloT = datapath + '/v3_dld_rs_ref11_reloT.txt'
v4_dld_rs_ref11_reloT = datapath + '/v4_dld_rs_ref11_reloT.txt'
v0_nx_rs_ref13_reloT = datapath + '/v0_nx_rs_ref13_reloT.txt'
dld_rs_ref11_mappT = datapath + '/dld_omni_ref13_mappT.txt'


# 读取数据
ref = pd.read_csv(dld_rs_ref11_mappT, sep=" ", header=None)
relo = pd.read_csv(v0_nx_rs_ref13_reloT, sep=" ", header=None)

ref = ref.iloc[1:]
relo = relo.iloc[1:260]

mean_ref_8 = ref[8].mean()
mean_ref_9 = ref[9].mean()
mean_ref_12 = ref[12].mean()
mean_ref_11 = ref[11].mean()
mean_ref_10 = ref[10].mean()

mean_relo_36 = relo[36].mean()
mean_relo_37 = relo[37].mean()
mean_relo_38 = relo[38].mean()

print("Superpoint: ", mean_ref_8)
print("projection: ", mean_ref_9 + mean_ref_12)
print("process: ", mean_ref_11)
print("add: ", mean_ref_10)

print("detect: ", mean_relo_36)
print("match: ", mean_relo_37)
print("pnp: ", mean_relo_38)
print("pgo: ", mean_ref_10)

# 创建图形和子图
fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_subplot(111)

ax1.plot(ref.index, ref[8], label='Time Superpoint Extraction')
ax1.plot(ref.index, ref[9], label='Time lidar projection')
ax1.plot(ref.index, ref[12], label='Time feature track')
ax1.plot(ref.index, ref[11], label='Time process feature and triangulation')
ax1.plot(ref.index, ref[10], label='Time add keyFrame')

ax1.set_xlabel('Index')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Mapping Time Statistics')
ax1.legend()

# 创建图形和子图
fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(111)
print(relo.columns)
ax2.plot(relo.index, relo[36], label='Time detect loop')
ax2.plot(relo.index, relo[37], label='Time brief match')
ax2.plot(relo.index, relo[38], label='Time pnpransac')

ax2.set_xlabel('Index')
ax2.set_ylabel('Time (ms)')
ax2.set_title('Relocalization Time Statistics')
ax2.legend()

plt.show()
