datapath= '/home/heron/Desktop/loop_output/exp/'
ref6_gt=datapath + '/ref6_gt.txt'
dld_usb_ref6_mappT =datapath + '/dld_omni_ref11_mappT.txt'
v2_dld_rs_usb_ref6_gt  =datapath + '/v2_r11_gt.txt'
v2_dld_rs_ref6_reloT=datapath + '/v2_dld_rs_ref11_reloT.txt'


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 读取数据
loopPdFrame = pd.read_csv(dld_usb_ref6_mappT, sep=" ", header=None)
# 提取前 8列作为lio frame并保存
lioFrame = loopPdFrame[[0, 1, 2, 3, 4, 5, 6, 7]]
lioFrame.to_csv(ref6_gt, sep=' ', header=False, index=False)  
print("lioFrame shape: ", lioFrame.shape)

# 读取数据
loopPdFrame1 = pd.read_csv(v2_dld_rs_ref6_reloT, sep=" ", header=None)
# 提取前 1,2-9列作为lio frame并保存
lioFrame = loopPdFrame1[[0, 1, 2, 3, 4, 5, 6, 7, 8]]
#去掉2列
lioFrame.drop(lioFrame.columns[1], axis=1, inplace=True)
lioFrame.to_csv(v2_dld_rs_usb_ref6_gt, sep=' ', header=False, index=False)
print("lioFrame shape: ", lioFrame.shape)