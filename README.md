# Global-Loop
global loop could associate Lidar scans and camera pixels to generate hybrid visual-lidar map, which could later be utilized to localization in the same scene.

## Quick Start
### Dependency
Cuda and Tensorrt are default for Jetson NX JetPack 5.1.1, Opencv and Ceres need to be installed separately.
- Cuda 11.4 
- Tensorrt 8.5.2.2 
- Opencv 4.2.0
- Ceres 1.14.0(If the environment does not conflict, it can be installed directly in the system path
)

For Desktop or Laptop users, we recommend you export cuda and tensorrt to PATH and LD_LIBRARY_PATH, so that CMakeLists could find them easily. Here is my exports in ~/.bashrc.
```shell
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/lib64
export CUDNN_ROOT_DIR=/usr/local/cuda-11.4/lib64

export PATH=/home/heron/Dependency/TensorRT-8.5.2.2/bin:$PATH
export LD_LIBRARY_PATH=/home/heron/Dependency/TensorRT-8.5.2.2/lib:$LD_LIBRARY_PATH
```
And as for Opencv and Ceres, we recommend you install them into system path which is convenient.
```shell
cd opencv #(or cd ceres)
mkdir build && cd build
cmake .. && sudo make install
```

### Build
```shell
mkdir -p global_loop/src
cd global_loop/src
git clone http://zjufast.tpddns.cn:38026/fy/global_loop.git
cd ..
catkin build -j
```
### Configs
The config files are in this [folder](/global_loop/config), where each subfolder contains the config files for different type of drone data. Take [fastlab_dld_omni](/global_loop/config/fastlab_dld_omni) for example, it contains intrinsics of four omni pinhole cameras, one realsense d430 stereo camera, one mapping config and one relocalization config. You could create one your own config folder and calibrate the intrinsics and extrinsics of your drone. Most of the default parameters work well, some specific parameters for mapping and relocalization would be introduced in each part respectively.

### Mapping
you could choose to record data to run later or run online with fresh data. You need to change those parameters to your own: onnx_file and engine_file in super_point , pattern_file in brief , num_of_cam , image_topic , cam0_calib , image_width , image_height , pose_graph_save_path and body_T_cam0. We recomment you use `detector:0`(superpoint) to detect keypoint and extract descriptors.

- run online

```shell
./hardware.sh (start hardware including livox,imu,camera)
# for omni-direction keyframes
roslaunch loop_fusion fl_dld_omni_mapping.launch
# for realsense monocular keyframes
roslaunch loop_fusion fl_dld_rs_mapping.launch
# fast-lio mapping
roslaunch fast_lio mapping_mid360_dld.launch
# rviz (optional)
roslaunch loop_fusion mapping_rviz.launch
```

- record data and run offline

```shell
# first
./hardware.sh (start hardware including livox,imu,camera)
./record.sh
# then
rosbag play xxx.bag
# for omni-direction keyframes
roslaunch loop_fusion fl_dld_omni_mapping.launch
# for realsense monocular keyframes
roslaunch loop_fusion fl_dld_rs_mapping.launch
# fast-lio mapping
roslaunch fast_lio mapping_mid360_dld.launch
# rviz (optional)
roslaunch loop_fusion mapping_rviz.launch
```

During this process, 'show_track' is turn on and you could see two picture like this, the first one is the tracking result while the second one is the visual-lidar keyframe.
![image](/global_loop/figs/track.png)
![image](/global_loop/figs/all.png)
When the mapping is over, input **'s'** into the shell and press `enter` to save the map. The map is saved in `pose_graph_save_path` in the [mapping.yaml](/global_loop/config/fastlab_dld_omni/mapping.yaml). It would create the folder if it does not exist, and `pose_graph.txt`, `x_keypoints.txt`, `x_sppdes.bin`,`1_image.png(optional)` could be found in the folder. Besides, you could copy the PCD map in FAST-LIO into the [folder](/global_loop/PCD) and select waypoints in the map, which would be explained later.

### Re-localization
- preparation

Before relocalization, you should compress the generated map and copy it into another platform, and set the `pose_graph_save_path` as the location where you unzip the map, which could be seen in [reloc.yaml](/global_loop/config/fastlab_dld_omni/reloc.yaml). If you test this algorithm on the same plane, then the `pose_graph_save_path` should be the same. Besides, you need to change those parameters to your own: onnx_file and engine_file in super_point and point_matcher, pattern_file in brief , num_of_cam , image_topic , cam0_calib , image_width , image_height , pose_graph_save_path and body_T_cam0.

- run re-localization
After map loaded, the rviz view should be look like this, in which the purple line is the reference trajectory, while the white ball is the keyframe pose.
![rviz_view](/global_loop/figs/reloc_rviz.png)
If you turn on the option `Mappts`, you could see the pointcloud of keypoints.
![rviz_map](/global_loop/figs/reloc_map.png)
Then you could run re-localization based on the loaded map. It should be noted that our method rely on a vio, and the frequency should be higher than 100 Hz as we give this to the drone controller. FAR_VINS is recommended here.
```shell
# launch reloc
roslaunch loop_fusion fl_dld_omni_reloc_far.launch
# far vins
roslaunch far_vins far_fast_lab_dld_vio_stereo.launch 
# play on real plane (choose this one or play on bag)
./hardware.sh (start hardware including livox,imu,camera)
# play on bag
rosbag play xxx.bag

# rviz (optional)
roslaunch loop_fusion reloc_rviz.launch
```
The relocalization result can be seen as follows, where the rected odom(orange arrow) and path (orange billboard) in LoopGroup demonstrate the result. The PnP result(axes) are used to debug, they are supposed to be close to the ground truth position.
![reloc_test](/global_loop/figs/reloc_test.png)
The 2D-3D matching result can be seen as below, you can turn on this option in `debug_image in reloc.yaml` to debug. 
![matching](/global_loop/figs/matching.png)
- select waypoints (If you need planner)

If you want to navigate the drone in the mapped scene, you could run the [scipt](/global_loop/scripts/select_point.py) to select waypoints in the lidar map, the result can be seen below. Parameters *ceiling_height* and *pcd_path* need to be change according to your map.
```shell
python3 global_loop/scripts/select_point.py
#Press [shift + left click] to select a point. And [shift + right click] to remove point.
```
![image](/global_loop/figs/select.png)

