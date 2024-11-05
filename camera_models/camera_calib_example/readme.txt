# help for checking input parameters.
rosrun camera_models Calibrations --help

# example pinhole model.
rosrun camera_models Calibrations -w 8 -h 5 -s 29 -i /home/nv/Pictures/front_1010/ --camera-model pinhole

# example mei model.
rosrun camera_models Calibrations -w 12 -h 8 -s 80 -i /home/nv/SJTU_cm_ws/src/global_loop/camera_models/camera_calib_example/calibrationdata --camera-model mei
