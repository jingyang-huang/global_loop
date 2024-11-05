
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
import argparse

def read_bag_and_extract_trajectory(bag_file, odom_topic_name):
    # read bag file
    bag = rosbag.Bag(bag_file)
    
    x_vals = []
    y_vals = []
    z_vals = []
    time_stamps = []

    # read nav_msgs/Odometry message
    for topic, msg, t in bag.read_messages(topics=[odom_topic_name]):
        
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)
        time_stamps.append(t.to_sec())

    bag.close()
    
    return np.array(x_vals), np.array(y_vals), np.array(z_vals), np.array(time_stamps)

def plot_trajectory_and_deltas(x_vals, y_vals, z_vals, time_stamps):
    # compute position delta
    delta_x = np.diff(x_vals)
    delta_y = np.diff(y_vals)
    delta_z = np.diff(z_vals)
    
    # plot 2d trajectory
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals, label='Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('2D Trajectory')
    plt.legend()
    plt.grid(True)
    # plt.show()
    ylim = (-0.07, 0.07)
    # plot xyz position
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(time_stamps, x_vals, label='X axis')
    plt.xlabel('Time [s]')
    plt.ylabel('X Position')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.subplot(3, 2, 3)
    plt.plot(time_stamps, y_vals, label='Y axis')
    plt.xlabel('Time [s]')
    plt.ylabel('Y Position')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.subplot(3, 2, 5)
    plt.plot(time_stamps, z_vals, label='Z axis')
    plt.xlabel('Time [s]')
    plt.ylabel('Z Position')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    # plt.tight_layout()
    # plt.show()
    
    # plot trajectory jump
    # plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 2, 2)
    plt.plot(time_stamps[1:], delta_x, label='Delta X')
    plt.xlabel('Time [s]')
    plt.ylabel('Delta X')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.subplot(3, 2, 4)
    plt.plot(time_stamps[1:], delta_y, label='Delta Y')
    plt.xlabel('Time [s]')
    plt.ylabel('Delta Y')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.subplot(3, 2, 6)
    plt.plot(time_stamps[1:], delta_z, label='Delta Z')
    plt.xlabel('Time [s]')
    plt.ylabel('Delta Z')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Analyze trajectories''')
    # parser.add_argument('--bag_path', help="Bag Path", default="/media/heron/2E46-68F7/hjy-bag/0829/v1_dld_rs_ref6_3loops.bag")
    parser.add_argument('--bag_path', help="Bag Path", default="/home/heron/SJTU/Codes/VIO/cross_modal_ws/v3_reloc.bag")
    parser.add_argument('--odom_topic', help='Imu Rate Odometry', default="/ros_reloc_node/odometry_rect_ws")
    args = parser.parse_args()
    if (args.bag_path == None):
        print("Please Check Your Bag FILE Path")
        exit()
    print("Odometry Topic is: ", args.odom_topic)
    bag_file = args.bag_path
    odom_topic = args.odom_topic
    
    # read bag and extractor odometry message
    x_vals, y_vals, z_vals, time_stamps = read_bag_and_extract_trajectory(bag_file, odom_topic)
    
    # plot result
    plot_trajectory_and_deltas(x_vals, y_vals, z_vals, time_stamps)
