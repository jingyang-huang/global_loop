import open3d as o3d
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2

def read_and_downsample_pcd(file_path, voxel_size=0.5):
    # 使用open3d读取PCD文件
    pcd = o3d.io.read_point_cloud(file_path)
    
    # 使用体素降采样进行降采样
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    return downsampled_pcd

def convert_open3d_to_ros(downsampled_pcd, frame_id="map"):
    # 从open3d的点云中提取点
    points = np.asarray(downsampled_pcd.points)
    
    # 创建PointCloud2消息
    header = rospy.Header(frame_id=frame_id)
    ros_cloud = pcl2.create_cloud_xyz32(header, points)
    
    return ros_cloud

def publish_point_cloud(ros_cloud):
    # 初始化ROS节点
    rospy.init_node('pcd_publisher', anonymous=True)
    pub = rospy.Publisher('Global_map', PointCloud2, queue_size=10)
    
    # 等待连接
    rospy.sleep(1)
    
    # 发布点云
    pub.publish(ros_cloud)
    rospy.loginfo("Published point cloud to Global_map topic.")

if __name__ == "__main__":
    file_path = "/home/heron/Desktop/OmniMatch_ws/src/FAST_LIO_SAM/PCD/scan_shenge.pcd"  # 替换为你的PCD文件路径
    voxel_size = 0.5  # 体素大小
    
    # 读取并降采样PCD文件
    downsampled_pcd = read_and_downsample_pcd(file_path, voxel_size)
    
    # 将open3d点云转换为ROS消息
    ros_cloud = convert_open3d_to_ros(downsampled_pcd)
    
    # 发布到ROS话题
    publish_point_cloud(ros_cloud)