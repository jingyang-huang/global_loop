/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <vector>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <ros/package.h>
#include <mutex>
#include <queue>
#include <thread>
#include <pthread.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include "parameters.h"
#include "feature_manager.h"
#include <pcl/common/transforms.h> // Include this header

#define SKIP_FIRST_CNT 10
using namespace std;

queue<sensor_msgs::ImageConstPtr> img0_buf, img1_buf, img2_buf, img3_buf;
queue<std::tuple<double, cv::Mat, cv::Mat>> stereo_image_mat_buf;
queue<std::tuple<double, cv::Mat, cv::Mat, cv::Mat, cv::Mat>> omni_image_mat_buf;
queue<cv::Mat> image_mat_buf;
// typedef

queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>>> featureBuf;
queue<sensor_msgs::PointCloudConstPtr> imgpoint_buf;
queue<sensor_msgs::PointCloud2ConstPtr> lidar_point_buf;
queue<nav_msgs::Odometry::ConstPtr> pose_buf;
queue<Eigen::Vector3d> odometry_buf;
std::mutex m_buf, m_image_Buf, img_buf_mutex;
std::mutex m_process;
std::mutex m_command;
int frame_index = 0;
int sequence = 1;
PoseGraph posegraph;
int skip_first_cnt = 0;
int SKIP_CNT;
int skip_cnt = 0;
bool load_flag = 0;
bool start_flag = 0;
int BUILD_KEYFRAME;
const int WINDOW_SIZE = 4;
int MIN_QUERY_GAP = 50;
int FREQ_FACTOR_IMG = 6;
int FREQ_FACTOR_ODOM = 20; // 10hz

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE = 0;
int MAPPING_MODE;

Eigen::Vector3d last_lio_t;
Eigen::Quaterniond last_lio_q;

std::vector<camodocal::CameraPtr> m_camera;
std::vector<Eigen::Vector3d> t_i_c;
std::vector<Eigen::Matrix3d> R_i_c;
Eigen::Vector3d til;
Eigen::Matrix3d ril;
std::vector<Eigen::Matrix4d> i_T_c_vec;
std::deque<double> img_blur_deq;
Vector3d ts[(WINDOW_SIZE + 1)];
Matrix3d Rs[(WINDOW_SIZE + 1)];

// Eigen::Matrix3d R_i_c[];
ros::Publisher pub_match_img;
ros::Publisher pub_image_track;
ros::Publisher pub_camera_pose_visual;
ros::Publisher pub_odometry_rect;

std::string POSE_GRAPH_SAVE_PATH;
std::string LOOP_RESULT_PATH;
std::string MAPPING_TIME_LOG_PATH;
std::string RELO_TIME_LOG_PATH;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_T(-100, -100, -100);
Eigen::Matrix3d last_R;
double last_image_time = -1;

deque<double> time_buffer;               // 记录lidar时间
deque<PointCloudXYZI::Ptr> lidar_buffer; // 储存处理后的lidar特征
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
FeatureTracker featureTracker;
FeatureManager f_manager;
PointCloudXYZI::Ptr points_triangulate(new PointCloudXYZI);
PointCloudXYZI::Ptr points_lidar_corner(new PointCloudXYZI);
ros::Publisher pub_imgpoint_cloud, pub_margin_cloud, pub_point_cloud_cam, pub_point_cloud_inFOV, pub_triangulate_cloud, pub_lidar_corner_cloud, pub_oldKeyframe_cloud;
ros::Publisher pub_map_lidarpts;
bool COMMAND_KEYFRAME = false;
bool initialized_pass = false;

enum MarginalizationFlag
{
    MARGIN_OLD = 0,
    MARGIN_SECOND_NEW = 1
};
MarginalizationFlag marginalization_flag;
int frame_count = 0;
int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
int inputImageCnt;
const int total_steps = 50;
double pos_smooth_rate = 0.02; // 5 m/s
double rot_smooth_rate = 0.02;  // 10 deg/s

void new_sequence()
{
    // printf("new sequence\n");
    // sequence++;
    // printf("sequence cnt %d \n", sequence);
    // if (sequence > 5)
    // {
    //     ROS_WARN("only support 5 sequences since it's boring to copy code for more sequences.");
    //     ROS_BREAK();
    // }
    // posegraph.posegraph_visualization->reset();
    // posegraph.publish();
    // m_buf.lock();
    // while (!img0_buf.empty())
    //     img0_buf.pop();
    // while (!imgpoint_buf.empty())
    //     imgpoint_buf.pop();
    // while (!pose_buf.empty())
    //     pose_buf.pop();
    // while (!odometry_buf.empty())
    //     odometry_buf.pop();
    // m_buf.unlock();
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // m_buf.lock();
    // // double preprocess_start_time = omp_get_wtime();
    // if (last_timestamp_imu > 0.0 && imu_msg->header.stamp.toSec() < last_timestamp_imu)
    // {
    //     ROS_ERROR("IMU loop back, clear buffer");
    //     // imu_buffer.clear();
    // }
    // last_timestamp_imu = imu_msg->header.stamp.toSec();
    // // imu_buffer.push_back(imu_msg);
    // m_buf.unlock();
}

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    // ROS_INFO("img0_callback!");
    static short img_cnt = 0;
    if (++img_cnt >= FREQ_FACTOR_IMG) // 1/3
    {
        img_cnt = 0;
        m_buf.lock();
        img0_buf.push(img_msg);
        m_buf.unlock();
    }
    // printf(" image time %f \n", img_msg->header.stamp.toSec());

    // detect unstable camera stream
    if (last_image_time == -1)
        last_image_time = img_msg->header.stamp.toSec();
    else if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        // ROS_WARN("image discontinue! detect a new sequence!");
        new_sequence();
    }
    last_image_time = img_msg->header.stamp.toSec();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    static short img_cnt = 0;
    if (++img_cnt >= FREQ_FACTOR_IMG) // 1/3
    {
        img_cnt = 0;
        m_buf.lock();
        img1_buf.push(img_msg);
        m_buf.unlock();
    }
}

void img2_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    static short img_cnt = 0;
    if (++img_cnt >= FREQ_FACTOR_IMG) // 1/3
    {
        img_cnt = 0;
        m_buf.lock();
        img2_buf.push(img_msg);
        m_buf.unlock();
    }
}

void img3_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    static short img_cnt = 0;
    if (++img_cnt >= FREQ_FACTOR_IMG) // 1/3
    {
        img_cnt = 0;
        m_buf.lock();
        img3_buf.push(img_msg);
        m_buf.unlock();
    }
}

// 关键帧的位姿，会实际参与优化
void pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    // ROS_INFO("pose_callback!");
    static short pos_cnt = 0;
    if (++pos_cnt >= FREQ_FACTOR_ODOM) // 1/3
    {
        pos_cnt = 0;
        m_buf.lock();
        pose_buf.push(pose_msg);
        // printf("pose callback, posebuf size %d \n", pose_buf.size());
        m_buf.unlock();
    }
    /*
    printf("pose t: %f, %f, %f   q: %f, %f, %f %f \n", pose_msg->pose.pose.position.x,
                                                       pose_msg->pose.pose.position.y,
                                                       pose_msg->pose.pose.position.z,
                                                       pose_msg->pose.pose.orientation.w,
                                                       pose_msg->pose.pose.orientation.x,
                                                       pose_msg->pose.pose.orientation.y,
                                                       pose_msg->pose.pose.orientation.z);
    */
    return;
}

// for navigation
void vio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    // ROS_INFO("vio_callback!");
    nav_msgs::Odometry odometry;
    Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    Quaterniond vio_q(pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x, pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z);
    // TODO: write as a function
    static Eigen::Vector3d last_t_drift = posegraph.t_drift;
    static bool bDriftUpdated = false;
    bool bPosReached = false;
    bool bRotReached = false;

    // calculate the corrected pose
    posegraph.t_corrected = posegraph.r_drift * vio_t + posegraph.t_drift;
    posegraph.r_corrected = posegraph.r_drift * vio_q;

    // TODO: change posegraph.r_drift_pub
    if (USE_TRAJ_SMOOTH)
    {
        // detect change
        if ((last_t_drift - posegraph.t_drift).norm() > 1e-3)
        {
            // std::cout << "T_loop_drift_opt_bef_ changed, new posegraph.t_corrected " << posegraph.t_corrected.transpose() << std::endl;
            last_t_drift = posegraph.t_drift;
            bDriftUpdated = true;
        }
        // set the corrected pose for control
        if (bDriftUpdated) // 改成两种区间，一种是正在靠近，一种是中间纯用vio推，第一种才需要平滑
        {
            const Eigen::Vector3d goal_T = posegraph.t_corrected;
            Eigen::Vector3d now_T = posegraph.t_corrected_pub;
            const Eigen::Vector3d arrow_T = goal_T - now_T;
            posegraph.r_corrected_pub = posegraph.r_corrected; // may angle jump
            // smooth the publish pose
            if (arrow_T.norm() > 0.01) // wait for t_corrected_pub to reach
            {
                Eigen::Vector3d delta = arrow_T / arrow_T.norm() * pos_smooth_rate;
                posegraph.t_corrected_pub = now_T + delta;
                // std::cout<<"now_T: "<<now_T.transpose()<< " goal_T: "<<goal_T.transpose() << " arrow_T: "<<arrow_T.transpose() << " delta: "<<delta.transpose() << " posegraph.t_corrected_pub: "<<posegraph.t_corrected_pub.transpose()<<std::endl;
            }
            else
            {
                std::cout << BLUE << "bPosReached = true" << TAIL << std::endl;
                bPosReached = true;                 // approach over
                posegraph.t_corrected_pub = goal_T; // 1cm跳变直接过去
            }

            const Eigen::Quaterniond goal_Q(posegraph.r_corrected);
            Eigen::Quaterniond now_Q(posegraph.r_corrected_pub);
            const Eigen::Quaterniond gap_Q = goal_Q * now_Q.inverse();
            if (gap_Q.angularDistance(Eigen::Quaterniond::Identity()) > 0.01) // wait for t_corrected_pub to reach
            {
                // delta_Q * delta_Q * delta_Q * ... * delta_Q = gap_Q = goal_Q * now_Q.inverse()
                // gap_Q -> delta_rot = delta_Q
                Eigen::AngleAxisd rotation_vector(gap_Q);
                Eigen::AngleAxisd delta_rot(rot_smooth_rate, rotation_vector.axis());
                Eigen::Quaterniond delta_Q(delta_rot);
                posegraph.r_corrected_pub = delta_Q * now_Q;
            }
            else
            {
                // std::cout << PURPLE << "bRotReached = true" << TAIL << std::endl;
                bRotReached = true; // approach over
                posegraph.r_corrected_pub = goal_Q;
            }

            if (bPosReached && bRotReached)
            {
                bDriftUpdated = false;
            }
        }
        else // fully rely on vio
        {
            posegraph.t_corrected_pub = posegraph.t_corrected;
            posegraph.r_corrected_pub = posegraph.r_corrected;
        }
    }
    else
    {
        posegraph.t_corrected_pub = posegraph.t_corrected;
        posegraph.r_corrected_pub = posegraph.r_corrected;
    }

    rected_vio_t = posegraph.t_corrected_pub;
    rected_vio_q = posegraph.r_corrected_pub;

    odometry = *pose_msg;
    // odometry.header = pose_msg->header;
    // odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = rected_vio_t.x();
    odometry.pose.pose.position.y = rected_vio_t.y();
    odometry.pose.pose.position.z = rected_vio_t.z();
    odometry.pose.pose.orientation.x = rected_vio_q.x();
    odometry.pose.pose.orientation.y = rected_vio_q.y();
    odometry.pose.pose.orientation.z = rected_vio_q.z();
    odometry.pose.pose.orientation.w = rected_vio_q.w();

    pub_odometry_rect.publish(odometry);

    static int path_cnt = 0;
    if (path_cnt++ % 20 == 0)
    {
        // double x_diff = rected_vio_t.x() - pose_msg->pose.pose.position.x;
        // double y_diff = rected_vio_t.y() - pose_msg->pose.pose.position.y;
        // double z_diff = rected_vio_t.z() - pose_msg->pose.pose.position.z;
        // double q_w_diff = rected_vio_q.w() - pose_msg->pose.pose.orientation.w;
        // double q_x_diff = rected_vio_q.x() - pose_msg->pose.pose.orientation.x;
        // double q_y_diff = rected_vio_q.y() - pose_msg->pose.pose.orientation.y;
        // double q_z_diff = rected_vio_q.z() - pose_msg->pose.pose.orientation.z;
        // std::cout << std::setprecision(9) << "x_diff: " << x_diff << " y_diff: " << y_diff << " z_diff: " << z_diff << std::endl;
        // std::cout << std::setprecision(9) << "q_w_diff: " << q_w_diff << " q_x_diff: " << q_x_diff << " q_y_diff: " << q_y_diff << " q_z_diff: " << q_z_diff << std::endl;
        path_cnt = 0;
        // std::cout << "t_rect: " << t_rect << std::endl;
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = pose_msg->header;
        pose_stamped.pose = odometry.pose.pose;
        posegraph.path[sequence].poses.push_back(pose_stamped);
        posegraph.path[sequence].header = pose_msg->header;

        posegraph.pub_pg_path.publish(posegraph.path[sequence]);

        if (DEBUG_IMAGE)
            posegraph.publish();
    }
    // return;
    // Vector3d vio_t_cam;
    // Quaterniond vio_q_cam;
    // vio_t_cam = vio_t + vio_q * t_i_c[0];
    // vio_q_cam = vio_q * R_i_c[0];

    // cameraposevisual.reset();
    // cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    // cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);
}

void lio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    // ROS_INFO("lio_callback!");
    lio_t.x() = pose_msg->pose.pose.position.x;
    lio_t.y() = pose_msg->pose.pose.position.y;
    lio_t.z() = pose_msg->pose.pose.position.z;
    Quaterniond msg_lio_q;
    msg_lio_q.w() = pose_msg->pose.pose.orientation.w;
    msg_lio_q.x() = pose_msg->pose.pose.orientation.x;
    msg_lio_q.y() = pose_msg->pose.pose.orientation.y;
    msg_lio_q.z() = pose_msg->pose.pose.orientation.z;
    lio_R = msg_lio_q.toRotationMatrix();

    // nav_msgs::Odometry odometry;
    // odometry.header = pose_msg->header;
    // odometry.header.frame_id = "world";
    // odometry.pose.pose.position.x = lio_t.x();
    // odometry.pose.pose.position.y = lio_t.y();
    // odometry.pose.pose.position.z = lio_t.z();
    // odometry.pose.pose.orientation.x = lio_q.x();
    // odometry.pose.pose.orientation.y = lio_q.y();
    // odometry.pose.pose.orientation.z = lio_q.z();
    // odometry.pose.pose.orientation.w = lio_q.w();
    // pub_odometry_rect.publish(odometry);

    // 跑出来lio也是在fcu坐标系下, 不需要发布cameraposevisual
    // Vector3d lio_t_cam;
    // Quaterniond lio_q_cam;
    // lio_t_cam = lio_t + lio_q * t_i_c[0];
    // lio_q_cam = lio_q * R_i_c;

    // cameraposevisual.reset();
    // cameraposevisual.add_pose(lio_t_cam, lio_q_cam);
    // cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);
}

void extrinsic_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
}

bool Initialize(const sensor_msgs::ImageConstPtr &img_msg)
{
    Eigen::Vector3d T = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    // ROS_INFO("Initialize!");
    std::cout << "Initializing" << std::endl;
    cv::Mat img = cv_bridge::toCvShare(img_msg, "mono8")->image;

    // extract and compute
    FeatureData query_keypoints_data;
    // Eigen::Matrix<float, 259, Eigen::Dynamic> _spp_features;
    std::vector<Eigen::Matrix<float, 256, 1>> spp_descriptors;
    std::vector<cv::Mat> brief_descriptors;
    KeyFramePtr query_keyframe;
    if (0 == DETECTOR)
    {
        feature_detector->DetectSpp(img, query_keypoints_data, spp_descriptors);
        query_keyframe = std::make_shared<KeyFrame>(img_msg->header.stamp.toSec(), frame_index, T, R, img, query_keypoints_data, sequence);
        query_keyframe->query_spp_descriptors = spp_descriptors;
    }
    else if (1 == DETECTOR)
    {
        feature_detector->DetectBrief(img, query_keypoints_data, brief_descriptors);
        // feature_detector->extractBriefDescriptor(image, query_keypoints_data, brief_descriptors);
        query_keyframe = std::make_shared<KeyFrame>(img_msg->header.stamp.toSec(), frame_index, T, R, img, query_keypoints_data, sequence);
        query_keyframe->query_brief_descriptors = brief_descriptors; // 222
    }

    m_process.lock();
    start_flag = 1;
    posegraph.initialize(query_keyframe);
    m_process.unlock();
    frame_index++;
    last_T = T;
    last_R = R;
    return true;
}
void getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = ts[frame_count];
}

void getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = ts[index];
}

double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                         Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                         double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

double reprojectionError(Vector3d Pi_3d_world,
                         Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                         double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Pi_3d_world;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

/**
 * Laplacian 梯度函数
 *
 * Inputs:
 * @param image:
 * Return: double
 */
double laplacian(cv::Mat &gray_img)
{
    cv::Mat lap_image;
    cv::Laplacian(gray_img, lap_image, CV_32FC1);

    cv::Scalar lap_mean, lap_stddev;
    cv::meanStdDev(lap_image, lap_mean, lap_stddev);

    return lap_stddev[0]; // Assuming you want to return the standard deviation value
}

double isImageBlurryUsingFFT(const cv::Mat gray)
{
    // 将图像扩展到最佳的尺寸，边界用0补充
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(gray.rows);
    int n = cv::getOptimalDFTSize(gray.cols);
    cv::copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // 为傅里叶变换的结果(实部和虚部)分配存储空间。
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);

    // 进行傅里叶变换
    cv::dft(complexI, complexI);

    // 计算幅度并转换到对数尺度
    cv::split(complexI, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magI = planes[0];

    magI += cv::Scalar::all(1);
    cv::log(magI, magI);

    // 剪切和重分布幅度图象限
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    // 计算中心外围区域的平均值作为高频能量指标
    cv::Scalar meanVal = cv::mean(magI);

    // 设定一个阈值，根据实际情况调整
    double threshold = 10.0; // 示例阈值
    return meanVal[0];
}

void testPnPRANSAC(const FeatureData &points, std::vector<uchar> &status,
                   Matrix3d R_inital, Vector3d P_inital,
                   Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
    // for (int i = 0; i < matched_3d.size(); i++)
    //	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
    // printf("match size %d \n", matched_3d.size());
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    std::vector<cv::Point3f> matched_3d;
    std::vector<cv::Point2f> matched_2d_old_norm;

    for (int i = 0; i < points.size(); i++)
    {
        // 生成一个0-1的随机数
        double random = rand() / double(RAND_MAX);
        if (random > 0.1)
            continue;
        // 只保留25%的点
        matched_3d.push_back(cv::Point3f(points[i].point_3d_world.x, points[i].point_3d_world.y, points[i].point_3d_world.z));
        matched_2d_old_norm.push_back(cv::Point2f(points[i].keypoint_2d_norm.pt.x, points[i].keypoint_2d_norm.pt.y));
    }
    P_inital = P_inital + Vector3d(1.0, 0.2, 1.0);
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;
    TicToc t_pnp_ransac;

    if (CV_MAJOR_VERSION < 3)
        solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
    else
    {
        if (CV_MINOR_VERSION < 2)
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
        else
            // this
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, false, 100, 10.0 / 460.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
    }

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);

    for (int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

    // cj_T_w1
    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    // R_w_c_old = R_pnp;
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);
    // T_w_c_old = R_w_c_old * (T_pnp);

    // w1_T_bj
    PnP_R_old = R_w_c_old * R_i_c[0].transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * t_i_c[0];
}

struct Vector2iCompare
{
    bool operator()(const Eigen::Vector2i &lhs, const Eigen::Vector2i &rhs) const
    {
        if (lhs.x() != rhs.x())
            return lhs.x() < rhs.x();
        return lhs.y() < rhs.y();
    }
};

void process_reloc()
{

    while (1)
    {
        bool bSync = false;
        sensor_msgs::ImageConstPtr img_msg = NULL;
        sensor_msgs::PointCloudConstPtr imgpoint_msg = NULL;
        nav_msgs::Odometry::ConstPtr pose_msg = NULL;
        // if (img0_buf.size() != 0 && pose_buf.size() != 0)
        // printf("image size %d, point size %d, pose size %d \n", img0_buf.size(), pose_buf.size());
        // find out the messages with same time stamp
        {
            std::lock_guard<std::mutex> lock(m_buf);

            if (1)
            {
                // 等于把pose_msg夹在img_msg.front()和img_msg.back()之间，然后从左边旧的不断pop来同步
                if (!img0_buf.empty() && !pose_buf.empty())
                {
                    if (img0_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
                    {
                        pose_buf.pop();
                        // printf("throw pose at beginning\n");
                    }
                    // 只要最新的img和pose同步的，所以处理的快recall就会更多
                    else if (img0_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec())
                    {
                        pose_msg = pose_buf.front();
                        pose_buf.pop();
                        while (!pose_buf.empty())
                            pose_buf.pop();
                        while (img0_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                            img0_buf.pop();
                        img_msg = img0_buf.front();
                        img0_buf.pop();
                    }
                }
            }
            else
            {
                double time = 0;
                if (!img0_buf.empty() && !pose_buf.empty())
                {
                    bSync = true;
                    double time0 = img0_buf.front()->header.stamp.toSec();
                    // double time1 = img1_buf.front()->header.stamp.toSec();
                    // double time2 = img2_buf.front()->header.stamp.toSec();
                    // double time3 = img3_buf.front()->header.stamp.toSec();
                    // double time4 = lidar_point_buf.front()->header.stamp.toSec();
                    double time5 = pose_buf.front()->header.stamp.toSec();

                    double max_time = std::max(time0, time5);
                    if (time0 < max_time - 0.001)
                    {
                        bSync = false;
                        img0_buf.pop();
                        // ROS_WARN("throw img0");
                    }
                    if (time5 < max_time - 0.001)
                    {
                        bSync = false;
                        pose_buf.pop();
                        // ROS_WARN("throw pose");
                    }
                    if (bSync && !img0_buf.empty())
                    {
                        time = img0_buf.front()->header.stamp.toSec();
                        img_msg = img0_buf.front();
                        img0_buf.pop();
                        pose_msg = pose_buf.front();
                        pose_buf.pop();
                        // std::cout << "omni sync pass!" << std::endl;
                    }
                }
            }
        }

        if (pose_msg != NULL)
        {
            std::cout << " img0_buf size " << img0_buf.size() << " pose_buf size " << pose_buf.size() << std::endl;
            printf(" pose time %f \n", pose_msg->header.stamp.toSec());
            printf(" point time %f \n", imgpoint_msg->header.stamp.toSec());
            printf(" image time %f \n", img_msg->header.stamp.toSec());
            cv_bridge::CvImageConstPtr ptr;
            if (img_msg->encoding == "8UC1")
            {
                sensor_msgs::Image img;
                img.header = img_msg->header;
                img.height = img_msg->height;
                img.width = img_msg->width;
                img.is_bigendian = img_msg->is_bigendian;
                img.step = img_msg->step;
                img.data = img_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

            cv::Mat image = ptr->image;
            // build keyframe， using keyframe pose from vio
            Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  pose_msg->pose.pose.position.z);
            Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                     pose_msg->pose.pose.orientation.x,
                                     pose_msg->pose.pose.orientation.y,
                                     pose_msg->pose.pose.orientation.z)
                             .toRotationMatrix();

            double move_dis = (T - last_T).norm();
            double move_angle = Utility::normalizeAngle(Utility::R2ypr(R).x() - Utility::R2ypr(last_R).x());
            if (move_dis > SKIP_DIST || move_angle > SKIP_ANGLE)
            {
                printf("move_dis %f, move_angle %f\n", move_dis, move_angle);

                // 关键帧除了位姿，还有三维点，二维点，二维点的id，二维点的归一化坐标
                // KeyFrame *query_keyframe = new KeyFrame(pose_msg->header.stamp.toSec(), frame_index, T, R, image, sequence);
                // extract and compute
                FeatureData query_keypoints_data;
                // Eigen::Matrix<float, 259, Eigen::Dynamic> _spp_features;
                std::vector<Eigen::Matrix<float, 256, 1>> spp_descriptors;
                std::vector<cv::Mat> brief_descriptors;
                KeyFramePtr query_keyframe;
                if (0 == DETECTOR)
                {
                    feature_detector->DetectSpp(image, query_keypoints_data, spp_descriptors);
                    query_keyframe = std::make_shared<KeyFrame>(pose_msg->header.stamp.toSec(), frame_index, T, R, image, query_keypoints_data, sequence);
                    query_keyframe->query_spp_descriptors = spp_descriptors;
                }
                else if (1 == DETECTOR)
                {
                    feature_detector->DetectBrief(image, query_keypoints_data, brief_descriptors);
                    // feature_detector->extractBriefDescriptor(image, query_keypoints_data, brief_descriptors);
                    query_keyframe = std::make_shared<KeyFrame>(pose_msg->header.stamp.toSec(), frame_index, T, R, image, query_keypoints_data, sequence);
                    query_keyframe->query_brief_descriptors = brief_descriptors; // 222
                }

                m_process.lock();
                start_flag = 1;
                posegraph.addKeyFrame(query_keyframe);
                m_process.unlock();
                frame_index++;
                last_T = T;
                last_R = R;
            }

            // save loop
            fout_loopRes.setf(ios::fixed, ios::floatfield);
            fout_loopRes.precision(9);
            fout_loopRes << pose_msg->header.stamp.toSec() << " ";
            // fout_loopRes.precision(6);
            fout_loopRes << rected_vio_t.x() << " "
                         << rected_vio_t.y() << " "
                         << rected_vio_t.z() << " "
                         << rected_vio_q.x() << " "
                         << rected_vio_q.y() << " "
                         << rected_vio_q.z() << " "
                         << rected_vio_q.w() << endl;
        }
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }

    fout_reloTime.close();
    fout_loopRes.close();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "loop_fusion");
    ros::NodeHandle n("~");
    posegraph.registerPub(n);

    VISUALIZATION_SHIFT_X = 0;
    VISUALIZATION_SHIFT_Y = 0;
    SKIP_CNT = 0;
    SKIP_DIST = 0;
    // WINDOW_SIZE = 10;

    if (argc != 2)
    {
        printf("please intput: rosrun loop_fusion loop_fusion_node [config file] \n"
               "for example: rosrun loop_fusion loop_fusion_node "
               "/home/tony-ws1/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 0;
    }

    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);

    std::string IMAGE0_TOPIC, IMAGE1_TOPIC, IMAGE2_TOPIC, IMAGE3_TOPIC, IMU_TOPIC, RELO_IMAGE0_TOPIC;
    int LOAD_PREVIOUS_POSE_GRAPH;

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];

    // read cam parameters
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        fsSettings["image0_topic"] >> IMAGE0_TOPIC;
        fsSettings["image1_topic"] >> IMAGE1_TOPIC;
        fsSettings["image2_topic"] >> IMAGE2_TOPIC;
        fsSettings["image3_topic"] >> IMAGE3_TOPIC;
        fsSettings["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
        fsSettings["loop_output_path"] >> LOOP_RESULT_PATH;
        // fsSettings["mapping_time_log_path"] >> MAPPING_TIME_LOG_PATH;
        fsSettings["relo_time_log_path"] >> RELO_TIME_LOG_PATH;
        fsSettings["debug_image"] >> DEBUG_IMAGE;
        fsSettings["mapping_mode"] >> MAPPING_MODE;
        fsSettings["detector"] >> DETECTOR;
        fsSettings["matcher"] >> MATCHER;
        printf("MAPPING_MODE: %d\n", MAPPING_MODE);
        printf("DETECTOR: %d\n", DETECTOR);
        printf("MATCHER: %d\n", MATCHER);

        fsSettings["num_of_cam"] >> NUM_OF_CAM;
        printf("camera number %d\n", NUM_OF_CAM);

        int pn = config_file.find_last_of('/');
        std::string configPath = config_file.substr(0, pn);
        std::string cam0Calib;
        fsSettings["cam0_calib"] >> cam0Calib;
        std::string cam0Path = configPath + "/" + cam0Calib;
        printf("cam calib path: %s\n", cam0Path.c_str());
        CAM_NAMES.push_back(cam0Path);

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        R_i_c.push_back(T.block<3, 3>(0, 0));
        t_i_c.push_back(Eigen::Vector3d(T.block<3, 1>(0, 3)));
        i_T_c_vec.push_back(T);

        if (NUM_OF_CAM == 2)
        {
            // STEREO = 1;
            std::string cam1Calib;
            fsSettings["cam1_calib"] >> cam1Calib;
            std::string cam1Path = configPath + "/" + cam1Calib;
            // printf("%s cam1 path\n", cam1Path.c_str() );
            CAM_NAMES.push_back(cam1Path);
            fsSettings["body_T_cam1"] >> cv_T;
            cv::cv2eigen(cv_T, T);
            R_i_c.push_back(T.block<3, 3>(0, 0));
            t_i_c.push_back(T.block<3, 1>(0, 3));
            i_T_c_vec.push_back(T);
        }
        else if (NUM_OF_CAM == 4)
        {
            // OMNI = 1;
            std::string camCalib;
            fsSettings["cam1_calib"] >> camCalib;
            std::string cam1Path = configPath + "/" + camCalib;
            fsSettings["cam2_calib"] >> camCalib;
            std::string cam2Path = configPath + "/" + camCalib;
            fsSettings["cam3_calib"] >> camCalib;
            std::string cam3Path = configPath + "/" + camCalib;
            CAM_NAMES.push_back(cam1Path);
            CAM_NAMES.push_back(cam2Path);
            CAM_NAMES.push_back(cam3Path);

            fsSettings["body_T_cam1"] >> cv_T;
            cv::cv2eigen(cv_T, T);
            R_i_c.push_back(T.block<3, 3>(0, 0));
            t_i_c.push_back(T.block<3, 1>(0, 3));
            i_T_c_vec.push_back(T);
            fsSettings["body_T_cam2"] >> cv_T;
            cv::cv2eigen(cv_T, T);
            R_i_c.push_back(T.block<3, 3>(0, 0));
            t_i_c.push_back(T.block<3, 1>(0, 3));
            i_T_c_vec.push_back(T);
            fsSettings["body_T_cam3"] >> cv_T;
            cv::cv2eigen(cv_T, T);
            R_i_c.push_back(T.block<3, 3>(0, 0));
            t_i_c.push_back(T.block<3, 1>(0, 3));
            i_T_c_vec.push_back(T);
        }

        printf("camera number: %d\n", CAM_NAMES.size());

        // fsSettings["body_T_lidar"] >> cv_T;
        // cv::cv2eigen(cv_T, T);
        // qil = T.block<3, 3>(0, 0);
        // til = T.block<3, 1>(0, 3);
        // cout << " exitrinsic lidar " << endl
        //      << qil << endl
        //      << til.transpose() << endl;

        f_manager.setRic(R_i_c);

        // m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0Path.c_str());
        featureTracker.readIntrinsicParameter(CAM_NAMES);
    }

    LOAD_PREVIOUS_POSE_GRAPH = fsSettings["load_previous_pose_graph"];
    // LOOP_RESULT_PATH = LOOP_RESULT_PATH;
    std::cout << "loop_output_path path " << LOOP_RESULT_PATH << std::endl;
    // std::ofstream fout_loopRes(LOOP_RESULT_PATH, std::ios::out);
    // fout_loopRes.close();
    // std::ofstream fout_reloTime(RELO_TIME_LOG_PATH, std::ios::out);
    // fout_reloTime.close();

    fout_reloTime.open(RELO_TIME_LOG_PATH, ios::out);
    fout_loopRes.open(LOOP_RESULT_PATH, ios::out);

    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    MIN_DIST_LIDARPT = fsSettings["min_dist_lidarpt"];
    F_THRESHOLD = fsSettings["F_threshold"];
    FLOW_BACK = fsSettings["flow_back"];
    SKIP_DIST = fsSettings["skip_dist"];
    RELATIVE_THRESHOLD = fsSettings["relative_threshold"];
    std::cout << "SKIP_DIST " << SKIP_DIST << std::endl;
    std::cout << "RELATIVE_THRESHOLD " << RELATIVE_THRESHOLD << std::endl;
    pos_smooth_rate = fsSettings["pos_smooth_rate"];
    rot_smooth_rate = fsSettings["rot_smooth_rate"];
    std::cout << "pos_smooth_rate " << pos_smooth_rate << std::endl;
    std::cout << "rot_smooth_rate " << rot_smooth_rate << std::endl;

    USE_PG_OPTIMIZE = fsSettings["use_pg_optimize"];
    USE_TRAJ_SMOOTH = fsSettings["use_traj_smooth"];
    USE_IMU = fsSettings["imu"];
    if (USE_PG_OPTIMIZE)
        posegraph.setIMUFlag(USE_IMU);
    BUILD_KEYFRAME = fsSettings["build_keyframe"];
    cout << "BUILD_KEYFRAME: " << BUILD_KEYFRAME << endl;
    SuperPointConfig spp_config;
    BriefConfig brief_config;
    PointMatcherConfig pmconfig;
    std::string pkg_path = ros::package::getPath("loop_fusion");

    if (0 == DETECTOR)
    {
        fsSettings["superpoint"]["max_keypoints"] >> spp_config.max_keypoints;
        fsSettings["superpoint"]["keypoint_threshold"] >> spp_config.keypoint_threshold;
        fsSettings["superpoint"]["remove_borders"] >> spp_config.remove_borders;
        fsSettings["superpoint"]["dla_core"] >> spp_config.dla_core;
        fsSettings["superpoint"]["onnx_file"] >> spp_config.onnx_file;
        fsSettings["superpoint"]["engine_file"] >> spp_config.engine_file;
    }
    else if (1 == DETECTOR)
    {
        fsSettings["brief"]["max_keypoints"] >> brief_config.max_keypoints;
        fsSettings["brief"]["keypoint_threshold"] >> brief_config.keypoint_threshold;
        fsSettings["brief"]["remove_borders"] >> brief_config.remove_borders;
        fsSettings["brief"]["pattern_file"] >> brief_config.pattern_file;
        // brief_config.pattern_file = pkg_path + fsSettings["brief"]["pattern_file"];
        cout << "BRIEF_PATTERN_FILE" << brief_config.pattern_file << endl;
    }

    if (0 == MATCHER)
    {

        fsSettings["point_matcher"]["image_width"] >> pmconfig.image_width;
        fsSettings["point_matcher"]["image_height"] >> pmconfig.image_height;
        fsSettings["point_matcher"]["onnx_file"] >> pmconfig.onnx_file;
        fsSettings["point_matcher"]["engine_file"] >> pmconfig.engine_file;
    }

    fsSettings.release();

    ros::Subscriber sub_vio = n.subscribe("/vins_estimator/imu_propagate", 10, vio_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_lio = n.subscribe("/mapping/Odometry", 1000, lio_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 1000, img0_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_img1, sub_img2, sub_img3;
    if (BUILD_KEYFRAME)
    {
        if (2 == NUM_OF_CAM)
        {
            printf("sub_img 1 \n");
            sub_img1 = n.subscribe(IMAGE1_TOPIC, 1000, img1_callback, ros::TransportHints().tcpNoDelay()); // image1 for tracker, but not necessary
        }

        if (4 == NUM_OF_CAM)
        {
            printf("sub_img 1 \n");
            sub_img1 = n.subscribe(IMAGE1_TOPIC, 1000, img1_callback, ros::TransportHints().tcpNoDelay());
            sub_img2 = n.subscribe(IMAGE2_TOPIC, 1000, img2_callback, ros::TransportHints().tcpNoDelay());
            sub_img3 = n.subscribe(IMAGE3_TOPIC, 1000, img3_callback, ros::TransportHints().tcpNoDelay());
        }
    }
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 1000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_pose = n.subscribe("/vins_estimator/odometry", 1000, pose_callback, ros::TransportHints().tcpNoDelay());
    // ros::Subscriber sub_margin_point = n.subscribe("/vins_estimator/margin_cloud", 1000, margin_point_callback);

    pub_image_track = n.advertise<sensor_msgs::Image>("image_track", 1000);
    pub_match_img = n.advertise<sensor_msgs::Image>("match_image", 1000);
    // pub_imgpoint_cloud = n.advertise<sensor_msgs::PointCloud>("imgpoint_cloud", 1000);
    // pub_point_cloud_cam = n.advertise<sensor_msgs::PointCloud2>("point_cloud_inimu", 1000);
    // pub_point_cloud_inFOV = n.advertise<sensor_msgs::PointCloud2>("point_cloud_inFOV", 1000);
    // pub_triangulate_cloud = n.advertise<sensor_msgs::PointCloud2>("triangulate_cloud", 1000);
    // pub_lidar_corner_cloud = n.advertise<sensor_msgs::PointCloud2>("lidar_corner_cloud", 1000);
    pub_oldKeyframe_cloud = n.advertise<sensor_msgs::PointCloud2>("oldKeyframe_cloud", 1000);
    pub_map_lidarpts = n.advertise<sensor_msgs::PointCloud2>("map_lidarpts", 1000);
    pub_globalkf_pose = n.advertise<sensor_msgs::PointCloud2>("globalkf_pose", 1000);
    pub_pnp_pose = n.advertise<nav_msgs::Odometry>("pnp_pose", 10);
    pub_odometry_rect = n.advertise<nav_msgs::Odometry>("rected_imu_propagate", 10);

    if (0 == DETECTOR)
    {
        initializeSuperpointDetector(spp_config);
        string vocabulary_file = pkg_path + "/../support_files/point_voc_L4.bin";
        // cout << "vocabulary_file " << vocabulary_file << endl;
        posegraph.initDatabase(vocabulary_file, DETECTOR);
        initializePointMatcher(pmconfig);
    }
    else if (1 == DETECTOR)
    {
        initializeBriefDetector(brief_config);
        string vocabulary_file = pkg_path + "/../support_files/ORBvoc.txt.bin";
        // string vocabulary_file = "/home/heron/SJTU/Codes/VIO/cross_modal_ws/src/global_loop/support_files/brief_k10L6.bin";
        // cout << "vocabulary_file " << vocabulary_file << endl;
        posegraph.initDatabase(vocabulary_file, DETECTOR);
    }

    if (LOAD_PREVIOUS_POSE_GRAPH)
    {
        printf("load pose graph\n");
        m_process.lock();
        posegraph.loadPoseGraph();
        m_process.unlock();
        printf("load pose graph finish\n");
        load_flag = 1;
    }
    else
    {
        printf("no previous pose graph\n");
        load_flag = 1;
    }

    std::thread measurement_process;
    std::thread stereo_sync_thread, omni_sync_thread;

    printf("measurement_process is process_reloc\n");
    measurement_process = std::thread(process_reloc);
    pthread_setname_np(measurement_process.native_handle(), "measurement_process");

    ros::spin();

    return 0;
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}