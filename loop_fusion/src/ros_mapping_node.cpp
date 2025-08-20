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
double SKIP_DIST = 0;
double MAPP_DIS_FACTOR = 0;
double MAPP_ROT_FACTOR = 0;
int BUILD_KEYFRAME;
const int WINDOW_SIZE = 4;
int MIN_QUERY_GAP = 50;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE = 0;
int MAPPING_MODE;
int DETECTOR;
int MATCHER;
int SHOW_TRACK;
int FAST_PROJECTION;
Eigen::Vector3d last_lio_t;
Eigen::Quaterniond last_lio_q;

std::vector<camodocal::CameraPtr> m_camera;
std::vector<Eigen::Vector3d> t_i_c;
std::vector<Eigen::Matrix3d> R_i_c;
std::vector<Eigen::Matrix3d> cam_rot_R;
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
ros::Publisher pub_odometry_0, pub_odometry_1, pub_odometry_2, pub_odometry_3, pub_odometry_testPnp;

// std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string LOOP_RESULT_PATH;
std::string MAPPING_TIME_LOG_PATH;
std::string RELO_TIME_LOG_PATH;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_T(-100, -100, -100);
Eigen::Matrix3d last_R;
std::vector<cv::Mat> last_image_seq;
std::vector<cv::Mat> prev_image_seq;
double last_image_time = -1;
const short PWA_size = 3;

deque<double> time_buffer;               // 记录lidar时间
deque<PointCloudXYZI::Ptr> lidar_buffer; // 储存处理后的lidar特征
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
FeatureTracker featureTracker;
FeatureManager f_manager;

PointCloudXYZI::Ptr points_lidar_corner(new PointCloudXYZI);
ros::Publisher pub_imgpoint_cloud, pub_margin_cloud, pub_point_cloud_cam, pub_point_cloud_inFOV, pub_triangulate_cloud, pub_lidar_corner_cloud, pub_oldKeyframe_cloud, pub_map_lidarpts;
bool COMMAND_KEYFRAME = false;

enum MarginalizationFlag
{
    MARGIN_OLD = 0,
    MARGIN_SECOND_NEW = 1
};
MarginalizationFlag marginalization_flag;
int frame_count = 0;
int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
int inputImageCnt;

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

void pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    // ROS_INFO("pose_callback!");
    m_buf.lock();
    pose_buf.push(pose_msg);
    // printf("pose callback, posebuf size %d \n", pose_buf.size());
    m_buf.unlock();
    /*
    printf("pose t: %f, %f, %f   q: %f, %f, %f %f \n", pose_msg->pose.pose.position.x,
                                                       pose_msg->pose.pose.position.y,
                                                       pose_msg->pose.pose.position.z,
                                                       pose_msg->pose.pose.orientation.w,
                                                       pose_msg->pose.pose.orientation.x,
                                                       pose_msg->pose.pose.orientation.y,
                                                       pose_msg->pose.pose.orientation.z);
    */
}

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    // ROS_INFO("img0_callback!");
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
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
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}

void img2_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img2_buf.push(img_msg);
    m_buf.unlock();
}

void img3_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img3_buf.push(img_msg);
    m_buf.unlock();
}

void lidar_point_callback(const sensor_msgs::PointCloud2ConstPtr &point_msg)
{
    m_buf.lock();
    lidar_point_buf.push(point_msg);
    m_buf.unlock();
}

// 这个不是point，而是包含了feature的msg
void imgpoint_callback(const sensor_msgs::PointCloudConstPtr &imgpoint_msg)
{
    // ROS_INFO("point_callback!");
    m_buf.lock();
    imgpoint_buf.push(imgpoint_msg);
    m_buf.unlock();
    /*
    for (unsigned int i = 0; i < imgpoint_msg->points.size(); i++)
    {
        printf("%d, 3D point: %f, %f, %f 2D point %f, %f \n",i , imgpoint_msg->points[i].x,
                                                     imgpoint_msg->points[i].y,
                                                     imgpoint_msg->points[i].z,
                                                     imgpoint_msg->channels[i].values[0],
                                                     imgpoint_msg->channels[i].values[1]);
    }
    */
    // for visualization
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = imgpoint_msg->header;
    for (unsigned int i = 0; i < imgpoint_msg->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = imgpoint_msg->points[i].x;
        p_3d.y = imgpoint_msg->points[i].y;
        p_3d.z = imgpoint_msg->points[i].z;
        Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
        geometry_msgs::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    pub_imgpoint_cloud.publish(point_cloud);
}

// only for visualization
void margin_point_callback(const sensor_msgs::PointCloudConstPtr &point_msg)
{
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = point_msg->header;
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg->points[i].x;
        p_3d.y = point_msg->points[i].y;
        p_3d.z = point_msg->points[i].z;
        Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
        geometry_msgs::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    // pub_margin_cloud.publish(point_cloud);
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
void omni_sync_process()
{
    bool bSync = false;
    while (1)
    {

        std_msgs::Header header;
        double time = 0;
        std::vector<cv::Mat> image_seq(NUM_OF_CAM);
        img_buf_mutex.lock(); //  m_buf
        if (!img0_buf.empty() && !img1_buf.empty() && !img2_buf.empty() && !img3_buf.empty())
        {
            bSync = true;
            double time0 = img0_buf.front()->header.stamp.toSec();
            double time1 = img1_buf.front()->header.stamp.toSec();
            double time2 = img2_buf.front()->header.stamp.toSec();
            double time3 = img3_buf.front()->header.stamp.toSec();

            double max_time = std::max(std::max(time0, time1), std::max(time2, time3));
            if (time0 < max_time - 0.1)
            {
                bSync = false;
                img0_buf.pop();
                ROS_WARN("throw img0");
            }
            if (time1 < max_time - 0.1)
            {
                bSync = false;
                img1_buf.pop();
                ROS_WARN("throw img1");
            }
            if (time2 < max_time - 0.1)
            {
                bSync = false;
                img2_buf.pop();
                ROS_WARN("throw img2");
            }
            if (time3 < max_time - 0.1)
            {
                bSync = false;
                img3_buf.pop();
                ROS_WARN("throw img3");
            }
            if (bSync && !img0_buf.empty() && !img1_buf.empty() && !img2_buf.empty() && !img3_buf.empty())
            {

                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                image_seq[0] = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
                image_seq[1] = getImageFromMsg(img1_buf.front());
                img1_buf.pop();
                image_seq[2] = getImageFromMsg(img2_buf.front());
                img2_buf.pop();
                image_seq[3] = getImageFromMsg(img3_buf.front());
                img3_buf.pop();
                // std::cout << "omni sync pass!" << std::endl;
                // hcon_image = cv::Mat();
                // cv::hconcat(image_seq, hcon_image);
                // cv::imshow("omni", con_image);
                // cv::waitKey(1);
            }
        }

        img_buf_mutex.unlock();
        if (bSync && 4 == image_seq.size())
            omni_image_mat_buf.push(make_tuple(time, image_seq[0], image_seq[1], image_seq[2], image_seq[3]));

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

// extract images with same timestamp from two topics
void stereo_sync_process()
{

    while (1)
    {
        // printf("stereo start! \n");
        std::vector<cv::Mat> image_seq(2);
        std_msgs::Header header;
        double time = 0;
        m_buf.lock();
        if (!img0_buf.empty() && !img1_buf.empty())
        {
            double time0 = img0_buf.front()->header.stamp.toSec();
            double time1 = img1_buf.front()->header.stamp.toSec();
            // 0.003s sync tolerance
            if (time0 < time1 - 0.003)
            {
                img0_buf.pop();
                printf("throw img0\n");
            }
            else if (time0 > time1 + 0.003)
            {
                img1_buf.pop();
                printf("throw img1\n");
            }
            else
            {
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                image_seq[0] = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
                image_seq[1] = getImageFromMsg(img1_buf.front());
                img1_buf.pop();
                // printf("find img0 and img1\n");
            }
        }
        m_buf.unlock();
        if (!image_seq[0].empty())
            stereo_image_mat_buf.push(make_tuple(time, image_seq[0], image_seq[1]));
        // inputImage(time, image0, image1);

        std::chrono::milliseconds dura(10);
        std::this_thread::sleep_for(dura);
    }
}

void slideWindowNew()
{
    sum_of_front++; // 删除最新帧的帧数
    f_manager.removeFront(frame_count);
}

void slideWindowOld()
{
    sum_of_back++; // 删除最老帧的帧数
    f_manager.removeBack();

    // bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    // if (shift_depth)
    // {
    //     Matrix3d R0, R1;
    //     Vector3d P0, P1;
    //     R0 = back_R0 * R_i_c[0];
    //     R1 = Rs[0] * R_i_c[0];
    //     P0 = back_P0 + back_R0 * t_i_c[0];
    //     P1 = ts[0] + Rs[0] * t_i_c[0];
    //     f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    // }
    // else
    //     f_manager.removeBack();
}

void slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD) // 删除最老帧
    {
        // double t_0 = Headers[0];
        // back_R0 = Rs[0];
        // back_P0 = ts[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                // Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                ts[i].swap(ts[i + 1]);
            }
            // Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            ts[WINDOW_SIZE] = ts[WINDOW_SIZE - 1]; // ts[WINDOW_SIZE] 这一帧记录的位姿交换后是最后的，被覆盖掉，其实也没有用到这一帧
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            slideWindowOld();
        }
    }
    else // SECOND NEW 删除次新帧
    {
        if (frame_count == WINDOW_SIZE)
        {
            // Headers[frame_count - 1] = Headers[frame_count];
            ts[frame_count - 1] = ts[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            slideWindowNew();
        }
    }
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

void outliersRejection(set<int> &removeIndex)
{
    // return;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        feature_index++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            if (depth < 0.1)
                continue;

            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                // double tmp_error = reprojectionError(Rs[imu_i], ts[imu_i], R_i_c[0], t_i_c[0],
                //                                      Rs[imu_j], ts[imu_j], R_i_c[0], t_i_c[0],
                //                                      depth, pts_i, pts_j);
                double tmp_error = reprojectionError(it_per_id.position,
                                                     Rs[imu_j], ts[imu_j], R_i_c[0], t_i_c[0],
                                                     depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if (STEREO && it_per_frame.is_stereo)
            {

                Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j)
                {
                    double tmp_error = reprojectionError(it_per_id.position,
                                                         Rs[imu_j], ts[imu_j], R_i_c[1], t_i_c[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(it_per_id.position,
                                                         Rs[imu_j], ts[imu_j], R_i_c[1], t_i_c[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
            }
        }
        double ave_err = err / errCnt;
        if (ave_err * FOCAL_LENGTH > 3)
        {
            removeIndex.insert(it_per_id.feature_id);
            // printf("removeIndex %d, error %f\n", it_per_id.feature_id, ave_err);
        }
    }
    // printf("removeIndex size %d\n", removeIndex.size());
}

void addFeature(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &feature, const double header)
{
    ROS_DEBUG("new feature coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", feature.size());

    if (f_manager.addFeatureCheckParallax(frame_count, feature, 0.0))
    {
        marginalization_flag = MARGIN_OLD;
        // printf("MARGIN_OLD keyframe\n");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        // printf("MARGIN_SECOND_NEW non-keyframe\n");
    }

    if (frame_count < WINDOW_SIZE)
    {
        frame_count++;
        int prev_frame = frame_count - 1;
        ts[frame_count] = ts[prev_frame];
        Rs[frame_count] = Rs[prev_frame];
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());

    // check feature and determine if it is the keyframe

    // Headers[frame_count] = header;

    // prepare output of VINS
    // key_poses.clear();
    // for (int i = 0; i <= WINDOW_SIZE; i++)
    //     key_poses.push_back(ts[i]);

    // last_R = Rs[WINDOW_SIZE];
    // last_P = ts[WINDOW_SIZE];
    // last_R0 = Rs[0];
    // last_P0 = ts[0];
    // updateLatestStates();
}

void processFeature(PointCloudXYZI::Ptr points_triangulate)
{
    TicToc t_solve;
    FeatureData extracted_points;
    f_manager.triangulate(frame_count, ts, Rs, t_i_c, R_i_c, points_triangulate, extracted_points);
    featureTracker.final_points[0] = extracted_points;
    set<int> removeIndex;
    outliersRejection(removeIndex);
    f_manager.removeOutlier(removeIndex);
    featureTracker.removeOutliers(removeIndex);
    // predictPtsInNextFrame();

    ROS_DEBUG("solver costs: %fms", t_solve.toc());

    slideWindow();
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

void updateDataParted(const std::vector<FeatureData> &data, const std::vector<Eigen::Matrix<float, 259, Eigen::Dynamic>> &track_feature_seq, std::vector<FeatureDataParted> &data_part)
{

    for (int k = 0; k < NUM_OF_CAM; k++)
    {
        data_part[k].points_with_depth.clear();
        data_part[k].points_no_depth.clear();
        std::cout << "data size:" << data.size() << " track_feature_seq cols:" << track_feature_seq[k].cols() << std::endl;
        for (int j = 0; j < data[k].size(); j++)
        {
            auto it = data[k][j];
            it.score = track_feature_seq[k](0, j);
            // it.keypoint_2d_uv.pt = cv::Point2f((track_feature_seq[k](1, j)), (track_feature_seq[k](2, j)));
            it.spp_feature = track_feature_seq[k].col(j);
            if (it.point_depth > 0)
            {
                data_part[k].points_with_depth.push_back(it);
            }
            else
            {
                data_part[k].points_no_depth.push_back(it);
            }
        }
    }
}

void updateDataParted(const std::vector<FeatureData> &data, std::vector<FeatureDataParted> &data_part)
{
    for (int k = 0; k < NUM_OF_CAM; k++)
    {
        data_part[k].points_with_depth.clear();
        data_part[k].points_no_depth.clear();
        std::cout << "data size:" << data.size() << std::endl;
        for (int j = 0; j < data[k].size(); j++)
        {
            auto it = data[k][j];
            // it.score = shiTomasiScore(it.keypoint_2d_uv.pt, data[k][j].image);
            // it.keypoint_2d_uv.pt = cv::Point2f((track_feature_seq[k](1, j)), (track_feature_seq[k](2, j)));
            // it.spp_feature = track_feature_seq[k].col(j);
            if (it.point_depth > 0)
            {
                data_part[k].points_with_depth.push_back(it);
            }
            else
            {
                data_part[k].points_no_depth.push_back(it);
            }
        }
    }
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

// TODO: 自己做同步
void process_mapping()
{
    std::vector<std::vector<cv::Point2f>> cur_pts_seq(NUM_OF_CAM);
    prev_image_seq.resize(NUM_OF_CAM);
    // std::vector<FeatureDataParted> prev_spp_points;
    std::vector<FeatureDataParted> track_spp_points(NUM_OF_CAM);
    // prev_spp_points.resize(NUM_OF_CAM);
    std::vector<FeatureData> cur_final_points(NUM_OF_CAM);

    while (true)
    {
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> feature;
        std::tuple<double, cv::Mat, cv::Mat> stereo_image_mat;
        std::tuple<double, cv::Mat, cv::Mat, cv::Mat, cv::Mat> omni_image_mat;
        // cv::Mat image;
        // cv::Mat image_seq[0], image_seq[1], image_seq[2], image_seq[3];
        std::vector<std::map<Eigen::Vector2i, FeatureProperty, Vector2iCompare>> image_depth_map_vec(NUM_OF_CAM);
        std::vector<std::map<Eigen::Vector2i, FeatureProperty, Vector2iCompare>> image_spp_map_vec(NUM_OF_CAM);
        std::vector<cv::Mat> image_seq(NUM_OF_CAM);
        image_seq.resize(NUM_OF_CAM);
        sensor_msgs::PointCloud2ConstPtr lidar_point_msg = NULL;
        sensor_msgs::PointCloud2 pub_point_msg;
        sensor_msgs::ImageConstPtr img_msg = NULL;
        nav_msgs::Odometry::ConstPtr pose_msg = NULL;
        bool IS_KEYFRAME = false;
        double t_track = 0, t_sppextract = 0, t_lidarproject = 0., t_addKeyFrame = 0., t_processFeature = 0., t_featureTrack = 0.;
        bool bSync = false;
        // find out the messages with same time stamp
        // 根据相机数量写不同的同步，最好格式统一，但是不再需要共视 NUM_OF_CAM

        if (1 == NUM_OF_CAM) // mono
        {
            m_buf.lock(); //  m_buf
            // ros::Header header;
            double time = 0;

            if (!img0_buf.empty() && !lidar_point_buf.empty() && !pose_buf.empty())
            {
                bSync = true;
                double time0 = img0_buf.front()->header.stamp.toSec();
                // double time1 = img1_buf.front()->header.stamp.toSec();
                // double time2 = img2_buf.front()->header.stamp.toSec();
                // double time3 = img3_buf.front()->header.stamp.toSec();
                double time4 = lidar_point_buf.front()->header.stamp.toSec();
                double time5 = pose_buf.front()->header.stamp.toSec();

                double max_time = std::max(std::max(time0, time4), time5);
                if (time0 < max_time - 0.04)
                {
                    bSync = false;
                    img0_buf.pop();
                    ROS_WARN("throw img0");
                }
                if (time4 < max_time - 0.04)
                {
                    bSync = false;
                    lidar_point_buf.pop();
                    ROS_WARN("throw point");
                }
                if (time5 < max_time - 0.04)
                {
                    bSync = false;
                    pose_buf.pop();
                    ROS_WARN("throw pose");
                }
                if (bSync && !img0_buf.empty())
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    image_seq[0] = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    pose_msg = pose_buf.front();
                    pose_buf.pop();
                    lidar_point_msg = lidar_point_buf.front();
                    lidar_point_buf.pop();
                    std::cout << "omni sync pass!" << std::endl;
                }
            }
            m_buf.unlock();
        }
        else if (2 == NUM_OF_CAM)
        {
            m_buf.lock(); //  m_buf
            // ros::Header header;
            double time = 0;

            if (!img0_buf.empty() && !img1_buf.empty() && !lidar_point_buf.empty() && !pose_buf.empty())
            {
                bSync = true;
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();
                // double time2 = img2_buf.front()->header.stamp.toSec();
                // double time3 = img3_buf.front()->header.stamp.toSec();
                double time4 = lidar_point_buf.front()->header.stamp.toSec();
                double time5 = pose_buf.front()->header.stamp.toSec();

                double max_time = std::max(std::max(time0, time1), std::max(time4, time5));
                if (time0 < max_time - 0.04)
                {
                    bSync = false;
                    img0_buf.pop();
                    // ROS_WARN("throw img0");
                }
                if (time1 < max_time - 0.04)
                {
                    bSync = false;
                    img1_buf.pop();
                    // ROS_WARN("throw img1");
                }

                if (time4 < max_time - 0.04)
                {
                    bSync = false;
                    lidar_point_buf.pop();
                    // ROS_WARN("throw point");
                }
                if (time5 < max_time - 0.04)
                {
                    bSync = false;
                    pose_buf.pop();
                    // ROS_WARN("throw pose");
                }
                if (bSync && !img0_buf.empty() && !img1_buf.empty())
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    image_seq[0] = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image_seq[1] = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
                    pose_msg = pose_buf.front();
                    pose_buf.pop();
                    lidar_point_msg = lidar_point_buf.front();
                    lidar_point_buf.pop();
                    // std::cout << "omni sync pass!" << std::endl;
                }
            }
            m_buf.unlock();
        }
        else if (4 == NUM_OF_CAM)
        {
            m_buf.lock(); //  m_buf
            // ros::Header header;
            double time = 0;

            if (!img0_buf.empty() && !img1_buf.empty() && !img2_buf.empty() && !img3_buf.empty() && !lidar_point_buf.empty() && !pose_buf.empty())
            {
                bSync = true;
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();
                double time2 = img2_buf.front()->header.stamp.toSec();
                double time3 = img3_buf.front()->header.stamp.toSec();
                double time4 = lidar_point_buf.front()->header.stamp.toSec();
                double time5 = pose_buf.front()->header.stamp.toSec();

                double max_time = std::max(std::max(std::max(time0, time1), std::max(time2, time3)), std::max(time4, time5));
                if (time0 < max_time - 0.04)
                {
                    bSync = false;
                    img0_buf.pop();
                    // ROS_WARN("throw img0");
                }
                if (time1 < max_time - 0.04)
                {
                    bSync = false;
                    img1_buf.pop();
                    // ROS_WARN("throw img1");
                }
                if (time2 < max_time - 0.04)
                {
                    bSync = false;
                    img2_buf.pop();
                    // ROS_WARN("throw img2");
                }
                if (time3 < max_time - 0.04)
                {
                    bSync = false;
                    img3_buf.pop();
                    // ROS_WARN("throw img3");
                }
                if (time4 < max_time - 0.04)
                {
                    bSync = false;
                    lidar_point_buf.pop();
                    // ROS_WARN("throw point");
                }
                if (time5 < max_time - 0.04)
                {
                    bSync = false;
                    pose_buf.pop();
                    // ROS_WARN("throw pose");
                }
                if (bSync && !img0_buf.empty() && !img1_buf.empty() && !img2_buf.empty() && !img3_buf.empty())
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    image_seq[0] = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image_seq[1] = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
                    image_seq[2] = getImageFromMsg(img2_buf.front());
                    img2_buf.pop();
                    image_seq[3] = getImageFromMsg(img3_buf.front());
                    img3_buf.pop();
                    pose_msg = pose_buf.front();
                    pose_buf.pop();
                    lidar_point_msg = lidar_point_buf.front();
                    lidar_point_buf.pop();
                    // std::cout << "omni sync pass!" << std::endl;
                }
            }
            m_buf.unlock();
        }

        if (pose_msg != NULL)
        {
            // printf(" sync done \n");
            // printf(" pose time %f \n", pose_msg->header.stamp.toSec());
            // printf(" point time %f \n", lidar_point_msg->header.stamp.toSec());

            // build keyframe， using keyframe pose from vio
            Vector3d lio_T = Vector3d(pose_msg->pose.pose.position.x,
                                      pose_msg->pose.pose.position.y,
                                      pose_msg->pose.pose.position.z);
            Matrix3d lio_R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                         pose_msg->pose.pose.orientation.x,
                                         pose_msg->pose.pose.orientation.y,
                                         pose_msg->pose.pose.orientation.z)
                                 .toRotationMatrix();
            Matrix4f T_ijw_i0w = Matrix4f::Identity();
            T_ijw_i0w.block<3, 3>(0, 0) = lio_R.cast<float>();
            T_ijw_i0w.block<3, 1>(0, 3) = lio_T.cast<float>();
            // blur judge if extract
            double blur_score = 0, relative_blur_score = -100;
            double mean_spp_score = 0, tracking_score = 0.;
            for (int k = 0; k < NUM_OF_CAM; k++)
            {
                blur_score += laplacian(image_seq[k]);
            }
            img_blur_deq.push_back(blur_score);
            if (img_blur_deq.size() > WINDOW_SIZE)
            {
                img_blur_deq.pop_front();
                relative_blur_score = 0;
                for (int i = 0; i < img_blur_deq.size(); i++)
                {
                    relative_blur_score += (blur_score - img_blur_deq[i]);
                }
            }
            TicToc tic_track;
            std::vector<cv::Mat> image_seq_track(NUM_OF_CAM);
            std::vector<std::vector<cv::KeyPoint>> track_keypoints_seq(NUM_OF_CAM);
            std::vector<std::vector<cv::Point2f>> prev_pts_seq(NUM_OF_CAM);
            std::vector<cv::Mat> track_desc_mat_seq(NUM_OF_CAM);
            cv::Mat imgTrack;

            // update cur_pts_seq
            for (int k = 0; k < NUM_OF_CAM; k++)
            {
                for (int i = 0; i < cur_final_points[k].size(); i++)
                {
                    prev_pts_seq[k].emplace_back(cur_final_points[k][i].keypoint_2d_uv.pt);
                    cv::KeyPoint kp(cur_final_points[k][i].keypoint_2d_uv.pt, 1.0f); // octave改变了会有问题
                    track_keypoints_seq[k].emplace_back(kp);
                }
            }
            // std::vector<FeatureData> cur_track_points(NUM_OF_CAM);
            // prev_pts_seq.swap(cur_pts_seq); // update prev_pts_seq before tracking
            // prev_pts_seq = cur_pts_seq;
            // compare last_image_seq and image_seq using LK-OpticalFlow
            if (prev_image_seq.size())
            {
                for (int k = 0; k < NUM_OF_CAM; k++)
                {
                    cv::Mat prev_img = prev_image_seq[k];
                    cv::Mat cur_img = image_seq[k];
                    cur_pts_seq[k].clear();
                    cv::cvtColor(image_seq[k], image_seq_track[k], CV_GRAY2RGB);
                    assert(image_seq_track[k].rows == ROW && image_seq_track[k].cols == COL);
                    if (prev_pts_seq[k].size() > 0)
                    {
                        int prev_size = prev_pts_seq[k].size();
                        TicToc t_o;
                        vector<uchar> status;
                        vector<float> err;
                        cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts_seq[k], cur_pts_seq[k], status, err, cv::Size(21, 21), 3);
                        // reverse check
                        if (FLOW_BACK)
                        {
                            vector<uchar> reverse_status;
                            vector<cv::Point2f> reverse_pts = prev_pts_seq[k];
                            cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts_seq[k], reverse_pts, reverse_status, err, cv::Size(21, 21), 1,
                                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                            // cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts_seq[k], reverse_pts, reverse_status, err, cv::Size(21, 21), 3);
                            for (size_t i = 0; i < status.size(); i++)
                            {
                                if (status[i] && reverse_status[i] && featureTracker.distance(prev_pts_seq[k][i], reverse_pts[i]) <= 0.5)
                                {
                                    status[i] = 1;
                                    // 更新pt坐标

                                    cur_final_points[k][i].keypoint_2d_uv.pt = cur_pts_seq[k][i];
                                    cur_final_points[k][i].track_cnt++;

                                    if (1 == DETECTOR)
                                    {
                                        track_keypoints_seq[k][i].pt = cur_pts_seq[k][i];
                                    }
                                }
                                else
                                    status[i] = 0;
                            }
                        }

                        // for (int i = 0; i < int(cur_pts_seq[k].size()); i++)
                        //     if (status[i] && !featureTracker.inBorder(cur_pts_seq[k][i]))
                        //         status[i] = 0;
                        reduceVector(prev_pts_seq[k], status);
                        reduceVector(cur_pts_seq[k], status);
                        reduceVector(cur_final_points[k], status);

                        if (1 == DETECTOR)
                        {
                            reduceVector(track_keypoints_seq[k], status);
                            feature_detector->_brief_extractor->compute(image_seq[k], track_keypoints_seq[k], track_desc_mat_seq[k]);
                            // TODO 可以再加速，避免clear和pushback
                            int j = 0;
                            vector<uchar> compute_status;
                            for (int i = 0; i < cur_final_points[k].size(); i++)
                            {
                                if (j < track_keypoints_seq[k].size())
                                {
                                    if (cur_final_points[k][i].keypoint_2d_uv.pt == track_keypoints_seq[k][j].pt)
                                    {
                                        cur_final_points[k][i].score = shiTomasiScore(image_seq[k], cur_final_points[k][i].keypoint_2d_uv.pt.x, cur_final_points[k][i].keypoint_2d_uv.pt.y);
                                        cur_final_points[k][i].brief_descriptor = track_desc_mat_seq[k].row(j);
                                        compute_status.push_back(1);
                                        j++;
                                    }
                                    else
                                    {
                                        compute_status.push_back(0);
                                    }
                                }
                            }
                            reduceVector(cur_final_points[k], compute_status);
                        }

                        // reduceVector(ids, status);
                        // reduceVector(track_cnt, status);
                        // ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
                        // printf("track cnt %d\n", (int)ids.size());
                        if (SHOW_TRACK)
                        {
                            for (int j = 0; j < cur_final_points[k].size(); j++)
                            {
                                double len = std::min(1.0, 1.0 * cur_final_points[k][j].track_cnt / 4);                                                    // 3次就认为很多
                                cv::circle(image_seq_track[k], cur_final_points[k][j].keypoint_2d_uv.pt, 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 1); // blue->purple->red
                            }
                        }
                        int tracked_cur_size = cur_final_points[k].size();
                        tracking_score += prev_size - tracked_cur_size;
                    }
                }
                if (SHOW_TRACK)
                {
                    cv::hconcat(image_seq_track, imgTrack);
                    cv::imshow("Track", imgTrack);
                }
            }

            prev_image_seq = image_seq;
            t_track = tic_track.toc();

            double move_dis = (lio_T - last_T).norm();
            Eigen::Matrix3d relative_R = last_R.transpose() * lio_R;
            double turn_yaw = Utility::normalizeAngle(Utility::R2ypr(relative_R).x());
            // TODO: 距离+模糊+视差/公视
            // 检查模糊和跟踪的score对不对，现在不太影响建图
            double keyframe_score = MAPP_DIS_FACTOR * (move_dis + 1) + MAPP_ROT_FACTOR * abs(turn_yaw) + 10 * relative_blur_score + 2 * tracking_score;

            if (1 == BUILD_KEYFRAME) // auto keyframe
            {
                printf("keyframe_score %f, map_dis_score%f, map_yaw_thresh%f, blur_score %f, tracking_score%f \n", keyframe_score, MAPP_DIS_FACTOR * (move_dis + 1), MAPP_ROT_FACTOR * abs(turn_yaw), 10 * relative_blur_score, 2 * tracking_score);
                if (keyframe_score > 1500 - MAPPING_MODE * 500) // lidar : 1500
                {
                    IS_KEYFRAME = true;
                }
                else
                {
                    IS_KEYFRAME = false;
                }
            }
            else if (2 == BUILD_KEYFRAME) // mannual keyframe , for debug
            {
                IS_KEYFRAME = COMMAND_KEYFRAME;
            }

            FeatureDataParted extracted_points;
            std::vector<uchar> projected_pts_status; // if some point project into same pixel
            // extracted_points.reserve(MAX_CNT);       // reverse space to save time
            // image_depth_map_vec.clear();
            // image_depth_map_vec.resize(NUM_OF_CAM);
            // image_spp_map_vec.clear();
            // image_spp_map_vec.resize(NUM_OF_CAM);

            // convert lidar point to image_seq[0] (需要注意这里的lidar_point_msg就是在机体坐标系下的点云，不过pose和点云是同步的，也好转)
            PointCloudXYZI::Ptr points_inFOV(new PointCloudXYZI);
            PointCloudXYZI::Ptr points_corner(new PointCloudXYZI);
            // printf("image_seq[0] size %d, %d \n", image_seq[0].rows, image_seq[0].cols);
            // cv::Mat image_rgb;
            // cv::cvtColor(image_seq[0], image_rgb, CV_GRAY2RGB);
            TicToc tic_sppextract;
            std::vector<Eigen::Matrix<float, 259, Eigen::Dynamic>> extract_features_seq(NUM_OF_CAM); // 259 = score + uv + descriptor
            std::vector<Eigen::Matrix<float, 259, Eigen::Dynamic>> track_feature_seq(NUM_OF_CAM);
            std::vector<std::vector<cv::KeyPoint>> extract_keypoint_seq(NUM_OF_CAM);
            // blur score and spp features

            std::vector<cv::Mat> image_seq_extract(NUM_OF_CAM);
            cv::Mat imgExtract;
            std::vector<cv::Mat> desc_mat_seq(NUM_OF_CAM);
            std::vector<cv::Mat> wait_project_mask_vec(NUM_OF_CAM);
            if (IS_KEYFRAME)
            {
                // use gpu in one time
                if (0 == DETECTOR)
                {
                    feature_detector->DetectMultiwithTrack(image_seq, cur_final_points, extract_features_seq, track_feature_seq);
                    updateDataParted(cur_final_points, track_feature_seq, track_spp_points); // track_feature_seq -> 只要有深度的点 or 全都要，根据score筛选
                    for (int k = 0; k < NUM_OF_CAM; k++)
                    {
                        // feature_detector->Detect(image_seq[k], extract_features_seq[k]);
                        FeatureProperty tmp_property;
                        cv::cvtColor(image_seq[k], image_seq_extract[k], CV_GRAY2RGB);
                        assert(image_seq_extract[k].rows == ROW && image_seq_extract[k].cols == COL);
                        cv::Mat wait_project_mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(0));
                        wait_project_mask_vec[k] = wait_project_mask;
                        for (int j = 0; j < extract_features_seq[k].cols(); j++)
                        {
                            // auto this_spp_feature = extract_features_seq[k].col(j);
                            tmp_property.cam_id = k;
                            tmp_property.score = extract_features_seq[k](0, j);
                            tmp_property.keypoint_2d_uv.pt = cv::Point2f(round(extract_features_seq[k](1, j)), round(extract_features_seq[k](2, j)));

                            // tmp_property.spp_feature = extract_features_seq[k].block<256, 1>(3, j);
                            tmp_property.spp_feature = extract_features_seq[k].col(j);
                            mean_spp_score += tmp_property.score;

                            cv::circle(wait_project_mask_vec[k], tmp_property.keypoint_2d_uv.pt, 2, 255, -1);

                            // 对邻格也加入mask
                            for (int u = -PWA_size; u <= PWA_size; u++)
                            {
                                for (int v = -PWA_size; v <= PWA_size; v++)
                                {
                                    Eigen::Vector2i tmp_uv_int(round(extract_features_seq[k](1, j)) + u, round(extract_features_seq[k](2, j)) + v);
                                    image_spp_map_vec[k][tmp_uv_int] = tmp_property;
                                }
                            }

                            if (SHOW_TRACK)
                                cv::circle(image_seq_extract[k], cv::Point(tmp_property.keypoint_2d_uv.pt.x, tmp_property.keypoint_2d_uv.pt.y), 2, cv::Scalar(0, 255, 255), 2);
                        }
                        mean_spp_score /= extract_features_seq[k].cols();
                    }
                }
                else if (1 == DETECTOR)
                {
                    // std::cout << " image_vec.size " << image_seq.size() << std::endl;
                    feature_detector->DetectMulti(image_seq, extract_keypoint_seq, desc_mat_seq);
                    updateDataParted(cur_final_points, track_spp_points);
                    for (int k = 0; k < NUM_OF_CAM; k++)
                    {
                        // feature_detector->Detect(image_seq[k], extract_features_seq[k]);
                        FeatureProperty tmp_property;
                        cv::cvtColor(image_seq[k], image_seq_extract[k], CV_GRAY2RGB);
                        assert(image_seq_extract[k].rows == ROW && image_seq_extract[k].cols == COL);
                        cv::Mat wait_project_mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(0));
                        wait_project_mask_vec[k] = wait_project_mask;
                        for (int j = 0; j < extract_keypoint_seq[k].size(); j++)
                        {
                            // auto this_spp_feature = extract_features_seq[k].col(j);
                            tmp_property.cam_id = k;
                            tmp_property.score = extract_keypoint_seq[k][j].response;
                            tmp_property.keypoint_2d_uv.pt = extract_keypoint_seq[k][j].pt;
                            tmp_property.brief_descriptor = desc_mat_seq[k].row(j);
                            // tmp_property.spp_feature = extract_features_seq[k].block<256, 1>(3, j);
                            // tmp_property.spp_feature = extract_features_seq[k].col(j);
                            mean_spp_score += tmp_property.score;

                            cv::circle(wait_project_mask_vec[k], tmp_property.keypoint_2d_uv.pt, 2, 255, -1);

                            // 对邻格也加入mask
                            for (int u = -PWA_size; u <= PWA_size; u++)
                            {
                                for (int v = -PWA_size; v <= PWA_size; v++)
                                {
                                    Eigen::Vector2i tmp_uv_int(round(extract_keypoint_seq[k][j].pt.x) + u, round(extract_keypoint_seq[k][j].pt.y) + v);
                                    image_spp_map_vec[k][tmp_uv_int] = tmp_property;
                                }
                            }

                            if (SHOW_TRACK)
                                cv::circle(image_seq_extract[k], cv::Point(tmp_property.keypoint_2d_uv.pt.x, tmp_property.keypoint_2d_uv.pt.y), 2, cv::Scalar(0, 255, 255), 2);
                        }
                        mean_spp_score /= extract_features_seq[k].cols();
                    }

                    // std::cout << "extract_features_seq size:" << extract_features_seq[k].cols() << " image_spp_map_vec size: " << image_spp_map_vec[k].size() << std::endl;
                }
            }

            t_sppextract = tic_sppextract.toc();
            // std::cout << "blur_score: " << blur_score << " relative_blur_score: " << relative_blur_score << std::endl;

            // double blur_score = laplacian(image_seq[0]); //laplacian(image_seq[0]);
            // printf("featureTracker time: %f\n", featureTrackerTime.toc());

            // lidar points project
            PointCloudXYZI::Ptr local_map_inworld(new PointCloudXYZI);
            PointCloudXYZI::Ptr cloud_inimu(new PointCloudXYZI);
            PointCloudXYZI::Ptr project_cloud_inworld(new PointCloudXYZI);
            std::vector<FeatureDataParted> extract_spp_points(NUM_OF_CAM);

            // ablation here
            TicToc tic_lidarproject;
            if (0 == MAPPING_MODE && IS_KEYFRAME)
            {
                pcl::fromROSMsg(*lidar_point_msg, *local_map_inworld);
                *cloud_inimu = *local_map_inworld;
                // 转到body坐标系
                pcl::transformPointCloud(*cloud_inimu, *cloud_inimu, T_ijw_i0w.inverse());
                std::cout << "cloud_inimu size: " << cloud_inimu->points.size() << std::endl;
                for (int i = 0; i < cloud_inimu->points.size(); i++)
                {
                    short cam_id = 0;
                    PointType point = cloud_inimu->points[i];
                    Vector3d point_imu(point.x, point.y, point.z);
                    Vector3d point_cam;                                   // point_cam
                    float angle = atan2(point.y, point.x) * 180.0 / M_PI; // 计算点的角度
                    if (angle < 0)
                        angle += 360; // 转换为0到360度的范围
                    
                    // four cam cases
                    // if (angle <= 45 || angle >= 315)
                    // {
                    //     // cloud_incam[0]->points.push_back(point); // 前
                    //     cam_id = 0;
                    // }
                    // else if (angle >= 135 && angle <= 225) // 后
                    // {
                    //     // cloud_incam[1]->points.push_back(point);
                    //     cam_id = 1;
                    // }
                    // else if (angle >= 45 && angle <= 135) // 左
                    // {
                    //     // cloud_incam[2]->points.push_back(point);
                    //     cam_id = 2;
                    // }
                    // else if (angle >= 225 && angle <= 315) // 右
                    // {
                    //     // cloud_incam[3]->points.push_back(point);
                    //     cam_id = 3;
                    // }
                    
                    // puyuan case
                    if(angle >= 45 && angle <= 135)
                    {
                        // cloud_incam[0]->points.push_back(point); // 左
                        cam_id = 0;
                    }

                    if (cam_id >= NUM_OF_CAM) // exceed max cam
                        continue;

                    // pcl::transformPointCloud(*cloud_incam[k], *cloud_incam[k], i_T_c.inverse());
                    point_cam = R_i_c[cam_id].transpose() * (point_imu - t_i_c[cam_id]); // imu-> cam
                    if (point_cam(2) < 0 || point_cam(2) >= 100)                         // 后面的看不到 , 太远的也过滤
                        continue;
                    FeatureProperty tmp_property;

                    Vector2d tmp_uv;
                    m_camera[cam_id]->spaceToPlane(point_cam, tmp_uv);
                    cv::Point2f tmp_uv_cv(tmp_uv(0), tmp_uv(1));
                    Vector2d tmp_2d_normal(point_cam(0) / point_cam(2), point_cam(1) / point_cam(2));
                    // std::cout<<tmp_2d_normal<<endl;
                    // 判断是否在fov内且mask内有值
                    if ((tmp_uv(0) > 0 && tmp_uv(0) < COL && tmp_uv(1) > 0 && tmp_uv(1) < ROW) && (!FAST_PROJECTION || wait_project_mask_vec[cam_id].at<uchar>(tmp_uv_cv) == 255)) //(isInFrame(tmp_uv))
                    {
                        // const float score = featureTracker.shiTomasiScore(image_seq[cam_id], tmp_uv(0), tmp_uv(1));
                        // if (score < SHI_THRESHOLD_SCORE)
                        //     continue;
                        Vector2i tmp_uv_int(round(tmp_uv(0)), round(tmp_uv(1)));
                        if (SHOW_TRACK)
                        {
                            double len = std::min(1.0, 1.0 * point_cam(2) / 20);
                            // 先col后row
                            // cv::Point2f hcon_point_uv(final_points[j].keypoint_2d_uv.pt.x + cam_id*COL , final_points[j].keypoint_2d_uv.pt.y  );
                            // 越绿表示深度越大
                            cv::circle(image_seq_extract[cam_id], cv::Point(round(tmp_uv(0)), round(tmp_uv(1))), 2, cv::Scalar(255 * (1 - len), 0, 255 * (len)), 1); // BGR
                        }

                        // second filter from extract_features_seq
                        if (image_spp_map_vec[cam_id].find(tmp_uv_int) == image_spp_map_vec[cam_id].end())
                            continue;

                        // tmp_property = image_spp_map_vec[cam_id][tmp_uv_int]; //not init here
                        tmp_property.keypoint_2d_uv.pt = cv::Point2f(round(tmp_uv(0)), round(tmp_uv(1)));
                        tmp_property.keypoint_2d_norm.pt = cv::Point2f(tmp_2d_normal(0), tmp_2d_normal(1));
                        tmp_property.point_3d_world = cv::Point3f(local_map_inworld->points[i].x, local_map_inworld->points[i].y, local_map_inworld->points[i].z);
                        tmp_property.point_intensity = local_map_inworld->points[i].intensity;
                        tmp_property.point_depth = point_cam(2);
                        // depth approximation

                        // tmp_property.cam_id = cam_id;
                        if (image_depth_map_vec[cam_id].find(tmp_uv_int) == image_depth_map_vec[cam_id].end() || -1 == image_depth_map_vec[cam_id][tmp_uv_int].point_depth) //
                        {
                            image_depth_map_vec[cam_id][tmp_uv_int] = tmp_property;
                        }
                        // 如果有重复的点，保留最近的点
                        else
                        {
                            if (point_cam(2) < image_depth_map_vec[cam_id][tmp_uv_int].point_depth)
                                // printf("replace point!!!\n");
                                image_depth_map_vec[cam_id][tmp_uv_int] = tmp_property;
                        }
                    }
                }

                for (int k = 0; k < NUM_OF_CAM; k++)
                {
                    // auto image_depth_map = image_depth_map_vec[k];
                    // int cnt_with_depth = 0;
                    int extract_size;
                    if (0 == DETECTOR)
                    {
                        extract_size = extract_features_seq[k].cols();
                    }
                    else if (1 == DETECTOR)
                    {
                        extract_size = extract_keypoint_seq[k].size();
                    }
                    for (int j = 0; j < extract_size; j++)
                    {
                        double depth_avg = 0.;
                        cv::Point3f point_3d_avg = cv::Point3f(0, 0, 0);
                        short cnt = 0;
                        bool bValid = false;
                        std::vector<double> depth_vec;
                        std::vector<cv::Point3f> point_3d_vec;
                        FeatureProperty tmp_spp_property, tmp_depth_property;
                        // tmp_depth_property for right depth
                        // 采用周围9个像素的平均值(非0)来近似
                        Vector2d tmp_uv;
                        if (0 == DETECTOR)
                        {
                            tmp_uv << extract_features_seq[k](1, j), extract_features_seq[k](2, j);
                        }
                        else if (1 == DETECTOR)
                        {
                            tmp_uv << extract_keypoint_seq[k][j].pt.x, extract_keypoint_seq[k][j].pt.y;
                        }

                        for (int u = -PWA_size; u <= PWA_size; u++)
                        {
                            for (int v = -PWA_size; v <= PWA_size; v++)
                            {
                                Eigen::Vector2i tmp_uv_int(round(tmp_uv(0)) + u, round(tmp_uv(1)) + v);
                                tmp_depth_property = image_depth_map_vec[k][tmp_uv_int];
                                if (tmp_depth_property.point_depth == -1) // no depth, skip this
                                    continue;
                                double this_depth = tmp_depth_property.point_depth;
                                depth_vec.push_back(this_depth);
                                point_3d_vec.push_back(tmp_depth_property.point_3d_world);
                                cnt++;
                            }
                        }

                        if (depth_vec.size() > 0)
                        {
                            double mean_depth = std::accumulate(depth_vec.begin(), depth_vec.end(), 0.0) / depth_vec.size();
                            double stdvar_depth = 0.0;
                            for (auto dv : depth_vec)
                                stdvar_depth += (dv - mean_depth) * (dv - mean_depth);
                            stdvar_depth = sqrt(stdvar_depth / depth_vec.size());

                            // 去除异常值
                            for (int d = 0; d < depth_vec.size(); d++)
                            {
                                if (depth_vec[d] > mean_depth + 3 * stdvar_depth || depth_vec[d] < mean_depth - 3 * stdvar_depth)
                                {
                                    depth_vec.erase(depth_vec.begin() + d);
                                    point_3d_vec.erase(point_3d_vec.begin() + d);
                                    d--;
                                }
                            }
                        }
                        if (depth_vec.size() > 0)
                        {
                            for (int d = 0; d < depth_vec.size(); d++)
                            {
                                depth_avg += depth_vec[d];
                                point_3d_avg += point_3d_vec[d];
                            }

                            point_3d_avg /= (short)depth_vec.size();
                            depth_avg /= depth_vec.size();
                            bValid = true;
                            // if(depth_vec.size() > 1)
                            // {
                            //     std::cout << "depth_vec.size() > 1" << std::endl;
                            // }
                        }

                        Eigen::Vector2i tmp_uv_int(round(tmp_uv(0)), round(tmp_uv(1)));
                        tmp_spp_property = image_spp_map_vec[k][tmp_uv_int];
                        if (bValid)
                        {
                            // assert(tmp_spp_property.spp_feature == extract_features_seq[k].block<256, 1>(3, j));
                            // update right depth and 3d point
                            // cnt_with_depth++;
                            tmp_spp_property.point_depth = depth_avg;
                            tmp_spp_property.point_3d_world = point_3d_avg;
                            Vector3d point_in_cam = tmp_spp_property.point_depth * Vector3d(tmp_spp_property.keypoint_2d_norm.pt.x, tmp_spp_property.keypoint_2d_norm.pt.y, 1);
                            PointType pointxyzi;
                            pointxyzi.x = point_in_cam.x();
                            pointxyzi.y = point_in_cam.y();
                            pointxyzi.z = point_in_cam.z();
                            pointxyzi.intensity = tmp_depth_property.point_intensity;
                            // 绘制符合要求的点
                            points_corner->push_back(pointxyzi);
                            tmp_spp_property.point_id = featureTracker.n_id++;
                            extract_spp_points[k].points_with_depth.push_back(tmp_spp_property); // 包括了有深度的
                            // featureTracker.track_cnt.push_back(1);
                        }
                        else
                        {
                            tmp_spp_property.point_id = featureTracker.n_id++;
                            extract_spp_points[k].points_no_depth.push_back(tmp_spp_property); // 包括了无深度的
                            // featureTracker.track_cnt.push_back(1);
                        }
                        //
                    }
                }
            }

            t_lidarproject = tic_lidarproject.toc();

            // pcl::transformPointCloud(*cloud_inimu, *cloud_inimu, i_T_c);
            // pcl::toROSMsg(*cloud_inimu, pub_point_msg);
            // pub_point_msg.header.frame_id = "world";
            // pub_point_cloud_cam.publish(pub_point_msg);
            // pcl::transformPointCloud(*points_corner, *points_corner, i_T_c_vec[0]);
            // pcl::toROSMsg(*points_corner, pub_point_msg);
            // pub_point_msg.header.frame_id = "world";
            // pub_point_cloud_inFOV.publish(pub_point_msg);
            // cloud_inimu->clear();

            // image reference_keypoints extract
            TicToc tic_featureTrack;

            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;

            double time = pose_msg->header.stamp.toSec();

            if (0 == MAPPING_MODE && IS_KEYFRAME)
            {
                // featureTracker.add_extract_points(extract_spp_points);
                // split cur_final_points into track_spp_points
                // updateDataParted(cur_final_points, track_spp_points);
                // track_feature_seq -> 只要有深度的点 or 全都要，根据score筛选
                // featureTracker.add_track_points(track_spp_points);
                // 每次KEYFRAME都会清空cur_pts_seq
                featureTracker.setMask(ROW, COL, extract_spp_points, track_spp_points, cur_final_points); // TODO: 感觉不太好优化，因为需要根据score来排序，筛的时候也许可以顺便排序，但不太好弄
                if (1 == DETECTOR)
                {
                    // feature_detector->extractBriefDescriptor(image_seq, cur_final_points);
                }
                if (SHOW_TRACK)
                {
                    featureTracker.drawLidarProjectPts(image_seq_extract);
                }
            }
            else if (1 == MAPPING_MODE)
            {
                featureFrame = featureTracker.trackImage(time, image_seq[0]);
                if (SHOW_TRACK)
                {
                    featureTracker.drawVisualTrackPts(image_seq_extract);
                }
            }

            if (SHOW_TRACK && IS_KEYFRAME)
            {
                // cv::putText(imgExtract, "blur_score:" + std::to_string(blur_score), cv::Point(10, 60), CV_FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2, 8);
                // cv::putText(imgExtract, "mean_shitomasi_score:" + std::to_string(mean_shitomasi_score), cv::Point(10, 60), CV_FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 2, 8);
                cv::hconcat(image_seq_extract, imgExtract);
                std_msgs::Header header;
                header.frame_id = "world";
                header.stamp = pose_msg->header.stamp;
                sensor_msgs::ImagePtr imgExtractMsg = cv_bridge::CvImage(header, "bgr8", imgExtract).toImageMsg();
                pub_image_track.publish(imgExtractMsg);
                // cv::resize(imgExtract,imgExtract, cv::Size(),0.5,0.5);
                cv::imshow("Project and Extract", imgExtract);
                // ostringstream path;
                // static int img_id = 0;
                // path << "/home/nv/Desktop/loop_output/loop_images/"
                //      << img_id++ << "-"
                //      << "imgExtract.jpg";
                // cv::imwrite(path.str().c_str(), imgExtract);
                // printf("featureFrame size %d\n", featureFrame.size());
            }

            t_featureTrack = tic_featureTrack.toc();

            // printf(" feature_list size %d \n", featureFrame.size());
            // printf(" lidar point size %d \n", lidar_point_msg->data.size());

            // triangulate
            // FeatureData merged_keyframe_points;
            // merged_keyframe_points = featureTracker.final_points;
            if (1 == MAPPING_MODE)
            {
                PointCloudXYZI::Ptr points_triangulate(new PointCloudXYZI);
                ts[frame_count] = lio_T;
                Rs[frame_count] = lio_R;
                addFeature(featureFrame, time);
                if (!IS_KEYFRAME)
                {
                    slideWindow();
                }
                else if (IS_KEYFRAME) // (lio_T - last_T).norm() > SKIP_DIST
                {
                    TicToc tic_processFeature;
                    processFeature(points_triangulate);
                    t_processFeature = tic_processFeature.toc();
                    // printf("points_triangulate size %d, extracted_points size %d \n", points_triangulate->size(), extracted_points.size());
                }
            }

            // pcl::transformPointCloud(*points_corner, *points_corner, T_ijw_i0w);
            // pcl::toROSMsg(*points_corner, pub_point_msg);
            // pub_point_msg.header.frame_id = "world";
            // pub_lidar_corner_cloud.publish(pub_point_msg);
            if (IS_KEYFRAME)
            {
                // searchByBRIEFDes 只依赖point_2d_uv，point_2d_norm， 这里预先跑一遍BRIEF描述子生成就可以了，这样就能过第一个筛选
                TicToc tic_addKeyFrame;

                printf("cur_lidar_pts size %d ", featureTracker.num_lidar_pts);
                // printf("merged_keyframe_points size %d \n", merged_keyframe_points.size());
                printf("Generating KeyFrame \n");

                // 关键帧除了位姿，还有三维点，二维点，二维点的id，二维点的归一化坐标
                for (int k = 0; k < NUM_OF_CAM; k++)
                {
                    // KeyFramePtr reference_keyframe = new KeyFrame(pose_msg->header.stamp.toSec(), frame_index, lio_T, lio_R, image_seq[k], cur_final_points[k], sequence);

                    // lio_R rots to cam_R (in imu coordinate)
                    Vector3d cam_t = lio_T;
                    Matrix3d cam_R = lio_R * cam_rot_R[k]; //* cam_rot_R[k]
                    KeyFramePtr reference_keyframe = std::make_shared<KeyFrame>(pose_msg->header.stamp.toSec(), frame_index, cam_t, cam_R, image_seq[k], cur_final_points[k], sequence, k);
                    // 先直接从vio拿到T和R
                    m_process.lock();
                    start_flag = 1;
                    // posegraph.addKeyFrame(reference_keyframe, 0); // 只建立关键帧，不进行优化
                    // posegraph.addKeyFrameIntoVoc(reference_keyframe); // 只建立关键帧，不进行优化
                    posegraph.addRefKeyFrame(reference_keyframe); // 只建立关键帧，不进行优化

                    // 其实提取描述子和加入database可以放在一起，现在是分开的，不过也没啥问题
                    m_process.unlock();

                    Eigen::Quaterniond cam_q(cam_R);
                    nav_msgs::Odometry::Ptr odom_msg(new nav_msgs::Odometry);
                    *odom_msg = *pose_msg;
                    odom_msg->pose.pose.orientation.x = cam_q.x();
                    odom_msg->pose.pose.orientation.y = cam_q.y();
                    odom_msg->pose.pose.orientation.z = cam_q.z();
                    odom_msg->pose.pose.orientation.w = cam_q.w();
                    if (0 == k)
                    {
                        pub_odometry_0.publish(odom_msg);
                    }
                    else if (1 == k)
                    {
                        pub_odometry_1.publish(odom_msg);
                    }
                    else if (2 == k)
                    {
                        pub_odometry_2.publish(odom_msg);
                    }
                    else if (3 == k)
                    {
                        pub_odometry_3.publish(odom_msg);
                    }
                }
                t_addKeyFrame = tic_addKeyFrame.toc();
                frame_index++;
                last_T = lio_T;
                last_R = lio_R;
                last_image_seq = image_seq;
                m_command.lock();
                COMMAND_KEYFRAME = false;
                m_command.unlock();
                IS_KEYFRAME = false;
            }
            cv::waitKey(1);
            Eigen::Quaterniond lio_q(lio_R);
            std::ofstream fout_mappingTime(MAPPING_TIME_LOG_PATH, std::ios::app);
            fout_mappingTime.setf(ios::fixed, ios::floatfield);
            fout_mappingTime.precision(9);
            fout_mappingTime << pose_msg->header.stamp.toSec() << " ";
            fout_mappingTime.precision(9);
            fout_mappingTime << lio_T(0) << " " << lio_T(1) << " " << lio_T(2) << " "                                                                    // 1-3
                             << lio_q.x() << " " << lio_q.y() << " " << lio_q.z() << " " << lio_q.w() << " "                                             // 4-7
                             << t_sppextract << " " << t_lidarproject << " " << t_addKeyFrame << " " << t_processFeature << " " << t_featureTrack << " " // 8-12
                             << endl;

            fout_mappingTime.close();
            // std::cout << " t_track " << t_track << " t_sppextract " << t_sppextract << " t_lidarproject " << t_lidarproject << " t_featureTrack " << t_featureTrack << " t_processFeature " << t_processFeature << " t_addKeyFrame " << t_addKeyFrame << std::endl;
            // test 1 pnp
            // std::vector<uchar> status;
            // Eigen::Vector3d PnP_T_old;
            // Eigen::Matrix3d PnP_R_old;
            // testPnPRANSAC(track_spp_points, status, R, T, PnP_T_old, PnP_R_old);
            // std::cout << "PnP_T_old: " << PnP_T_old.transpose() << std::endl;
            // std::cout << "PnP_R_old: \n"
            //           << PnP_R_old << std::endl;
            // std::cout << "True T: " << T.transpose() << std::endl;
            // std::cout << "True R: \n"
            //           << R << std::endl;

            // // test2 extract + pnp + show odom
            // // fast 提出来point_2d_uv，point_2d_normal
            // std::vector<cv::Point3f> point_3d_world;
            // std::vector<cv::Point2f> keypoint_2d_uv.pt;
            // std::vector<cv::Point2f> keypoint_2d_norm.pt;
            // std::vector<int> point_id;
            // // 新建一个query_test_kf，看下
            // KeyFramePtr query_test_kf = new KeyFrame(pose_msg->header.stamp.toSec(), 10000 + frame_index, T, R, image_seq[0], sequence);

            // // search in reference_keypoints
            // query_test_kf->findConnection(reference_keyframe);
            // Vector3d w_t_old, w_t_cur_calc, w_t_old_calc, vio_t_cur;
            // Matrix3d w_R_old, w_R_cur_calc, w_R_old_calc, vio_R_cur;
            // reference_keyframe->getVioPose(w_t_old, w_R_old);
            // Vector3d relative_t;
            // Quaterniond relative_q;
            // relative_t = query_test_kf->getLoopRelativeT();
            // relative_q = (query_test_kf->getLoopRelativeQ()).toRotationMatrix();

            // // bi_T_b0(w2)
            // w_t_old_calc = w_R_old * relative_t + w_t_old;
            // w_R_old_calc = w_R_old * relative_q;
            // // std::cout << "w_t_old_calc " << w_t_old_calc.transpose() << std::endl
            // //           << " w_R_old_calc \n"
            // //           << w_R_old_calc << std::endl;
            // Eigen::Quaterniond q(w_R_old_calc);
            // nav_msgs::Odometry odom;
            // odom.header.stamp = pose_msg->header.stamp;
            // odom.header.frame_id = "world";
            // odom.pose.pose.position.x = w_t_old_calc(0);
            // odom.pose.pose.position.y = w_t_old_calc(1);
            // odom.pose.pose.position.z = w_t_old_calc(2);
            // odom.pose.pose.orientation.x = q.x();
            // odom.pose.pose.orientation.y = q.y();
            // odom.pose.pose.orientation.z = q.z();
            // odom.pose.pose.orientation.w = q.w();
            // pub_odometry_testPnp.publish(odom);
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void command()
{
    while (1)
    {
        char c = getchar();
        if (c == 's')
        {
            m_process.lock();
            posegraph.savePoseGraph();
            m_process.unlock();
            printf("save pose graph finish\nyou can set 'load_previous_pose_graph' to 1 in the config file to reuse it next time\n");
            // printf("program shutting down...\n");
            // ros::shutdown();
        }
        if (c == 'n')
            new_sequence();
        if (c == 'k')
        {
            m_command.lock();
            COMMAND_KEYFRAME = true;
            m_command.unlock();
            printf("COMMAND_KEYFRAME is TRUE! \n");
        }

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
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
    MAPP_DIS_FACTOR = 100;
    MAPP_ROT_FACTOR = 20;
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
    std::string pkg_path = ros::package::getPath("loop_fusion");
    // string vocabulary_file = pkg_path + "/../support_files/point_voc_L4.bin";
    // cout << "vocabulary_file" << vocabulary_file << endl;
    // posegraph.initSppDatabase(vocabulary_file);

    // read cam parameters
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        fsSettings["image0_topic"] >> IMAGE0_TOPIC;
        fsSettings["image1_topic"] >> IMAGE1_TOPIC;
        fsSettings["image2_topic"] >> IMAGE2_TOPIC;
        fsSettings["image3_topic"] >> IMAGE3_TOPIC;
        fsSettings["relo_image0_topic"] >> RELO_IMAGE0_TOPIC;
        fsSettings["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
        // fsSettings["loop_output_path"] >> LOOP_RESULT_PATH;
        fsSettings["mapping_time_log_path"] >> MAPPING_TIME_LOG_PATH;
        // fsSettings["relo_time_log_path"] >> RELO_TIME_LOG_PATH;
        fsSettings["debug_image"] >> DEBUG_IMAGE;
        fsSettings["show_track"] >> SHOW_TRACK;
        fsSettings["fast_projection"] >> FAST_PROJECTION;
        fsSettings["mapping_mode"] >> MAPPING_MODE;
        fsSettings["detector"] >> DETECTOR;
        // fsSettings["matcher"] >> MATCHER;
        printf("MAPPING_MODE: %d\n", MAPPING_MODE);
        printf("DETECTOR: %d\n", DETECTOR);
        // printf("MATCHER: %d\n", MATCHER);

        // 检查POSE_GRAPH_SAVE_PATH文件夹是否存在，如果不存在则创建
        printf(" POSE_GRAPH_SAVE_PATH is %s \n", POSE_GRAPH_SAVE_PATH.c_str());
        if (access(POSE_GRAPH_SAVE_PATH.c_str(), 0) == -1)
        {
            int flag = mkdir(POSE_GRAPH_SAVE_PATH.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            if (flag == 0)
            {
                printf("pose graph path not exist, create success\n");
            }
            else
            {
                printf("pose graph path not exist, create failed!!\n");
            }
        }

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
        cam_rot_R.push_back(Matrix3d::Identity());
        double yaw = 0.;

        if (NUM_OF_CAM == 2)
        {
            // STEREO = 1;
            std::string cam1Calib;
            fsSettings["cam1_calib"] >> cam1Calib;
            std::string cam1Path = configPath + "/" + cam1Calib;
            // printf("%s cam1 path\n", cam1Path.c_str() );
            CAM_NAMES.push_back(cam1Path);
            fsSettings["body_T_cam1"] >> cv_T;
            cv::cv2eigen(cv_T, T); // lidar

            R_i_c.push_back(T.block<3, 3>(0, 0));
            t_i_c.push_back(T.block<3, 1>(0, 3));
            i_T_c_vec.push_back(T);
            Eigen::Matrix3d R_c1_c0 = R_i_c[1] * R_i_c[0].transpose();
            cam_rot_R.push_back(R_c1_c0);
        }
        else if (NUM_OF_CAM == 4)
        {
            OMNI = 1;
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
            Eigen::Matrix3d R_c1_c0_given = Utility::ypr2R(Vector3d(180.0, 0, 0));
            Eigen::Matrix3d R_c1_c0 = R_i_c[1] * R_i_c[0].transpose(); //??
            std::cout << "R_c1_c0_given " << R_c1_c0_given << std::endl
                      << " R_c1_c0 " << R_c1_c0 << std::endl;
            Eigen::Matrix3d R_i_c1_calc = R_c1_c0_given * R_i_c[0]; //?? 公式不对?结果对
            std::cout << " R_i_c1 " << R_i_c[1] << std::endl
                      << "R_i_c1_calc " << R_i_c1_calc << std::endl;
            cam_rot_R.push_back(R_c1_c0);

            fsSettings["body_T_cam2"] >> cv_T;
            cv::cv2eigen(cv_T, T);
            R_i_c.push_back(T.block<3, 3>(0, 0));
            t_i_c.push_back(T.block<3, 1>(0, 3));
            i_T_c_vec.push_back(T);
            Eigen::Matrix3d cam_rot_R2 = Utility::ypr2R(Vector3d(90.0, 0, 0));
            Eigen::Matrix3d R_c2_c0 = R_i_c[2] * R_i_c[0].transpose();
            // std::cout << "cam_rot_R2 " << cam_rot_R2 << std::endl << " R_c2_c0 " << R_c2_c0 << std::endl;
            cam_rot_R.push_back(R_c2_c0);

            fsSettings["body_T_cam3"] >> cv_T;
            cv::cv2eigen(cv_T, T);
            R_i_c.push_back(T.block<3, 3>(0, 0));
            t_i_c.push_back(T.block<3, 1>(0, 3));
            i_T_c_vec.push_back(T);
            Eigen::Matrix3d R_c3_c0 = R_i_c[3] * R_i_c[0].transpose();
            cam_rot_R.push_back(R_c3_c0);
            // std::cout << "cam_rot_R " << cam_rot_R.back() << std::endl;
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

    std::ofstream fout_mappingTime(MAPPING_TIME_LOG_PATH, std::ios::out);
    fout_mappingTime.close();

    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    MIN_DIST_LIDARPT = fsSettings["min_dist_lidarpt"];
    F_THRESHOLD = fsSettings["F_threshold"];
    FLOW_BACK = fsSettings["flow_back"];
    // SKIP_DIST = fsSettings["skip_dist"];
    MAPP_DIS_FACTOR = fsSettings["mapp_dis_factor"];
    MAPP_ROT_FACTOR = fsSettings["mapp_rot_factor"];
    fsSettings["fast"]["use.fast"] >> USE_GRID_FAST;
    fsSettings["fast"]["shi.min.score"] >> SHI_THRESHOLD_SCORE;
    fsSettings["fast"]["grid.size"] >> FAST_GRID_SIZE;
    std::cout << "Use FAST " << USE_GRID_FAST << std::endl;
    std::cout << "SHI MIN Score " << SHI_THRESHOLD_SCORE << std::endl;
    std::cout << "GRID Size " << FAST_GRID_SIZE << std::endl;

    USE_IMU = 0;
    posegraph.setIMUFlag(USE_IMU);
    BUILD_KEYFRAME = fsSettings["build_keyframe"];
    cout << "BUILD_KEYFRAME: " << BUILD_KEYFRAME << endl;
    SuperPointConfig spp_config;
    BriefConfig brief_config;
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
        // cout << "BRIEF_PATTERN_FILE" << brief_config.pattern_file << endl;
    }

    fsSettings.release();

    // ros::Subscriber sub_vio = n.subscribe("/vins_estimator/odometry", 2000, vio_callback);
    ros::Subscriber sub_lio = n.subscribe("/mapping/Odometry", 2000, lio_callback);
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 2000, img0_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_img1, sub_img2, sub_img3;
    if (BUILD_KEYFRAME)
    {
        if (2 == NUM_OF_CAM)
        {
            printf("sub_img 1 \n");
            sub_img1 = n.subscribe(IMAGE1_TOPIC, 2000, img1_callback, ros::TransportHints().tcpNoDelay()); // image1 for tracker, but not necessary
        }
        else if (4 == NUM_OF_CAM)
        {
            printf("sub_img 1 \n");
            sub_img1 = n.subscribe(IMAGE1_TOPIC, 2000, img1_callback, ros::TransportHints().tcpNoDelay());
            sub_img2 = n.subscribe(IMAGE2_TOPIC, 2000, img2_callback, ros::TransportHints().tcpNoDelay());
            sub_img3 = n.subscribe(IMAGE3_TOPIC, 2000, img3_callback, ros::TransportHints().tcpNoDelay());
        }
    }
    // ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_pose = n.subscribe("/mapping/Odometry", 2000, pose_callback);
    // ros::Subscriber sub_extrinsic = n.subscribe("/vins_estimator/extrinsic", 2000, extrinsic_callback);
    // ros::Subscriber sub_margin_point = n.subscribe("/vins_estimator/margin_cloud", 2000, margin_point_callback);
    ros::Subscriber sub_lidar = n.subscribe("/mapping/Submap", 2000, lidar_point_callback);
    pub_image_track = n.advertise<sensor_msgs::Image>("image_track", 1000);
    // pub_match_img = n.advertise<sensor_msgs::Image>("match_image", 1000);
    // pub_imgpoint_cloud = n.advertise<sensor_msgs::PointCloud>("imgpoint_cloud", 1000);
    // pub_point_cloud_cam = n.advertise<sensor_msgs::PointCloud2>("point_cloud_inimu", 1000);
    // pub_point_cloud_inFOV = n.advertise<sensor_msgs::PointCloud2>("point_cloud_inFOV", 1000);
    // pub_triangulate_cloud = n.advertise<sensor_msgs::PointCloud2>("triangulate_cloud", 1000);
    // pub_lidar_corner_cloud = n.advertise<sensor_msgs::PointCloud2>("lidar_corner_cloud", 1000);
    // pub_oldKeyframe_cloud = n.advertise<sensor_msgs::PointCloud2>("oldKeyframe_cloud", 1000);
    pub_odometry_0 = n.advertise<nav_msgs::Odometry>("odometry_0", 1000);
    pub_odometry_1 = n.advertise<nav_msgs::Odometry>("odometry_1", 1000);
    pub_odometry_2 = n.advertise<nav_msgs::Odometry>("odometry_2", 1000);
    pub_odometry_3 = n.advertise<nav_msgs::Odometry>("odometry_3", 1000);

    if (0 == DETECTOR)
    {
        initializeSuperpointDetector(spp_config);
    }
    else if (1 == DETECTOR)
    {
        initializeBriefDetector(brief_config);
    }

    std::thread measurement_process;
    std::thread keyboard_command_process;
    std::thread stereo_sync_thread, omni_sync_thread;

    printf("measurement_process is record!\n");
    measurement_process = std::thread(process_mapping);

    keyboard_command_process = std::thread(command);
    pthread_setname_np(keyboard_command_process.native_handle(), "keyboard_command_process");

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