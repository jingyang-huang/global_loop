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

#pragma once

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "sys_configs.h"
#include "feature_detector.h"
#include "point_matcher.h"


extern ros::Publisher pub_match_img;
extern int VISUALIZATION_SHIFT_X;
extern int VISUALIZATION_SHIFT_Y;
extern std::string POSE_GRAPH_SAVE_PATH;
// extern std::string BRIEF_PATTERN_FILE;
extern int ROW;
extern int COL;
extern std::string LOOP_RESULT_PATH;
extern std::string MAPPING_TIME_LOG_PATH;
extern std::string RELO_TIME_LOG_PATH;
extern std::ofstream fout_reloTime;
extern std::ofstream fout_mappingTime;
extern std::ofstream fout_loopRes;
extern Eigen::Vector3d rected_vio_t;
extern Eigen::Quaterniond rected_vio_q;
extern int RELATIVE_THRESHOLD;
extern int DEBUG_IMAGE;
extern int MAPPING_MODE;
extern int DETECTOR;
extern int MATCHER;

extern Eigen::Vector3d last_lio_t;
extern Eigen::Quaterniond last_lio_q;

// extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern int MIN_DIST_LIDARPT;
extern double F_THRESHOLD;
extern int FLOW_BACK;
extern double FOCAL_LENGTH;
extern int MIN_QUERY_GAP;
extern Eigen::Matrix4d body_T_lidar;

extern double TD;
extern int NUM_OF_CAM;
extern int STEREO;
extern int OMNI;
extern int USE_IMU;
extern int BUILD_KEYFRAME;
extern int MULTIPLE_THREAD;
extern double INIT_DEPTH;
extern const int WINDOW_SIZE;
extern double MIN_PARALLAX;
extern int USE_INITIALIZE;
extern int USE_PG_OPTIMIZE;
extern int USE_TRAJ_SMOOTH;
extern int FREQ_FACTOR;
extern int USE_GRID_FAST;
extern int FAST_GRID_SIZE;
extern double SHI_THRESHOLD_SCORE;

// 声明全局的FeatureDetector实例
extern std::shared_ptr<FeatureDetector> feature_detector;
void initializeSuperpointDetector(const SuperPointConfig &sppconfig);
void initializeBriefDetector(const BriefConfig &briefconfig) ;

extern std::shared_ptr<PointMatcher> point_matcher;
void initializePointMatcher(const PointMatcherConfig &pmconfig);