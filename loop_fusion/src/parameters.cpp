/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

// int WINDOW_SIZE;
double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.7964};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
// std::string EX_CALIB_RESULT_PATH;
// std::string LOOP_RESULT_PATH;
// std::string OUTPUT_FOLDER;
// std::string IMU_TOPIC;
// int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int OMNI;
int USE_IMU;
int MULTIPLE_THREAD;
int USE_INITIALIZE;
int USE_PG_OPTIMIZE;
int USE_TRAJ_SMOOTH;
std::map<int, Eigen::Vector3d> pts_gt;

std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
int MIN_DIST_LIDARPT;
double F_THRESHOLD;
int FLOW_BACK;
double FOCAL_LENGTH = 460.0;
int USE_GRID_FAST;
int FAST_GRID_SIZE;
double SHI_THRESHOLD_SCORE;

// 定义并初始化全局的FeatureDetector实例
std::shared_ptr<FeatureDetector> feature_detector;

void initializeSuperpointDetector(const SuperPointConfig &sppconfig) {
    feature_detector = std::make_shared<FeatureDetector>(sppconfig);
}

void initializeBriefDetector(const BriefConfig &briefconfig) {
    feature_detector = std::make_shared<FeatureDetector>(briefconfig);
}


std::shared_ptr<PointMatcher> point_matcher;
void initializePointMatcher(const PointMatcherConfig &pmconfig) {
    point_matcher = std::make_shared<PointMatcher>(pmconfig);
}