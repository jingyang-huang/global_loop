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

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "parameters.h"
#include "utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
public:
    FeatureTracker();
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    void setMask(int row, int col, std::vector<FeatureData> &cur_final_points);
    void setMask(int row, int col, std::vector<FeatureDataParted> &extract_spp_points, std::vector<FeatureDataParted> &track_spp_points, std::vector<FeatureData> &cur_final_points);

    void setMaskVisual(int row, int col);
    void readIntrinsicParameter(const vector<string> &calib_file);
    void showUndistortion(const string &name);
    void rejectWithF();
    void undistortedPoints();

    cv::Point2f undistortedPt(Eigen::Vector2d pt_vec, camodocal::CameraPtr cam);
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2,
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(cv::Mat &imTrack);
    void drawVisualTrackPts(std::vector<cv::Mat> &image_seq_rgb);
    void drawLidarProjectPts(std::vector<cv::Mat> &image_seq_rgb);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    bool inBorder(const cv::Point2f &pt);
    void add_extract_points(std::vector<FeatureDataParted> &pts_in);
    void add_track_points(std::vector<FeatureDataParted> &pts_in);

    int row, col;
    cv::Mat imTrack;
    // cv::Mat mask;
    std::vector<cv::Mat> mask_vec;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img;
    cv::Mat rightImg;
    std::vector<FeatureDataParted> track_spp_points;
    std::vector<FeatureDataParted> extract_spp_points;
    FeatureData tracking_points;
    std::vector<FeatureData> final_points;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> track_cnt;
    vector<uchar> type; // 1-projected, 2-predicted

    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;

    vector<cv::Point2f> cur_lidar_pts, cur_lidar_norm_pts;
    map<int, cv::Point2f> cur_lidar_norm_pts_map;
    vector<int> ids_lidar;
    vector<int> track_cnt_lidar;
    vector<cv::Point3f> point_3d_lidar;

    double cur_time;
    double prev_time;
    bool stereo_cam;
    bool omni_cam;
    int n_id;
    bool hasPrediction;
    int num_lidar_pts = 0;
    int num_visual_pts = 0;
};
