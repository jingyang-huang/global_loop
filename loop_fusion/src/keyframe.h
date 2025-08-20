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
#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <memory>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "utility/Grider_FAST.h"
#include "utility/dls_pnp.h"
#include "parameters.h"

#include <bitset>

#define MIN_LOOP_NUM 20

// using namespace Eigen;
// using namespace std;
// using namespace DVision;

typedef std::vector<FeatureProperty> KeyPointsData;

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
	int j = 0;
	for (int i = 0; i < int(v.size()); i++)
		if (status[i])
			v[j++] = v[i];
	v.resize(j);
}

// class BriskExtractor
// {
// public:
// 	BriskExtractor();
// 	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = new cv::BriskDescriptorExtractor(true, true);
// };

class KeyFrame
{
public:
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_t_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal,
			 vector<int> &_point_id, int _sequence);
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_t_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1> &_loop_info,
			 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brisk_descriptors);
	// add by hjy
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_t_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
			 vector<int> &_point_id, int _sequence);
	// create keyframe database, for visual mapping
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_t_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 FeatureData &keyframe_points, int _sequence, int _cam_id);
	// load previous keyframe, as reference
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_t_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, short _cam_id, Eigen::Matrix<double, 8, 1> &_loop_info,
			 KeyPointsData &_reference_keypoints_data);
	// create keyframe online, for query, extract online
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_t_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image, int _sequence);
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_t_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image, FeatureData &_keyframe_points, int _sequence);
	void computeReferencexxxPoint(const FeatureData &keyframe_points);
	void computeReferenceBRIEFPoint(const FeatureData &keyframe_points);
	void ComputeQueryBRIEFPoint();
	void extractComputeQueryBRIEFPoint();
	void computeReferenceBRISKPoint(const FeatureData &keyframe_points);
	void ComputeQueryBRISKPoint();
	void extractComputeQueryBRISKPoint();
	void computeReferenceSuperPoint(const FeatureData &keyframe_points);
	void ComputeQuerySuperPoint();
	void extractComputeQuerySuperPoint();
	// void extractBrief();
	int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
	void searchInAera(const BRIEF::bitset keypoints_descriptor,
					  const std::vector<BRIEF::bitset> &descriptors_old,
					  int &idx);
	void searchInAera(const cv::Mat keypoints_descriptor,
					  const FeatureData &reference_keypoints_data,
					  int &idx);
	void searchByBRIEFDes(const std::shared_ptr<KeyFrame> old_kf,
						  std::vector<cv::Point2f> &matched_cur_2d,
						  std::vector<cv::Point2f> &matched_cur_2d_norm,
						  std::vector<cv::Point2f> &matched_old_2d,
						  std::vector<cv::Point3f> &matched_old_3d,
						  std::vector<uchar> &status);
	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
								const std::vector<cv::Point2f> &matched_2d_old_norm,
								vector<uchar> &status);
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
				   const std::vector<cv::Point3f> &matched_3d,
				   std::vector<uchar> &status,
				   Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old, double &reprojection_error);
	void PnPRANSAC2(const vector<cv::Point2f> &matched_2d_old_norm,
					const std::vector<cv::Point3f> &matched_3d,
					std::vector<uchar> &status,
					Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info);

	// cv::Ptr<cv::BRISK> brisk_detector = cv::BRISK::create();
	// cv::Ptr<cv::ORB> brief_detector = cv::ORB::create();
	// cv::Ptr<cv::FREAK> freak_detector = cv::FREAK::create();

	// cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = new cv::BriskDescriptorExtractor(true, true);
	Eigen::Vector3d getLoopRelativeT();
	double getLoopRelativeYaw();
	Eigen::Quaterniond getLoopRelativeQ();

	double time_stamp;
	int index;

	short cam_id;
	Eigen::Vector3d vio_t_w_i;
	Eigen::Matrix3d vio_R_w_i;
	Eigen::Vector3d T_w_i;
	Eigen::Matrix3d R_w_i;
	Eigen::Quaterniond q_w_i;
	Eigen::Vector3d origin_vio_T;
	Eigen::Matrix3d origin_vio_R;
	Eigen::Vector3d old_vio_T;
	Eigen::Matrix3d old_vio_R;
	Eigen::Vector3d PnP_T;
	Eigen::Matrix3d PnP_R;
	cv::Mat image;
	// std::vector<cv::Mat> image_seq;
	// cv::Mat thumbnail;
	// vector<cv::Point3f> query_point_3d; //
	// vector<cv::Point2f> query_point_2d_uv;
	// vector<cv::Point2f> query_point_2d_norm;
	// vector<int> point_id;
	KeyPointsData query_keypoints_data; // TODO: check query的keypoint可以用小数
	KeyPointsData reference_keypoints_data;

	// std::vector<Eigen::Matrix<float, 256, 1>> reference_spp_descriptors; // for DBow2
	std::vector<Eigen::Matrix<float, 256, 1>> query_spp_descriptors;
	std::vector<cv::Mat> query_brief_descriptors;
	// std::vector<BRIEF::bitset> query_brief_descriptors;
	// std::vector<BRIEF::bitset> reference_brief_descriptors;

	// vector<std::bitset<384>> query_brief_descriptors;
	// std::vector<cv::Mat> reference_brief_descriptors;
	// vector<std::bitset<384>> reference_brief_descriptors;

	int sequence;

	bool has_loop;
	int loop_index;
	int optimize_buf_index = -1;
	Eigen::Matrix<double, 8, 1> loop_info;
};
typedef std::shared_ptr<KeyFrame> KeyFramePtr;
