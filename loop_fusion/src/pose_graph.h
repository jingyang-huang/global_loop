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

#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <string>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <queue>
#include <assert.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <stdio.h>
#include <ros/ros.h>
#include "keyframe.h"
#include "feature_tracker.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/tic_toc.h"
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "feature_manager.h"
#include "database.h"
// #include "ThirdParty/DBoW2/DBoW2.h"
// #include "ThirdParty/DVision/DVision.h"
// #include "ThirdParty/DBoW2/TemplatedDatabase.h"
// #include "ThirdParty/DBoW2/TemplatedVocabulary.h"

#define SHOW_S_EDGE true
#define SHOW_L_EDGE true

using namespace DVision;
using namespace DBoW2;
extern ros::Publisher pub_oldKeyframe_cloud;
extern ros::Publisher pub_map_lidarpts;
extern ros::Publisher pub_globalkf_pose;
extern ros::Publisher pub_pnp_pose;
extern Eigen::Vector3d lio_t;
extern Eigen::Matrix3d lio_R;
extern double t_PnPRANSAC;
extern double t_match;
extern double t_detectLoop;
extern double SKIP_DIST;
extern double SKIP_ANGLE;

class PoseGraph
{
public:
	PoseGraph();
	~PoseGraph();
	void registerPub(ros::NodeHandle &n);
	void addKeyFrame(KeyFramePtr cur_kf);
	void addRefKeyFrame(KeyFramePtr keyframe);
	void loadKeyFrame(KeyFramePtr cur_kf);
	void loadKeyFrame(KeyFramePtr cur_kf, const std::vector<cv::Mat>& reference_brief_descriptors);
	void loadKeyFrame(KeyFramePtr cur_kf, const std::vector<Eigen::Matrix<float, 256, 1>>& reference_spp_descriptors);
	void addKeyFrameIntoVoc(KeyFramePtr keyframe);
	void initBasePath(KeyFramePtr cur_kf);
	void initialize(KeyFramePtr cur_kf);
	void initDatabase(std::string voc_path, int DE);
	void initBriefDatabase(std::string voc_path);
	void setIMUFlag(bool _use_imu);
	KeyFramePtr getKeyFrame(int index);
	KeyFramePtr getLocalKeyFrame(int index);

	nav_msgs::Path path[10];
	nav_msgs::Path base_path;
	PointCloudXYZI::Ptr base_pose_cloud;
	CameraPoseVisualization *posegraph_visualization;
	void savePoseGraph();
	void loadPoseGraph();
	bool getLoopFactRes(int index, KeyFramePtr cur_kf);
	void publish();
	Vector3d t_drift;
	double yaw_drift;
	Matrix3d r_drift;
	Vector3d last_t_drift;
	double last_yaw_drift;
	Matrix3d last_r_drift;
	Vector3d t_corrected;
	double yaw_corrected;
	Matrix3d r_corrected;

	Vector3d t_corrected_pub;
	double yaw_corrected_pub;
	Matrix3d r_corrected_pub;

	Vector3d t_drift_pub;
	double yaw_drift_pub;
	Matrix3d r_drift_pub;
	bool bUpdated = false;
	int pub_step;
	// world frame( base sequence or first sequence)<----> cur sequence frame
	Vector3d w_t_vio;
	Matrix3d w_r_vio;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	Eigen::Vector3d loop_error_t;
	Quaterniond loop_error_q;
	double relative_yaw;
	short loop_res = 0; // 0- no loop; 1- good loop; 2- bad loop
	int final_matched_num = 0;
	PointCloudXYZI::Ptr map_lidarpts;
	std::mutex m_updated;
	KeyFramePtr neighbor_kf;
	KeyFramePtr optimize_end_kf;
	double Z_THRESHOLD = 0.5;

	ros::Publisher pub_pg_path;
	ros::Publisher pub_base_path, pub_base_pose;
	ros::Publisher pub_pose_graph;
	ros::Publisher pub_path[10];

private:
	int detectLoop(KeyFramePtr keyframe, int frame_index);
	int detectLoop(KeyFramePtr keyframe, std::vector<int> &candidates);
	bool findConnection(KeyFramePtr cur_kf, const KeyFramePtr old_kf);
	void kNNMatcher(FeatureData &cur_query, FeatureData &old_ref, std::vector<cv::DMatch> &matches, bool crossCheck);

	void slideWindow_keyframelist();
	void optimize4DoF();
	// void optimize6DoF();
	void updatePath();
	// deque<KeyFramePtr > keyframelist;
	std::deque<KeyFramePtr> local_keyframelist_window;
	// std::deque<KeyFramePtr > global_keyframelist;
	PointCloudXYZI::Ptr globalkf_cloud;
	std::deque<KeyFramePtr> ref_keyframelist;
	// marginalized_poses
	std::mutex m_keyframelist;
	std::mutex m_optimize_sig;
	std::mutex m_path;
	std::mutex m_drift;
	std::thread t_optimization;
	std::deque<std::pair<int, int>> loop_edges; // cur - old

	bool optimize_signal = 0;
	int global_index;
	int sequence_cnt;
	vector<int> sequence_global_index;
	vector<bool> sequence_loop;
	map<int, cv::Mat> image_pool;
	int earliest_loop_index;
	int base_sequence;
	bool use_imu;
	bool bLoopKeepLost = true; // 一开始和后面跟踪丢失
	short cLostCount = 0;
	short LOCAL_KF_WINDOW_SIZE = 100;

	// BriefDatabase db;
	// BriefVocabulary *voc;
	DatabaseInterfacePtr dbi;
};

void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

template <typename T>
inline void QuaternionInverse(const T q[4], T q_inverse[4])
{
	q_inverse[0] = q[0];
	q_inverse[1] = -q[1];
	q_inverse[2] = -q[2];
	q_inverse[3] = -q[3];
};

template <typename T>
T NormalizeAngle(const T &angle_degrees)
{
	if (angle_degrees > T(180.0))
		return angle_degrees - T(360.0);
	else if (angle_degrees < T(-180.0))
		return angle_degrees + T(360.0);
	else
		return angle_degrees;
};

class AngleLocalParameterization
{
public:
	template <typename T>
	bool operator()(const T *theta_radians, const T *delta_theta_radians,
					T *theta_radians_plus_delta) const
	{
		*theta_radians_plus_delta =
			NormalizeAngle(*theta_radians + *delta_theta_radians);

		return true;
	}

	static ceres::LocalParameterization *Create()
	{
		return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization,
														 1, 1>);
	}
};

template <typename T>
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll, T R[9])
{

	T y = yaw / T(180.0) * T(M_PI);
	T p = pitch / T(180.0) * T(M_PI);
	T r = roll / T(180.0) * T(M_PI);

	R[0] = cos(y) * cos(p);
	R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
	R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
	R[3] = sin(y) * cos(p);
	R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
	R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
	R[6] = -sin(p);
	R[7] = cos(p) * sin(r);
	R[8] = cos(p) * cos(r);
};

template <typename T>
void RotationMatrixTranspose(const T R[9], T inv_R[9])
{
	inv_R[0] = R[0];
	inv_R[1] = R[3];
	inv_R[2] = R[6];
	inv_R[3] = R[1];
	inv_R[4] = R[4];
	inv_R[5] = R[7];
	inv_R[6] = R[2];
	inv_R[7] = R[5];
	inv_R[8] = R[8];
};

template <typename T>
void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3])
{
	r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
	r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
	r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
};

struct FourDOFError
{
	FourDOFError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
		: t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i) {}

	template <typename T>
	bool operator()(const T *const yaw_i, const T *ti, const T *yaw_j, const T *tj, T *residuals) const
	{
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		// euler to rotation
		T w_R_i[9];
		YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
		// rotation transpose
		T i_R_w[9];
		RotationMatrixTranspose(w_R_i, i_R_w);
		// rotation matrix rotate point
		T t_i_ij[3];
		RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

		residuals[0] = (t_i_ij[0] - T(t_x));
		residuals[1] = (t_i_ij[1] - T(t_y));
		residuals[2] = (t_i_ij[2] - T(t_z));
		residuals[3] = NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw));

		return true;
	}

	static ceres::CostFunction *Create(const double t_x, const double t_y, const double t_z,
									   const double relative_yaw, const double pitch_i, const double roll_i)
	{
		return (new ceres::AutoDiffCostFunction<
				FourDOFError, 4, 1, 3, 1, 3>(
			new FourDOFError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
	}

	double t_x, t_y, t_z;
	double relative_yaw, pitch_i, roll_i;
};

struct FourDOFWeightError
{
	FourDOFWeightError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
		: t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i)
	{
		weight = 5;
	}

	template <typename T>
	bool operator()(const T *const yaw_i, const T *ti, const T *yaw_j, const T *tj, T *residuals) const
	{
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		// euler to rotation
		T w_R_i[9];
		YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
		// rotation transpose
		T i_R_w[9];
		RotationMatrixTranspose(w_R_i, i_R_w);
		// rotation matrix rotate point
		T t_i_ij[3];
		RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

		residuals[0] = (t_i_ij[0] - T(t_x)) * T(weight);
		residuals[1] = (t_i_ij[1] - T(t_y)) * T(weight);
		residuals[2] = (t_i_ij[2] - T(t_z)) * T(weight);
		residuals[3] = NormalizeAngle((yaw_j[0] - yaw_i[0] - T(relative_yaw))) * T(weight) / T(10.0);

		return true;
	}

	static ceres::CostFunction *Create(const double t_x, const double t_y, const double t_z,
									   const double relative_yaw, const double pitch_i, const double roll_i)
	{
		return (new ceres::AutoDiffCostFunction<
				FourDOFWeightError, 4, 1, 3, 1, 3>(
			new FourDOFWeightError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
	}

	double t_x, t_y, t_z;
	double relative_yaw, pitch_i, roll_i;
	double weight;
};

struct RelativeRTError
{
	RelativeRTError(double t_x, double t_y, double t_z,
					double q_w, double q_x, double q_y, double q_z,
					double t_var, double q_var)
		: t_x(t_x), t_y(t_y), t_z(t_z),
		  q_w(q_w), q_x(q_x), q_y(q_y), q_z(q_z),
		  t_var(t_var), q_var(q_var) {}

	template <typename T>
	bool operator()(const T *const w_q_i, const T *ti, const T *w_q_j, const T *tj, T *residuals) const
	{
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		T i_q_w[4];
		QuaternionInverse(w_q_i, i_q_w);

		T t_i_ij[3];
		ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);

		residuals[0] = (t_i_ij[0] - T(t_x)) / T(t_var);
		residuals[1] = (t_i_ij[1] - T(t_y)) / T(t_var);
		residuals[2] = (t_i_ij[2] - T(t_z)) / T(t_var);

		T relative_q[4];
		relative_q[0] = T(q_w);
		relative_q[1] = T(q_x);
		relative_q[2] = T(q_y);
		relative_q[3] = T(q_z);

		T q_i_j[4];
		ceres::QuaternionProduct(i_q_w, w_q_j, q_i_j);

		T relative_q_inv[4];
		QuaternionInverse(relative_q, relative_q_inv);

		T error_q[4];
		ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q);

		residuals[3] = T(2) * error_q[1] / T(q_var);
		residuals[4] = T(2) * error_q[2] / T(q_var);
		residuals[5] = T(2) * error_q[3] / T(q_var);

		return true;
	}

	static ceres::CostFunction *Create(const double t_x, const double t_y, const double t_z,
									   const double q_w, const double q_x, const double q_y, const double q_z,
									   const double t_var, const double q_var)
	{
		return (new ceres::AutoDiffCostFunction<
				RelativeRTError, 6, 4, 3, 4, 3>(
			new RelativeRTError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, t_var, q_var)));
	}

	double t_x, t_y, t_z, t_norm;
	double q_w, q_x, q_y, q_z;
	double t_var, q_var;
};