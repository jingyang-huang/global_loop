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

#include "keyframe.h"

// create keyframe online, for query, extract online
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_t_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image, FeatureData &_keyframe_points, int _sequence)
{
	time_stamp = _time_stamp;
	index = _index;
	vio_t_w_i = _vio_t_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_t_w_i; // 回环的位姿初值也用vio的
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_t_w_i; // 刚进来的时候vio的T，还未回环
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	// cv::resize(image, thumbnail, cv::Size(80, 60));
	has_loop = false;
	loop_index = -1;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;

	query_keypoints_data = _keyframe_points;
	// computeReferenceBRIEFPoint(); // 计算vins提取特征点的描述子
	// ComputeQueryBRIEFPoint();	   // 计算fast提取特征点的描述子
	// extractComputeQueryBRIEFPoint();

	if (!DEBUG_IMAGE)
		image.release();
}

// create reference keyframe database, for visual mapping
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_t_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
				   FeatureData &_keyframe_points, int _sequence, int _cam_id)
{
	time_stamp = _time_stamp;
	index = _index;
	vio_t_w_i = _vio_t_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_t_w_i; // 回环的位姿初值也用vio的
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_t_w_i; // 刚进来的时候vio的T，还未回环
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	// cv::resize(image, thumbnail, cv::Size(80, 60));s
	has_loop = false;
	loop_index = -1;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	cam_id = _cam_id;

	// computeReferenceBRIEFPoint(_keyframe_points);
	// computeReferencexxxPoint(_keyframe_points);
	reference_keypoints_data = _keyframe_points;

	// ComputeQueryBRIEFPoint();
	if (!DEBUG_IMAGE)
		image.release();
}

// load previous keyframe, as reference
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_t_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
				   cv::Mat &_image, int _loop_index, short _cam_id, Eigen::Matrix<double, 8, 1> &_loop_info,
				   KeyPointsData &_reference_keypoints_data)
{
	time_stamp = _time_stamp;
	index = _index;
	// vio_t_w_i = _vio_t_w_i;
	// vio_R_w_i = _vio_R_w_i;
	vio_t_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
	q_w_i = Eigen::Quaterniond(_R_w_i);
	if (DEBUG_IMAGE)
	{
		image = _image.clone();
		// cv::resize(image, thumbnail, cv::Size(80, 60));
	}
	if (_loop_index != -1)
		has_loop = true;
	else
		has_loop = false;
	loop_index = _loop_index;
	loop_info = _loop_info;
	sequence = 0;
	cam_id = _cam_id;
	// TODO!
	reference_keypoints_data = _reference_keypoints_data;
	// reference_spp_descriptors = _reference_spp_descriptors;
	// reference_brief_descriptors = _brief_descriptors;
}

// void KeyFrame::computeReferenceBRIEFPoint(const FeatureData &fd_in)
// {
// 	// BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
// 	FeatureProperty kpp;
// 	std::vector<cv::KeyPoint> reference_keypoints;
// 	for (int i = 0; i < (int)fd_in.size(); i++)
// 	{
// 		short cam_id = fd_in[i].cam_id;
// 		kpp.keypoint_2d_uv.pt = fd_in[i].keypoint_2d_uv.pt;
// 		kpp.keypoint_2d_norm.pt = fd_in[i].keypoint_2d_norm.pt; // 可能需要删
// 		kpp.point_3d_world = fd_in[i].point_3d_world;
// 		kpp.cam_id = fd_in[i].cam_id;
// 		reference_keypoints.push_back(kpp.keypoint_2d_uv);
// 		reference_keypoints_data.push_back(kpp);
// 	}

// 	// extractor(image, reference_keypoints, reference_brief_descriptors);
// }

// void KeyFrame::computeReferencexxxPoint(const FeatureData &fd_in)
// {
// 	reference_keypoints_data = fd_in;
// 	// reference_brief_descriptors
// }

void KeyFrame::extractComputeQueryBRISKPoint()
{
	// const int fast_th = 20; // corner detector response threshold
	// const int agast_th = 20;
	// if (0)
	// {
	// 	cv::AGAST(image, query_keypoints, agast_th, true);
	// 	// cv::FAST(image, query_keypoints, fast_th, true);

	// 	// Grider_FAST::perform_griding(image, query_keypoints, 200, 1, 1, fast_th, true);
	// 	// printf("fast query_keypoints size %d\n", (int)query_keypoints.size());
	// }
	// else
	// {
	// 	vector<cv::Point2f> tmp_pts;
	// 	cv::goodFeaturesToTrack(image, tmp_pts, 800, 0.01, 3);
	// 	for (int i = 0; i < (int)tmp_pts.size(); i++)
	// 	{
	// 		cv::KeyPoint key;
	// 		key.pt = tmp_pts[i];
	// 		query_keypoints.push_back(key); // 这里换成我自己提的
	// 	}
	// }

	// for (int i = 0; i < (int)query_keypoints.size(); i++)
	// {
	// 	Eigen::Vector3d tmp_p;
	// 	m_camera[0]->liftProjective(Eigen::Vector2d(query_keypoints[i].pt.x, query_keypoints[i].pt.y), tmp_p);
	// 	cv::KeyPoint tmp_norm;
	// 	tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
	// 	query_keypoints_norm.push_back(tmp_norm);
	// 	query_point_2d_uv.push_back(query_keypoints[i].pt);
	// 	query_point_2d_norm.push_back(tmp_norm.pt);
	// }
	// cv::Mat descriptors_cv;
	// cv::Ptr<cv::BRISK> brisk_detector = cv::BRISK::create();
	// // brief_detector->compute(image, reference_keypoints, descriptors_cv); // 32*8 = 256,cv 是按照一个字节来计算的
	// brisk_detector->compute(image, query_keypoints, descriptors_cv); // 64*8 = 512,所以正常的BRISK是512bit
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int KeyFrame::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
	const int *pa = a.ptr<int32_t>();
	const int *pb = b.ptr<int32_t>();

	int dist = 0;

	for (int i = 0; i < 8; i++, pa++, pb++)
	{
		unsigned int v = *pa ^ *pb;
		v = v - ((v >> 1) & 0x55555555);
		v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
		dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	}

	return dist;
}

void KeyFrame::searchInAera(const cv::Mat keypoints_descriptor,
							const FeatureData &reference_keypoints_data,
							int &idx)
{
	cv::Point2f best_pt;
	int bestDist = 64;
	int bestIndex = -1;
	// 在old descriptors里面循环找与当前reference_descriptor最近的
	for (int i = 0; i < (int)reference_keypoints_data.size(); i++)
	{

		// int dis = HammingDis(keypoints_descriptor, reference_keypoints_data[i].brief_descriptor); // 222
		int dis = DescriptorDistance(keypoints_descriptor, reference_keypoints_data[i].brief_descriptor); // 222
		if (dis < bestDist)
		{
			bestDist = dis;
			bestIndex = i;
		}
	}

	idx = bestIndex;

	// // printf("best dist %d", bestDist);
	// if (bestIndex != -1 && bestDist < 60)
	// {
	// 	tmp_kpp = ref_kpd[bestIndex];
	// 	return true;
	// }
	// else
	// 	return false;
}

void KeyFrame::searchByBRIEFDes(const KeyFramePtr old_kf,
								std::vector<cv::Point2f> &matched_cur_2d,
								std::vector<cv::Point2f> &matched_cur_2d_norm,
								std::vector<cv::Point2f> &matched_old_2d,
								std::vector<cv::Point3f> &matched_old_3d,
								std::vector<uchar> &status)
{
	for (int i = 0; i < (int)query_keypoints_data.size(); i++) // reference_brisk_descriptors换成brief_descriptors，从当前提的里找
	{
		FeatureProperty tmp_kpp;
		auto old_kpd = old_kf->reference_keypoints_data;
		int idx = -1;
		searchInAera(this->query_keypoints_data[i].brief_descriptor, old_kf->reference_keypoints_data, idx); // 222
		// 对每个brief_descriptors都能找到最近的old keypoint的uv,norm,3d
		if (-1 != idx)
		{
			matched_cur_2d.push_back(query_keypoints_data[i].keypoint_2d_uv.pt);
			matched_cur_2d_norm.push_back(query_keypoints_data[i].keypoint_2d_norm.pt);
			matched_old_2d.push_back(old_kpd[idx].keypoint_2d_uv.pt);
			matched_old_3d.push_back(old_kpd[idx].point_3d_world);
			status.push_back(1);
		}
		else
			status.push_back(0);
	}
}

void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
									  const std::vector<cv::Point2f> &matched_2d_norm,
									  vector<uchar> &status)
{
	int n = (int)matched_2d_cur_norm.size();
	for (int i = 0; i < n; i++)
		status.push_back(0);
	if (n >= 8)
	{
		vector<cv::Point2f> tmp_cur(n), tmp_old(n);
		for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
		{
			double FOCAL_LENGTH = 460.0;
			double tmp_x, tmp_y;
			tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
			tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
			tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

			tmp_x = FOCAL_LENGTH * matched_2d_norm[i].x + COL / 2.0;
			tmp_y = FOCAL_LENGTH * matched_2d_norm[i].y + ROW / 2.0;
			tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
		}
		cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
	}
}

void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_norm,
						 const std::vector<cv::Point3f> &matched_3d,
						 std::vector<uchar> &status,
						 Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old,
						 double &reprojection_error)
{
	// for (int i = 0; i < matched_3d.size(); i++)
	//	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
	// printf("match size %d \n", matched_3d.size());
	cv::Mat r, rvec, t, D, tmp_r;
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
	Matrix3d R_inital;
	Vector3d P_inital;
	// Matrix3d R_w_c = origin_vio_R * R_i_c[0];
	// Vector3d T_w_c = origin_vio_T + origin_vio_R * t_i_c[0];
	Matrix3d R_w_c = old_vio_R * R_i_c[0];
	Vector3d T_w_c = old_vio_T + old_vio_R * t_i_c[0];

	R_inital = R_w_c.inverse();
	P_inital = -(R_inital * T_w_c);

	cv::eigen2cv(R_inital, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_inital, t);

	cv::Mat inliers;
	TicToc t_pnp_ransac;

	// this
	try
	{
		solvePnPRansac(matched_3d, matched_2d_norm, K, D, rvec, t, false, 1000, 15.0 / 389.0, 0.6, inliers, cv::SOLVEPNP_ITERATIVE);
	}
	catch (std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}

	for (int i = 0; i < (int)matched_2d_norm.size(); i++)
		status.push_back(0);

	// cj_T_w1
	cv::Rodrigues(rvec, r);
	Matrix3d R_pnp, R_w_c_old;
	Vector3d T_pnp, T_w_c_old;
	cv::cv2eigen(r, R_pnp);
	cv::cv2eigen(t, T_pnp);

	for (int i = 0; i < inliers.rows; i++)
	{
		int k = inliers.at<int>(i);
		status[k] = 1;
		Eigen::Vector2d pt_uv(matched_2d_norm[k].x, matched_2d_norm[k].y);
		Eigen::Vector2d repro_uv;
		Eigen::Vector3d pt_3d(matched_3d[k].x, matched_3d[k].y, matched_3d[k].z);
		// m_camera[0]->spaceToPlane(pt_3d, repro_uv); // u是x，v是y，所以u是列，v是行
	}

	R_w_c_old = R_pnp.transpose();
	// R_w_c_old = R_pnp;
	T_w_c_old = R_w_c_old * (-T_pnp);
	// T_w_c_old = R_w_c_old * (T_pnp);

	// w1_T_bj
	PnP_R_old = R_w_c_old * R_i_c[0].transpose();
	PnP_T_old = T_w_c_old - PnP_R_old * t_i_c[0];
	// chech ok --hjy
}

int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
	BRIEF::bitset xor_of_bitset = a ^ b;
	int dis = xor_of_bitset.count();
	return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
	_T_w_i = vio_t_w_i;
	_R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
	_T_w_i = T_w_i;
	_R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	// 只更新回环的位姿
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	// 同时更新vio和回环的位姿，都是进行sequence间的变换
	vio_t_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_t_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
	return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
	return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
	return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info)
{
	if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
	{
		// printf("update loop info\n");
		loop_info = _loop_info;
	}
}

// void KeyFrame::BriskExtract()
// {
// 	// (detector)->detect(currentImage->image, currentImage->keypoint_2d_uv);
//     // get the descriptors
//     descriptorExtractor->compute(currentImage->image, currentImage->keypoint_2d_uv,
//                                  currentImage->descriptors);
// }
