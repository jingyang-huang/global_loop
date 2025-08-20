/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int k = 0; k < NUM_OF_CAM; k++)
        R_i_c[k].setIdentity();
}

FeatureManager::FeatureManager()
{
}

void FeatureManager::setRic(std::vector<Eigen::Matrix3d> _ric)
{
    for (int i = 0; i < _ric.size(); i++)
    {
        R_i_c[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {
        it.used_num = it.feature_per_frame.size();
        if (it.used_num >= 4)
        {
            cnt++;
        }
    }
    return cnt;
}

bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    last_average_parallax = 0;
    new_feature_num = 0;
    long_track_num = 0;
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);
        assert(id_pts.second[0].first == 0);
        if (id_pts.second.size() == 2)
        {
            f_per_fra.rightObservation(id_pts.second[1].second);
            assert(id_pts.second[1].first == 1);
        }

        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          { return it.feature_id == feature_id; });

        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count)); // frame_count 赋值给start_frame
            feature.back().feature_per_frame.push_back(f_per_fra);
            new_feature_num++;
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
            if (it->feature_per_frame.size() >= 4)
                long_track_num++;
        }
    }

    // if (frame_count < 2 || last_track_num < 20)
    // if (frame_count < 2 || last_track_num < 20 || new_feature_num > 0.5 * last_track_num)
    if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 || new_feature_num > 0.5 * last_track_num)
        return true;

    for (auto &it_per_id : feature)
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;

            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        // ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth()
{
    for (auto &it_per_id : feature)
        it_per_id.estimated_depth = -1;
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

// Triangulates N views by computing SVD that minimizes the error.
void FeatureManager::TriangulateNViewSVD(const std::vector<Matrix3x4d> &poses,
                                         const std::vector<Vector3d> &points,
                                         Vector3d &point_3d)
{
    assert(poses.size() >= 2);
    assert(poses.size() == points.size());

    MatrixXd design_matrix(3 * points.size(), 4 + points.size());

    for (int i = 0; i < points.size(); i++)
    {
        design_matrix.block<3, 4>(3 * i, 0) = -poses[i].matrix();
        design_matrix.block<3, 1>(3 * i, 4 + i) = points[i];
    }
    Vector4d triangulated_point;
    triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>().head(4);
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                                      Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
        design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

void FeatureManager::triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], std::vector<Vector3d> t_i_c, std::vector<Matrix3d> R_i_c, PointCloudXYZI::Ptr points_triangulate, FeatureData &extracted_points)
{
    int stereo_pts_count = 0, left_multi_pts_count = 0;

    for (auto &it_per_id : feature) // 这个是检测所有滑窗内的，所以有的老点不应该被拿出来
    {
        FeatureProperty tmp_feature;
        tmp_feature.point_id = it_per_id.feature_id;
        tmp_feature.keypoint_2d_uv.pt = cv::Point2f(it_per_id.feature_per_frame[0].uv.x(), it_per_id.feature_per_frame[0].uv.y());
        tmp_feature.keypoint_2d_norm.pt = cv::Point2f(it_per_id.feature_per_frame[0].point.x(), it_per_id.feature_per_frame[0].point.y());

        int endframe = it_per_id.endFrame();
        Eigen::Vector3d t0_end = Ps[endframe] + Rs[endframe] * t_i_c[0];
        Eigen::Matrix3d R0_end = Rs[endframe] * R_i_c[0];

        if (it_per_id.estimated_depth > 0) // 已经三角化过了， 但是只要这一帧看到了就应该放进去
        {
            // 老特征点可能会被看到过，只需要看它endFrame的位置是不是frame_count，这样就能表明这次有没有被看到
            // 只有onTracking的点才会被看到
            if (frameCnt == it_per_id.endFrame() && it_per_id.start_frame >5)
            {
                PointType p;
                p.x = it_per_id.position.x();
                p.y = it_per_id.position.y();
                p.z = it_per_id.position.z();
                p.intensity = 1000.0;
                points_triangulate->points.push_back(p);
                tmp_feature.point_depth = it_per_id.estimated_depth;
                tmp_feature.point_3d_world = cv::Point3f(it_per_id.position.x(), it_per_id.position.y(), it_per_id.position.z());
                extracted_points.push_back(tmp_feature);
            }

            continue;
        }
        if (0)
        {
            if (it_per_id.feature_per_frame.size() > 1)
            {
                int frame_size = it_per_id.feature_per_frame.size();
                std::vector<Matrix3x4d> Poses;
                std::vector<Vector3d> points;
                Eigen::Vector3d point3d_world;
                Eigen::Matrix<double, 3, 4> leftPose;
                Eigen::Matrix<double, 3, 4> rightPose;
                // for (int imu_i = it_per_id.start_frame; imu_i < frame_size; imu_i++)
                int imu_i = it_per_id.start_frame;
                {
                    if (it_per_id.feature_per_frame[imu_i].is_stereo)
                    {
                        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * t_i_c[0];
                        Eigen::Matrix3d R0 = Rs[imu_i] * R_i_c[0];
                        leftPose.leftCols<3>() = R0.transpose();
                        leftPose.rightCols<1>() = -R0.transpose() * t0;

                        Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * t_i_c[1];
                        Eigen::Matrix3d R1 = Rs[imu_i] * R_i_c[1];
                        rightPose.leftCols<3>() = R1.transpose();
                        rightPose.rightCols<1>() = -R1.transpose() * t1;

                        Eigen::Vector3d point0, point1;

                        point0 = it_per_id.feature_per_frame[imu_i].point;
                        point1 = it_per_id.feature_per_frame[imu_i].pointRight;
                        Poses.push_back(leftPose);
                        Poses.push_back(rightPose);
                        points.push_back(point0);
                        points.push_back(point1);
                    }
                    else // 单目的，要不要加入？
                    {
                        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * t_i_c[0];
                        Eigen::Matrix3d R0 = Rs[imu_i] * R_i_c[0];
                        leftPose.leftCols<3>() = R0.transpose();
                        leftPose.rightCols<1>() = -R0.transpose() * t0;

                        Eigen::Vector3d point0;

                        point0 = it_per_id.feature_per_frame[imu_i].point;
                        Poses.push_back(leftPose);
                        points.push_back(point0);
                    }
                }
                if (Poses.size() < 2)
                    continue;
                // triangulatePoint(leftPose, rightPose, point0, point1, point3d_world);
                TriangulateNViewSVD(Poses, points, point3d_world);
                Eigen::Vector3d localPoint;
                localPoint = leftPose.leftCols<3>() * point3d_world + leftPose.rightCols<1>(); // 最新帧的相机坐标系
                double depth = localPoint.z();
                if (depth > 0)
                {
                    it_per_id.estimated_depth = depth;
                    it_per_id.position = point3d_world;
                    PointType p;
                    p.x = point3d_world.x();
                    p.y = point3d_world.y();
                    p.z = point3d_world.z();
                    p.intensity = 255.0;
                    points_triangulate->points.push_back(p);
                    tmp_feature.point_3d_world = cv::Point3f(point3d_world.x(), point3d_world.y(), point3d_world.z());
                    tmp_feature.point_depth = it_per_id.estimated_depth;
                    left_multi_pts_count++;
                    extracted_points.push_back(tmp_feature);
                }
                else
                {
                    it_per_id.estimated_depth = INIT_DEPTH;
                }
                
                continue;
                /*
                Vector3d ptsGt = pts_gt[it_per_id.feature_id];
                printf("motion  %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                                ptsGt.x(), ptsGt.y(), ptsGt.z());
                */
                //    printf("left multi tri pts size %d \n", points_triangulate->points.size());
            }
        }
        else
        {
            if (STEREO && it_per_id.feature_per_frame[0].is_stereo) // 左右目三角化
            {
                int imu_i = it_per_id.start_frame;
                Eigen::Matrix<double, 3, 4> leftPose;
                Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * t_i_c[0];
                Eigen::Matrix3d R0 = Rs[imu_i] * R_i_c[0];
                leftPose.leftCols<3>() = R0.transpose();
                leftPose.rightCols<1>() = -R0.transpose() * t0;
                // cout << "left pose " << leftPose << endl;

                Eigen::Matrix<double, 3, 4> rightPose;
                Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * t_i_c[1];
                Eigen::Matrix3d R1 = Rs[imu_i] * R_i_c[1];
                rightPose.leftCols<3>() = R1.transpose();
                rightPose.rightCols<1>() = -R1.transpose() * t1;
                // cout << "right pose " << rightPose << endl;

                Eigen::Vector2d point0, point1;
                Eigen::Vector3d point3d_world;
                point0 = it_per_id.feature_per_frame[0].point.head(2);
                point1 = it_per_id.feature_per_frame[0].pointRight.head(2);
                // cout << "point0 " << point0.transpose() << endl;
                // cout << "point1 " << point1.transpose() << endl;

                triangulatePoint(leftPose, rightPose, point0, point1, point3d_world);
                // Triangulate(leftPose, rightPose, point0, point1, &point3d_world);
                Eigen::Vector3d localPoint;
                localPoint = leftPose.leftCols<3>() * point3d_world + leftPose.rightCols<1>();
                double depth = localPoint.z();
                if (depth > 0)
                {
                    it_per_id.estimated_depth = depth;
                    it_per_id.position = point3d_world;
                    PointType p;
                    p.x = point3d_world.x();
                    p.y = point3d_world.y();
                    p.z = point3d_world.z();
                    points_triangulate->points.push_back(p);
                    tmp_feature.point_3d_world = cv::Point3f(point3d_world.x(), point3d_world.y(), point3d_world.z());
                    tmp_feature.point_depth = it_per_id.estimated_depth;
                    stereo_pts_count++;
                    extracted_points.push_back(tmp_feature);
                }
                else
                    it_per_id.estimated_depth = INIT_DEPTH; // 暂时设为-1，不对还要继续算
                /*
                Vector3d ptsGt = pts_gt[it_per_id.feature_id];
                printf("stereo %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                                ptsGt.x(), ptsGt.y(), ptsGt.z());
                */
                
                continue;
            }
            else if (it_per_id.feature_per_frame.size() > 1)
            {
                int imu_i = it_per_id.start_frame;
                Eigen::Matrix<double, 3, 4> leftPose;
                Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * t_i_c[0];
                Eigen::Matrix3d R0 = Rs[imu_i] * R_i_c[0];
                leftPose.leftCols<3>() = R0.transpose();
                leftPose.rightCols<1>() = -R0.transpose() * t0;

                imu_i++;
                Eigen::Matrix<double, 3, 4> rightPose;
                Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * t_i_c[0];
                Eigen::Matrix3d R1 = Rs[imu_i] * R_i_c[0];
                rightPose.leftCols<3>() = R1.transpose();
                rightPose.rightCols<1>() = -R1.transpose() * t1;

                Eigen::Vector2d point0, point1;
                Eigen::Vector3d point3d_world;
                point0 = it_per_id.feature_per_frame[0].point.head(2);
                point1 = it_per_id.feature_per_frame[1].point.head(2);
                triangulatePoint(leftPose, rightPose, point0, point1, point3d_world);
                Eigen::Vector3d localPoint;
                localPoint = leftPose.leftCols<3>() * point3d_world + leftPose.rightCols<1>();
                double depth = localPoint.z();
                if (depth > 0)
                {
                    it_per_id.estimated_depth = depth;
                    it_per_id.position = point3d_world;
                    PointType p;
                    p.x = point3d_world.x();
                    p.y = point3d_world.y();
                    p.z = point3d_world.z();
                    p.intensity = 255.0;
                    points_triangulate->points.push_back(p);
                    tmp_feature.point_3d_world = cv::Point3f(point3d_world.x(), point3d_world.y(), point3d_world.z());
                    tmp_feature.point_depth = it_per_id.estimated_depth;
                    left_multi_pts_count++;
                    extracted_points.push_back(tmp_feature);
                }
                else
                    it_per_id.estimated_depth = INIT_DEPTH;
                /*
                Vector3d ptsGt = pts_gt[it_per_id.feature_id];
                printf("motion  %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                                ptsGt.x(), ptsGt.y(), ptsGt.z());
                */
                //    printf("left multi tri pts size %d \n", points_triangulate->points.size());
                // extracted_points.push_back(tmp_feature);
                continue;
            }
        }
        // 正常三角化到这里就结束了，下面是计算？

        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * t_i_c[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * R_i_c[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * t_i_c[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * R_i_c[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        // it_per_id->estimated_depth = -b / A;
        // it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        // it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }
        // extracted_points.point_id.push_back(tmp_feature.point_id);
    }
    // printf("stereo tri pts size %d, left multi tri pts size %d \n", stereo_pts_count, left_multi_pts_count);
}

void FeatureManager::removeOutlier(set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        int index = it->feature_id;
        itSet = outlierIndex.find(index);
        if (itSet != outlierIndex.end())
        {
            feature.erase(it);
            // printf("remove outlier %d \n", index);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else //start_frame == 0
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            // ？
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    // check the second last frame is keyframe or not
    // parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    // int r_i = frame_count - 2;
    // int r_j = frame_count - 1;
    // p_i_comp = R_i_c[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * R_i_c[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}