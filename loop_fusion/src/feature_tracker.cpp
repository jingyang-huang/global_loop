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

#include "feature_tracker.h"
int SHOW_TRACK;
bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
}

// 修改了函数，还会使用projected_points
// 这里就会把extract和curr的点用mask加进去
// if (cur_pts_seq[k].size())
// {
//     cur_pts_seq[k].clear();
//     prev_pts_seq[k].clear();
// }
cv::Mat mask_extract = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
// mask_vec_tmp.push_back(ini_mask.clone());

// // 根据score排序
// sort(track_spp_points.begin(), track_spp_points.end(), [](const FeatureProperty &a, const FeatureProperty &b)
//      { return a.score > b.score; });

// cur_pts.clear();
// cur_un_pts.clear();
// ids.clear();
// track_cnt.clear();

// // cur_lidar_pts.clear();
// // cur_lidar_norm_pts.clear();
// // ids_lidar.clear();
// // track_cnt_lidar.clear();

// for (auto it : track_spp_points)
// {
//     short cam_id = it.cam_id;
//     if (lidar_cnt_vec[cam_id] >= MAX_CNT) // 只保留最大的MAX_CNT个特征点 //没有深度信息的点不要
//         continue;
//     // lidar的点选择性放进去
//     if (mask_vec_tmp[cam_id].at<uchar>(it.keypoint_2d_uv.pt) == 255) // 如果对应像素是白的，那么就会先把自己放进去，不过之后在它周围的都不会放进去
//     {
//         final_points[cam_id].push_back(it);
//         lidar_cnt_vec[cam_id]++;
//         // cur_lidar_pts.push_back(it.keypoint_2d_uv.pt);
//         // cur_lidar_norm_pts.push_back(it.keypoint_2d_norm.pt);
//         // point_3d_lidar.push_back(it.point_3d_world);
//         // ids_lidar.push_back(it.point_id);
//         // track_cnt_lidar.push_back(1); // track_cnt_lidar怎么用

//         num_lidar_pts++;
//         if (it.point_depth > 0)                                                              // draw mask for points with depth
//             cv::circle(mask_vec_tmp[cam_id], it.keypoint_2d_uv.pt, MIN_DIST_LIDARPT, 0, -1); // 至少是4

//         // std::cout<< " score " << it.score << std::endl;
//     }
// }
void FeatureTracker::setMask(int row, int col, std::vector<FeatureDataParted> &extract_spp_points, std::vector<FeatureDataParted> &track_spp_points, std::vector<FeatureData> &cur_final_points)
{
    std::vector<int> lidar_cnt_vec;
    std::vector<cv::Mat> mask_vec_tmp;
    final_points.clear();
    final_points.resize(NUM_OF_CAM);
    cur_final_points.clear();
    cur_final_points.resize(NUM_OF_CAM);
    cv::Mat ini_mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));
    num_lidar_pts = 0;
    mask_vec_tmp.reserve(NUM_OF_CAM);
    // if (mask_vec.size())
    //     mask_vec_tmp.clear();
    // mask_vec_tmp.reserve(NUM_OF_CAM);
    for (int k = 0; k < NUM_OF_CAM; k++)
    {
        mask_vec_tmp.push_back(ini_mask.clone());
        lidar_cnt_vec.push_back(0);
        // cur_pts_seq[k].clear();
    }

    // 根据score和depth排序
    for (int k = 0; k < NUM_OF_CAM; k++)
    {
        sort(extract_spp_points[k].points_no_depth.begin(), extract_spp_points[k].points_no_depth.end(), [](const FeatureProperty &a, const FeatureProperty &b)
             { return a.score > b.score; });
        sort(track_spp_points[k].points_no_depth.begin(), track_spp_points[k].points_no_depth.end(), [](const FeatureProperty &a, const FeatureProperty &b)
             { return a.score > b.score; });
    }

    for (int k = 0; k < NUM_OF_CAM; k++)
    {
        FeatureData toadd_spp_points_all;
        int cnt_with_depth = 0;
        toadd_spp_points_all.insert(toadd_spp_points_all.end(), extract_spp_points[k].points_with_depth.begin(), extract_spp_points[k].points_with_depth.end());
        toadd_spp_points_all.insert(toadd_spp_points_all.end(), track_spp_points[k].points_with_depth.begin(), track_spp_points[k].points_with_depth.end());
        toadd_spp_points_all.insert(toadd_spp_points_all.end(), extract_spp_points[k].points_no_depth.begin(), extract_spp_points[k].points_no_depth.end());
        toadd_spp_points_all.insert(toadd_spp_points_all.end(), track_spp_points[k].points_no_depth.begin(), track_spp_points[k].points_no_depth.end());

        // 先把有深度的拿进来
        for (auto it : toadd_spp_points_all)
        {
            // short cam_id = it.cam_id;
            if (lidar_cnt_vec[k] >= MAX_CNT) // 1
                continue;
            // lidar的点选择性放进去
            if (mask_vec_tmp[k].at<uchar>(it.keypoint_2d_uv.pt) == 255) // 如果对应像素是白的，那么就会先把自己放进去，不过之后在它周围的都不会放进去
            {
                final_points[k].push_back(it);
                // cur_pts_seq[k].push_back(it.keypoint_2d_uv.pt);
                lidar_cnt_vec[k]++;
                if (it.point_depth > 0)
                    cnt_with_depth++;
                // cur_lidar_pts.push_back(it.keypoint_2d_uv.pt);
                // cur_lidar_norm_pts.push_back(it.keypoint_2d_norm.pt);
                // point_3d_lidar.push_back(it.point_3d_world);
                // ids_lidar.push_back(it.point_id);
                // track_cnt_lidar.push_back(1); // track_cnt_lidar怎么用

                num_lidar_pts++;
                // draw mask for points with depth
                cv::circle(mask_vec_tmp[k], it.keypoint_2d_uv.pt, MIN_DIST_LIDARPT, 0, -1); // 至少是4

                // std::cout<< " score " << it.score << std::endl;
            }
        }
        std::cout <<  "extract_spp_points size " << extract_spp_points[k].points_with_depth.size() << " track_spp_points size " << track_spp_points[k].points_with_depth.size() << " cnt_with_depth " << cnt_with_depth << std::endl;
    }

    cur_final_points = final_points;
    // if (SHOW_TRACK)
    // {
    //     cv::Mat hcon_mask;
    //     cv::hconcat(mask_vec_tmp, hcon_mask);
    //     cv::imshow("lidar_mask", hcon_mask);
    // }

    // mask_vec = mask_vec_tmp;

    // cur_lidar_pts.clear();
    // cur_lidar_norm_pts.clear();
    // ids_lidar.clear();
    // track_cnt_lidar.clear();
}

void FeatureTracker::setMaskVisual(int row, int col)
{
    std::vector<int> visual_cnt_vec;

    cv::Mat ini_mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));
    num_visual_pts = 0;
    std::vector<cv::Mat> mask_vec_tmp;
    // if (mask_vec.size())
    //     mask_vec.clear();
    mask_vec_tmp.reserve(NUM_OF_CAM);
    for (int k = 0; k < NUM_OF_CAM; k++)
    {
        mask_vec_tmp.push_back(ini_mask.clone());
        visual_cnt_vec.push_back(0);
    }

    // prefer to keep features that are tracked for long time

    // 根据track_cnt的大小进行排序
    sort(tracking_points.begin(), tracking_points.end(), [](const FeatureProperty &a, const FeatureProperty &b)
         { return a.track_cnt > b.track_cnt; });

    cur_pts.clear();
    cur_un_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto it : tracking_points)
    {
        short cam_id = it.cam_id;
        if (visual_cnt_vec[cam_id] >= MAX_CNT) // 只保留最大的MAX_CNT个特征点 //没有深度信息的点不要
            continue;
        // lidar的点选择性放进去
        if (mask_vec_tmp[cam_id].at<uchar>(it.keypoint_2d_uv.pt) == 255) // 如果对应像素是白的，那么就会先把自己放进去，不过之后在它周围的都不会放进去
        {
            final_points[cam_id].push_back(it);
            visual_cnt_vec[cam_id]++;
            // cur_lidar_pts.push_back(it.keypoint_2d_uv.pt);
            // cur_lidar_norm_pts.push_back(it.keypoint_2d_norm.pt);
            // point_3d_lidar.push_back(it.point_3d_world);
            // ids_lidar.push_back(it.point_id);
            // track_cnt_lidar.push_back(1); // track_cnt_lidar怎么用

            // num_lidar_pts++;
            if (it.point_depth > 0)                                                      // draw mask for points with depth
                cv::circle(mask_vec_tmp[cam_id], it.keypoint_2d_uv.pt, MIN_DIST, 0, -1); // 至少是4

            // std::cout<< " score " << it.score << std::endl;
        }
    }
    mask_vec = mask_vec_tmp;
    // cv::Mat hcon_mask;
    // cv::hconcat(mask_vec_tmp, hcon_mask);
    // cv::imshow("visual_mask", hcon_mask);
    
}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

// 此时只是把projected_points入参添加到类的成员变量中，还没有进行特征点的合并
void FeatureTracker::add_extract_points(std::vector<FeatureDataParted> &points_in)
{
    extract_spp_points = points_in;
    // add projected points
}

// 此时只是把projected_points入参添加到类的成员变量中，还没有进行特征点的合并
void FeatureTracker::add_track_points(std::vector<FeatureDataParted> &points_in)
{
    track_spp_points = points_in;
    // add projected points
}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    TicToc t_r;
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    rightImg = _img1;
    /*
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */

    cur_pts.clear();

    if (prev_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        if (hasPrediction)
        {
            cur_pts = predict_pts;
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1,
                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

            int succ_num = 0;
            for (size_t i = 0; i < status.size(); i++)
            {
                if (status[i])
                    succ_num++;
            }
            if (succ_num < 10)
            { // 还是采用普通方法跟踪
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
            }
        }
        else
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
        // reverse check
        if (FLOW_BACK)
        {
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = prev_pts;
            cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1,
                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            // cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3);
            for (size_t i = 0; i < status.size(); i++)
            {
                if (status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                {
                    status[i] = 1;
                }
                else
                    status[i] = 0;
            }
        }

        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        // ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        // printf("track cnt %d\n", (int)ids.size());
        for (auto &n : track_cnt)
            n++;
    }

    // rejectWithF();
    ROS_DEBUG("set mask begins");
    TicToc t_m;
    // 修改了函数，还会使用projected_points
    setMaskVisual(row, col);
    // cv::imshow("mask", mask);
    ROS_DEBUG("set mask costs %fms", t_m.toc());

    ROS_DEBUG("detect feature begins");
    TicToc t_t;
    int n_max_cnt = MAX_CNT;
    // printf("current feature cnt %d\n", (int)cur_pts.size());
    if (n_max_cnt > 0)
    {
        if (mask_vec[0].empty())
            cout << "mask is empty " << endl;
        if (mask_vec[0].type() != CV_8UC1)
            cout << "mask type wrong " << endl;
        cv::goodFeaturesToTrack(cur_img, n_pts, n_max_cnt, 0.01, MIN_DIST, mask_vec[0]); // 后续换成grider fast试试
    }
    else
        n_pts.clear();
    ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

    for (auto &p : n_pts)
    {
        cur_pts.push_back(p);
        ids.push_back(n_id++);
        track_cnt.push_back(1);
    }
    // printf("feature cnt after add %d\n", (int)ids.size());

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    // mean shitomasi score

    // mean_shitomasi_score = 0;
    // for (unsigned int i = 0; i < ids.size(); i++)
    // {
    //     mean_shitomasi_score += shiTomasiScore(cur_img, cur_pts[i].x, cur_pts[i].y);
    // }
    // mean_shitomasi_score /= ids.size();

    // 对右目进行提取其实就是改变带有_right的vector
    if (!_img1.empty() && stereo_cam)
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if (!cur_pts.empty())
        {
            // printf("stereo image; track feature on right image\n");
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;
            vector<float> err;
            // cur left ---- cur right
            cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);
            // reverse check cur right ---- cur left
            if (FLOW_BACK)
            {
                cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
                for (size_t i = 0; i < status.size(); i++)
                {
                    if (status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                        status[i] = 1;
                    else
                        status[i] = 0;
                }
            }

            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            // only keep left-right pts
            /*
            reduceVector(cur_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            reduceVector(pts_velocity, status);
            */
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }

    // generate featureFrame for feature manager
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y, z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }

    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y, z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
        }
    }

    // 并不需要用视觉的管理器来manage它
    // if (1)
    // {
    //     for (size_t i = 0; i < ids_lidar.size(); i++)
    //     {
    //         int feature_id = ids_lidar[i];
    //         double x, y;
    //         x = cur_lidar_norm_pts[i].x;
    //         y = cur_lidar_norm_pts[i].y;
    //         double p_u, p_v;
    //         p_u = cur_lidar_pts[i].x;
    //         p_v = cur_lidar_pts[i].y;
    //         int camera_id = 100;
    //         double position_x, position_y, position_z;
    //         position_x = point_3d_lidar[i].x;
    //         position_y = point_3d_lidar[i].y;
    //         position_z = point_3d_lidar[i].z;

    //         Eigen::Matrix<double, 7, 1> xy_uv_position;
    //         xy_uv_position << x, y, p_u, p_v, position_x, position_y, position_z;
    //         featureFrame[feature_id].emplace_back(camera_id, xy_uv_position);
    //     }
    // }

    // save as previous image and features for LK flow tracking
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    // 不需要保留上一帧是角点的lidar点
    // prev_pts.insert(prev_pts.end(), cur_lidar_pts.begin(), cur_lidar_pts.end());
    // prev_un_pts.insert(prev_un_pts.end(), cur_lidar_norm_pts.begin(), cur_lidar_norm_pts.end());
    // ids.insert(ids.end(), ids_lidar.begin(), ids_lidar.end());
    // track_cnt.insert(track_cnt.end(), track_cnt_lidar.begin(), track_cnt_lidar.end());

    prev_un_pts_map = cur_un_pts_map; // 计算feature的速度，我用不到
    prev_time = cur_time;
    hasPrediction = false;

    // draw line in tracking
    prevLeftPtsMap.clear();

    // for (size_t i = 0; i < track_spp_points.size(); i++)
    //     prevLeftPtsMap[track_spp_points[i].point_id] = track_spp_points[i].keypoint_2d_uv.pt;

    for (size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    // printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}

void FeatureTracker::drawTrack(cv::Mat &imTrack)
{
    // drawLidarProjectPts(imTrack);
    // drawVisualTrackPts(imTrack);
}

void FeatureTracker::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
    else if (calib_file.size() == 4)
        omni_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        // cout << trackerData[0].K << endl;
        // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

cv::Point2f FeatureTracker::undistortedPt(Eigen::Vector2d pt_vec, camodocal::CameraPtr cam)
{
    cv::Point2f un_pts;

    Eigen::Vector2d a(pt_vec);
    Eigen::Vector3d b;
    cam->liftProjective(a, b);
    un_pts = cv::Point2f(b.x() / b.z(), b.y() / b.z());

    return un_pts;
}

// cv::Point2f FeatureTracker::normalizePts2D(cv::Point2f &pts, camodocal::CameraPtr cam)
// {
//     cv::Point2f normalized_pts;

//     Eigen::Vector2d a(pts.x, pts.y);
//     Eigen::Vector3d b;
//     cam->liftProjective(a, b);
//     un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));

//     return normalized_pts;
// }

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                                map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;

        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    else // 无法估计，给一个0，0
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

void FeatureTracker::drawLidarProjectPts(std::vector<cv::Mat> &image_seq_rgb)
{
    for (int k = 0; k < NUM_OF_CAM; k++)
    {
        for (size_t j = 0; j < final_points[k].size(); j++)
        {
            double depth = final_points[k][j].point_depth;
            if (depth <= 0)
            {
                // cv::circle(image_seq_rgb[k], final_points[k][j].keypoint_2d_uv.pt, 1, cv::Scalar(0, 255, 255), 2);
                continue;
            }
            else
            {
                if (final_points[k][j].track_cnt < 2) // 新的点，绿色
                {
                    double len = std::min(1.0, 1.0 * depth / 20);
                    // 先col后row
                    cv::circle(image_seq_rgb[k], final_points[k][j].keypoint_2d_uv.pt, 1, cv::Scalar(0, 255 * len, 0), 2); // 越绿表示深度越大
                }
                else // 老track的点，红色
                {
                    double len = std::min(1.0, 1.0 * depth / 20);
                    cv::circle(image_seq_rgb[k], final_points[k][j].keypoint_2d_uv.pt, 1, cv::Scalar(255 * (1 - len), 0, 255 * len), 2); // 越红表示深度越大
                }
            }
        }
    }
}

void FeatureTracker::drawVisualTrackPts(std::vector<cv::Mat> &image_seq_rgb)
{
    auto curLeftIds = this->ids;
    auto curLeftPts = this->cur_pts;
    auto curRightPts = this->cur_right_pts;
    auto prevLeftPtsMap = this->prevLeftPtsMap;
    cv::Mat imTrack = image_seq_rgb[0];

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 3); // 3次就认为很多
        cv::circle(imTrack, curLeftPts[j], 1, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!rightImg.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cur_img.cols;
            cv::circle(imTrack, rightPt, 1, cv::Scalar(0, 255, 0), 2);
            // cv::Point2f leftPt = curLeftPtsTrackRight[i];
            // cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }

    // map<int, cv::Point2f>::iterator mapIt;
    // for (size_t i = 0; i < curLeftIds.size(); i++)
    // {
    //     int id = curLeftIds[i];
    //     mapIt = prevLeftPtsMap.find(id);
    //     if (mapIt != prevLeftPtsMap.end())
    //     {
    //         cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    //     }
    // }

    // draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    // printf("predict pts size %d \n", (int)predict_pts_debug.size());

    // cv::Mat imCur2Compress;
    // cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}

// void FeatureTracker::predictPtsInNextFrame()
// {
//     //printf("predict pts in next frame\n");
//     if(frame_count < 2)
//         return;
//     // predict next pose. Assume constant velocity motion
//     Eigen::Matrix4d curT, prevT, nextT;
//     getPoseInWorldFrame(curT);
//     getPoseInWorldFrame(frame_count - 1, prevT);
//     nextT = curT * (prevT.inverse() * curT);
//     map<int, Eigen::Vector3d> predictPts;

//     for (auto &it_per_id : f_manager.feature)
//     {
//         if(it_per_id.estimated_depth > 0)
//         {
//             int firstIndex = it_per_id.start_frame;
//             int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
//             //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
//             if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
//             {
//                 double depth = it_per_id.estimated_depth;
//                 Vector3d pts_j = R_i_c[0] * (depth * it_per_id.feature_per_frame[0].point) + t_i_c[0];
//                 Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
//                 Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
//                 Vector3d pts_cam = R_i_c[0].transpose() * (pts_local - t_i_c[0]);
//                 int ptsIndex = it_per_id.feature_id;
//                 predictPts[ptsIndex] = pts_cam;
//             }
//         }
//     }
//     setPrediction(predictPts);
//     //printf("estimator output %d predict pts\n",(int)predictPts.size());
// }

void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        // printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]);
    }
}

void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if (itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}

cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}