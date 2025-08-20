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

#include "pose_graph.h"
float RECALL = 0, PRECISION = 0, F1 = 0;
int TP = 0, FP = 0, FN = 0;
bool bDetectRes = 0, bFactRes = 0;
std::vector<bool> DetectRes, FactRes;
Eigen::Vector3d lio_t;
Eigen::Matrix3d lio_R;
ros::Publisher pub_oldKeyframe_cloud, pub_map_lidarpts, pub_globalkf_pose, pub_pnp_pose;
double t_PnPRANSAC = 0., t_match = 0., t_detectLoop = 0.;
double SKIP_DIST = 0;
double SKIP_ANGLE = 30.0;
int RELATIVE_THRESHOLD = 0;
std::ofstream fout_reloTime;
std::ofstream fout_loopRes;
Vector3d rected_vio_t;
Quaterniond rected_vio_q;
int DEBUG_IMAGE = 0;
int DETECTOR;
int MATCHER;

PoseGraph::PoseGraph()
{
    posegraph_visualization = new CameraPoseVisualization(1.0, 0.0, 1.0, 1.0);
    posegraph_visualization->setScale(0.1);
    posegraph_visualization->setLineWidth(0.01);
    earliest_loop_index = -1;
    t_drift = Eigen::Vector3d(0, 0, 0);
    yaw_drift = 0;
    r_drift = Eigen::Matrix3d::Identity();
    w_t_vio = Eigen::Vector3d(0, 0, 0);
    w_r_vio = Eigen::Matrix3d::Identity();
    global_index = 0;
    sequence_cnt = 0;
    sequence_global_index.emplace_back(0);
    sequence_loop.push_back(0);
    base_sequence = 1;
    use_imu = 0;
    globalkf_cloud.reset(new PointCloudXYZI);
    base_pose_cloud.reset(new PointCloudXYZI);
}

PoseGraph::~PoseGraph()
{
    t_optimization.detach();
}

void PoseGraph::registerPub(ros::NodeHandle &n)
{
    pub_pg_path = n.advertise<nav_msgs::Path>("pose_graph_path", 100000); // 这个也许不需要那么大的
    pub_base_path = n.advertise<nav_msgs::Path>("base_path", 100000);
    pub_base_pose = n.advertise<sensor_msgs::PointCloud2>("base_pose", 10000);
    pub_pose_graph = n.advertise<visualization_msgs::MarkerArray>("pose_graph", 10000);
    for (int i = 1; i < 10; i++)
        pub_path[i] = n.advertise<nav_msgs::Path>("path_" + to_string(i), 10000);
}

void PoseGraph::setIMUFlag(bool _use_imu)
{
    use_imu = _use_imu;
    if (use_imu)
    {
        printf("VIO input, perfrom 4 DoF (x, y, z, yaw) pose graph optimization\n");
        t_optimization = std::thread(&PoseGraph::optimize4DoF, this); // 原来这个优化的线程一直在跑
    }
    // else
    // {
    //     printf("VO input, perfrom 6 DoF pose graph optimization\n");
    //     t_optimization = std::thread(&PoseGraph::optimize6DoF, this);
    // }
}

void PoseGraph::initDatabase(std::string voc_path, int DETECTOR)
{
    dbi = std::make_shared<DatabaseInterface>(voc_path, DETECTOR);
}

void PoseGraph::initBriefDatabase(std::string voc_path)
{
}

void PoseGraph::initialize(KeyFramePtr cur_kf)
{
    bDetectRes = 0;
    std::vector<int> candidates;
    // printf(" index in last sequence %d \n", global_index - sequence_global_index[0]);
    int loop_num = -1, loop_index = -1;
    KeyFramePtr old_kf;

    // bFactRes = getLoopFactRes(global_index, cur_kf); // 找在过去的帧中是哪个
    // printf(" loop fact res %d \n", bFactRes);
    // printf("RECALL: %f, PRECISION: %f, F1: %f\n", RECALL, PRECISION, 2 * RECALL * PRECISION / (RECALL + PRECISION));
    loop_num = detectLoop(cur_kf, candidates);

    if (loop_num != -1)
    {
        std::cout << " candidates size " << candidates.size();
        std::cout << "candidates index: ";
        for (const auto &value : candidates)
        {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        bool bFind_best_loop = false;
        for (int i = min((int)candidates.size() - 1, 2); i >= 0; i--)
        {
            loop_index = candidates[i];
            // printf(" %d detect loop with %d , local_keyframelist_window size %d \n", cur_kf->index, loop_index, local_keyframelist_window.size());
            old_kf = getKeyFrame(loop_index);
            if (old_kf == nullptr)
            {
                printf("old kf is null\n");
                continue;
            }

            sensor_msgs::PointCloud2 pc;
            pcl::PointCloud<pcl::PointXYZ> cloud;

            // for (int i = 0; i < old_kf->reference_keypoints_data.size(); i++)
            // {
            //     cloud.points.push_back(pcl::PointXYZ(old_kf->reference_keypoints_data[i].point_3d_world.x, old_kf->reference_keypoints_data[i].point_3d_world.y, old_kf->reference_keypoints_data[i].point_3d_world.z));
            // }
            // pcl::toROSMsg(cloud, pc);
            // pc.header.frame_id = "world";
            // pc.header.stamp = ros::Time::now();
            // pub_oldKeyframe_cloud.publish(pc);

            bFind_best_loop = findConnection(cur_kf, old_kf);
        }

        if (bFind_best_loop && cur_kf->has_loop) // 计算相对位姿
        {
            if (loop_index < earliest_loop_index || earliest_loop_index == -1)
                earliest_loop_index = loop_index;

            if (1)
            {
                nav_msgs::Odometry odom_msg;
                odom_msg.header.stamp = ros::Time(cur_kf->time_stamp);
                odom_msg.header.frame_id = "world";
                odom_msg.pose.pose.position.x = cur_kf->PnP_T(0);
                odom_msg.pose.pose.position.y = cur_kf->PnP_T(1);
                odom_msg.pose.pose.position.z = cur_kf->PnP_T(2);
                Eigen::Quaterniond PnP_q(cur_kf->PnP_R);
                odom_msg.pose.pose.orientation.x = PnP_q.x();
                odom_msg.pose.pose.orientation.y = PnP_q.y();
                odom_msg.pose.pose.orientation.z = PnP_q.z();
                odom_msg.pose.pose.orientation.w = PnP_q.w();
                pub_pnp_pose.publish(odom_msg);
            }
            cout << "find best loop" << endl;
            bDetectRes = 1;
            bLoopKeepLost = false;
            cLostCount = 0;
            Vector3d w_t_old, w_t_cur_calc, vio_t_cur_calc, vio_t_cur;
            Matrix3d w_R_old, w_R_cur_calc, vio_R_cur_calc, vio_R_cur;
            old_kf->getVioPose(w_t_old, w_R_old);
            cur_kf->getVioPose(vio_t_cur, vio_R_cur);

            // test: 跑同一个包，理想情况下t是0，q是单位阵
            //  relative_t = Vector3d(0, 0, 0);
            //  relative_q = Eigen::Matrix3d::Identity();

            // bi_T_b0(w2)
            vio_t_cur_calc = cur_kf->PnP_T;
            vio_R_cur_calc = cur_kf->PnP_R;

            double shift_yaw;
            Matrix3d shift_r;
            Vector3d shift_t;
            if (use_imu)
            {
                shift_yaw = Utility::R2ypr(vio_R_cur_calc).x() - Utility::R2ypr(vio_R_cur).x();
                shift_r = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));
            }
            else
                shift_r = vio_R_cur_calc * vio_R_cur.transpose();
            shift_t = vio_t_cur_calc - vio_R_cur_calc * vio_R_cur.transpose() * vio_t_cur;
            // test

            r_drift = shift_r;
            t_drift = shift_t;

            loop_res = 1;
        }
    }
}

void PoseGraph::addKeyFrame(KeyFramePtr cur_kf)
{
    // shift to base frame
    Vector3d vio_t_cur, global_t_cur;
    Matrix3d vio_R_cur, global_R_cur;
    loop_res = 0;
    t_detectLoop = t_match = t_PnPRANSAC = 0;
    if (sequence_cnt != cur_kf->sequence)
    {
        sequence_cnt++;
        sequence_global_index.emplace_back(0);
        sequence_loop.push_back(0);
        w_t_vio = Eigen::Vector3d(0, 0, 0);
        w_r_vio = Eigen::Matrix3d::Identity();
        m_drift.lock();
        t_drift = Eigen::Vector3d(0, 0, 0);
        r_drift = Eigen::Matrix3d::Identity();
        m_drift.unlock();
    }
    // 方便调用vio的位姿，vio_t_cur和vio_R_cur
    cur_kf->getVioPose(vio_t_cur, vio_R_cur);
    // 由于两个坐标系不一样，所以进行sequence间的变换 (不过同一次回环的应该等于没有变换)
    // 在optimize之前，借助之前的drift进行变换(不过其实没有影响，因为优化的边是相对的)、
    // 之后所有的关键帧一直以这个默认的drift为准；实测下来一般z偏差比较大
    // 这里选择不加，避免过于复杂
    // vio_t_cur = w_r_vio * vio_t_cur + w_t_vio;
    // vio_R_cur = w_r_vio * vio_R_cur;

    // 同时更新vio和回环的位姿，都是进行sequence间的变换
    cur_kf->updateVioPose(vio_t_cur, vio_R_cur);
    global_t_cur = r_drift * vio_t_cur + t_drift;
    global_R_cur = r_drift * vio_R_cur;
    cur_kf->updatePose(global_t_cur, global_R_cur); // 先给出一个大概消除drift的位姿
    cur_kf->index = global_index;
    global_index++;
    sequence_global_index[sequence_cnt]++;
    bDetectRes = 0;
    std::vector<int> candidates;
    // printf(" index in last sequence %d \n", global_index - sequence_global_index[0]);

    int loop_num = -1, loop_index = -1;
    KeyFramePtr old_kf;

    TicToc tic_detectLoop;
    // bFactRes = getLoopFactRes(global_index, cur_kf); // 找在过去的帧中是哪个
    // printf(" loop fact res %d \n", bFactRes);
    // printf("RECALL: %f, PRECISION: %f, F1: %f\n", RECALL, PRECISION, 2 * RECALL * PRECISION / (RECALL + PRECISION));
    loop_num = detectLoop(cur_kf, candidates);
    t_detectLoop = tic_detectLoop.toc();

    if (loop_num != -1)
    {
        std::cout << " candidates size " << candidates.size();
        std::cout << "candidates index: ";
        for (const auto &value : candidates)
        {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        bool bFind_best_loop = false;
        for (int i = min((int)candidates.size() - 1, 2); i >= 0; i--)
        {
            loop_index = candidates[i];
            // printf(" %d detect loop with %d , local_keyframelist_window size %d \n", cur_kf->index, loop_index, local_keyframelist_window.size());
            old_kf = getKeyFrame(loop_index);
            if (old_kf == nullptr)
            {
                printf("old kf is null\n");
                continue;
            }

            sensor_msgs::PointCloud2 pc;
            pcl::PointCloud<pcl::PointXYZ> cloud;

            // for (int i = 0; i < old_kf->reference_keypoints_data.size(); i++)
            // {
            //     cloud.points.push_back(pcl::PointXYZ(old_kf->reference_keypoints_data[i].point_3d_world.x, old_kf->reference_keypoints_data[i].point_3d_world.y, old_kf->reference_keypoints_data[i].point_3d_world.z));
            // }
            // pcl::toROSMsg(cloud, pc);
            // pc.header.frame_id = "world";
            // pc.header.stamp = ros::Time::now();
            // pub_oldKeyframe_cloud.publish(pc);

            bFind_best_loop = findConnection(cur_kf, old_kf);
        }

        if (bFind_best_loop && cur_kf->has_loop) // 计算相对位姿
        {
            if (loop_index < earliest_loop_index || earliest_loop_index == -1)
                earliest_loop_index = loop_index;

            if (1)
            {
                nav_msgs::Odometry odom_msg;
                odom_msg.header.stamp = ros::Time(cur_kf->time_stamp);
                odom_msg.header.frame_id = "world";
                odom_msg.pose.pose.position.x = cur_kf->PnP_T(0);
                odom_msg.pose.pose.position.y = cur_kf->PnP_T(1);
                odom_msg.pose.pose.position.z = cur_kf->PnP_T(2);
                Eigen::Quaterniond PnP_q(cur_kf->PnP_R);
                odom_msg.pose.pose.orientation.x = PnP_q.x();
                odom_msg.pose.pose.orientation.y = PnP_q.y();
                odom_msg.pose.pose.orientation.z = PnP_q.z();
                odom_msg.pose.pose.orientation.w = PnP_q.w();
                pub_pnp_pose.publish(odom_msg);
            }
            cout << "find best loop" << endl;
            bDetectRes = 1;
            bLoopKeepLost = false;
            cLostCount = 0;
            Vector3d w_t_old, w_t_cur_calc, vio_t_cur_calc, vio_t_cur;
            Matrix3d w_R_old, w_R_cur_calc, vio_R_cur_calc, vio_R_cur;
            old_kf->getVioPose(w_t_old, w_R_old);
            cur_kf->getVioPose(vio_t_cur, vio_R_cur);

            Vector3d relative_t;
            Quaterniond relative_q;
            relative_t = cur_kf->getLoopRelativeT();
            relative_q = (cur_kf->getLoopRelativeQ()).toRotationMatrix();

            double relative_yaw = cur_kf->loop_info(7);

            // test: 跑同一个包，理想情况下t是0，q是单位阵
            //  relative_t = Vector3d(0, 0, 0);
            //  relative_q = Eigen::Matrix3d::Identity();

            // bi_T_b0(w2)
            vio_t_cur_calc = cur_kf->PnP_T;
            vio_R_cur_calc = cur_kf->PnP_R;
            // std::cout << "vio_t_cur_calc " << vio_t_cur_calc.transpose() << std::endl
            //           << " vio_R_cur_calc \n"
            //           << vio_R_cur_calc << std::endl;

            double shift_yaw;
            Matrix3d shift_r;
            Vector3d shift_t;
            if (use_imu)
            {
                shift_yaw = Utility::R2ypr(vio_R_cur_calc).x() - Utility::R2ypr(vio_R_cur).x();
                shift_r = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));
            }
            else
                shift_r = vio_R_cur_calc * vio_R_cur.transpose();
            shift_t = vio_t_cur_calc - vio_R_cur_calc * vio_R_cur.transpose() * vio_t_cur;
            // test
            if (!USE_PG_OPTIMIZE)
            {
                r_drift = shift_r;
                t_drift = shift_t;
            }

            // shift vio pose of whole sequence to the world frame
            // 是两段不同的sequence，同时这段sequence没回环过，推测是后边加上去的
            // 如果还没有loop过，那么给之前的所有帧都加一个，用PnP算出来的shift，算是有一个优化的初值，但是其实不是必须的
            // 觉得没影响，选择不加
            // if (old_kf->sequence != cur_kf->sequence && sequence_loop[cur_kf->sequence] == 0)
            // {
            //     printf("first best loop occur!!!!!!!!!!!!!!!!!!!!\n");
            //     w_r_vio = shift_r;
            //     w_t_vio = shift_t;
            //     //
            //     vio_t_cur = shift_r * vio_t_cur + shift_t;
            //     vio_R_cur = shift_r * vio_R_cur;
            //     cur_kf->updateVioPose(vio_t_cur, vio_R_cur);
            //     deque<KeyFramePtr >::iterator it = local_keyframelist_window.begin();
            //     for (; it != local_keyframelist_window.end(); it++)
            //     {
            //         if ((*it)->sequence == cur_kf->sequence) // 更新w2系下所有的帧
            //         {
            //             Vector3d vio_t_cur;
            //             Matrix3d vio_R_cur;
            //             (*it)->getVioPose(vio_t_cur, vio_R_cur);
            //             vio_t_cur = shift_r * vio_t_cur + w_t_vio;
            //             vio_R_cur = shift_r * vio_R_cur;
            //             (*it)->updateVioPose(vio_t_cur, vio_R_cur);
            //         }
            //     }
            //     sequence_loop[cur_kf->sequence] = 1;
            // }

            loop_res = 1;

            if (USE_PG_OPTIMIZE)
            {
                m_optimize_sig.lock();
                optimize_signal = 1;
                optimize_end_kf = cur_kf;
                // loop_edges.push_back(std::make_pair(cur_kf->index, old_kf->index));
                m_optimize_sig.unlock();
            }
        }
    }
    // TP += bFactRes & bDetectRes;
    // FN += bFactRes & (!bDetectRes);
    // FP += (!bFactRes) & bDetectRes;
    // RECALL = (float)TP / (TP + FN);
    // PRECISION = (float)TP / (TP + FP);
    // printf("RECALL: %f, PRECISION: %f, F1: %f\n", RECALL, PRECISION, 2 * RECALL * PRECISION / (RECALL + PRECISION));
    // DetectRes.push_back(bDetectRes);
    // FactRes.push_back(bFactRes);
    lock_guard<std::mutex> guard(m_keyframelist);

    Eigen::Quaterniond lio_q(lio_R);
    Eigen::Quaterniond vio_q_cur(vio_R_cur);
    Eigen::Quaterniond PnP_q(cur_kf->PnP_R);

    fout_reloTime.setf(ios::fixed, ios::floatfield);
    fout_reloTime.precision(9);
    fout_reloTime << cur_kf->time_stamp << " ";
    fout_reloTime.precision(9);
    fout_reloTime << loop_res << " "
                  << lio_t(0) << " " << lio_t(1) << " " << lio_t(2) << " " // 2-4
                  << lio_q.x() << " " << lio_q.y() << " " << lio_q.z() << " " << lio_q.w() << " "
                  << vio_t_cur(0) << " " << vio_t_cur(1) << " " << vio_t_cur(2) << " " // 9 -11
                  << vio_q_cur.x() << " " << vio_q_cur.y() << " " << vio_q_cur.z() << " " << vio_q_cur.w() << " "
                  << cur_kf->PnP_T(0) << " " << cur_kf->PnP_T(1) << " " << cur_kf->PnP_T(2) << " " // 16-18
                  << PnP_q.x() << " " << PnP_q.y() << " " << PnP_q.z() << " " << PnP_q.w() << " "  // 21-22
                  << rected_vio_t(0) << " " << rected_vio_t(1) << " " << rected_vio_t(2) << " "    // 23-25
                  << rected_vio_q.x() << " " << rected_vio_q.y() << " " << rected_vio_q.z() << " " << rected_vio_q.w() << " "
                  << relative_t(0) << " " << relative_t(1) << " " << relative_t(2) << " " << relative_yaw << " "
                  << final_matched_num << " " << cur_kf->index << " "                     // 34-35
                  << t_detectLoop << " " << t_match << " " << t_PnPRANSAC << " " << endl; // 36-38

    Vector3d t_corrected;
    Matrix3d R_corrected;
    cur_kf->getVioPose(t_corrected, R_corrected); // t就是vio_t_cur因为更新过了，但是这里需要再转换一次r_drift
    t_corrected = r_drift * t_corrected + t_drift;
    R_corrected = r_drift * R_corrected;
    // m_updated.lock();
    // bUpdated = true;
    // printf("bUpdated %d !\n", bUpdated);
    // m_updated.unlock();
    // t = vio_t_cur;
    // R_corrected = vio_R_cur;
    cur_kf->updatePose(t_corrected, R_corrected);

    if (globalkf_cloud->points.size())
    {
        sensor_msgs::PointCloud2 pc2;
        pcl::toROSMsg(*globalkf_cloud, pc2);
        pc2.header.frame_id = "world";
        pc2.header.stamp = ros::Time(cur_kf->time_stamp);
        pub_globalkf_pose.publish(pc2);
    }
    // draw local connection
    // if (SHOW_S_EDGE)
    // {
    //     deque<KeyFramePtr>::reverse_iterator rit = local_keyframelist_window.rbegin(); // 从后往前建立边的连接

    //     if (rit != local_keyframelist_window.rend())
    //     {
    //         Vector3d conncected_t;
    //         Matrix3d connected_R;
    //         if ((*rit)->sequence == cur_kf->sequence)
    //         {
    //             (*rit)->getPose(conncected_t, connected_R);
    //             posegraph_visualization->add_edge(t_corrected, conncected_t);
    //         }
    //         rit++;
    //     }
    // }
    // if (SHOW_L_EDGE)
    // {
    //     if (cur_kf->has_loop)
    //     {
    //         // printf("has loop \n");
    //         KeyFramePtr connected_KF = getKeyFrame(cur_kf->loop_index);
    //         Vector3d connected_t, P0;
    //         Matrix3d connected_R, R0;
    //         connected_KF->getPose(connected_t, connected_R);
    //         // cur_kf->getVioPose(P0, R0);
    //         // cur_kf->getPose(P0, R0);
    //         if (cur_kf->sequence > 0)
    //         {
    //             // printf("add loop into visual \n");
    //             posegraph_visualization->add_loopedge(cur_kf->PnP_T, connected_t + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
    //         }
    //     }
    // }

    // keyframelist.push_back(cur_kf);
    local_keyframelist_window.push_back(cur_kf);
    slideWindow_keyframelist();
    // global_keyframelist.push_back(old_kf);Window_keyframelist();
    // slideWindow();
}

void PoseGraph::slideWindow_keyframelist()
{
    if (local_keyframelist_window.size() > LOCAL_KF_WINDOW_SIZE)
    {
        KeyFramePtr old_kf = local_keyframelist_window.front();
        local_keyframelist_window.pop_front();
        // TODO: USE PARRALEX
        // use corrected pose
        Vector3d old_t;
        Matrix3d old_R;
        old_kf->getPose(old_t, old_R);

        if (!globalkf_cloud->points.size())
        {
            // global_keyframelist.push_back(old_kf);
            PointType p;
            p.x = old_t.x();
            p.y = old_t.y();
            p.z = old_t.z();
            p.intensity = old_kf->index;
            globalkf_cloud->points.push_back(p); // 没初始化会报错
        }

        // globalkf_cloud->points.back()->getPose(near_global_t, front_global_R);
        auto npc = globalkf_cloud->points.back();
        Vector3d near_global_t(npc.x, npc.y, npc.z);
        double dis = (near_global_t - old_t).norm();
        if (dis > 3.0)
        {
            // global_keyframelist.push_back(old_kf); // TODO: global_keyframelist最优调整，一直绕圈圈的情况
            PointType p;
            p.x = old_t.x();
            p.y = old_t.y();
            p.z = old_t.z();
            p.intensity = old_kf->index;
            globalkf_cloud->points.push_back(p);
        }
        // delete old_kf;
    }
}
// 相当与db.add(keyframe->query_brief_descriptors); + keyframelist.push_back(cur_kf);
void PoseGraph::loadKeyFrame(KeyFramePtr cur_kf, const std::vector<Eigen::Matrix<float, 256, 1>> &reference_spp_descriptors)
{
    cur_kf->index = global_index;
    global_index++;
    sequence_global_index[sequence_cnt]++;
    int loop_index = -1;
    // addKeyFrameIntoVoc(cur_kf);

    {
        lock_guard<std::mutex> guard(m_keyframelist);
        ref_keyframelist.push_back(cur_kf);
    }
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    if (DEBUG_IMAGE)
    {
        int feature_num = cur_kf->reference_keypoints_data.size();
        cv::resize(cur_kf->image, compressed_image, cv::Size(376, 240));
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        image_pool[cur_kf->index] = compressed_image;
    }
    Vector3d global_t_cur;
    Matrix3d global_R_cur;
    cur_kf->getPose(global_t_cur, global_R_cur); // 这个是从文件里读出来的

    dbi->spp_db->addRefPos(global_t_cur);
    dbi->spp_db->add(reference_spp_descriptors); // 参考airslam

    // DBoW2::WordIdToFeatures word_features;
    // DBoW2::BowVector bow_vector;
    // std::vector<DBoW2::WordId> word_of_features;
    // spp_db->FrameToBow(cur_kf->reference_keypoints_data, word_features, bow_vector, word_of_features);
    // spp_db->addFeatures(cur_kf->reference_keypoints_data);

    initBasePath(cur_kf);
}

void PoseGraph::loadKeyFrame(KeyFramePtr old_kf, const std::vector<cv::Mat> &reference_brief_descriptors)
{
    old_kf->index = global_index;
    global_index++;
    sequence_global_index[sequence_cnt]++;
    int loop_index = -1;
    // addKeyFrameIntoVoc(old_kf);

    {
        lock_guard<std::mutex> guard(m_keyframelist);
        ref_keyframelist.push_back(old_kf);
    }
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    if (DEBUG_IMAGE)
    {
        int feature_num = old_kf->reference_keypoints_data.size();
        cv::resize(old_kf->image, compressed_image, cv::Size(376, 240));
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        image_pool[old_kf->index] = compressed_image;
    }
    Vector3d global_t_cur;
    Matrix3d global_R_cur;
    old_kf->getPose(global_t_cur, global_R_cur); // 这个是从文件里读出来的
    // old_kf->reference_brief_descriptors = reference_brief_descriptors;
    dbi->orb_db->addRefPos(global_t_cur);
    dbi->orb_db->add(reference_brief_descriptors); // 参考airslam 222

    // reference_brief_descriptors
    // DBoW2::WordIdToFeatures word_features;
    // DBoW2::BowVector bow_vector;
    // std::vector<DBoW2::WordId> word_of_features;
    // spp_db->FrameToBow(old_kf->reference_keypoints_data, word_features, bow_vector, word_of_features);
    // spp_db->addFeatures(old_kf->reference_keypoints_data);

    initBasePath(old_kf);
}

void PoseGraph::initBasePath(KeyFramePtr old_kf)
{
    Vector3d t;
    Matrix3d R;
    old_kf->getPose(t, R);
    Quaterniond Q{R};
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(old_kf->time_stamp);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = t.x() + VISUALIZATION_SHIFT_X;
    pose_stamped.pose.position.y = t.y() + VISUALIZATION_SHIFT_Y;
    pose_stamped.pose.position.z = t.z();
    pose_stamped.pose.orientation.x = Q.x();
    pose_stamped.pose.orientation.y = Q.y();
    pose_stamped.pose.orientation.z = Q.z();
    pose_stamped.pose.orientation.w = Q.w();
    base_path.poses.push_back(pose_stamped);
    base_path.header = pose_stamped.header;

    // global_keyframelist.push_back(old_kf); // TODO: global_keyframelist最优调整，一直绕圈圈的情况
    PointType p;
    p.x = t.x();
    p.y = t.y();
    p.z = t.z();
    p.intensity = 0;
    base_pose_cloud->points.push_back(p);
}

KeyFramePtr PoseGraph::getKeyFrame(int index)
{
    //    unique_lock<mutex> lock(m_keyframelist);
    if (1)
    {
        return ref_keyframelist[index]; // fix bug
    }
    else
    {
        deque<KeyFramePtr>::iterator it = ref_keyframelist.begin();
        for (; it != ref_keyframelist.end(); it++)
        {
            if ((*it)->index == index)
                break;
        }
        if (it != ref_keyframelist.end())
            return *it;
        else
            return nullptr;
    }
}

KeyFramePtr PoseGraph::getLocalKeyFrame(int index)
{
    //    unique_lock<mutex> lock(m_keyframelist);
    deque<KeyFramePtr>::iterator it = local_keyframelist_window.begin();
    for (; it != local_keyframelist_window.end(); it++)
    {
        if ((*it)->index == index)
            break;
    }
    if (it != local_keyframelist_window.end())
        return *it;
    else
        return nullptr;
}

int PoseGraph::detectLoop(KeyFramePtr keyframe, std::vector<int> &candidates)
{
    int frame_index = keyframe->index;
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    if (DEBUG_IMAGE)
    {
        int feature_num = keyframe->query_keypoints_data.size();
        cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        image_pool[frame_index] = compressed_image;
    }
    TicToc tmp_t;
    // first query; then add this frame into database!
    QueryResults ret;
    TicToc t_query;
    bool bQueryPosGuess = 0;
    if (cLostCount > 100)
    {
        bLoopKeepLost = true;
        std::cout << "bLoopKeepLost! " << " cLostCount " << cLostCount << std::endl;
    }

    if (!bLoopKeepLost)
    {
        bQueryPosGuess = 1;
    }

    if (bQueryPosGuess)
    {
        Vector3d global_t_cur;
        Matrix3d global_R_cur;
        keyframe->getPose(global_t_cur, global_R_cur);
        if (0 == DETECTOR)
        {
            dbi->spp_db->setQueryPosGuess(global_t_cur);
        }
        else
        {
            dbi->orb_db->setQueryPosGuess(global_t_cur);
        }
    }

    double neighbor_score;
    if (0 == DETECTOR)
    {
        dbi->spp_db->query(keyframe->query_spp_descriptors, ret, 9, frame_index - MIN_QUERY_GAP);
    }
    else
    {
        dbi->orb_db->query(keyframe->query_brief_descriptors, ret, 9, frame_index - MIN_QUERY_GAP); // 222
    }
    // dbi->spp_db->query(keyframe->query_spp_descriptors, ret, 9, frame_index - MIN_QUERY_GAP);
    // printf("query time: %f", t_query.toc());
    // cout << "Searching for Image " << frame_index << ". " << ret << endl;

    TicToc t_add;

    // dbi->spp_db->add(keyframe->query_brief_descriptors); // 先不加，只用地图
    // 判断有没有被赋值
    // 暂时不能用，测试会扰乱query，暂时不明确原因
    double threshold = 0.006;
    if (RELATIVE_THRESHOLD) //
    {
        if (local_keyframelist_window.size())
            neighbor_kf = local_keyframelist_window.back();

        if (neighbor_kf != nullptr)
        {
            // dbi->spp_db->calcNeighborScore(keyframe->query_brief_descriptors, neighbor_kf->query_brief_descriptors, neighbor_score);
        }
        else
        {
            std::cout << "init!" << std::endl;
            return -1; // skip this frame
        }
        threshold = neighbor_score * 0.06;
        std::cout << " relative threshold: " << threshold << std::endl;
    }

    // printf("add feature time: %f", t_add.toc());
    //  ret[0] is the nearest neighbour's score. threshold change with neighour score
    bool find_loop = false;
    cv::Mat loop_result;
    if (DEBUG_IMAGE)
    {
        loop_result = compressed_image.clone();
        if (ret.size() > 0)
            putText(loop_result, "index:  " + to_string(frame_index), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255)); //
        if (RELATIVE_THRESHOLD)                                                                                                            //
        {
            cv::resize(neighbor_kf->image, compressed_image, cv::Size(376, 240));
            cv::Mat tmp_image = compressed_image.clone();
            putText(tmp_image, "neighbor image score:" + to_string(neighbor_score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
            cv::hconcat(tmp_image, loop_result, loop_result);
        }
    }

    // a good match with its nerghbour

    // if (ret.size() >= 1 && ret[0].Score > 0.05)
    for (unsigned int i = 0; i < ret.size(); i++) // loop through all the candidates for rotation
    {
        if (ret[i].Score > threshold) // 0.20*0.5 = 0.010
        // if (ret[i].Score > 0.015)
        {
            find_loop = true;
            int tmp_index = ret[i].Id;
            // visual loop result
            if (DEBUG_IMAGE) // do not show here, record the index and set find_loop true, will show later
            {
                auto it = image_pool.find(tmp_index);
                cv::Mat tmp_image = (it->second).clone();
                putText(tmp_image, "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
                cv::hconcat(loop_result, tmp_image, loop_result);
            }
        }
    }
    // debug release
    if (DEBUG_IMAGE)
    {
        ostringstream path;
        path << "/home/ubuntu/Documents/dataplace/huzhou/loop_output/loop_detect/"
             << frame_index << ".jpg";
        cv::imwrite(path.str().c_str(), loop_result);

        cv::imshow("loop_result", loop_result);
        cv::waitKey(20);
    }

    if (find_loop && frame_index > MIN_QUERY_GAP)
    {
        int min_index = -1, best_index = -1;
        float max_score = 0;

        if (1)
        {
            for (unsigned int i = 0; i < ret.size(); i++)
            {
                if (ret[i].Score < threshold)
                    continue;

                candidates.push_back(ret[i].Id); // 返回candidate id
            }
            return candidates.size(); // 当有多个匹配时，返回最小的index
        }
        else
        {
            // divide into 3 groups according to the location
            std::vector<int> groups[3];
            std::vector<Eigen::Vector3d> groups_t[3];
            for (unsigned int i = 0; i < ret.size(); i++)
            {
                if (ret[i].Score < threshold)
                    continue;

                KeyFramePtr loop_kf = getKeyFrame(ret[i].Id);
                if (loop_kf == nullptr)
                    continue;
                Vector3d loop_t;
                Matrix3d loop_R;
                loop_kf->getPose(loop_t, loop_R);
                for (int group_index = 0; group_index < 3; group_index++)
                {
                    if (groups_t[group_index].empty())
                    {
                        groups_t[group_index].push_back(loop_t);
                        groups[group_index].push_back(ret[i].Id);
                        break;
                    }
                    else
                    {
                        if ((groups_t[group_index].front() - loop_t).norm() < 5)
                        {
                            groups_t[group_index].push_back(loop_t);
                            groups[group_index].push_back(ret[i].Id);
                            break;
                        }
                    }
                }
            }

            // return 3 best in each group
            for (int i = 0; i < 3; i++)
            {
                if (groups[i].empty())
                    continue;

                candidates.push_back(groups[i].front());
                // // if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > threshold))
                // //     min_index = ret[i].Id;
                // if (ret[i].Score < threshold)
                //     continue;

                // if (ret[i].Score > max_score ||)
                // {
                //     candidates.push_back(ret[i].Id); // 返回candidate id
                // }
            }
            return candidates.size(); // 当有多个匹配时，返回最小的index
        }
    }
    else
    {
        loop_res = 0;
        cLostCount++;
        return -1;
    }
}

void PoseGraph::addRefKeyFrame(KeyFramePtr keyframe)
{
    lock_guard<std::mutex> guard(m_keyframelist);
    ref_keyframelist.push_back(keyframe);
}

void PoseGraph::addKeyFrameIntoVoc(KeyFramePtr keyframe)
{
}

void PoseGraph::kNNMatcher(FeatureData &cur_query, FeatureData &old_ref, std::vector<cv::DMatch> &matches, bool crossCheck)
{
    matches.clear();
    cv::Mat query;
    cv::Mat train;

    for (size_t i = 0; i < cur_query.size(); i++)
    {
        query.push_back(cur_query[i].brief_descriptor);
    }
    for (size_t i = 0; i < old_ref.size(); i++)
    {
        train.push_back(old_ref[i].brief_descriptor);
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> vmatches;
    matcher.knnMatch(query, train, vmatches, 2);

    for (size_t i = 0; i < vmatches.size(); i++)
    {
        if (vmatches[i][0].distance < 0.8 * vmatches[i][1].distance || vmatches[i].size() == 1)
        {
            matches.push_back(vmatches[i][0]);
        }
    }
}

// 判断是否有效
bool PoseGraph::findConnection(KeyFramePtr cur_kf, const KeyFramePtr old_kf)
{
    // printf("find Connection\n");
    vector<uchar> status;
    // KeyPointsData matched_cur_kpd = cur_kf->query_keypoints_data;
    // KeyPointsData matched_old_kpd; // wait push back
    std::vector<cv::Point2f> matched_cur_2d;
    std::vector<cv::Point2f> matched_cur_2d_norm;
    std::vector<cv::Point2f> matched_old_2d;
    std::vector<cv::Point3f> matched_old_3d;
    std::vector<cv::DMatch> matches;
    // matched_3d_cur = query_point_3d;
    // matched_2d_cur = cur_kf->query_keypoints_data.keypoint_2d_uv.pt; // 这个一定是整数的，因为输入的时候就是整数
    // matched_2d_cur_norm = cur_kf->query_point_2d_norm;
    // matched_id = point_id;

    // 删除无效点
    // if (old_kf->sequence == sequence)
    // {
    // 	// printf("same sequence \n");
    // 	return false;
    // }

    TicToc tic_match;
    // printf("search by des\n");
    // first three old and status are output, the rest are input

    if (0 == DETECTOR)
    {
        point_matcher->MatchingPoints(cur_kf->query_keypoints_data, old_kf->reference_keypoints_data, matches, true);
        for (int i = 0; i < matches.size(); i++)
        {
            matched_cur_2d.push_back(cur_kf->query_keypoints_data[matches[i].queryIdx].keypoint_2d_uv.pt);
            matched_cur_2d_norm.push_back(cur_kf->query_keypoints_data[matches[i].queryIdx].keypoint_2d_norm.pt);
            matched_old_2d.push_back(old_kf->reference_keypoints_data[matches[i].trainIdx].keypoint_2d_uv.pt);
            matched_old_3d.push_back(old_kf->reference_keypoints_data[matches[i].trainIdx].point_3d_world);
        }
    }
    else if (1 == DETECTOR)
    {
        // cur_kf->searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, matched_3d_old, status, old_kf->reference_brief_descriptors, old_kf->reference_keypoints, old_kf->reference_keypoints_norm, old_kf->reference_keypoints_3d_world);
        if (1)
        {
            cur_kf->searchByBRIEFDes(old_kf, matched_cur_2d, matched_cur_2d_norm, matched_old_2d, matched_old_3d, status);
        }
        else
        {
            kNNMatcher(cur_kf->query_keypoints_data, old_kf->reference_keypoints_data, matches, true);
            for (int i = 0; i < matches.size(); i++)
            {
                matched_cur_2d.push_back(cur_kf->query_keypoints_data[matches[i].queryIdx].keypoint_2d_uv.pt);
                matched_cur_2d_norm.push_back(cur_kf->query_keypoints_data[matches[i].queryIdx].keypoint_2d_norm.pt);
                matched_old_2d.push_back(old_kf->reference_keypoints_data[matches[i].trainIdx].keypoint_2d_uv.pt);
                matched_old_3d.push_back(old_kf->reference_keypoints_data[matches[i].trainIdx].point_3d_world);
            }
        }
    }

    // 出来的vector一开始都是和brief_descriptors长度一样的，其中status是0的是无效的
    // 而matched_2d_cur等_cur的本来就和brief_descriptors一样长
    // reduceVector(matched_cur_kpd, status);

    // printf("search by des finish\n");
    t_match += tic_match.toc();

#if 1
    if (DEBUG_IMAGE)
    {

        int gap = 10;
        cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
        cv::Mat gray_img, loop_match_img;
        cv::Mat old_img = old_kf->image;
        cv::hconcat(cur_kf->image, gap_image, gap_image);
        cv::hconcat(gap_image, old_img, gray_img);
        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
        // draw  match
        for (int i = 0; i < (int)matched_cur_2d.size(); i++)
        {
            cv::Point2f cur_pt = matched_cur_2d[i];
            cv::Point2f old_pt = matched_old_2d[i];
            cv::circle(loop_match_img, cur_pt, 3, cv::Scalar(255, 0, 0), -1);
            old_pt.x += (COL + gap);
            cv::circle(loop_match_img, old_pt, 3, cv::Scalar(255, 0, 0), -1);
            cv::line(loop_match_img, cur_pt, old_pt, cv::Scalar(0, 255, 0), 2, cv::LINE_AA, 0);
        }

        // ostringstream path, path1, path2;
        // path << "/home/ubuntu/Documents/dataplace/huzhou/loop_output/loop_image1/"
        // 		"seq"
        // 	 << cur_kf->sequence << "-"
        // 	 << cur_kf->index << "-"
        // 				 "oseq"
        // 	 << old_kf->sequence << "-"
        // 	 << old_kf->index << "-" << "1descriptor_match.jpg";
        // cv::imwrite(path.str().c_str(), loop_match_img);

        // debug
        // printf("loop connection show\n");
        cv::imshow("descriptor_match", loop_match_img);
        cv::waitKey(10);
    }
#endif
    status.clear();
    /*
    FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_cur_norm, status);
    reduceVector(matched_2d_old_norm, status);
    reduceVector(matched_3d_cur, status);
    reduceVector(matched_id, status);
    */

    Eigen::Vector3d PnP_T_old;
    Eigen::Matrix3d PnP_R_old;

    // old_kf->T_w_i = old_kf->T_w_i; // 感觉用回环的更好一点，反正激光的一定是一样的
    // old_kf->R_w_i = old_kf->R_w_i; // 可能有地方把它改变了..

    // debug
    // printf("loop matched 2d num %d \n", (int)matched_2d_cur.size());
    cv::Mat gray_img, loop_match_img;

    if ((int)matched_cur_2d.size() > MIN_LOOP_NUM)
    {
        // printf("PNP RANSAC\n");
        status.clear();
        TicToc tic_PnPRANSAC;
        double reprojection_error = 0;
        if (1)
        {
            cur_kf->PnPRANSAC(matched_cur_2d_norm, matched_old_3d, status, PnP_T_old, PnP_R_old, reprojection_error);
            // PnPRANSAC2(matched_2d_old_norm, matched_3d_cur, status, PnP_T_old, PnP_R_old);
            // 如果自己和自己配的话，不会有ransac，基本都是对的；(还有一个现象也很正常，pnp R都是[0,0,1;-1,0,0;0,-1,0]，T[0,0,0]附近)这是因为没有tranpose...
        }
        else
        {
            // cur_kf->PnPRANSAC(matched_old_kpd, matched_cur_kpd, status, PnP_T_old, PnP_R_old, reprojection_error);
        }

        reduceVector(matched_cur_2d, status);
        reduceVector(matched_cur_2d_norm, status);
        reduceVector(matched_old_2d, status);
        reduceVector(matched_old_3d, status);

        t_PnPRANSAC += tic_PnPRANSAC.toc();

#if 1

        if (DEBUG_IMAGE)
        {
            // printf(" DEBUG_IMAGE\n");
            int gap = 10;
            cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));

            cv::Mat old_img = old_kf->image;
            cv::hconcat(cur_kf->image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);

            for (int i = 0; i < (int)matched_cur_2d.size(); i++)
            {
                cv::Point2f cur_pt = matched_cur_2d[i];
                cv::Point2f old_pt = matched_old_2d[i];
                cv::circle(loop_match_img, cur_pt, 3, cv::Scalar(0, 0, 255), -1);
                old_pt.x += (COL + gap);
                cv::circle(loop_match_img, old_pt, 3, cv::Scalar(0, 0, 255), -1);
                cv::line(loop_match_img, cur_pt, old_pt, cv::Scalar(0, 255, 0), 2, cv::LINE_AA, 0);
            }

            cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
            putText(notation, "current frame: " + to_string(cur_kf->index) + "  sequence: " + to_string(cur_kf->sequence), cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

            putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
            cv::vconcat(notation, loop_match_img, loop_match_img);

            // printf("after RANSAC matched 2d num %d \n", (int)matched_2d_cur.size());

            if ((int)matched_cur_2d.size() > MIN_LOOP_NUM)
            {
                // ostringstream path;
                // path << "/home/ubuntu/Documents/dataplace/huzhou/loop_output/loop_image1/"
                // 		"seq"
                // 	 << cur_kf->sequence << "-"
                // 	 << cur_kf->index << "-"
                // 				 "oseq "
                // 	 << old_kf->sequence << "-"
                // 	 << old_kf->index << "-" << "3pnp_match.jpg";
                // cv::imwrite(path.str().c_str(), loop_match_img);

                // debug
                // printf("loop connection show\n");
                cv::imshow("loop connection", loop_match_img);
                cv::waitKey(1);

                cv::Mat thumbimage;
                cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
                msg->header.stamp = ros::Time(cur_kf->time_stamp);
                pub_match_img.publish(msg);
            }
        }
#endif
    }

    if ((int)matched_cur_2d.size() > MIN_LOOP_NUM)
    {

        // cout << "origin_vio_T " << origin_vio_T.transpose() << endl;
        // cout << "origin_vio_R " << endl
        // 	 << origin_vio_R << endl;

        // cout << "old_vio_T " << old_kf->T_w_i.transpose() << endl;
        // cout << "old_vio_R " << endl
        // 	 << old_kf->R_w_i << endl;
        // 通过loop_error_t来判断是否值得进行回环
        loop_error_t = cur_kf->vio_R_w_i.transpose() * (PnP_T_old - cur_kf->vio_t_w_i);
        loop_error_q = cur_kf->vio_R_w_i.transpose() * PnP_R_old;

        // bi_T_bj(w1) ?
        relative_t = old_kf->R_w_i.transpose() * (PnP_T_old - old_kf->T_w_i);
        relative_q = old_kf->R_w_i.transpose() * PnP_R_old;
        relative_yaw = Utility::normalizeAngle(Utility::R2ypr(relative_q.toRotationMatrix()).x());
        // relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
        // relative_q = PnP_R_old.transpose() * origin_vio_R;
        final_matched_num = (int)matched_cur_2d.size();

        // printf("PNP relative\n");
        // cout << "pnp relative_t " << relative_t.transpose() << endl;
        // cout << "pnp relative_yaw " << relative_yaw << endl;

        if (cLostCount > 10)
        {
            Z_THRESHOLD = 2.0;
        }
        else
        {
            Z_THRESHOLD = 0.8;
        }

        // std::cout << "old_kf->cam_id " << old_kf->cam_id << " old_kf->index " << old_kf->index << " old_kf->sequence " << old_kf->sequence << std::endl;
        // Eigen::Quaterniond old_q(old_kf->R_w_i);
        // old_q.normalize();
        // std::cout << " old_kf->R_w_i " << old_q.coeffs().transpose() << std::endl;
        // std::cout << " old_kf->R_w_i" << old_q.x() << " " << old_q.y() << " " << old_q.z() << " " << old_q.w() << std::endl;

        relative_q.normalize();
        std::cout << " pnp relative_q  " << relative_q.coeffs().transpose() << std::endl;
        std::cout << " pnp relative_q " << relative_q.x() << " " << relative_q.y() << " " << relative_q.z() << " " << relative_q.w() << std::endl;
        cout << "pnp relative_yaw " << relative_yaw << endl;
        cout << "pnp relative_t " << relative_t.transpose() << endl;
        bool yaw_good = abs(relative_yaw) < 30.0;
        // loop_error_t.norm() > 0.1 && 约束仍然要有
        if (yaw_good && relative_t.norm() < 8.0 && abs(relative_t.z()) < Z_THRESHOLD) //&& abs(relative_t.z()) < 0.5
        {
            // 根据score选择最好的:pnp分布最在两边而且重投影误差小

            cur_kf->has_loop = true;
            cur_kf->PnP_T = PnP_T_old;
            cur_kf->PnP_R = PnP_R_old;
            cur_kf->loop_index = old_kf->index;
            cur_kf->loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
                relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
                relative_yaw;

            // debug
            // 求出和激光的相对位姿
            Eigen::Vector3d relative_t_lidar = PnP_T_old - lio_t;
            // cout << "relative_t_lidar norm " << relative_t_lidar.norm() << endl;
            // std::cout << "PnP_T_old " << PnP_T_old.transpose() << std::endl;
            //   << " PnP_R_old \n"
            //   << PnP_R_old << std::endl;
            // if (DEBUG_IMAGE)
            // {
            //     if (relative_t_lidar.norm() > 1.0)
            //     {
            //         // cout << "relative_t_lidar " << relative_t_lidar.transpose() << endl;
            //         // cout << "relative_t_lidar norm " << relative_t_lidar.norm() << endl;
            //         ostringstream path;
            //         path << "/home/ubuntu/Desktop/loop_output/loop_image_FP/"
            //              << cur_kf->sequence << "-"
            //              << cur_kf->index << "-"
            //                                  "oseq "
            //              << old_kf->sequence << "-"
            //              << old_kf->index << "-"
            //              << "tnorm-" << relative_t_lidar.norm() << "-"
            //              << "3pnp_match.jpg";
            //         cv::imwrite(path.str().c_str(), loop_match_img);
            //     }
            //     else
            //     {
            //         ostringstream path;
            //         path << "/home/ubuntu/Desktop/loop_output/loop_image_TP/"
            //              << cur_kf->sequence << "-"
            //              << cur_kf->index << "-"
            //                                  "oseq "
            //              << old_kf->sequence << "-"
            //              << old_kf->index << "-"
            //              << "tnorm-" << relative_t_lidar.norm() << "-"
            //              << "3pnp_match.jpg";
            //         cv::imwrite(path.str().c_str(), loop_match_img);
            //     }
            // }

            // cout << "pnp relative_t " << relative_t.transpose() << endl;
            // cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
            return true;
        }
        else
        {
            printf("PNP relative fail!\n");

            // if (DEBUG_IMAGE)
            // {
            //     ostringstream path;
            //     path << "/home/nv/laji_ws/vins_output/loop_match_images/"
            //          << cur_kf->sequence << "-"
            //          << cur_kf->index << "-"
            //                              "oseq "
            //          << old_kf->sequence << "-"
            //          << old_kf->index << "-"
            //          << "tnorm-" << relative_t.norm() << "-"
            //          << "yaw-" << relative_yaw << "-"
            //          << "3pnp_match.jpg";
            //     cv::imwrite(path.str().c_str(), loop_match_img);

            //     // cv::imshow("loop_result", loop_result);
            //     // cv::waitKey(20);
            // }

            loop_res = 2;
            cLostCount++;
        }
    }

    // printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());
    return false;
}

void PoseGraph::optimize4DoF()
{
    while (true)
    {
        int optimize_end_index = -1;
        int first_looped_index = -1;

        if (optimize_signal)
        {
            optimize_end_index = optimize_end_kf->index;
            first_looped_index = earliest_loop_index;

            double local_kf_dis = SKIP_DIST * local_keyframelist_window.size();
            if (local_kf_dis > 4.0) //&& local_kf_dis > 4.0
            {
                printf("optimize pose graph \n");
                TicToc tmp_t;
                lock_guard<std::mutex> guard(m_keyframelist);

                ceres::Problem problem;
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                // options.minimizer_progress_to_stdout = true;
                // options.max_solver_time_in_seconds = SOLVER_TIME * 3;
                options.max_num_iterations = 30;
                ceres::Solver::Summary summary;
                ceres::LossFunction *loss_function;
                loss_function = new ceres::HuberLoss(0.1);
                // loss_function = new ceres::CauchyLoss(1.0);
                ceres::LocalParameterization *angle_local_parameterization =
                    AngleLocalParameterization::Create();

                deque<KeyFramePtr>::iterator it;
                std::deque<int> oldkfindex_in_array, curkfindex_in_array;
                for (int i = 0; i < local_keyframelist_window.size(); i++)
                {
                    curkfindex_in_array.push_back(local_keyframelist_window[i]->index);

                    if (local_keyframelist_window[i]->has_loop)
                    {
                        bool bUnique = 1;
                        // int kkkk = loop_edges[i].second;
                        int old_kf_index = local_keyframelist_window[i]->loop_index;
                        auto old_iia_it = std::find(oldkfindex_in_array.begin(), oldkfindex_in_array.end(), old_kf_index);
                        if (old_iia_it != oldkfindex_in_array.end()) // 重复
                        {
                            local_keyframelist_window[i]->optimize_buf_index = old_iia_it - oldkfindex_in_array.begin();
                            bUnique = 0;
                        }
                        if (bUnique)
                        {
                            oldkfindex_in_array.push_back(old_kf_index); // 不重复的，旧的用第一次的index
                            local_keyframelist_window[i]->optimize_buf_index = oldkfindex_in_array.size() - 1;
                        }
                    }
                }

                // for (int i = 0; i < kfindex_in_array.size(); i++)
                // {
                //     cur_kf->optimize_buf_index = i;
                // }

                int max_length = oldkfindex_in_array.size() + curkfindex_in_array.size();

                // w^t_i   w^q_i
                double t_array[max_length][3];
                Quaterniond q_array[max_length];
                double euler_array[max_length][3];
                double sequence_array[max_length];

                for (int i = 0; i < oldkfindex_in_array.size(); i++)
                {
                    // old kf

                    // 从最早回环上的开始找，拿出之前的位姿参与构建残差
                    //  首先是加入t_array和q_array以及冗余的euler_array
                    // (*it)->optimize_buf_index = i;
                    Quaterniond tmp_q;
                    Matrix3d tmp_r;
                    Vector3d tmp_t;
                    auto old_kf = getKeyFrame(oldkfindex_in_array[i]);
                    old_kf->getVioPose(tmp_t, tmp_r);
                    tmp_q = tmp_r;
                    int array_index = i;
                    t_array[array_index][0] = tmp_t(0);
                    t_array[array_index][1] = tmp_t(1);
                    t_array[array_index][2] = tmp_t(2);
                    q_array[array_index] = tmp_q;

                    Vector3d euler_angle = Utility::R2ypr(tmp_q.toRotationMatrix());
                    euler_array[array_index][0] = euler_angle.x();
                    euler_array[array_index][1] = euler_angle.y();
                    euler_array[array_index][2] = euler_angle.z();

                    sequence_array[array_index] = old_kf->sequence;

                    // 加入PGO优化问题
                    problem.AddParameterBlock(euler_array[array_index], 1, angle_local_parameterization); // 优化从过去第一帧到现在的所有帧的位姿
                    problem.AddParameterBlock(t_array[array_index], 3);

                    // 固定前三帧以及第一个序列(参考序列的所有帧)
                    if (old_kf->index <= first_looped_index || old_kf->sequence == 0)
                    {
                        problem.SetParameterBlockConstant(euler_array[array_index]);
                        problem.SetParameterBlockConstant(t_array[array_index]);
                    }
                }

                for (int i = 0; i < curkfindex_in_array.size(); i++)
                {
                    auto cur_kf = getLocalKeyFrame(curkfindex_in_array[i]);
                    Quaterniond tmp_q;
                    Matrix3d tmp_r;
                    Vector3d tmp_t;
                    cur_kf->getVioPose(tmp_t, tmp_r);
                    //  提前减去vio的漂移，降低优化难度
                    tmp_t = r_drift * tmp_t + t_drift;
                    tmp_r = r_drift * tmp_r;
                    tmp_q = tmp_r;
                    int array_index = oldkfindex_in_array.size() + i;
                    t_array[array_index][0] = tmp_t(0);
                    t_array[array_index][1] = tmp_t(1);
                    t_array[array_index][2] = tmp_t(2);
                    q_array[array_index] = tmp_q;

                    Vector3d euler_angle = Utility::R2ypr(tmp_q.toRotationMatrix());
                    euler_array[array_index][0] = euler_angle.x();
                    euler_array[array_index][1] = euler_angle.y();
                    euler_array[array_index][2] = euler_angle.z();

                    sequence_array[array_index] = cur_kf->sequence;

                    // 加入PGO优化问题
                    problem.AddParameterBlock(euler_array[array_index], 1, angle_local_parameterization); // 优化从过去第一帧到现在的所有帧的位姿
                    problem.AddParameterBlock(t_array[array_index], 3);
                    int cur_index_in_array = array_index;

                    if (cur_kf->has_loop)
                    {
                        int old_kf_index = cur_kf->loop_index;
                        // auto old_kf = getKeyFrame(old_kf_index);
                        int loop_index_in_array = cur_kf->optimize_buf_index;

                        // add loop edge
                        if (old_kf_index >= first_looped_index)
                        {
                            // std::cout << " loop index " << (*it)->loop_index << std::endl;
                            // add edge
                            Vector3d euler_conncected = Utility::R2ypr(tmp_r);
                            Vector3d relative_t;
                            relative_t = cur_kf->getLoopRelativeT();
                            double relative_yaw = cur_kf->getLoopRelativeYaw();
                            ceres::CostFunction *cost_function = FourDOFWeightError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                                            relative_yaw, euler_conncected.y(), euler_conncected.z());
                            problem.AddResidualBlock(cost_function, loss_function, euler_array[loop_index_in_array],
                                                     t_array[loop_index_in_array],
                                                     euler_array[cur_index_in_array],
                                                     t_array[cur_index_in_array]);
                        }
                    }

                    // add edge
                    // int j = 1;
                    for (int j = 1; j <= 3; j++)
                    {
                        if (sequence_array[cur_index_in_array] == sequence_array[cur_index_in_array - j])
                        {
                            Vector3d euler_conncected = Utility::R2ypr(q_array[cur_index_in_array - j].toRotationMatrix());
                            Vector3d relative_t(t_array[cur_index_in_array][0] - t_array[cur_index_in_array - j][0], t_array[cur_index_in_array][1] - t_array[cur_index_in_array - j][1], t_array[cur_index_in_array][2] - t_array[cur_index_in_array - j][2]);
                            relative_t = q_array[cur_index_in_array - j].inverse() * relative_t;
                            double relative_yaw = euler_array[cur_index_in_array][0] - euler_array[cur_index_in_array - j][0];
                            ceres::CostFunction *cost_function = FourDOFError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                                      relative_yaw, euler_conncected.y(), euler_conncected.z());
                            problem.AddResidualBlock(cost_function, NULL, euler_array[cur_index_in_array - j],
                                                     t_array[cur_index_in_array - j],
                                                     euler_array[cur_index_in_array],
                                                     t_array[cur_index_in_array]);
                        }
                    }

                    if (cur_kf->index == optimize_end_index)
                        break;
                }

                ceres::Solve(options, &problem, &summary);
                std::cout << summary.BriefReport() << "\n";

                // printf("pose optimization time: %f \n", tmp_t.toc());
                /*
                for (int j = 0 ; j < i; j++)
                {
                    printf("optimize i: %d p: %f, %f, %f\n", j, t_array[j][0], t_array[j][1], t_array[j][2] );
                }
                */

                int i = oldkfindex_in_array.size(); // update frome curkfindex_in_array
                for (it = local_keyframelist_window.begin(); it != local_keyframelist_window.end(); it++)
                {
                    if ((*it)->index < first_looped_index)
                        continue;
                    Quaterniond tmp_q;
                    tmp_q = Utility::ypr2R(Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
                    Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
                    Matrix3d tmp_r = tmp_q.toRotationMatrix();
                    (*it)->updatePose(tmp_t, tmp_r); // 把优化后的结果更新回去，这时候就只更新回环的位姿了

                    if ((*it)->index == optimize_end_index)
                        break;
                    i++;
                }

                Vector3d cur_t, vio_t;
                Matrix3d cur_r, vio_r;
                // KeyFramePtr cur_kf = local_keyframelist_window.back();
                optimize_end_kf->getPose(cur_t, cur_r);
                optimize_end_kf->getVioPose(vio_t, vio_r);
                // throw degenerate solution
                Vector3d t_diff = cur_t - optimize_end_kf->PnP_T;
                std::cout << RED << "t_diff " << t_diff.norm() << TAIL << std::endl;
                if (t_diff.norm() < 5.0 && summary.IsSolutionUsable())
                {
                    lock_guard<std::mutex> guard(m_drift);
                    // res -> delta_t_drift
                    // t_drift = t_drift + delta_t_drift;
                    last_t_drift = t_drift;
                    last_r_drift = r_drift;
                    yaw_drift = (Utility::R2ypr(cur_r).x() - Utility::R2ypr(vio_r).x());
                    r_drift = Utility::ypr2R(Vector3d(yaw_drift, 0, 0));
                    t_drift = cur_t - r_drift * vio_t;
                    pub_step = 0;
                    cout << BLUE << "t_drift " << t_drift.transpose() << endl;
                    cout << "r_drift " << Utility::R2ypr(r_drift).transpose() << endl;
                    cout << "yaw drift " << yaw_drift << TAIL << endl;
                    // updatePath();
                }
                // no more need to update
                // it++;
                // for (; it != local_keyframelist_window.end(); it++)
                // {
                //     Vector3d t;
                //     Matrix3d R;
                //     (*it)->getVioPose(t, R);
                //     t = r_drift * t + t_drift;
                //     R = r_drift * R;
                //     (*it)->updatePose(t, R);
                // }
            }
        }

        m_optimize_sig.lock();
        optimize_signal = 0;
        m_optimize_sig.unlock();

        std::chrono::milliseconds dura(1000);
        std::this_thread::sleep_for(dura);
    }

    return;
}

// void PoseGraph::optimize6DoF()
// {
//     while (true)
//     {
//         int optimize_end_index = -1;
//         int first_looped_index = -1;
//         m_optimize_sig.lock();
//         while (!optimize_buf.empty())
//         {
//             optimize_end_index = optimize_buf.front();
//             first_looped_index = earliest_loop_index;
//             optimize_buf.pop();
//         }
//         m_optimize_sig.unlock();
//         if (optimize_end_index != -1)
//         {
//             printf("optimize pose graph \n");
//             TicToc tmp_t;
//             m_keyframelist.lock();
//             KeyFramePtr cur_kf = getKeyFrame(optimize_end_index);

//             int max_length = optimize_end_index + 1;

//             // w^t_i   w^q_i
//             double t_array[max_length][3];
//             double q_array[max_length][4];
//             double sequence_array[max_length];

//             ceres::Problem problem;
//             ceres::Solver::Options options;
//             options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//             // ptions.minimizer_progress_to_stdout = true;
//             // options.max_solver_time_in_seconds = SOLVER_TIME * 3;
//             options.max_num_iterations = 5;
//             ceres::Solver::Summary summary;
//             ceres::LossFunction *loss_function;
//             loss_function = new ceres::HuberLoss(0.1);
//             // loss_function = new ceres::CauchyLoss(1.0);
//             ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();

//             deque<KeyFramePtr >::iterator it;

//             int i = 0;
//             for (it = keyframelist.begin(); it != keyframelist.end(); it++)
//             {
//                 if ((*it)->index < first_looped_index)
//                     continue;
//                 (*it)->optimize_buf_index = i;
//                 Quaterniond tmp_q;
//                 Matrix3d tmp_r;
//                 Vector3d tmp_t;
//                 (*it)->getVioPose(tmp_t, tmp_r);
//                 tmp_q = tmp_r;
//                 t_array[i][0] = tmp_t(0);
//                 t_array[i][1] = tmp_t(1);
//                 t_array[i][2] = tmp_t(2);
//                 q_array[i][0] = tmp_q.w();
//                 q_array[i][1] = tmp_q.x();
//                 q_array[i][2] = tmp_q.y();
//                 q_array[i][3] = tmp_q.z();

//                 sequence_array[i] = (*it)->sequence;

//                 problem.AddParameterBlock(q_array[i], 4, local_parameterization);
//                 problem.AddParameterBlock(t_array[i], 3);

//                 if ((*it)->index == first_looped_index || (*it)->sequence == 0)
//                 {
//                     problem.SetParameterBlockConstant(q_array[i]);
//                     problem.SetParameterBlockConstant(t_array[i]);
//                 }

//                 // add edge
//                 for (int j = 1; j < 5; j++)
//                 {
//                     if (i - j >= 0 && sequence_array[i] == sequence_array[i - j])
//                     {
//                         Vector3d relative_t(t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1], t_array[i][2] - t_array[i - j][2]);
//                         Quaterniond q_i_j = Quaterniond(q_array[i - j][0], q_array[i - j][1], q_array[i - j][2], q_array[i - j][3]);
//                         Quaterniond q_i = Quaterniond(q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]);
//                         relative_t = q_i_j.inverse() * relative_t;
//                         Quaterniond relative_q = q_i_j.inverse() * q_i;
//                         ceres::CostFunction *vo_function = RelativeRTError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
//                                                                                    relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
//                                                                                    0.1, 0.01);
//                         problem.AddResidualBlock(vo_function, NULL, q_array[i - j], t_array[i - j], q_array[i], t_array[i]);
//                     }
//                 }

//                 // add loop edge

//                 if ((*it)->has_loop)
//                 {
//                     assert((*it)->loop_index >= first_looped_index);
//                     int loop_index = getKeyFrame((*it)->loop_index)->optimize_buf_index;
//                     Vector3d relative_t;
//                     relative_t = (*it)->getLoopRelativeT();
//                     Quaterniond relative_q;
//                     relative_q = (*it)->getLoopRelativeQ();
//                     ceres::CostFunction *loop_function = RelativeRTError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
//                                                                                  relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
//                                                                                  0.1, 0.01);
//                     problem.AddResidualBlock(loop_function, loss_function, q_array[loop_index], t_array[loop_index], q_array[i], t_array[i]);
//                 }

//                 if ((*it)->index == optimize_end_index)
//                     break;
//                 i++;
//             }
//             m_keyframelist.unlock();

//             ceres::Solve(options, &problem, &summary);
//             // std::cout << summary.BriefReport() << "\n";

//             // printf("pose optimization time: %f \n", tmp_t.toc());
//             /*
//             for (int j = 0 ; j < i; j++)
//             {
//                 printf("optimize i: %d p: %f, %f, %f\n", j, t_array[j][0], t_array[j][1], t_array[j][2] );
//             }
//             */
//             m_keyframelist.lock();
//             i = 0;
//             for (it = keyframelist.begin(); it != keyframelist.end(); it++)
//             {
//                 if ((*it)->index < first_looped_index)
//                     continue;
//                 Quaterniond tmp_q(q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]);
//                 Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
//                 Matrix3d tmp_r = tmp_q.toRotationMatrix();
//                 (*it)->updatePose(tmp_t, tmp_r);

//                 if ((*it)->index == optimize_end_index)
//                     break;
//                 i++;
//             }

//             Vector3d cur_t, vio_t;
//             Matrix3d cur_r, vio_r;
//             cur_kf->getPose(cur_t, cur_r);
//             cur_kf->getVioPose(vio_t, vio_r);
//             m_drift.lock();
//             r_drift = cur_r * vio_r.transpose();
//             t_drift = cur_t - r_drift * vio_t;
//             m_drift.unlock();
//             // cout << "t_drift " << t_drift.transpose() << endl;
//             // cout << "r_drift " << Utility::R2ypr(r_drift).transpose() << endl;

//             it++;
//             for (; it != keyframelist.end(); it++)
//             {
//                 Vector3d t;
//                 Matrix3d R;
//                 (*it)->getVioPose(t, R);
//                 t = r_drift * t + t_drift;
//                 R = r_drift * R;
//                 (*it)->updatePose(t, R);
//             }
//             m_keyframelist.unlock();
//             // updatePath();
//         }

//         std::chrono::milliseconds dura(2000);
//         std::this_thread::sleep_for(dura);
//     }
//     return;
// }

void PoseGraph::updatePath()
{
    deque<KeyFramePtr>::iterator it;
    posegraph_visualization->reset();
    // for (int i = 1; i <= sequence_cnt; i++)
    // {
    //     path[i].poses.clear();
    // }

    if (0) // 全部清除然后重新写 SAVE_LOOP_PATH
    {
        ofstream loop_path_file_tmp(LOOP_RESULT_PATH, ios::out);
        loop_path_file_tmp.close();
    }

    for (int i = local_keyframelist_window.size() - 1; i > 0; i--)
    {
        it = local_keyframelist_window.begin() + i;
        Vector3d t;
        Matrix3d R;
        (*it)->getPose(t, R);
        Quaterniond Q;
        Q = R;
        //        printf("path p: %f, %f, %f\n",  t.x(),  t.z(),  t.y() );

        // geometry_msgs::PoseStamped pose_stamped;
        // pose_stamped.header.stamp = ros::Time((*it)->time_stamp);
        // pose_stamped.header.frame_id = "world";
        // pose_stamped.pose.position.x = t.x() + VISUALIZATION_SHIFT_X;
        // pose_stamped.pose.position.y = t.y() + VISUALIZATION_SHIFT_Y;
        // pose_stamped.pose.position.z = t.z();
        // pose_stamped.pose.orientation.x = Q.x();
        // pose_stamped.pose.orientation.y = Q.y();
        // pose_stamped.pose.orientation.z = Q.z();
        // pose_stamped.pose.orientation.w = Q.w();
        // if ((*it)->sequence != 0)
        // {
        //     path[(*it)->sequence].poses.push_back(pose_stamped);
        //     path[(*it)->sequence].header = pose_stamped.header;
        // }

        if (0) // SAVE_LOOP_PATH
        {
            ofstream fout_loopRes(LOOP_RESULT_PATH, ios::app);
            fout_loopRes.setf(ios::fixed, ios::floatfield);
            fout_loopRes.precision(9);
            fout_loopRes << (*it)->time_stamp << " ";
            fout_loopRes.precision(6);
            fout_loopRes << t.x() << " "
                         << t.y() << " "
                         << t.z() << " "
                         << Q.x() << " "
                         << Q.y() << " "
                         << Q.z() << " "
                         << Q.w() << endl;
            fout_loopRes.close();
        }
        // draw local connection
        if (SHOW_S_EDGE)
        {
            deque<KeyFramePtr>::iterator it_before = it;
            if (i != 1) // 1 和0 之间连线 会有问题
            {
                it_before--;
                Vector3d t_before;
                Matrix3d R_before;
                (*it_before)->getPose(t_before, R_before);
                posegraph_visualization->add_edge(t, t_before);
            }
        }
        if (SHOW_L_EDGE)
        {
            if ((*it)->has_loop && (*it)->loop_index >= earliest_loop_index && (*it)->sequence == sequence_cnt)
            {
                const KeyFramePtr connected_KF = getKeyFrame((*it)->loop_index);
                Vector3d connected_t;
                Matrix3d connected_R;
                connected_KF->getPose(connected_t, connected_R);
                //(*it)->getVioPose(t, R);
                (*it)->getPose(t, R);
                if ((*it)->sequence > 0)
                {
                    posegraph_visualization->add_loopedge(t, connected_t);
                }
            }
        }
    }
    // publish();
}

void PoseGraph::savePoseGraph()
{
    lock_guard<std::mutex> guard(m_keyframelist);
    TicToc tmp_t;
    // FILE *pFile;
    string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
    std::shared_ptr<std::ofstream> pFile = std::make_shared<std::ofstream>(file_path.c_str());
    pFile->precision(9);
    int pose_graph_size = ref_keyframelist.size();
    printf("pose graph path: %s\n", POSE_GRAPH_SAVE_PATH.c_str());
    printf("pose graph saving..., pose graph size is %d\n", pose_graph_size);

    // pFile = fopen(file_path.c_str(), "w");
    // fprintf(pFile, "index time_stamp Tx Ty Tz Qw Qx Qy Qz loop_index loop_info\n");
    deque<KeyFramePtr>::iterator it;
    unsigned int idx = 0;
    for (it = ref_keyframelist.begin(); it != ref_keyframelist.end(); it++)
    {
        std::string image_path, descriptor_path, spp_path, keypoints_path;
        Quaterniond VIO_tmp_Q{(*it)->vio_R_w_i};
        Quaterniond PG_tmp_Q{(*it)->R_w_i};
        Vector3d VIO_tmp_t = (*it)->vio_t_w_i;
        Vector3d PG_tmp_t = (*it)->T_w_i;

        // fprintf(pFile, " %d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %f %f %f %f %f %f %f %f %d\n", idx, (*it)->cam_id , (*it)->time_stamp,
        //         VIO_tmp_t.x(), VIO_tmp_t.y(), VIO_tmp_t.z(),
        //         PG_tmp_t.x(), PG_tmp_t.y(), PG_tmp_t.z(),
        //         VIO_tmp_Q.w(), VIO_tmp_Q.x(), VIO_tmp_Q.y(), VIO_tmp_Q.z(),
        //         PG_tmp_Q.w(), PG_tmp_Q.x(), PG_tmp_Q.y(), PG_tmp_Q.z(),
        //         (*it)->loop_index,
        //         (*it)->loop_info(0), (*it)->loop_info(1), (*it)->loop_info(2), (*it)->loop_info(3),
        //         (*it)->loop_info(4), (*it)->loop_info(5), (*it)->loop_info(6), (*it)->loop_info(7),
        //         (int)(*it)->reference_keypoints_data.size());
        if (pFile->is_open())
        {
            *pFile << idx << " " << (*it)->cam_id << " " << (*it)->time_stamp << " "
                   << VIO_tmp_t.x() << " " << VIO_tmp_t.y() << " " << VIO_tmp_t.z() << " "
                   << PG_tmp_t.x() << " " << PG_tmp_t.y() << " " << PG_tmp_t.z() << " "
                   << VIO_tmp_Q.w() << " " << VIO_tmp_Q.x() << " " << VIO_tmp_Q.y() << " " << VIO_tmp_Q.z() << " "
                   << PG_tmp_Q.w() << " " << PG_tmp_Q.x() << " " << PG_tmp_Q.y() << " " << PG_tmp_Q.z() << " "
                   << (*it)->loop_index << " "
                   << (*it)->loop_info(0) << " " << (*it)->loop_info(1) << " " << (*it)->loop_info(2) << " " << (*it)->loop_info(3) << " "
                   << (*it)->loop_info(4) << " " << (*it)->loop_info(5) << " " << (*it)->loop_info(6) << " " << (*it)->loop_info(7) << " "
                   << static_cast<int>((*it)->reference_keypoints_data.size()) << "\n";
        }

        // TODO: crop map

        // save image for debug
        if (DEBUG_IMAGE)
        {
            image_path = POSE_GRAPH_SAVE_PATH + to_string(idx) + "_image.png";
            imwrite(image_path.c_str(), (*it)->image);
        }
        // write reference_keypoints, reference_brief_descriptors   vector<cv::KeyPoint> reference_keypoints vector<BRIEF::bitset> reference_brief_descriptors;
        spp_path = POSE_GRAPH_SAVE_PATH + to_string(idx) + "_sppdes.bin";
        std::ofstream spp_file(spp_path, std::ios::binary | std::ios::out);
        std::string brief_path = POSE_GRAPH_SAVE_PATH + to_string(idx) + "_briefdes.dat";
        // std::ofstream brief_file(brief_path, std::ios::binary | std::ios::out);
        cv::FileStorage brief_file(brief_path, cv::FileStorage::WRITE_BASE64);
        if (1 == DETECTOR)
        {
            brief_file << "descriptors" << "[";
        }

        keypoints_path = POSE_GRAPH_SAVE_PATH + to_string(idx) + "_keypoints.txt";
        FILE *reference_keypoints_file;
        reference_keypoints_file = fopen(keypoints_path.c_str(), "w");

        auto refkd = (*it)->reference_keypoints_data;

        for (int i = 0; i < (int)refkd.size(); i++)
        {
            if (0 == DETECTOR)
            {
                auto matrix = refkd[i].spp_feature;
                spp_file.write(reinterpret_cast<const char *>(matrix.data()), matrix.size() * sizeof(float));
            }
            else if (1 == DETECTOR)
            {
                // brief_file << (*it)->reference_brief_descriptors;
                brief_file << refkd[i].brief_descriptor;
            }
            fprintf(reference_keypoints_file, "%d %f %f %f %f %f %f\n", refkd[i].cam_id, refkd[i].keypoint_2d_uv.pt.x, refkd[i].keypoint_2d_uv.pt.y,
                    refkd[i].point_depth, refkd[i].point_3d_world.x, refkd[i].point_3d_world.y, refkd[i].point_3d_world.z);
        }

        spp_file.close();
        if (1 == DETECTOR)
        {
            brief_file << "]";
        }
        
        brief_file.release();
        fclose(reference_keypoints_file);
        idx++;
    }
    // fclose(pFile);
    pFile->close();

    printf("save pose graph time: %f s\n", tmp_t.toc() / 1000);
}
void PoseGraph::loadPoseGraph()
{
    TicToc tmp_t;
    string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
    printf("lode pose graph from: %s \n", file_path.c_str());
    printf("pose graph loading...\n");
    std::ifstream pFile(file_path.c_str());
    if (!pFile.is_open())
    {
        printf("lode previous pose graph error: wrong previous pose graph path or no previous pose graph \n the system will start with new pose graph \n");
        return;
    }
    int index;
    double time_stamp;
    double VIO_tx, VIO_ty, VIO_tz;
    double PG_tx, PG_ty, PG_tz;
    double VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz;
    double PG_Qw, PG_Qx, PG_Qy, PG_Qz;
    double loop_info_0, loop_info_1, loop_info_2, loop_info_3;
    double loop_info_4, loop_info_5, loop_info_6, loop_info_7;
    int loop_index;
    short cam_id;
    int keypoints_num;
    Eigen::Matrix<double, 8, 1> loop_info;
    int cnt = 0;
    map_lidarpts.reset(new PointCloudXYZI);

    std::string line;
    while (std::getline(pFile, line))
    {
        std::stringstream ss(line);
        ss >> index >> cam_id >> time_stamp >>
            VIO_tx >> VIO_ty >> VIO_tz >> PG_tx >> PG_ty >> PG_tz >>
            VIO_Qw >> VIO_Qx >> VIO_Qy >> VIO_Qz >> PG_Qw >> PG_Qx >> PG_Qy >> PG_Qz >> loop_index >> loop_info_0 >> loop_info_1 >> loop_info_2 >>
            loop_info_3 >> loop_info_4 >> loop_info_5 >> loop_info_6 >> loop_info_7 >> keypoints_num;
        cv::Mat image;
        std::string image_path, descriptor_path;
        if (DEBUG_IMAGE)
        {
            image_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_image.png";
            image = cv::imread(image_path.c_str(), 0);
        }

        // VIO_tz = std::max(VIO_tz, 0.3);
        Vector3d VIO_t(VIO_tx, VIO_ty, VIO_tz);
        Vector3d PG_t(PG_tx, PG_ty, PG_tz);
        Quaterniond VIO_Q;
        VIO_Q.w() = VIO_Qw;
        VIO_Q.x() = VIO_Qx;
        VIO_Q.y() = VIO_Qy;
        VIO_Q.z() = VIO_Qz;
        Quaterniond PG_Q;
        PG_Q.w() = PG_Qw;
        PG_Q.x() = PG_Qx;
        PG_Q.y() = PG_Qy;
        PG_Q.z() = PG_Qz;
        Matrix3d VIO_R, PG_R;
        VIO_R = VIO_Q.toRotationMatrix();
        PG_R = PG_Q.toRotationMatrix();
        Eigen::Matrix<double, 8, 1> loop_info;
        loop_info << loop_info_0, loop_info_1, loop_info_2, loop_info_3, loop_info_4, loop_info_5, loop_info_6, loop_info_7;

        if (loop_index != -1)
            if (earliest_loop_index > loop_index || earliest_loop_index == -1)
            {
                earliest_loop_index = loop_index;
            }

        // load reference_keypoints, reference_brief_descriptors
        string spp_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_sppdes.bin";
        std::ifstream spp_file(spp_path, std::ios::binary);
        std::string brief_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_briefdes.dat";
        // std::ifstream brief_file(brief_path, std::ios::binary);

        string reference_keypoints_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_keypoints.txt";
        FILE *reference_keypoints_file;
        reference_keypoints_file = fopen(reference_keypoints_path.c_str(), "r");
        //
        KeyPointsData reference_keypoints_data;
        std::vector<Eigen::Matrix<float, 256, 1>> reference_spp_descriptors;
        Eigen::Matrix<float, 259, 1> spp_feature;
        // Eigen::Matrix<float, 259, Eigen::Dynamic> spp_feature_frame;
        // spp_feature_frame.resize(259, keypoints_num);
        std::vector<cv::Mat> reference_brief_descriptors;

        if (1 == DETECTOR)
        {
            cv::FileStorage brief_file(brief_path, cv::FileStorage::READ);
            cv::FileNode n = brief_file["descriptors"];
            for (auto it = n.begin(); it != n.end(); ++it)
            {
                cv::Mat brief_des;
                *it >> brief_des;
                reference_brief_descriptors.push_back(brief_des);
            }
            brief_file.release();
        }

        for (int i = 0; i < keypoints_num; i++)
        {
            PointType p;

            if (0 == DETECTOR)
            {
                spp_file.read(reinterpret_cast<char *>(spp_feature.data()), spp_feature.size() * sizeof(float));
            }

            FeatureProperty tmp_property;

            float point_depth, p_x, p_y, p_x_norm, p_y_norm;
            float p_x_world, p_y_world, p_z_world;
            short pt_cam_id;
            if (!fscanf(reference_keypoints_file, "%d %f %f %f %f %f %f", &pt_cam_id, &p_x, &p_y, &point_depth, &p_x_world, &p_y_world, &p_z_world))
                printf(" fail to load pose graph \n");
            // if (!fscanf(reference_keypoints_file, "%lf %lf %lf %lf %lf %lf %lf", &p_x, &p_y, &p_x_norm, &p_y_norm, &p_x_world, &p_y_world, &p_z_world))
            //     printf(" fail to load pose graph \n");
            tmp_property.cam_id = pt_cam_id;
            tmp_property.point_depth = point_depth;
            tmp_property.keypoint_2d_uv.pt = cv::Point2f(p_x, p_y);
            tmp_property.point_3d_world = cv::Point3f(p_x_world, p_y_world, p_z_world);
            if (0 == DETECTOR)
            {
                tmp_property.spp_feature = spp_feature;
                reference_spp_descriptors.push_back(spp_feature.tail(256)); // for DBoW2 ; all points are taken
            }
            else if (1 == DETECTOR)
            {
                tmp_property.brief_descriptor = reference_brief_descriptors[i];
                // reference_brief_descriptors.push_back(brief_des);
            }

            if (tmp_property.point_depth < 0) // for 2d-3d matching, only take the points with depth
                continue;

            reference_keypoints_data.push_back(tmp_property);

            p.x = p_x_world;
            p.y = p_y_world;
            p.z = p_z_world;
            p.intensity = cam_id * 100;
            map_lidarpts->points.push_back(p);
        }
        spp_file.close();
        // brief_file.close();
        fclose(reference_keypoints_file);
        // KeyFramePtr keyframe = new KeyFrame(time_stamp, index, VIO_t, VIO_R, PG_t, PG_R, image, loop_index, loop_info, reference_keypoints_data, reference_spp_descriptors);
        KeyFramePtr keyframe = std::make_shared<KeyFrame>(time_stamp, index, VIO_t, VIO_R, PG_t, PG_R, image, loop_index, cam_id, loop_info, reference_keypoints_data);
        if (0 == DETECTOR)
        {
            loadKeyFrame(keyframe, reference_spp_descriptors);
        }
        else if (1 == DETECTOR)
        {
            loadKeyFrame(keyframe, reference_brief_descriptors);
        }

        if (cnt % 20 == 0)
        {
            publish();
        }
        cnt++;
    }

    pFile.close();
    publish();
    sensor_msgs::PointCloud2 pc;
    pcl::toROSMsg(*map_lidarpts, pc);
    pc.header.frame_id = "world";
    pc.header.stamp = ros::Time::now();
    if (pub_map_lidarpts)
    {
        pub_map_lidarpts.publish(pc);
    }
    else
    {
        ROS_ERROR("Publisher is invalid.");
    }

    if (map_lidarpts->size() > 0)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *map_lidarpts);
    }

    printf("load pose graph time: %f , pose graph num %d \n", tmp_t.toc() / 1000, cnt);
    // std::cout << " dbi size " << dbi.size() << std::endl;
    base_sequence = 0;
}
// header
void PoseGraph::publish()
{
    pub_base_path.publish(base_path);
    // posegraph_visualization->publish_by(pub_pose_graph, base_path.header);
    sensor_msgs::PointCloud2 pc2;
    pcl::toROSMsg(*base_pose_cloud, pc2);
    pc2.header = base_path.header;
    pub_base_pose.publish(pc2);
}
