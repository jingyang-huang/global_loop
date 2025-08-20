#ifndef READ_CONFIGS_H_
#define READ_CONFIGS_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "utils.h"
#include "ThirdParty/DVision/DVision.h"
// #include "ThirdParty/brisk/include/brisk/brisk.h"
#include <memory>
#include <vector>
#include <eigen3/Eigen/Dense>
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


typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
extern std::vector<camodocal::CameraPtr> m_camera;
extern std::vector<Eigen::Vector3d> t_i_c;
extern std::vector<Eigen::Matrix3d> R_i_c;

// reference SVO(https://github.com/uzh-rpg/rpg_svo)
inline float shiTomasiScore(const cv::Mat &img, int u, int v)
{
    assert(img.type() == CV_8UC1);

    float dXX = 0.0;
    float dYY = 0.0;
    float dXY = 0.0;
    const int halfbox_size = 4;
    const int box_size = 2 * halfbox_size;
    const int box_area = box_size * box_size;
    const int x_min = u - halfbox_size;
    const int x_max = u + halfbox_size;
    const int y_min = v - halfbox_size;
    const int y_max = v + halfbox_size;

    if (x_min < 1 || x_max >= img.cols - 1 || y_min < 1 || y_max >= img.rows - 1)
        return 0.0; // patch is too close to the boundary

    const int stride = img.step.p[0];
    for (int y = y_min; y < y_max; ++y)
    {
        const uint8_t *ptr_left = img.data + stride * y + x_min - 1;
        const uint8_t *ptr_right = img.data + stride * y + x_min + 1;
        const uint8_t *ptr_top = img.data + stride * (y - 1) + x_min;
        const uint8_t *ptr_bottom = img.data + stride * (y + 1) + x_min;
        for (int x = 0; x < box_size;
             ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom)
        {
            float dx = *ptr_right - *ptr_left;
            float dy = *ptr_bottom - *ptr_top;
            dXX += dx * dx;
            dYY += dy * dy;
            dXY += dx * dy;
        }
    }

    // Find and return smaller eigenvalue:
    dXX = dXX / (2.0 * box_area);
    dYY = dYY / (2.0 * box_area);
    dXY = dXY / (2.0 * box_area);
    return 0.5 * (dXX + dYY -
                  sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
}


struct FeatureProperty
{
  cv::KeyPoint keypoint_2d_uv;
  // cv::Point2i point_2d_uv_int;
  cv::KeyPoint keypoint_2d_norm;
  int point_id;
  short cam_id;
  float point_depth = -1;
  uchar point_intensity = 0;
  float score = 0;
  int track_cnt = 0;
  cv::Point3f point_3d_world = cv::Point3f(-1, -1, -1);
  Eigen::Matrix<float, 259, 1> spp_feature = Eigen::Matrix<float, 259, 1>::Zero(); // TODO: 可能会去掉以节省内存
  // DVision::BRIEF::bitset brief_descriptor;
  cv::Mat brief_descriptor;
};

// extern double mean_shitomasi_score;
typedef std::vector<FeatureProperty> FeatureData;

struct FeatureDataParted
{
  FeatureData points_with_depth;
  FeatureData points_no_depth;
};

struct SuperPointConfig
{
  SuperPointConfig() {}
  int max_keypoints;
  float keypoint_threshold;
  int remove_borders;
  int dla_core;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct BriefConfig
{
  BriefConfig() {}
  int max_keypoints;
  float keypoint_threshold;
  int remove_borders;
  // int dla_core;
  // std::vector<std::string> input_tensor_names;
  // std::vector<std::string> output_tensor_names;
  std::string pattern_file;
  // std::string engine_file;
};


struct PointMatcherConfig
{
  PointMatcherConfig() {}

  int image_width;
  int image_height;
  int dla_core;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};


#define RED "\033[0;1;31m"
#define GREEN "\033[0;1;32m"
#define YELLOW "\033[0;1;33m"
#define BLUE "\033[0;1;34m"
#define PURPLE "\033[0;1;35m"
#define DEEPGREEN "\033[0;1;36m"
#define WHITE "\033[0;1;37m"
#define RED_IN_WHITE "\033[0;47;31m"
#define GREEN_IN_WHITE "\033[0;47;32m"
#define YELLOW_IN_WHITE "\033[0;47;33m"

#define TAIL "\033[0m"

#endif // READ_CONFIGS_H_
