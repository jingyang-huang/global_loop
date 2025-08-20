#include <opencv2/opencv.hpp>

// #include "plnet.h"
#include "feature_detector.h"
#include "utils.h"
// int NUM_OF_CAM = 4;

FeatureDetector::FeatureDetector(const SuperPointConfig &superpoint_config) : _spp_config(superpoint_config)
{
  _spp_config.input_tensor_names.push_back("input");
  _spp_config.output_tensor_names.push_back("scores");
  _spp_config.output_tensor_names.push_back("descriptors");

  _superpoint = std::shared_ptr<SuperPoint>(new SuperPoint(_spp_config));
  if (!_superpoint->build())
  {
    std::cout << "Error in SuperPoint building" << std::endl;
    exit(0);
  }
  std::cout << "super_point building success!" << std::endl;
}

FeatureDetector::FeatureDetector(const BriefConfig &brief_config) : _brief_config(brief_config)
{
  // _brief_extractor = std::shared_ptr<BriefExtractor>(new BriefExtractor(_brief_config.pattern_file));
  _brief_extractor = cv::ORB::create(800, 1., 0);
  if (_brief_extractor == nullptr)
  {
    std::cout << "Error in brief building" << std::endl;
    exit(0);
  }
  std::cout << "brief building success!" << std::endl;
}

// track_featuredata_seq -> track_features_vec
bool FeatureDetector::DetectMulti(const std::vector<cv::Mat> &image_vec, std::vector<std::vector<cv::KeyPoint>> &extract_keypoint_seq, std::vector<cv::Mat> &desc_mat_seq)
{
  bool good_infer = true;
  // std::cout << " image_vec.size " << image_vec.size() << std::endl;
  for (int k = 0; k < image_vec.size(); k++)
  {
    // std::cout << k << ",    image_vec.size " << image_vec.size() << std::endl;
    // cv::FAST(image_vec[k], extract_keypoint_seq[k], this->_brief_config.keypoint_threshold, true);
    Grider_FAST::perform_griding(image_vec[k], extract_keypoint_seq[k], this->_brief_config.max_keypoints, 64, 48, this->_brief_config.keypoint_threshold, true);
    _brief_extractor->compute(image_vec[k], extract_keypoint_seq[k], desc_mat_seq[k]);
    printf("fast query_keypoints size %d\n", (int)extract_keypoint_seq[k].size());
  }
  return true;
}

void FeatureDetector::extractBriefDescriptor(std::vector<FeatureData> &all_featuredata_seq, std::vector<std::vector<cv::Mat>> &desc_vec_seq)
{
  for (int k = 0; k < all_featuredata_seq.size(); k++)
  {
    // std::vector<BRIEF::bitset> descriptors;
    std::vector<cv::Mat> desc_vec;
    for (int j = 0; j < all_featuredata_seq[k].size(); j++)
    {
      // all_featuredata_seq[k][j].keypoint_2d_uv -> keys
      // std::cout << j << std::endl;
      desc_vec_seq[k].emplace_back(all_featuredata_seq[k][j].brief_descriptor);
    }

    // _brief_extractor->compute(image_vec[k], keys, desc_mat_seq[k]);
    // if (!desc_mat_seq[k].empty())
    // {
    //   desc_vec = convertToDescriptorVector(desc_mat_seq[k]);
    // }
    // // _brief_extractor->extract(image_vec[k], keys, descriptors);
    // for (int j = 0; j < all_featuredata_seq[k].size(); j++)
    // {
    //   all_featuredata_seq[k][j].brief_descriptor = desc_vec[j];
    // }
  }
}

void FeatureDetector::extractBriefDescriptor(const cv::Mat &image, FeatureData &all_featuredata, std::vector<BRIEF::bitset> &brief_descriptors)
{
  std::vector<cv::KeyPoint> keys;
  for (int j = 0; j < all_featuredata.size(); j++)
  {
    // all_featuredata[j].keypoint_2d_uv -> keys
    keys.push_back(all_featuredata[j].keypoint_2d_uv);
  }
  // _brief_extractor->extract(image, keys, brief_descriptors);
  // for (int j = 0; j < all_featuredata.size(); j++)
  // {
  //   all_featuredata[j].brief_descriptor = brief_descriptors[j];
  // }
}

bool FeatureDetector::DetectMultiwithTrack(const std::vector<cv::Mat> &image_vec, std::vector<FeatureData> &track_featuredata_seq, std::vector<Eigen::Matrix<float, 259, Eigen::Dynamic>> &extract_features_vec, std::vector<Eigen::Matrix<float, 259, Eigen::Dynamic>> &track_features_vec)
{
  bool good_infer = true;
  for (int k = 0; k < image_vec.size(); k++)
  {
    bool good_infer_this = _superpoint->infer_withTrack(image_vec[k], track_featuredata_seq[k], extract_features_vec[k], track_features_vec[k]);
    // desc_map = _superpoint->desc_map
    if (!good_infer_this)
    {
      std::cout << "Failed when extracting point features !" << std::endl;
    }
    good_infer = good_infer & good_infer_this;
  }
  return good_infer;
}

void FeatureDetector::DetectBrief(cv::Mat &image, FeatureData &query_keypoints_data, std::vector<cv::Mat> &brief_descriptors)
{
  bool good_infer = false;
  std::vector<cv::KeyPoint> query_keypoints;
  cv::Mat desc_mat;
  // cv::FAST(image_vec[k], extract_keypoint_seq[k], this->_brief_config.keypoint_threshold, true);
  Grider_FAST::perform_griding(image, query_keypoints, this->_brief_config.max_keypoints, 64, 48, this->_brief_config.keypoint_threshold, true);
  _brief_extractor->compute(image, query_keypoints, desc_mat);
  printf("fast query_keypoints size %d\n", (int)query_keypoints.size());
  for (int i = 0; i < query_keypoints.size(); i++)
  {
    FeatureProperty kpp;
    Eigen::Vector3d tmp_p;
    m_camera[0]->liftProjective(Eigen::Vector2d(query_keypoints[i].pt.x, query_keypoints[i].pt.y), tmp_p);
    cv::Point2f tmp_norm;
    tmp_norm = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
    kpp.keypoint_2d_norm.pt = tmp_norm;
    kpp.keypoint_2d_uv = query_keypoints[i];
    kpp.score = query_keypoints[i].response;
    kpp.brief_descriptor = desc_mat.row(i);
    brief_descriptors.emplace_back(desc_mat.row(i));
    query_keypoints_data.push_back(kpp);
  }
}

void FeatureDetector::DetectSpp(cv::Mat &image, FeatureData &query_keypoints_data, std::vector<Eigen::Matrix<float, 256, 1>> &query_spp_descriptors)
{
  Eigen::Matrix<float, 259, Eigen::Dynamic> spp_features;
  this->Detect(image, spp_features);

  FeatureProperty kpp;
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, CV_GRAY2RGB);

  for (int j = 0; j < spp_features.cols(); j++)
  {
    cv::KeyPoint key;
    key.pt = cv::Point2f((spp_features(1, j)), (spp_features(2, j)));
    Eigen::Vector3d tmp_p;
    m_camera[0]->liftProjective(Eigen::Vector2d(key.pt.x, key.pt.y), tmp_p);
    cv::Point2f tmp_norm;
    tmp_norm = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
    kpp.score = spp_features(0, j);
    kpp.keypoint_2d_uv = key;
    kpp.keypoint_2d_norm.pt = tmp_norm;
    kpp.spp_feature = spp_features.col(j);
    query_keypoints_data.push_back(kpp);
    query_spp_descriptors.push_back(spp_features.col(j).tail(256));
    // if (DEBUG_IMAGE)
    // 	cv::circle(image_rgb, cv::Point(kpp.keypoint_2d_uv.pt.x, kpp.keypoint_2d_uv.pt.y), 1, cv::Scalar(0, 200, 200), 1);
  }
  // if (DEBUG_IMAGE)
  // 	cv::imshow("image_rgb", image_rgb);
}

bool FeatureDetector::Detect(cv::Mat &image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features)
{
  bool good_infer = false;

  good_infer = _superpoint->infer(image, features);

  if (!good_infer)
  {
    std::cout << "Failed when extracting point features !" << std::endl;
  }
  return good_infer;
}

bool FeatureDetector::Detect(cv::Mat &image_left, cv::Mat &image_right,
                             Eigen::Matrix<float, 259, Eigen::Dynamic> &left_features,
                             Eigen::Matrix<float, 259, Eigen::Dynamic> &right_features)
{
  bool good_infer_left = Detect(image_left, left_features);
  bool good_infer_right = Detect(image_right, right_features);
  bool good_infer = good_infer_left & good_infer_right;
  if (!good_infer)
  {
    std::cout << "Failed when extracting point features !" << std::endl;
  }
  return good_infer;
}

std::vector<cv::Mat> FeatureDetector::convertToDescriptorVector(const cv::Mat &descriptors)
{
  assert(descriptors.rows > 0);
  std::vector<cv::Mat> desc;
  desc.reserve(descriptors.rows);
  for (int j = 0; j < descriptors.rows; j++)
  {
    desc.emplace_back(descriptors.row(j));
  }
  return desc;
}