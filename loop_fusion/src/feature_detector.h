#ifndef FEATURE_DETECTOR_H_
#define FEATURE_DETECTOR_H_

#include "super_point.h"
// #include "ThirdParty/DBoW2/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
// #include "ThirdParty/brisk/include/brisk/brisk.h"
#include <memory>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "sys_configs.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "utility/Grider_FAST.h"
#include "utility/dls_pnp.h"

using namespace Eigen;
using namespace std;
using namespace DVision;

class BriefExtractor
{
public:
  DVision::BRIEF m_brief;
  void extract(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors)
  {
    m_brief.compute(im, keys, descriptors);
  };
  BriefExtractor(const std::string &pattern_file)
  {
    // The DVision::BRIEF extractor computes a random pattern by default when
    // the object is created.
    // We load the pattern that we used to build the vocabulary, to make
    // the descriptors compatible with the predefined vocabulary

    // loads the pattern
    cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
    if (!fs.isOpened())
      throw string("Could not open file ") + pattern_file;

    vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;
    fs["x2"] >> x2;
    fs["y1"] >> y1;
    fs["y2"] >> y2;

    m_brief.importPairs(x1, y1, x2, y2);
  };
};
typedef std::shared_ptr<BriefExtractor> BriefExtractorPtr;

class FeatureDetector
{
public:
  FeatureDetector(const SuperPointConfig &superpoint_config);
  FeatureDetector(const BriefConfig &brief_config);

  bool Detect(cv::Mat &image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features);
  bool Detect(cv::Mat &image_left, cv::Mat &image_right, Eigen::Matrix<float, 259, Eigen::Dynamic> &left_features,
              Eigen::Matrix<float, 259, Eigen::Dynamic> &right_features);
  void DetectBrief(cv::Mat &image, FeatureData &query_keypoints_data, std::vector<cv::Mat> &brief_descriptors);
  void DetectSpp(cv::Mat &image, FeatureData &query_keypoints_data, std::vector<Eigen::Matrix<float, 256, 1>> &query_spp_descriptors);

  bool DetectMultiwithTrack(const std::vector<cv::Mat> &image_vec, std::vector<FeatureData> &track_featuredata_seq, std::vector<Eigen::Matrix<float, 259, Eigen::Dynamic>> &extract_features_vec, std::vector<Eigen::Matrix<float, 259, Eigen::Dynamic>> &track_features_vec);
  bool DetectMulti(const std::vector<cv::Mat> &image_vec, std::vector<std::vector<cv::KeyPoint>> &extract_keypoint_seq);
  bool DetectMulti(const std::vector<cv::Mat> &image_vec, std::vector<std::vector<cv::KeyPoint>> &extract_keypoint_seq, std::vector<cv::Mat> &desc_mat_seq);
  void extractBriefDescriptor(const cv::Mat &image, FeatureData &all_featuredata, std::vector<BRIEF::bitset> &brief_descriptors);
  void extractBriefDescriptor(std::vector<FeatureData> &all_featuredata_seq, std::vector<std::vector<cv::Mat>> &desc_vec_seq);

  std::vector<cv::Mat> convertToDescriptorVector(const cv::Mat &descriptors);
  cv::Ptr<cv::ORB> _brief_extractor;

private:
  SuperPointConfig _spp_config;
  BriefConfig _brief_config;
  SuperPointPtr _superpoint;
  // BriefExtractorPtr _brief_extractor;

  // std::vector<std::vector<cv::Point2f>> & track_pts_seq;
  // descrpitors of four images
};

typedef std::shared_ptr<FeatureDetector> FeatureDetectorPtr;

#endif // FEATURE_DETECTOR_H_
