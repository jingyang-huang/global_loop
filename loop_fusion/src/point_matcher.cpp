#include "point_matcher.h"
#include <opencv2/opencv.hpp>

PointMatcher::PointMatcher(const PointMatcherConfig &config) : _config(config)
{
  // // lightglue
  // _config.dla_core = -1;
  // _config.input_tensor_names.push_back("keypoints_0");
  // _config.input_tensor_names.push_back("keypoints_1");
  // _config.input_tensor_names.push_back("descriptors_0");
  // _config.input_tensor_names.push_back("descriptors_1");
  // _config.output_tensor_names.push_back("scores");

  // _lightglue = std::shared_ptr<SuperPointLightGlue>(new SuperPointLightGlue(_config));
  // if (!_lightglue->build())
  // {
  //   std::cout << "Error lightglue building" << std::endl;
  // }
  // superglue
  _config.dla_core = -1;
  _config.input_tensor_names.push_back("keypoints_0");
  _config.input_tensor_names.push_back("scores_0");
  _config.input_tensor_names.push_back("descriptors_0");
  _config.input_tensor_names.push_back("keypoints_1");
  _config.input_tensor_names.push_back("scores_1");
  _config.input_tensor_names.push_back("descriptors_1");
  _config.output_tensor_names.push_back("scores");

  _superglue = std::shared_ptr<SuperGlue>(new SuperGlue(_config));
  if (!_superglue->build())
  {
    std::cout << "Erron superglue building" << std::endl;
  }
  std::cout<<" super_glue building success!" << std::endl;
}

void PointMatcher::NormalizeKeypoints(const Eigen::Matrix<float, 259, Eigen::Dynamic> &features,
                                      Eigen::Matrix<float, 259, Eigen::Dynamic> &normalized_features,
                                      int width, int height, float scale)
{
  normalized_features = features;
  float L_inv = 1.0 / std::max(width, height) * scale;
  for (int col = 0; col < features.cols(); ++col)
  {
    normalized_features(1, col) = (features(1, col) - width / 2) * L_inv;
    normalized_features(2, col) = (features(2, col) - height / 2) * L_inv;
  }
}

void PointMatcher::NormalizeKeypoints(const FeatureData &kfd,
                                      Eigen::Matrix<float, 259, Eigen::Dynamic> &normalized_features,
                                      int width, int height, float scale)
{
  float L_inv = 1.0 / std::max(width, height) * scale;
  Eigen::Matrix<float, 259, Eigen::Dynamic> features(259, kfd.size());
  for (int i = 0; i < kfd.size(); i++)
  {
    features.col(i) = kfd[i].spp_feature;
    features(1, i) = (kfd[i].keypoint_2d_uv.pt.x - width / 2) * L_inv;
    features(2, i) = (kfd[i].keypoint_2d_uv.pt.y - height / 2) * L_inv;
  }
  normalized_features = features;
}

int PointMatcher::MatchingPoints(const FeatureData &kfd0, const FeatureData &kfd1,
                                 std::vector<cv::DMatch> &matches, bool outlier_rejection)
{
  if (kfd0.size() < 1 || kfd1.size() < 1)
  {
    return 0;
  }

  Eigen::Matrix<float, 259, Eigen::Dynamic> normalized_features0, normalized_features1;
  float scale = 0.7;
  NormalizeKeypoints(kfd0, normalized_features0, _config.image_width, _config.image_height, scale);
  NormalizeKeypoints(kfd1, normalized_features1, _config.image_width, _config.image_height, scale);

  matches.clear();
  std::vector<cv::Point> points0, points1;
  // lightglue
  // Eigen::Matrix<int, Eigen::Dynamic, 2> matches_index;
  // Eigen::Matrix<float, Eigen::Dynamic, 1> matches_score;
  // _lightglue->infer(normalized_features0.bottomRows(258), normalized_features1.bottomRows(258), matches_index, matches_score);

  // for (size_t i = 0; i < matches_index.rows(); i++)
  // {
  //   matches.emplace_back(matches_index(i, 0), matches_index(i, 1), 1.0 - matches_score(i));
  //   if (outlier_rejection)
  //   {
  //     points0.emplace_back(kfd0[matches_index(i, 0)].keypoint_2d_uv.pt);
  //     points1.emplace_back(kfd1[matches_index(i, 1)].keypoint_2d_uv.pt);
  //     // points0.emplace_back(features0(1, matches_index(i, 0)), features0(2, matches_index(i, 0)));
  //     // points1.emplace_back(features1(1, matches_index(i, 1)), features1(2, matches_index(i, 1)));
  //   }
  // }

  // superglue
  Eigen::VectorXi indices0, indices1;
  Eigen::VectorXd mscores0, mscores1;
  _superglue->infer(normalized_features0, normalized_features1, indices0, indices1, mscores0, mscores1);
  int num_match = 0;
  std::vector<int> point_indexes;
  for (size_t i = 0; i < indices0.size(); i++)
  {
    if (indices0(i) < indices1.size() && indices0(i) >= 0 && indices1(indices0(i)) == i)
    {
      double d = 1.0 - (mscores0[i] + mscores1[indices0[i]]) / 2.0;
      matches.emplace_back(i, indices0[i], d);
      if (outlier_rejection)
      {
        points0.emplace_back(kfd0[i].keypoint_2d_uv.pt);
        points1.emplace_back(kfd1[indices0(i)].keypoint_2d_uv.pt);
        // points0.emplace_back(features0(1, i), features0(2, i));
        // points1.emplace_back(features1(1, indices0(i)), features1(2, indices0(i)));
      }
    }
  }

  // reject outliers
  if (outlier_rejection && matches.size() > 8)
  {
    std::vector<uchar> inliers;
    cv::findFundamentalMat(points0, points1, cv::FM_RANSAC, 20, 0.99, inliers);
    int j = 0;
    for (int i = 0; i < matches.size(); i++)
    {
      if (inliers[i])
      {
        matches[j++] = matches[i];
      }
    }
    matches.resize(j);
  }

  return matches.size();
}

// int PointMatcher::MatchingPoints(const Eigen::Matrix<float, 259, Eigen::Dynamic> &features0,
//                                  const Eigen::Matrix<float, 259, Eigen::Dynamic> &features1,
//                                  std::vector<cv::DMatch> &matches, bool outlier_rejection)
// {
//   if (features0.cols() < 1 || features1.cols() < 1)
//   {
//     return 0;
//   }

//   Eigen::Matrix<float, 259, Eigen::Dynamic> normalized_features0, normalized_features1;
//   float scale = 0.7;
//   NormalizeKeypoints(features0, normalized_features0, _config.image_width, _config.image_height, scale);
//   NormalizeKeypoints(features1, normalized_features1, _config.image_width, _config.image_height, scale);

//   matches.clear();
//   std::vector<cv::Point> points0, points1;
//   // lightglue
//   Eigen::Matrix<int, Eigen::Dynamic, 2> matches_index;
//   Eigen::Matrix<float, Eigen::Dynamic, 1> matches_score;
//   _lightglue->infer(normalized_features0.bottomRows(258), normalized_features1.bottomRows(258), matches_index, matches_score);

//   for (size_t i = 0; i < matches_index.rows(); i++)
//   {
//     matches.emplace_back(matches_index(i, 0), matches_index(i, 1), 1.0 - matches_score(i));
//     if (outlier_rejection)
//     {
//       points0.emplace_back(features0(1, matches_index(i, 0)), features0(2, matches_index(i, 0)));
//       points1.emplace_back(features1(1, matches_index(i, 1)), features1(2, matches_index(i, 1)));
//     }
//   }

//   // reject outliers
//   if (outlier_rejection && matches.size() > 8)
//   {
//     std::vector<uchar> inliers;
//     cv::findFundamentalMat(points0, points1, cv::FM_RANSAC, 20, 0.99, inliers);
//     int j = 0;
//     for (int i = 0; i < matches.size(); i++)
//     {
//       if (inliers[i])
//       {
//         matches[j++] = matches[i];
//       }
//     }
//     matches.resize(j);
//   }

//   return matches.size();
// }