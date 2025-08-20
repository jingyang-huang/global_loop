//
// Created by haoyuefan on 2021/9/22.
//

#ifndef SUPER_POINT_H_
#define SUPER_POINT_H_

#include <string>
#include <memory>
#include <Eigen/Core>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>

#include "ThirdParty/tensorrt_common/include/buffers.h"
#include "sys_configs.h"

using samplesCommon::SampleUniquePtr;

class SuperPoint
{
public:
    explicit SuperPoint(const SuperPointConfig &super_point_config);

    bool build();

    bool infer(const cv::Mat &image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features);
    bool infer_withTrack(const cv::Mat &image_, FeatureData &track_feature_seq, Eigen::Matrix<float, 259, Eigen::Dynamic> &extract_features, Eigen::Matrix<float, 259, Eigen::Dynamic> &track_features);
    void save_engine();

    bool deserialize_engine();

private:
    int input_width;
    int input_height;
    int resized_width;
    int resized_height;
    float w_scale;
    float h_scale;

    SuperPointConfig super_point_config_;
    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims semi_dims_{};
    nvinfer1::Dims desc_dims_{};
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<std::vector<int>> keypoints_;
    std::vector<std::vector<float>> descriptors_;

    bool construct_network(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                           SampleUniquePtr<nvinfer1::INetworkDefinition> &network,
                           SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                           SampleUniquePtr<nvonnxparser::IParser> &parser) const;

    bool process_input(const samplesCommon::BufferManager &buffers, const cv::Mat &image);

    bool process_output(const samplesCommon::BufferManager &buffers, Eigen::Matrix<float, 259, Eigen::Dynamic> &features);
    bool process_output_withTrack(const samplesCommon::BufferManager &buffers, FeatureData &track_feature_seq, Eigen::Matrix<float, 259, Eigen::Dynamic> &extract_features, Eigen::Matrix<float, 259, Eigen::Dynamic> &track_features);
    bool keypoints_decoder(const float *scores, const float *descriptors, Eigen::Matrix<float, 259, Eigen::Dynamic> &features);

    std::vector<size_t> sort_indexes(std::vector<float> &data);
    int clip(int val, int max);

    void detect_point(const float *heat_map, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, int h, int w, float threshold, int border, int top_k);
    void extract_descriptors(const float *descriptors, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, int h, int w, int s);
    void track_point(const float *heat_map, FeatureData &track_featuredata, Eigen::Matrix<float, 259, Eigen::Dynamic> &track_features, int h, int w, int border);
};

typedef std::shared_ptr<SuperPoint> SuperPointPtr;

#endif // SUPER_POINT_H_
