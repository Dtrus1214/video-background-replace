#pragma once
#include <opencv2/core.hpp>
#include <filesystem>
#include <memory>

// Single-object, FP32, fixed-resolution bundle from tools/export_matanyone2.py.
class MatAnyoneOnnx {
public:
    explicit MatAnyoneOnnx(const std::filesystem::path& bundle);
    ~MatAnyoneOnnx();
    cv::Mat initialize(const cv::Mat& bgr, const cv::Mat& firstMask);
    cv::Mat step(const cv::Mat& bgr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
