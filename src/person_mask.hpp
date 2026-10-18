#pragma once
#include <opencv2/core.hpp>
#include <filesystem>

// Generate the first-frame person mask using a local DeepLabV3 ONNX model.
cv::Mat generatePersonMask(const cv::Mat& bgr, const std::filesystem::path& modelPath);
