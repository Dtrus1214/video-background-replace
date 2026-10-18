#include "person_mask.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

cv::Mat generatePersonMask(const cv::Mat& bgr, const std::filesystem::path& modelPath) {
    if (bgr.empty() || bgr.type() != CV_8UC3)
        throw std::runtime_error("Automatic detection needs an 8-bit BGR frame.");
    if (!std::filesystem::is_regular_file(modelPath))
        throw std::runtime_error("Local person-segmentation ONNX model not found: " + modelPath.string());
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "person-detection");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4);
    Ort::Session session(env, modelPath.c_str(), options);
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
        throw std::runtime_error("Invalid person-segmentation ONNX model interface.");
    // torchvision's weights use shortest-side 520 and ImageNet RGB normalization.
    // The C++ implementation uses OpenCV's deterministic bilinear byte resize.
    const double scale = 520.0 / std::min(bgr.rows, bgr.cols);
    const int height = std::max(520, static_cast<int>(bgr.rows * scale));
    const int width = std::max(520, static_cast<int>(bgr.cols * scale));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR_EXACT);
    const auto plane = static_cast<size_t>(height) * width;
    std::vector<float> input(3 * plane);
    constexpr std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
    constexpr std::array<float, 3> deviation{0.229f, 0.224f, 0.225f};
    for (int y = 0; y < height; ++y) {
        const auto* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c)
                input[c * plane + static_cast<size_t>(y) * width + x] =
                    (row[x][2 - c] / 255.0f - mean[c]) / deviation[c];
    }
    const std::array<int64_t, 4> shape{1, 3, height, width};
    auto cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(cpu, input.data(), input.size(), shape.data(), shape.size());
    const char* inputName = "image";
    const char* outputName = "logits";
    auto result = session.Run(Ort::RunOptions{nullptr}, &inputName, &tensor, 1, &outputName, 1);
    const auto outputInfo = result[0].GetTensorTypeAndShapeInfo();
    const auto outputShape = outputInfo.GetShape();
    if (outputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        outputShape != std::vector<int64_t>{1, 21, height, width})
        throw std::runtime_error("Unexpected person-segmentation output dimensions/type.");
    const float* logits = result[0].GetTensorData<float>();
    cv::Mat best(bgr.size(), CV_32FC1, cv::Scalar(-std::numeric_limits<float>::infinity()));
    cv::Mat classes(bgr.size(), CV_8UC1, cv::Scalar(0));
    for (int c = 0; c < 21; ++c) {
        cv::Mat channel(height, width, CV_32FC1, const_cast<float*>(logits + c * plane));
        cv::Mat restored;
        cv::resize(channel, restored, bgr.size(), 0, 0, cv::INTER_LINEAR);
        if (!cv::checkRange(restored)) throw std::runtime_error("Person detector returned non-finite logits.");
        for (int y = 0; y < bgr.rows; ++y) {
            auto* scores = best.ptr<float>(y);
            auto* labels = classes.ptr<unsigned char>(y);
            const auto* values = restored.ptr<float>(y);
            for (int x = 0; x < bgr.cols; ++x) {
                if (values[x] > scores[x]) {
                    scores[x] = values[x];
                    labels[x] = static_cast<unsigned char>(c);
                }
            }
        }
    }
    // COCO_WITH_VOC_LABELS person index; all detected people share the foreground.
    cv::Mat mask = classes == 15;
    if (cv::countNonZero(mask) < std::max(16, static_cast<int>(mask.total() * 0.001)))
        throw std::runtime_error("No person detected in the first frame. Start with the person clearly visible.");
    return mask;
}
