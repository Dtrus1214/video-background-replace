#include "matanyone_onnx.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
    Tensor() = default;
    explicit Tensor(std::vector<int64_t> dimensions) : shape(std::move(dimensions)) {
        size_t count = 1;
        for (auto d : shape) {
            if (d <= 0) throw std::runtime_error("Invalid tensor shape.");
            count *= static_cast<size_t>(d);
        }
        data.resize(count);
    }
};
using Feed = std::map<std::string, Tensor>;

class Graph {
    Ort::Session session_;
    std::vector<std::string> inputs_, outputs_;
public:
    Graph(Ort::Env& env, const std::filesystem::path& path) : session_(nullptr) {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(4);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session(env, path.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session_.GetInputCount(); ++i) {
            inputs_.emplace_back(session_.GetInputNameAllocated(i, allocator).get());
            if (session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                throw std::runtime_error("Expected an FP32 MatAnyone 2 bundle.");
        }
        for (size_t i = 0; i < session_.GetOutputCount(); ++i)
            outputs_.emplace_back(session_.GetOutputNameAllocated(i, allocator).get());
    }
    Feed run(Feed& feed) {
        auto cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> values;
        std::vector<const char*> inputNames, outputNames;
        for (const auto& name : inputs_) {
            auto& tensor = feed.at(name);
            inputNames.push_back(name.c_str());
            values.push_back(Ort::Value::CreateTensor<float>(cpu, tensor.data.data(), tensor.data.size(),
                                                           tensor.shape.data(), tensor.shape.size()));
        }
        for (const auto& name : outputs_) outputNames.push_back(name.c_str());
        auto results = session_.Run(Ort::RunOptions{nullptr}, inputNames.data(), values.data(), values.size(),
                                    outputNames.data(), outputNames.size());
        Feed output;
        for (size_t i = 0; i < results.size(); ++i) {
            Tensor tensor(results[i].GetTensorTypeAndShapeInfo().GetShape());
            const float* data = results[i].GetTensorData<float>();
            std::copy(data, data + tensor.data.size(), tensor.data.begin());
            if (!std::all_of(tensor.data.begin(), tensor.data.end(), [](float x) { return std::isfinite(x); }))
                throw std::runtime_error("Model returned non-finite values.");
            output.emplace(outputs_[i], std::move(tensor));
        }
        return output;
    }
};

Tensor concatenate(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) throw std::runtime_error("Empty memory bank.");
    auto shape = tensors.front().shape;
    shape.back() = 0;
    for (const auto& tensor : tensors) shape.back() += tensor.shape.back();
    Tensor result(shape);
    const auto rows = result.data.size() / static_cast<size_t>(shape.back());
    for (size_t row = 0; row < rows; ++row) {
        size_t offset = row * static_cast<size_t>(shape.back());
        for (const auto& tensor : tensors) {
            auto width = static_cast<size_t>(tensor.shape.back());
            std::copy_n(tensor.data.data() + row * width, width, result.data.data() + offset);
            offset += width;
        }
    }
    return result;
}
}

struct MatAnyoneOnnx::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "matanyone2"};
    int width = 0, height = 0, sensoryDim = 0, memEvery = 0, maxFrames = 0, warmup = 0;
    int time = 0;
    bool initialized = false;
    std::unique_ptr<Graph> encoder, maskDeep, maskShallow, decodeFirst, decode;
    Tensor sensory, mask, lastPixel, lastValue, objectMemory;
    std::vector<Tensor> keys, shrinkages, values;
    cv::Size original;
    cv::Rect content;

    explicit Impl(const std::filesystem::path& bundle) {
        std::ifstream file(bundle / "model.txt");
        std::string signature;
        if (!(file >> signature >> width >> height >> sensoryDim >> memEvery >> maxFrames >> warmup) ||
            signature != "MATANYONE2_ONNX_V1" || width < 64 || height < 64 || width % 16 || height % 16 ||
            (static_cast<int64_t>(width) / 16) * (height / 16) < 30 ||
            sensoryDim <= 0 || memEvery != 5 || maxFrames != 5 || warmup != 10)
            throw std::runtime_error("Invalid or incompatible MatAnyone 2 ONNX manifest.");
        encoder = std::make_unique<Graph>(env, bundle / "encoder.onnx");
        maskDeep = std::make_unique<Graph>(env, bundle / "mask_deep.onnx");
        maskShallow = std::make_unique<Graph>(env, bundle / "mask_shallow.onnx");
        decodeFirst = std::make_unique<Graph>(env, bundle / "decode_first.onnx");
        decode = std::make_unique<Graph>(env, bundle / "decode.onnx");
    }

    Tensor imageTensor(const cv::Mat& bgr) {
        if (bgr.type() != CV_8UC3 || bgr.size() != original)
            throw std::runtime_error("Video frame dimensions/type changed; start a new inference session.");
        cv::Mat scaled, canvas(height, width, CV_8UC3, cv::Scalar::all(0));
        cv::resize(bgr, scaled, content.size(), 0, 0, cv::INTER_LINEAR_EXACT);
        scaled.copyTo(canvas(content));
        Tensor image({1, 3, height, width});
        const size_t plane = static_cast<size_t>(width) * height;
        for (int y = 0; y < height; ++y) {
            const auto* row = canvas.ptr<cv::Vec3b>(y);
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < 3; ++c)
                    image.data[c * plane + static_cast<size_t>(y) * width + x] = row[x][2 - c] / 255.0f;
        }
        return image;
    }

    void encodeMask(Tensor& image, Feed& features, bool deep, bool first) {
        Feed feed{{"image", image}, {"pixel", features.at("pixel")}, {"sensory", sensory}, {"mask", mask}};
        auto output = (deep ? maskDeep : maskShallow)->run(feed);
        lastValue = std::move(output.at("value"));
        if (deep) {
            sensory = std::move(output.at("new_sensory"));
            auto obj = std::move(output.at("object"));
            if (first) {
                keys.clear(); shrinkages.clear(); values.clear();
                objectMemory = std::move(obj);
            } else {
                if (objectMemory.shape != obj.shape) throw std::runtime_error("Object memory shape changed.");
                for (size_t i = 0; i < obj.data.size(); ++i) objectMemory.data[i] += obj.data[i];
            }
            keys.push_back(features.at("key"));
            shrinkages.push_back(features.at("shrinkage"));
            auto flat = lastValue;
            flat.shape = {1, 1, lastValue.shape.at(2), lastValue.shape.at(3) * lastValue.shape.at(4)};
            values.push_back(std::move(flat));
            if (keys.size() > static_cast<size_t>(maxFrames)) {
                // Keep permanent first-frame memory and the latest four memory frames.
                keys.erase(keys.begin() + 1); shrinkages.erase(shrinkages.begin() + 1); values.erase(values.begin() + 1);
            }
        }
        lastPixel = features.at("pixel");
    }

    void predict(Tensor& image, Feed& features, bool first) {
        Feed feed = features;
        feed["sensory"] = sensory;
        feed["last_mask"] = mask;
        feed["last_pixel"] = lastPixel;
        feed["last_value"] = lastValue;
        feed["object_memory"] = objectMemory;
        if (!first) {
            feed["memory_key"] = concatenate(keys);
            feed["memory_shrinkage"] = concatenate(shrinkages);
            feed["memory_value"] = concatenate(values);
        }
        auto prediction = (first ? decodeFirst : decode)->run(feed);
        mask = std::move(prediction.at("mask"));
        sensory = std::move(prediction.at("new_sensory"));
        encodeMask(image, features, first || time % memEvery == 0, first);
    }

    cv::Mat alpha() {
        if (mask.shape != std::vector<int64_t>{1, 1, height, width})
            throw std::runtime_error("Invalid model alpha dimensions.");
        cv::Mat small(height, width, CV_32FC1, mask.data.data()), restored;
        cv::resize(small(content), restored, original, 0, 0, cv::INTER_LINEAR);
        cv::max(restored, 0, restored);
        cv::min(restored, 1, restored);
        return restored;
    }
};

MatAnyoneOnnx::MatAnyoneOnnx(const std::filesystem::path& bundle) : impl_(std::make_unique<Impl>(bundle)) {}
MatAnyoneOnnx::~MatAnyoneOnnx() = default;

cv::Mat MatAnyoneOnnx::initialize(const cv::Mat& bgr, const cv::Mat& firstMask) {
    auto& s = *impl_;
    s.initialized = false;
    if (bgr.empty() || firstMask.type() != CV_8UC1 || firstMask.size() != bgr.size())
        throw std::runtime_error("First-frame mask must be grayscale and match the source frame.");
    s.original = bgr.size();
    const double scale = std::min(double(s.width) / bgr.cols, double(s.height) / bgr.rows);
    const int w = std::min(s.width, std::max(1, int(std::round(bgr.cols * scale))));
    const int h = std::min(s.height, std::max(1, int(std::round(bgr.rows * scale))));
    s.content = cv::Rect((s.width - w) / 2, (s.height - h) / 2, w, h);
    s.sensory = Tensor({1, 1, s.sensoryDim, s.height / 16, s.width / 16});
    s.mask = Tensor({1, 1, s.height, s.width});
    cv::Mat scaled;
    cv::resize(firstMask, scaled, s.content.size(), 0, 0, cv::INTER_NEAREST);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            s.mask.data[static_cast<size_t>(y + s.content.y) * s.width + x + s.content.x] = scaled.at<unsigned char>(y, x) / 255.0f;
    if (*std::max_element(s.mask.data.begin(), s.mask.data.end()) == 0)
        throw std::runtime_error("First-frame mask contains no foreground.");
    Tensor image = s.imageTensor(bgr);
    Feed input{{"image", image}};
    auto features = s.encoder->run(input);
    s.time = 0;
    s.encodeMask(image, features, true, true);
    // Upstream runs one prediction when seeding, then ten first-frame predictions.
    for (int i = 0; i <= s.warmup; ++i) s.predict(image, features, true);
    s.initialized = true;
    return s.alpha();
}

cv::Mat MatAnyoneOnnx::step(const cv::Mat& bgr) {
    auto& s = *impl_;
    if (!s.initialized) throw std::runtime_error("Initialize the person mask before inference.");
    Tensor image = s.imageTensor(bgr);
    Feed input{{"image", image}};
    auto features = s.encoder->run(input);
    ++s.time;
    s.predict(image, features, false);
    return s.alpha();
}
