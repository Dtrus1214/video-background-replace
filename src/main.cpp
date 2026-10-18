#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "matanyone_onnx.hpp"
#include "person_mask.hpp"

namespace fs = std::filesystem;

cv::Mat composite(const cv::Mat& frame, const cv::Mat& background, const cv::Mat& matte) {
    if (frame.size() != matte.size() || frame.size() != background.size() ||
        frame.type() != CV_8UC3 || background.type() != CV_8UC3 ||
        (matte.type() != CV_8UC1 && matte.type() != CV_32FC1))
        throw std::runtime_error("Composite requires matching dimensions, BGR images and a grayscale matte.");
    cv::Mat alpha, foreground, backdrop;
    matte.convertTo(alpha, CV_32FC1, matte.type() == CV_8UC1 ? 1.0 / 255.0 : 1.0);
    cv::cvtColor(alpha, alpha, cv::COLOR_GRAY2BGR);
    frame.convertTo(foreground, CV_32FC3);
    background.convertTo(backdrop, CV_32FC3);
    cv::Mat blended = foreground.mul(alpha) + backdrop.mul(cv::Scalar::all(1.0) - alpha);
    cv::Mat result;
    blended.convertTo(result, CV_8UC3);
    return result;
}

cv::Mat cover(const cv::Mat& image, cv::Size target) {
    const double scale = std::max(double(target.width) / image.cols, double(target.height) / image.rows);
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(std::max(target.width, int(std::ceil(image.cols * scale))),
                                      std::max(target.height, int(std::ceil(image.rows * scale)))));
    return resized(cv::Rect((resized.cols - target.width) / 2, (resized.rows - target.height) / 2,
                            target.width, target.height)).clone();
}

class BackgroundMedia {
    std::string path_;
    cv::Mat image_, frame_, fitted_;
    cv::VideoCapture video_;
    double fps_ = 0;
    std::size_t position_ = 0;
    std::size_t fittedPosition_ = 0;
public:
    explicit BackgroundMedia(const std::string& path) : path_(path) {
        if (cv::haveImageReader(path)) image_ = cv::imread(path);
        if (!image_.empty()) return;
        video_.open(path);
        fps_ = video_.get(cv::CAP_PROP_FPS);
        if (!video_.isOpened() || !std::isfinite(fps_) || fps_ <= 0 || !video_.read(frame_))
            throw std::runtime_error("Cannot read background image/video or its frame rate.");
    }
    cv::Mat at(std::size_t sourceIndex, double sourceFps, cv::Size size) {
        const auto target = image_.empty() ? static_cast<std::size_t>(std::floor(sourceIndex * fps_ / sourceFps + 1e-9)) : 0;
        if (target < position_) throw std::runtime_error("Background timestamps must be chronological.");
        while (position_ < target) {
            cv::Mat next;
            if (!video_.read(next)) {
                video_.release();
                video_.open(path_);
                if (!video_.read(next)) throw std::runtime_error("Cannot restart background video.");
            }
            frame_ = next;
            ++position_;
        }
        if (fitted_.empty() || fitted_.size() != size || fittedPosition_ != target) {
            fitted_ = cover(image_.empty() ? frame_ : image_, size);
            fittedPosition_ = target;
        }
        return fitted_;
    }
};

int selfTest() {
    cv::Mat person(1, 3, CV_8UC3, cv::Scalar(200, 100, 50));
    cv::Mat backdrop(1, 3, CV_8UC3, cv::Scalar(20, 40, 60));
    cv::Mat alpha = (cv::Mat_<unsigned char>(1, 3) << 0, 255, 128);
    const auto output = composite(person, backdrop, alpha);
    cv::Mat floatAlpha;
    alpha.convertTo(floatAlpha, CV_32FC1, 1.0 / 255.0);
    if (cv::norm(output, composite(person, backdrop, floatAlpha), cv::NORM_INF) != 0)
        throw std::runtime_error("Float alpha compositing differs from normalized byte alpha.");
    for (int channel = 0; channel < 3; ++channel) {
        if (output.at<cv::Vec3b>(0, 0)[channel] != backdrop.at<cv::Vec3b>(0, 0)[channel] ||
            output.at<cv::Vec3b>(0, 1)[channel] != person.at<cv::Vec3b>(0, 1)[channel])
            throw std::runtime_error("Opaque/transparent alpha test failed.");
        const double expected = person.at<cv::Vec3b>(0, 2)[channel] * (128.0 / 255.0) +
                                backdrop.at<cv::Vec3b>(0, 2)[channel] * (127.0 / 255.0);
        if (std::abs(output.at<cv::Vec3b>(0, 2)[channel] - expected) > 1)
            throw std::runtime_error("Soft alpha test failed.");
    }
    std::cout << "Composite tests passed.\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") return selfTest();
        if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
            std::cout << "Usage: video-background-replace --input clip.mp4 --background background.mp4 "
                         "--model models/matanyone2-onnx --output replaced.avi\n"
                         "People are detected automatically when --mask is omitted.\n"
                         "Optional --mask path overrides detection; --person-model selects local detector ONNX.\n"
                         "Optional --save-initial-mask path saves the initial mask for inspection.\n"
                         "Or use --mattes results/clip/pha instead of --model.\n"
                         "Optional --alpha-dir path saves inferred alpha as PNGs.\n"
                         "Background accepts an image or a video; videos loop and follow source timing.\n"
                         "Matte PNG names must be zero-padded frame numbers starting at 0000.\n"
                         "Output uses MJPEG AVI; audio is added separately with FFmpeg.\n";
            return 0;
        }
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            const std::string key = argv[i];
            if (i + 1 >= argc || (key != "--input" && key != "--background" && key != "--mattes" && key != "--output" &&
                                 key != "--model" && key != "--mask" && key != "--alpha-dir" &&
                                 key != "--person-model" && key != "--save-initial-mask"))
                throw std::runtime_error("Unknown option or missing value: " + key);
            if (!args.emplace(key, argv[i + 1]).second) throw std::runtime_error("Duplicate option: " + key);
        }
        for (const auto* key : {"--input", "--background", "--output"})
            if (!args.count(key)) throw std::runtime_error(std::string("Missing ") + key);
        const bool inference = args.count("--model") != 0;
        if (inference == (args.count("--mattes") != 0))
            throw std::runtime_error("Use either --model or --mattes.");
        if (!inference && (args.count("--mask") || args.count("--person-model") || args.count("--save-initial-mask")))
            throw std::runtime_error("Mask initialization options require --model.");
        if (args.count("--save-initial-mask") && fs::exists(args.at("--save-initial-mask")))
            throw std::runtime_error("Initial-mask output file already exists.");
        if (args.count("--alpha-dir")) {
            if (!inference) throw std::runtime_error("--alpha-dir requires --model.");
            if (fs::exists(args.at("--alpha-dir"))) throw std::runtime_error("Alpha output directory already exists.");
        }
        const fs::path outputPath = fs::absolute(args.at("--output")).lexically_normal();
        if (outputPath.extension() != ".avi") throw std::runtime_error("Output must use the .avi extension.");
        if (fs::exists(outputPath)) throw std::runtime_error("Output already exists; choose a new filename.");
        std::vector<fs::path> mattes;
        if (!inference) {
            for (const auto& entry : fs::directory_iterator(args.at("--mattes")))
                if (entry.is_regular_file() && entry.path().extension() == ".png") mattes.push_back(entry.path());
            std::sort(mattes.begin(), mattes.end());
            if (mattes.empty()) throw std::runtime_error("No matte PNGs found.");
            for (std::size_t i = 0; i < mattes.size(); ++i) {
                const auto stem = mattes[i].stem().string();
                if (stem.empty() || stem.find_first_not_of("0123456789") != std::string::npos || std::stoull(stem) != i)
                    throw std::runtime_error("Matte sequence must be consecutive, numerically ordered and start at zero.");
            }
        }
        std::unique_ptr<MatAnyoneOnnx> model;
        cv::Mat firstMask;
        if (inference) {
            model = std::make_unique<MatAnyoneOnnx>(fs::path(args.at("--model")));
            if (args.count("--mask")) {
                firstMask = cv::imread(args.at("--mask"), cv::IMREAD_GRAYSCALE);
                if (firstMask.empty()) throw std::runtime_error("Cannot read first-frame mask.");
            }
        }
        cv::VideoCapture input(args.at("--input"));
        if (!input.isOpened()) throw std::runtime_error("Cannot open input video.");
        const double fps = input.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps <= 0) throw std::runtime_error("Input has invalid frame rate.");
        BackgroundMedia background(args.at("--background"));
        if (args.count("--alpha-dir")) fs::create_directories(args.at("--alpha-dir"));
        cv::VideoWriter writer;
        cv::Mat frame, fitted;
        std::size_t index = 0;
        while (input.read(frame)) {
            if (inference && index == 0) {
                if (firstMask.empty()) {
                    const fs::path detector = args.count("--person-model") ? fs::path(args.at("--person-model")) :
                        fs::path(args.at("--model")).parent_path() / "person-segmentation.onnx";
                    std::cout << "Detecting people in the first frame..." << std::endl;
                    firstMask = generatePersonMask(frame, detector);
                }
                if (args.count("--save-initial-mask")) {
                    const fs::path maskPath(args.at("--save-initial-mask"));
                    if (maskPath.has_parent_path()) fs::create_directories(maskPath.parent_path());
                    if (!cv::imwrite(maskPath.string(), firstMask)) throw std::runtime_error("Cannot save initial person mask.");
                }
            }
            if (!inference && index >= mattes.size()) throw std::runtime_error("Video has more frames than the matte sequence.");
            cv::Mat alpha = inference ? (index == 0 ? model->initialize(frame, firstMask) : model->step(frame)) :
                                       cv::imread(mattes[index].string(), cv::IMREAD_GRAYSCALE);
            if (alpha.empty() || alpha.size() != frame.size())
                throw std::runtime_error("Unreadable matte or dimensions differ from input.");
            if (args.count("--alpha-dir")) {
                const auto number = std::to_string(index);
                const auto name = std::string(number.size() < 8 ? 8 - number.size() : 0, '0') + number + ".png";
                cv::Mat png;
                alpha.convertTo(png, CV_16UC1, 65535.0);
                if (!cv::imwrite((fs::path(args.at("--alpha-dir")) / name).string(), png))
                    throw std::runtime_error("Failed to save alpha PNG.");
            }
            if (!writer.isOpened()) {
                writer.open(outputPath.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame.size());
                if (!writer.isOpened()) throw std::runtime_error("Cannot create output video.");
            }
            fitted = background.at(index, fps, frame.size());
            writer.write(composite(frame, fitted, alpha));
            ++index;
        }
        if (index == 0 || (!inference && index != mattes.size())) throw std::runtime_error("Decoded video and matte frame counts differ.");
        writer.release();
        std::cout << "Wrote " << index << " frames to " << outputPath.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
