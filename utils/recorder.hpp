#pragma once
#include "thread_safe_queue.hpp"
#include <atomic>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <yaml-cpp/yaml.h>

namespace dart_vision {

class Recorder {
public:
    Recorder(const YAML::Node& config) {
        enable_ = config["enable"].as<bool>(false);
        if (!enable_) return;
        out_fps_ = config["fps"].as<int>(30);
        path_ = config["path"].as<std::string>("data");
        codec_ = config["codec"].as<std::string>("MJPG");
    }

    void init() {
        if (!enable_ || started_) return;
        started_ = true;
        rec_q_ = std::make_unique<ThreadSafeQueue<cv::Mat>>(2);
        writer_thread_ = std::thread(&Recorder::writeLoop, this);
    }

    bool push(cv::Mat frame) {
        if (!started_ || !rec_q_ || !rec_q_->running()) return false;
        rec_q_->push(frame.clone());
        return true;
    }

    void stop() {
        if (!started_) return;
        if (rec_q_) rec_q_->stop();
        if (writer_thread_.joinable()) writer_thread_.join();
        if (writer_.isOpened()) { writer_.release(); std::cout << "[Recorder] 已停止" << std::endl; }
        started_ = false;
    }

    bool enabled() const { return enable_ && started_; }

private:
    void writeLoop() {
        while (rec_q_->running()) {
            cv::Mat frame;
            if (!rec_q_->pop(frame, 500)) continue;

            if (!writer_.isOpened()) {
                std::string full_path = buildPath();
                mkdir(full_path.substr(0, full_path.find_last_of('/')).c_str(), 0755);
                int fourcc = cv::VideoWriter::fourcc(
                    codec_[0], codec_[1], codec_[2], codec_[3]);
                writer_.open(full_path, fourcc, out_fps_, frame.size());
                if (!writer_.isOpened()) {
                    std::cerr << "[Recorder] 无法打开: " << full_path << std::endl;
                    rec_q_->stop();
                    return;
                }
                std::cout << "[Recorder] 录制中: " << full_path << std::endl;
            }
            writer_.write(frame);
        }
        if (writer_.isOpened()) writer_.release();
    }

    std::string buildPath() {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << path_ << "/" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".avi";
        return oss.str();
    }

    bool enable_ = false;
    bool started_ = false;
    int out_fps_ = 30;
    std::string path_ = "data";
    std::string codec_ = "MJPG";

    std::unique_ptr<ThreadSafeQueue<cv::Mat>> rec_q_;
    std::thread writer_thread_;
    cv::VideoWriter writer_;
};

} // namespace dart_vision
