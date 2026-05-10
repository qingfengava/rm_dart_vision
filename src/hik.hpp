#pragma once
#include "MvCameraControl.h"
#include <algorithm>
#include <atomic>
#include <deque>
#include <iostream>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <unordered_map>
#include <yaml-cpp/yaml.h>
namespace dart_vision {
class HikCamera {
public:
    HikCamera(const YAML::Node& config) {
        config_ = config;
    }
    void init() {
        running_ = true;
        load(config_);
    }
    void load(const YAML::Node& config) {
        std::string target_sn = config["target_sn"].as<std::string>();
        if (!initialize_camera(target_sn)) {
            std::cout << "Failed to initialize camera with SN: " << target_sn << std::endl;
            return;
        }
        auto acquisition_frame_rate = config["acquisition_frame_rate"].as<double>();
        int r = MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", acquisition_frame_rate);
        if (r == MV_OK) {
            std::cout << "AcquisitionFrameRate set successfully!" << std::endl;
        } else {
            std::cout << "Failed to set AcquisitionFrameRate!" << std::endl;
        }
        auto exposure_time = config["exposure_time"].as<double>();
        r = MV_CC_SetFloatValue(camera_handle_, "ExposureTime", exposure_time);
        if (r == MV_OK) {
            std::cout << "ExposureTime set successfully!" << std::endl;
        } else {
            std::cout << "Failed to set ExposureTime!" << std::endl;
        }
        auto gain = config["gain"].as<double>();
        r = MV_CC_SetFloatValue(camera_handle_, "Gain", gain);
        if (r == MV_OK) {
            std::cout << "Gain set successfully!" << std::endl;
        } else {
            std::cout << "Failed to set Gain!" << std::endl;
        }
        r = MV_CC_SetBoolValue(camera_handle_, "AcquisitionFrameRateEnable", true);
    }
    void start() {
        int n_ret = MV_CC_StartGrabbing(camera_handle_);
        if (n_ret != MV_OK) {
            std::cout << "Failed to start camera grabbing!" << std::endl;
        }
        running_ = true;
        // daemon_thread_ = std::thread(&HikCamera::run_loop, this);
    }

    void run_loop() {
        while (running_) {
            hik_capture_loop();
            if (!running_) {
                break;
            }
            restart();
        }
    }
    struct MvGvspPixelTypeHash {
        std::size_t operator()(const MvGvspPixelType& v) const noexcept {
            return std::hash<int>()(static_cast<int>(v));
        }
    };
    const std::unordered_map<MvGvspPixelType, int, MvGvspPixelTypeHash> CVT_MAP_RGB = {
        { PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2BGR },
        { PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2BGR },
        { PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2BGR },
        { PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2BGR },
        { PixelType_Gvsp_RGB8_Packed, cv::COLOR_RGB2BGR },
        { PixelType_Gvsp_Mono8, cv::COLOR_GRAY2BGR },
    };
    cv::Mat read() {
        MV_FRAME_OUT out_frame;
        cv::Mat src;
        int n_ret = MV_CC_GetImageBuffer(camera_handle_, &out_frame, 1000);
        if (n_ret == MV_OK) {
            const auto current_time = std::chrono::steady_clock::now();
            const auto& info = out_frame.stFrameInfo;
            src = cv::Mat(cv::Size(info.nWidth, info.nHeight), CV_8U, out_frame.pBufAddr);
            const auto pixel_type = info.enPixelType;
            const auto& ref_cvt = CVT_MAP_RGB;
            int cvt_code = ref_cvt.at(pixel_type);
            cv::cvtColor(src, src, cvt_code);
            MV_CC_FreeImageBuffer(camera_handle_, &out_frame);
        } else {
            std::cout << "Failed to get image buffer!: " << n_ret << std::endl;
            restart();
        }
        // if (frames_.empty()) {
        //     return cv::Mat();
        // }
        // cv::Mat img = frames_.front();
        // frames_.pop_front();
        return src;
    }

    void hik_capture_loop() {
        std::cout << "Starting image capture loop!" << std::endl;
        MV_FRAME_OUT out_frame;
        while (running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            int n_ret = MV_CC_GetImageBuffer(camera_handle_, &out_frame, 1000);
            if (n_ret == MV_OK) {
                const auto current_time = std::chrono::steady_clock::now();
                const auto& info = out_frame.stFrameInfo;
                cv::Mat src(cv::Size(info.nWidth, info.nHeight), CV_8U, out_frame.pBufAddr);
                const auto pixel_type = info.enPixelType;
                const auto& ref_cvt = CVT_MAP_RGB;
                int cvt_code = ref_cvt.at(pixel_type);
                cv::cvtColor(src, src, cvt_code);
                {
                    std::lock_guard<std::mutex> lock(frames_mutex_);
                    frames_.push_back(std::move(src));
                }
                MV_CC_FreeImageBuffer(camera_handle_, &out_frame);
            } else {
                std::cout << "Failed to get image buffer!" << std::endl;
                break;
            }
        }
        std::cout << "Exiting image capture loop." << std::endl;
    }
    void stop() {
        if (!running_) {
            return;
        }
        running_ = false;
        if (daemon_thread_.joinable()) {
            daemon_thread_.join();
        }
        if (camera_handle_) {
            MV_CC_StopGrabbing(camera_handle_);
            MV_CC_CloseDevice(camera_handle_);
            MV_CC_DestroyHandle(&camera_handle_);
        }
        std::cout << "hik_camera has stop " << std::endl;
    }
    void restart() {
        std::cout << "Restarting camera" << std::endl;
        MV_CC_StopGrabbing(camera_handle_);
        MV_CC_CloseDevice(camera_handle_);
        MV_CC_DestroyHandle(&camera_handle_);
        camera_handle_ = nullptr;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        load(config_);

        int n_ret = MV_CC_StartGrabbing(camera_handle_);
        if (n_ret != MV_OK) {
            std::cout << "Failed to start grabbing after restart" << std::endl;
            return;
        }

        std::cout << "Camera restarted successfully!" << std::endl;
        return;
    }
    bool initialize_camera(const std::string& target_sn) {
        target_sn_ = target_sn;

        if (camera_handle_ != nullptr) {
            std::cout << "Closing previously opened camera" << std::endl;
            MV_CC_CloseDevice(camera_handle_);
            MV_CC_DestroyHandle(camera_handle_);
            camera_handle_ = nullptr;
        }

        while (running_) {
            MV_CC_DEVICE_INFO_LIST device_list = { 0 };
            int n_ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
            if (n_ret != MV_OK) {
                std::cout << "MV_CC_EnumDevices failed, error code: " << n_ret << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }

            if (device_list.nDeviceNum == 0) {
                std::cout << "No USB cameras found" << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }

            std::cout << "Found " << device_list.nDeviceNum << " USB camera(s):" << std::endl;
            for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
                auto info = device_list.pDeviceInfo[i];
                const char* sn =
                    reinterpret_cast<const char*>(info->SpecialInfo.stUsb3VInfo.chSerialNumber);
                std::cout << "[ " << i << " ] SN = " << sn << std::endl;
            }

            int sel = -1;
            for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
                auto info = device_list.pDeviceInfo[i];
                const char* sn =
                    reinterpret_cast<const char*>(info->SpecialInfo.stUsb3VInfo.chSerialNumber);
                if (target_sn == sn) {
                    sel = i;
                    break;
                }
            }

            if (sel < 0) {
                std::cout << "Camera with serial " << target_sn << " not found" << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }

            std::cout << "Selecting camera at index " << sel << " (SN=" << target_sn << ")"
                      << std::endl;
            n_ret = MV_CC_CreateHandle(&camera_handle_, device_list.pDeviceInfo[sel]);
            if (n_ret != MV_OK) {
                std::cout << "MV_CC_CreateHandle failed: " << n_ret << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }

            n_ret = MV_CC_OpenDevice(camera_handle_);
            if (n_ret != MV_OK) {
                std::cout << "MV_CC_OpenDevice failed: " << n_ret << std::endl;
                MV_CC_DestroyHandle(camera_handle_);
                camera_handle_ = nullptr;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }

            std::cout << "Camera initialized successfully" << std::endl;
            return true;
        }
        return false;
    }
    std::deque<cv::Mat> frames_;
    std::mutex frames_mutex_;
    YAML::Node config_;
    void* camera_handle_ = nullptr;
    std::string target_sn_;
    std::atomic<bool> running_ { false };
    std::thread daemon_thread_;
};
} // namespace dart_vision