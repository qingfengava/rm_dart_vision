#pragma once
#include "type.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <yaml-cpp/node/node.h>
#ifdef __ARM_NEON
    #include <arm_neon.h>
#endif
namespace dart_vision {
bool initializing = true;

// int D_threshold_ = 35;
int lowH = 35, highH = 85;
int lowS = 50, highS = 255;
int lowV = 80, highV = 255;
static void onTrackbar(int, void*) {
    if (initializing)
        return;
    lowH = cv::getTrackbarPos("LowH", "mask");
    highH = cv::getTrackbarPos("HighH", "mask");
    lowS = cv::getTrackbarPos("LowS", "mask");
    highS = cv::getTrackbarPos("HighS", "mask");
    lowV = cv::getTrackbarPos("LowV", "mask");
    highV = cv::getTrackbarPos("HighV", "mask");
    // D_threshold_ = cv::getTrackbarPos("D_threshold_", "mask");
}

void initGUI() {
    cv::namedWindow("mask", cv::WINDOW_NORMAL); // 允许调整大小
    cv::resizeWindow("mask", 400, 600); // 设置初始窗口大小
    // static int tmp_D_threshold;
    // cv::createTrackbar("D_threshold_", "mask", &tmp_D_threshold, 255, onTrackbar);

    // cv::setTrackbarPos("D_threshold_", "mask", tmp_D_threshold);
    cv::createTrackbar("LowH", "mask", &lowH, 179, onTrackbar);
    cv::createTrackbar("HighH", "mask", &highH, 179, onTrackbar);
    cv::createTrackbar("LowS", "mask", &lowS, 255, onTrackbar);
    cv::createTrackbar("HighS", "mask", &highS, 255, onTrackbar);
    cv::createTrackbar("LowV", "mask", &lowV, 255, onTrackbar);
    cv::createTrackbar("HighV", "mask", &highV, 255, onTrackbar);

    cv::setTrackbarPos("LowH", "mask", lowH);
    cv::setTrackbarPos("HighH", "mask", highH);
    cv::setTrackbarPos("LowS", "mask", lowS);
    cv::setTrackbarPos("HighS", "mask", highS);
    cv::setTrackbarPos("LowV", "mask", lowV);
    cv::setTrackbarPos("HighV", "mask", highV);
    initializing = false;
}
#ifdef __ARM_NEON
inline void neon_diff_threshold_bgr(const cv::Mat& src, cv::Mat& dst, uint8_t D_thresh) noexcept {
    CV_Assert(src.type() == CV_8UC3);
    CV_Assert(dst.type() == CV_8UC1);
    const int w = src.cols;
    const int h = src.rows;

    const uint8x16_t threshv = vdupq_n_u8(D_thresh);

    // Continuous memory fast-path
    if (src.isContinuous() && dst.isContinuous()) {
        const uint8_t* ps = src.data;
        uint8_t* pd = dst.data;
        size_t total = static_cast<size_t>(w) * h;
        size_t i = 0;

        // Process 32 pixels per iteration (2 x 16)
        for (; i + 31 < total; i += 32) {
            uint8x16x3_t v1 = vld3q_u8(ps);
            uint8x16x3_t v2 = vld3q_u8(ps + 48);

            uint8x16_t max1 = vmaxq_u8(v1.val[2], v1.val[0]);
            uint8x16_t max2 = vmaxq_u8(v2.val[2], v2.val[0]);

            uint8x16_t diff1 = vqsubq_u8(v1.val[1], max1);
            uint8x16_t diff2 = vqsubq_u8(v2.val[1], max2);

            vst1q_u8(pd, vcgtq_u8(diff1, threshv));
            vst1q_u8(pd + 16, vcgtq_u8(diff2, threshv));

            ps += 96; // 32 pixels * 3 bytes
            pd += 32;
        }

        // leftover 16 pixels
        for (; i + 15 < total; i += 16) {
            uint8x16x3_t v = vld3q_u8(ps);
            uint8x16_t maxv = vmaxq_u8(v.val[2], v.val[0]);
            uint8x16_t diff = vqsubq_u8(v.val[1], maxv);
            vst1q_u8(pd, vcgtq_u8(diff, threshv));
            ps += 48;
            pd += 16;
        }

        // tail
        for (; i < total; ++i) {
            uint8_t B = ps[0], G = ps[1], R = ps[2];
            uint8_t maxRB = (R > B) ? R : B;
            uint8_t diff = (G > maxRB) ? (G - maxRB) : 0;
            *pd = (diff > D_thresh) ? 255 : 0;
            ps += 3;
            pd += 1;
        }
        return;
    }

    // Row-by-row fallback
    for (int y = 0; y < h; ++y) {
        const uint8_t* ps = src.data + y * src.step;
        uint8_t* pd = dst.data + y * dst.step;
        int x = 0;

        for (; x + 31 < w; x += 32) {
            uint8x16x3_t v1 = vld3q_u8(ps);
            uint8x16x3_t v2 = vld3q_u8(ps + 48);

            uint8x16_t max1 = vmaxq_u8(v1.val[2], v1.val[0]);
            uint8x16_t max2 = vmaxq_u8(v2.val[2], v2.val[0]);

            uint8x16_t diff1 = vqsubq_u8(v1.val[1], max1);
            uint8x16_t diff2 = vqsubq_u8(v2.val[1], max2);

            vst1q_u8(pd, vcgtq_u8(diff1, threshv));
            vst1q_u8(pd + 16, vcgtq_u8(diff2, threshv));

            ps += 96;
            pd += 32;
        }

        for (; x + 15 < w; x += 16) {
            uint8x16x3_t v = vld3q_u8(ps);
            uint8x16_t maxv = vmaxq_u8(v.val[2], v.val[0]);
            uint8x16_t diff = vqsubq_u8(v.val[1], maxv);
            vst1q_u8(pd, vcgtq_u8(diff, threshv));
            ps += 48;
            pd += 16;
        }

        for (; x < w; ++x) {
            uint8_t B = ps[0], G = ps[1], R = ps[2];
            uint8_t maxRB = (R > B) ? R : B;
            uint8_t diff = (G > maxRB) ? (G - maxRB) : 0;
            *pd = (diff > D_thresh) ? 255 : 0;
            ps += 3;
            pd++;
        }
    }
}
#endif
class Detect {
public:
    Detect(const YAML::Node& config) {
        // D_threshold_ = config["D_threshold"].as<int>();
        max_area_ = config["contours"]["max_area"].as<double>();
        min_area_ = config["contours"]["min_area"].as<double>();
        one_one_diff_ = config["contours"]["one_one_diff"].as<double>();
        if (config["enable_gui"])
            enable_gui_ = config["enable_gui"].as<bool>();
        if (enable_gui_)
            initGUI();
    }
    cv::Mat preprocess(const cv::Mat& frame) {
        cv::Mat mask;

        // #ifdef __ARM_NEON

        //         mask = cv::Mat(frame.rows, frame.cols, CV_8UC1);

        //         neon_diff_threshold_bgr(frame, mask, static_cast<uint8_t>(D_threshold_));
        // #endif
        //         if (mask.empty()) {
        //             static cv::Mat channels[3];
        //             static cv::Mat maxRB;
        //             static cv::Mat diff;
        //             cv::split(frame, channels); // B G R

        //             cv::max(channels[2], channels[0], maxRB); // max(R,B)
        //             cv::subtract(channels[1], maxRB, diff); // G - max(R,B)

        //             cv::threshold(diff, mask, D_threshold_, 255, cv::THRESH_BINARY);
        //         }
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        cv::Scalar lower_green(lowH, lowS, lowV);
        cv::Scalar upper_green(highH, highS, highV);

        cv::inRange(hsv, lower_green, upper_green, mask);

        return mask;
    }
    std::vector<GreenLight> detect(cv::Mat& frame) {
        std::vector<GreenLight> lights;

        std::vector<std::vector<cv::Point>> contours;
        auto mask = preprocess(frame);
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& c: contours) {
            const cv::Moments m = cv::moments(c);
            const float area = m.m00;
            if (area < min_area_)
                continue;

            cv::Rect box = cv::boundingRect(c);
            const float ratio = float(box.width) / float(box.height);
            if (ratio < 1 - one_one_diff_ || ratio > 1 + one_one_diff_)
                continue;

            if (area <= 0.0f)
                continue;

            cv::Point2f center(
                static_cast<float>(m.m10 / m.m00),
                static_cast<float>(m.m01 / m.m00)
            );

            cv::Rect2f rect(box.x, box.y, box.width, box.height);

            lights.emplace_back(GreenLight {
                .score = area,
                .bbox = rect,
                .center = center,

            });

            cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2);
            cv::circle(frame, center, 3, cv::Scalar(0, 255, 0), -1);
        }

        if (enable_gui_) {
            cv::imshow("mask", mask);
            cv::waitKey(1);
        }

        return lights;
    }
    double min_area_ = 100;
    double max_area_ = 10000;
    double one_one_diff_;
    bool enable_gui_ = true;
};
} // namespace dart_vision