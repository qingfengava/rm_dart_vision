#include "detect.hpp"
#include "hik.hpp"
#include "recorder.hpp"
#include "serial_driver.hpp"
#include "thread_safe_queue.hpp"
#include "tracker.hpp"
#include "tracker_mode.hpp"
#include "type.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <opencv2/highgui.hpp>
#include <thread>
using namespace dart_vision;

struct SendDartCmdData {
    float diff_center_norm = 0;
    uint32_t sum;
} __attribute__((packed));

struct FramePacket {
    cv::Mat frame;
    std::chrono::steady_clock::time_point timestamp;
};

struct ResultPacket {
    std::vector<GreenLight> lights;
    cv::Mat annotated_frame;
    cv::Mat mask;
    std::chrono::steady_clock::time_point timestamp;
};

std::atomic<bool> app_running{true};

void signal_handler(int) { app_running = false; }

void handleKey(int key, Detect& detect) {
    if (key == 's') detect.printHSV();
    else if (key == 'r') detect.resetHSV();
}

void drawStatus(cv::Mat& img, const TrackerMode mode, const TrackInfo& info,
                float cx_norm, float cy_norm, float fps) {
    int y = 20;
    auto put = [&](const std::string& txt, cv::Scalar color = cv::Scalar(0, 255, 0)) {
        cv::putText(img, txt, cv::Point(8, y), cv::FONT_HERSHEY_SIMPLEX,
                    0.45, cv::Scalar(0, 0, 0), 2);
        cv::putText(img, txt, cv::Point(8, y), cv::FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1);
        y += 18;
    };

    put("Mode: " + std::string(mode == TrackerMode::STATIONARY ? "STATIONARY" : "MOVING"));

    if (info.valid) {
        put("Track: CONFIRMED  id=" + std::to_string(info.id));
        put("Hits: " + std::to_string(info.hits) + "  Age: " + std::to_string(info.age));
        char buf[64];
        snprintf(buf, sizeof(buf), "cx: %+.4f  cy: %+.4f", cx_norm, cy_norm);
        put(buf, cv::Scalar(0, 255, 255));
    } else {
        put("Track: NONE", cv::Scalar(0, 0, 255));
    }

    char fps_buf[32];
    snprintf(fps_buf, sizeof(fps_buf), "FPS: %.0f", fps);
    put(fps_buf);
}

void camera_thread_fn(HikCamera& camera, ThreadSafeQueue<FramePacket>& frame_q) {
    while (app_running) {
        cv::Mat src = camera.read();
        if (src.empty()) continue;
        FramePacket pkt;
        pkt.frame = src.clone();
        pkt.timestamp = std::chrono::steady_clock::now();
        frame_q.push(std::move(pkt));
    }
}

void detect_thread_fn(Detect& detect,
                      ThreadSafeQueue<FramePacket>& frame_q,
                      ThreadSafeQueue<ResultPacket>& result_q) {
    while (app_running) {
        FramePacket pkt;
        if (!frame_q.pop(pkt, 5)) continue;
        auto lights = detect.detect(pkt.frame);
        ResultPacket res;
        res.lights = std::move(lights);
        res.annotated_frame = pkt.frame;
        res.mask = detect.getMask();
        res.timestamp = pkt.timestamp;
        result_q.push(std::move(res));
    }
}

int main() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto config = YAML::LoadFile("config.yaml");
    Detect detect(config["detect"]);
    HikCamera camera(config["hik"]);
    SerialDriver serial(config["serial"]);
    KalmanTracker tracker(config["tracker"]);
    Recorder recorder(config["recorder"]);

    bool enable_gui = config["pipeline"]["enable_gui"].as<bool>(false);
    int frame_q_size = config["pipeline"]["frame_queue_size"].as<int>(2);
    int result_q_size = config["pipeline"]["result_queue_size"].as<int>(1);

    ThreadSafeQueue<FramePacket> frame_q(frame_q_size);
    ThreadSafeQueue<ResultPacket> result_q(result_q_size);

    cv::namedWindow("frame", cv::WINDOW_NORMAL);

    camera.init();
    camera.start();
    recorder.init();

    serial.start([&](const std::vector<uint8_t>& data) {
        if (data.size() >= 2 && data[0] == 0xAA) {
            switch (data[1]) {
                case 0: tracker.setTargetMode("fixed"); break;
                case 1: tracker.setTargetMode("fixed"); break;
                case 2: tracker.setTargetMode("random_fixed"); break;
                case 3: tracker.setTargetMode("random_moving"); break;
                case 4: tracker.setTargetMode("end_moving"); break;
            }
        }
    });

    std::thread cam_thread(camera_thread_fn, std::ref(camera), std::ref(frame_q));
    std::thread det_thread(detect_thread_fn, std::ref(detect),
                           std::ref(frame_q), std::ref(result_q));

    auto last_ts = std::chrono::steady_clock::now();
    int last_w = 640;
    int frame_count = 0;
    float fps = 0;
    auto fps_timer = std::chrono::steady_clock::now();

    while (app_running) {
        ResultPacket res;
        if (!result_q.pop(res, 10)) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_ts).count();
            last_ts = now;

            tracker.update({}, dt, 0);

            GreenLight best;
            if (tracker.pickLowest(best)) {
                float cx_norm = best.center.x / static_cast<float>(last_w) * 2.0f - 1.0f;
                SendDartCmdData send;
                send.diff_center_norm = cx_norm;
                send.sum = 0;
                for (uint8_t i = 0; i < 4; ++i)
                    send.sum += reinterpret_cast<uint8_t*>(&send)[i];
                serial.write(to_vector(send));
            } else {
                SendDartCmdData send{};
                serial.write(to_vector(send));
            }
            handleKey(cv::waitKey(1) & 0xFF, detect);
            continue;
        }

        // FPS
        frame_count++;
        auto now2 = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now2 - fps_timer).count();
        if (elapsed >= 2.0f) {
            fps = frame_count / elapsed;
            frame_count = 0;
            fps_timer = now2;
        }

        auto now = res.timestamp;
        float dt = std::chrono::duration<float>(now - last_ts).count();
        last_ts = now;
        if (dt <= 0.0f || dt > 1.0f) dt = 0.005f;

        int w = res.annotated_frame.size().width;
        int h = res.annotated_frame.size().height;
        last_w = w;
        tracker.update(res.lights, dt, w);

        GreenLight best;
        float cx_norm = 0, cy_norm = 0;
        bool tracking = tracker.pickLowest(best);
        if (tracking) {
            cx_norm = best.center.x / static_cast<float>(w) * 2.0f - 1.0f;
            cy_norm = best.center.y / static_cast<float>(h) * 2.0f - 1.0f;
            SendDartCmdData send;
            send.diff_center_norm = cx_norm;
            send.sum = 0;
            for (uint8_t i = 0; i < 4; ++i)
                send.sum += reinterpret_cast<uint8_t*>(&send)[i];
            serial.write(to_vector(send));
            cv::rectangle(res.annotated_frame, best.bbox, cv::Scalar(0, 0, 255), 2);
            cv::circle(res.annotated_frame, best.center, 4, cv::Scalar(0, 255, 255), -1);
        } else {
            SendDartCmdData send{};
            serial.write(to_vector(send));
        }

        if (enable_gui) {
            // ROI boundary
            if (detect.roiEnabled())
                cv::rectangle(res.annotated_frame, detect.getROIRect(res.annotated_frame.size()),
                              cv::Scalar(0, 255, 0), 1);

            // Track trajectory
            auto info = tracker.bestTrackInfo();
            if (info.valid && info.history.size() > 1) {
                std::vector<cv::Point> pts(info.history.begin(), info.history.end());
                for (size_t i = 1; i < pts.size(); ++i) {
                    float alpha = static_cast<float>(i) / pts.size();
                    cv::line(res.annotated_frame, pts[i - 1], pts[i],
                             cv::Scalar(0, 255 * alpha, 255 * (1 - alpha)), 1);
                }
            }

            drawStatus(res.annotated_frame, tracker.mode(), info, cx_norm, cy_norm, fps);

            if (!res.mask.empty()) cv::imshow("mask", res.mask);
            cv::imshow("frame", res.annotated_frame);
            handleKey(cv::waitKey(1) & 0xFF, detect);
        }

        if (recorder.enabled())
            recorder.push(res.annotated_frame);
    }

    recorder.stop();
    frame_q.stop();
    result_q.stop();
    cam_thread.join();
    det_thread.join();
    serial.stop();
    camera.stop();
    cv::destroyAllWindows();
    return 0;
}
