#include "detect.hpp"
#include "hik.hpp"
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
    std::chrono::steady_clock::time_point timestamp;
};

std::atomic<bool> app_running{true};

void signal_handler(int) { app_running = false; }

void handleKey(int key, Detect& detect) {
    if (key == 's') detect.printHSV();
    else if (key == 'r') detect.resetHSV();
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

    bool enable_gui = config["pipeline"]["enable_gui"].as<bool>(false);
    int frame_q_size = config["pipeline"]["frame_queue_size"].as<int>(2);
    int result_q_size = config["pipeline"]["result_queue_size"].as<int>(1);

    ThreadSafeQueue<FramePacket> frame_q(frame_q_size);
    ThreadSafeQueue<ResultPacket> result_q(result_q_size);

    cv::namedWindow("frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("frame", 640, 480);

    camera.init();
    camera.start();

    serial.start([&](const std::vector<uint8_t>& data) {
        if (data.size() >= 3 && data[0] == 0xA5 && data[2] == 0x5A) {
            switch (data[1]) {
                case 0x01: tracker.setMode(TrackerMode::STATIONARY); break;
                case 0x02: tracker.setMode(TrackerMode::MOVING); break;
            }
        }
    });

    std::thread cam_thread(camera_thread_fn, std::ref(camera), std::ref(frame_q));
    std::thread det_thread(detect_thread_fn, std::ref(detect),
                           std::ref(frame_q), std::ref(result_q));

    auto last_ts = std::chrono::steady_clock::now();

    while (app_running) {
        ResultPacket res;
        if (!result_q.pop(res, 10)) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_ts).count();
            last_ts = now;

            tracker.update({}, dt, 0);

            GreenLight best;
            if (tracker.bestTrack(best)) {
                float cx_norm = best.center.x / 640.0f * 2.0f - 1.0f;
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

        auto now = res.timestamp;
        float dt = std::chrono::duration<float>(now - last_ts).count();
        last_ts = now;
        if (dt <= 0.0f || dt > 1.0f) dt = 0.005f; // clamp

        int w = res.annotated_frame.size().width;
        tracker.update(res.lights, dt, w);

        GreenLight best;
        if (tracker.bestTrack(best)) {
            float cx_norm = best.center.x / static_cast<float>(w) * 2.0f - 1.0f;
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
            cv::imshow("frame", res.annotated_frame);
            handleKey(cv::waitKey(1) & 0xFF, detect);
        }
    }

    frame_q.stop();
    result_q.stop();
    cam_thread.join();
    det_thread.join();
    serial.stop();
    camera.stop();
    cv::destroyAllWindows();
    return 0;
}
