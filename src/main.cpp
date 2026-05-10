#include "detect.hpp"
#include "hik.hpp"
#include "serial_driver.hpp"
#include "type.hpp"
#include <chrono>
#include <cstdint>
#include <opencv2/highgui.hpp>
#include <thread>
using namespace dart_vision;
struct SendDartCmdData {
    float diff_center_norm = 0;
    uint32_t sum;
} __attribute__((packed));
int main() {
    auto config = YAML::LoadFile("config.yaml");
    Detect detect(config["detect"]);
    HikCamera camera(config["hik"]);
    SerialDriver serial(config["serial"]);
    camera.init();
    camera.start();
    cv::Mat src;
    cv::namedWindow("frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("frame", 640, 480);
    serial.start([&](const std::vector<uint8_t>& data) {

    });
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        src = camera.read();
        if (src.empty()) {
            continue;
        }
        auto lights = detect.detect(src);
        GreenLight best;
        float max_score = -1;
        for (const auto& light: lights) {
            if (light.score > max_score) {
                max_score = light.score;
                best = light;
            }
        }
        if (max_score > -1) {
            auto cx_norm = best.center.x / src.size().width * 2.0 - 1.0;
            SendDartCmdData send;
            send.diff_center_norm = cx_norm;
            send.sum = 0;
            for (uint8_t i = 0; i < 4; ++i) {
                send.sum += reinterpret_cast<uint8_t*>(&send)[i];
            }
            serial.write(to_vector(send));
            cv::rectangle(src, best.bbox, cv::Scalar(0, 0, 255), 2);
        } else {
            SendDartCmdData send;
            send.diff_center_norm = 0;
            send.sum = 0;
            for (uint8_t i = 0; i < 4; ++i) {
                send.sum += reinterpret_cast<uint8_t*>(&send)[i];
            }
            serial.write(to_vector(send));
        }
        cv::imshow("frame", src);
        cv::waitKey(1);
    }
    return 0;
}