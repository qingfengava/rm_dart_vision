#pragma once
#include "Hungarian.h"
#include "extended_kalman_filter.hpp"
#include "tracker_mode.hpp"
#include "type.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <deque>
#include <memory>
#include <vector>

namespace dart_vision {

constexpr double INF_COST = 1e9;
constexpr int X_N = 6, Z_N = 4;

struct Predict {
    double dt = 0.0;
    template<typename T>
    void operator()(const T x0[X_N], T x1[X_N]) const {
        x1[0] = x0[0] + x0[2] * T(dt);
        x1[1] = x0[1] + x0[3] * T(dt);
        x1[2] = x0[2];
        x1[3] = x0[3];
        x1[4] = x0[4];
        x1[5] = x0[5];
    }
};

struct Measure {
    template<typename T>
    void operator()(const T x[X_N], T z[Z_N]) const {
        z[0] = x[0]; // z_cx = cx
        z[1] = x[1]; // z_cy = cy
        z[2] = x[4]; // z_w  = w
        z[3] = x[5]; // z_h  = h
    }
};

using EKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;
using MatrixXX = EKF::MatrixXX;
using MatrixZZ = EKF::MatrixZZ;
using MatrixX1 = EKF::MatrixX1;
using MatrixZ1 = EKF::MatrixZ1;

struct TrackInfo {
    bool valid = false;
    int id = 0;
    int hits = 0;
    int age = 0;
    int state = 0; // 0=TENTATIVE, 1=CONFIRMED, 2=LOST
    cv::Point2f center;
    std::deque<cv::Point2f> history;
};

struct Track {
    int id = 0;
    enum State { TENTATIVE, CONFIRMED, LOST } state = TENTATIVE;
    int hits = 0;
    int misses = 0;
    int age = 0;
    std::unique_ptr<EKF> ekf;
    MatrixX1 x_pred = MatrixX1::Zero();
    std::deque<cv::Point2f> history;
};

class KalmanTracker {
public:
    KalmanTracker(const YAML::Node& config) {
        stationary_.load(config["stationary"]);
        moving_.load(config["moving"]);

        confirm_frames_ = config["confirm_frames"].as<int>(3);
        max_tentative_frames_ = config["max_tentative_frames"].as<int>(5);
        gate_threshold_ = config["gate_threshold"].as<double>(9.49);
        transition_alpha_ = config["transition_alpha"].as<double>(0.3);
        v_max_ = config["v_max"].as<double>(200.0);
        iteration_num_ = config["iteration_num"].as<int>(2);

        std::string mode_str = config["initial_mode"].as<std::string>("stationary");
        target_mode_ = modeFromString(mode_str);
        current_mode_ = target_mode_;

        if (config["auto_reset"]) {
            jump_distance_px_ = config["auto_reset"]["jump_distance_px"].as<double>(50.0);
            fast_confirm_frames_ = config["auto_reset"]["fast_confirm_frames"].as<int>(1);
            fast_confirm_duration_ = config["auto_reset"]["fast_confirm_duration"].as<int>(10);
        }

        // target_mode preset from config
        last_preset_ = config["target_mode"].as<std::string>("fixed");
        applyPreset(last_preset_);
        const auto& p = paramsFor(current_mode_);
        current_q_v_ = p.q_v;
        current_q_size_ = p.q_size;
        current_r_pos_ = p.r_pos;
        current_r_size_ = p.r_size;
    }

    void setMode(TrackerMode mode) {
        if (!enable_auto_maneuver_) target_mode_ = mode;
    }

    void setTargetMode(const std::string& preset) {
        if (last_preset_ == preset) return;
        last_preset_ = preset;
        applyPreset(preset);
        tracks_.clear();
        next_id_ = 0;
        frames_since_reset_ = -1;
        std::cout << "[Tracker] 模式切换: " << preset << std::endl;
    }

    TrackerMode mode() const { return target_mode_; }

    void update(const std::vector<GreenLight>& detections, float dt, int /*frame_width*/) {
        current_dt_ = dt;
        smoothParams();

        predictAll();

        if (!detections.empty()) {
            associateAndUpdate(detections);
        } else {
            for (auto& t : tracks_)
                t.misses++;
        }

        if (frames_since_reset_ >= 0)
            frames_since_reset_++;

        manageLifecycle();
    }

    bool bestTrack(GreenLight& out) const {
        const Track* best = nullptr;
        float best_score = -1.0f;
        for (const auto& t : tracks_) {
            if (t.state != Track::CONFIRMED) continue;
            float score = static_cast<float>(t.hits) / static_cast<float>(t.age + 1);
            if (score > best_score) {
                best_score = score;
                best = &t;
            }
        }
        if (!best) return false;
        const MatrixX1& x = best->x_pred;
        out.center = cv::Point2f(static_cast<float>(x[0]), static_cast<float>(x[1]));
        out.bbox = cv::Rect2f(
            static_cast<float>(x[0] - x[4] * 0.5f),
            static_cast<float>(x[1] - x[5] * 0.5f),
            static_cast<float>(x[4]),
            static_cast<float>(x[5])
        );
        out.score = best_score;
        return true;
    }

    bool pickLowest(GreenLight& out) const {
        const Track* lowest = nullptr;
        float max_y = -1;
        for (const auto& t : tracks_) {
            if (t.state != Track::CONFIRMED) continue;
            if (t.x_pred[1] > max_y) { max_y = t.x_pred[1]; lowest = &t; }
        }
        if (!lowest) return false;
        const MatrixX1& x = lowest->x_pred;
        out.center = cv::Point2f(static_cast<float>(x[0]), static_cast<float>(x[1]));
        out.bbox = cv::Rect2f(
            static_cast<float>(x[0] - x[4] * 0.5f),
            static_cast<float>(x[1] - x[5] * 0.5f),
            static_cast<float>(x[4]),
            static_cast<float>(x[5])
        );
        out.score = max_y;
        return true;
    }

    TrackInfo bestTrackInfo() const {
        TrackInfo info;
        const Track* best = nullptr;
        float best_score = -1.0f;
        for (const auto& t : tracks_) {
            if (t.state != Track::CONFIRMED) continue;
            float score = static_cast<float>(t.hits) / static_cast<float>(t.age + 1);
            if (score > best_score) { best_score = score; best = &t; }
        }
        if (!best) return info;
        info.valid = true;
        info.id = best->id;
        info.hits = best->hits;
        info.age = best->age;
        info.state = 1; // CONFIRMED
        info.center = cv::Point2f(best->x_pred[0], best->x_pred[1]);
        info.history = best->history;
        return info;
    }

    const std::vector<Track>& tracks() const { return tracks_; }

private:
    void applyPreset(const std::string& preset) {
        enable_auto_reset_ = false;
        enable_auto_maneuver_ = false;
        initial_mode_ = TrackerMode::STATIONARY;
        if (preset == "random_fixed") {
            enable_auto_reset_ = true;
        } else if (preset == "random_moving" || preset == "end_moving") {
            enable_auto_reset_ = true;
            enable_auto_maneuver_ = true;
            initial_mode_ = TrackerMode::MOVING;
        }
        target_mode_ = initial_mode_;
        current_mode_ = initial_mode_;
    }

    const ModeParams& paramsFor(TrackerMode m) const {
        return (m == TrackerMode::MOVING) ? moving_ : stationary_;
    }

    void smoothParams() {
        const auto& p = paramsFor(target_mode_);
        float alpha = transition_alpha_;
        current_q_v_ = current_q_v_ * (1.0f - alpha) + p.q_v * alpha;
        current_q_size_ = current_q_size_ * (1.0f - alpha) + p.q_size * alpha;
        current_r_pos_ = current_r_pos_ * (1.0f - alpha) + p.r_pos * alpha;
        current_r_size_ = current_r_size_ * (1.0f - alpha) + p.r_size * alpha;

        if (std::abs(current_q_v_ - p.q_v) < 0.01f) {
            current_q_v_ = p.q_v;
            current_q_size_ = p.q_size;
            current_r_pos_ = p.r_pos;
            current_r_size_ = p.r_size;
            current_mode_ = target_mode_;
        }
    }

    void predictAll() {
        Predict pred{current_dt_};
        for (auto& t : tracks_) {
            t.ekf->setPredictFunc(pred);
            t.x_pred = t.ekf->predict();
            t.age++;
        }
    }

    void associateAndUpdate(const std::vector<GreenLight>& detections) {
        size_t m = tracks_.size();
        size_t n = detections.size();

        if (m == 0) {
            for (size_t j = 0; j < n; ++j)
                createTrack(detections[j]);
            return;
        }

        H_mat_ << 1, 0, 0, 0, 0, 0,
                  0, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 1;

        MatrixZZ R = buildR();

        // Build cost matrix
        auto cost = std::vector<std::vector<double>>(m, std::vector<double>(n, 0.0));
        for (size_t i = 0; i < m; ++i) {
            const MatrixXX& P = tracks_[i].ekf->getPriorCovariance();
            MatrixZZ S = H_mat_ * P * H_mat_.transpose() + R;
            MatrixZZ S_inv = S.inverse();
            MatrixX1& xp = tracks_[i].x_pred;

            for (size_t j = 0; j < n; ++j) {
                MatrixZ1 z;
                z << detections[j].center.x, detections[j].center.y,
                     detections[j].bbox.width, detections[j].bbox.height;
                MatrixZ1 z_pred;
                z_pred << xp[0], xp[1], xp[4], xp[5];
                MatrixZ1 innov = z - z_pred;
                double mahal = innov.transpose() * S_inv * innov;
                cost[i][j] = (mahal < gate_threshold_) ? mahal : INF_COST;
            }
        }

        // Hungarian
        HungarianAlgorithm ha;
        std::vector<int> assignment(m, -1);
        ha.Solve(cost, assignment);

        // Update matched, mark misses
        for (size_t i = 0; i < m; ++i) {
            int j = assignment[i];
            if (j >= 0 && static_cast<size_t>(j) < n && cost[i][j] < INF_COST) {
                MatrixZ1 z;
                z << detections[j].center.x, detections[j].center.y,
                     detections[j].bbox.width, detections[j].bbox.height;
                tracks_[i].x_pred = tracks_[i].ekf->update(z);
                tracks_[i].hits++;
                tracks_[i].misses = 0;
                tracks_[i].history.push_back(cv::Point2f(tracks_[i].x_pred[0], tracks_[i].x_pred[1]));
                if (tracks_[i].history.size() > 50) tracks_[i].history.pop_front();
            } else {
                tracks_[i].misses++;
                tracks_[i].hits = 0;
            }
        }

        // Create new tracks for unmatched detections
        for (size_t j = 0; j < n; ++j) {
            bool matched = false;
            for (int a : assignment) {
                if (a == static_cast<int>(j)) { matched = true; break; }
            }
            if (!matched) createTrack(detections[j]);
        }
    }

    void createTrack(const GreenLight& det) {
        Track t;
        t.id = next_id_++;
        t.x_pred << det.center.x, det.center.y, 0.0, 0.0,
                    det.bbox.width, det.bbox.height;

        Predict pred{current_dt_};
        Measure meas{};

        auto uq = [this]() -> MatrixXX {
            float dt = current_dt_;
            float qv = current_q_v_, qs = current_q_size_;
            MatrixXX Q = MatrixXX::Zero();
            Q(0, 0) = qv * dt * dt * dt / 3.0;
            Q(1, 1) = qv * dt * dt * dt / 3.0;
            Q(2, 2) = qv * dt;
            Q(3, 3) = qv * dt;
            Q(4, 4) = qs * dt * dt * dt / 3.0;
            Q(5, 5) = qs * dt * dt * dt / 3.0;
            Q(0, 2) = Q(2, 0) = qv * dt * dt / 2.0;
            Q(1, 3) = Q(3, 1) = qv * dt * dt / 2.0;
            return Q;
        };

        auto ur = [this](const MatrixZ1&) -> MatrixZZ {
            return buildR();
        };

        MatrixXX P0 = MatrixXX::Identity();
        P0(0, 0) = current_r_pos_;
        P0(1, 1) = current_r_pos_;
        P0(2, 2) = v_max_ * v_max_;
        P0(3, 3) = v_max_ * v_max_;
        P0(4, 4) = current_r_size_;
        P0(5, 5) = current_r_size_;

        t.ekf = std::make_unique<EKF>(pred, meas, uq, ur, P0);
        t.ekf->setState(t.x_pred);
        t.ekf->setIterationNum(iteration_num_);
        tracks_.push_back(std::move(t));
    }

    void manageLifecycle() {
        cv::Point2f lost_pos(-1, -1);

        auto it = tracks_.begin();
        while (it != tracks_.end()) {
            // State transitions
            if (it->state == Track::TENTATIVE) {
                int cf = (enable_auto_reset_ && frames_since_reset_ >= 0 &&
                          frames_since_reset_ < fast_confirm_duration_)
                             ? fast_confirm_frames_ : confirm_frames_;
                if (it->hits >= cf)
                    it->state = Track::CONFIRMED;
                else if (it->age > max_tentative_frames_ || it->misses > 0) {
                    it = tracks_.erase(it);
                    continue;
                }
            } else if (it->state == Track::CONFIRMED) {
                int max_lost = (target_mode_ == TrackerMode::MOVING)
                                   ? moving_.max_lost_frames
                                   : stationary_.max_lost_frames;
                if (it->misses > max_lost) {
                    it = tracks_.erase(it);
                    continue;
                } else if (it->misses > 0) {
                    lost_pos = cv::Point2f(it->x_pred[0], it->x_pred[1]);
                    it->state = Track::LOST;
                }
            } else { // LOST
                int max_lost = (target_mode_ == TrackerMode::MOVING)
                                   ? moving_.max_lost_frames
                                   : stationary_.max_lost_frames;
                if (it->hits > 0)
                    it->state = Track::CONFIRMED;
                else if (it->misses > max_lost) {
                    it = tracks_.erase(it);
                    continue;
                }
            }
            ++it;
        }

        if (enable_auto_reset_ && lost_pos.x >= 0) {
            for (auto& t : tracks_) {
                if (t.state != Track::TENTATIVE) continue;
                float dx = t.x_pred[0] - lost_pos.x;
                float dy = t.x_pred[1] - lost_pos.y;
                if (dx * dx + dy * dy > jump_distance_px_ * jump_distance_px_) {
                    tracks_.clear();
                    next_id_ = 0;
                    frames_since_reset_ = 0;
                    target_mode_ = TrackerMode::STATIONARY;
                    std::cout << "[Tracker] 目标跳变, 自动重置 (快速确认 "
                              << fast_confirm_frames_ << "帧, 持续"
                              << fast_confirm_duration_ << "帧)" << std::endl;
                    return;
                }
            }
        }

        if (enable_auto_maneuver_) {
            const Track* best = nullptr;
            float best_score = -1.0f;
            for (const auto& t : tracks_) {
                if (t.state != Track::CONFIRMED) continue;
                float s = static_cast<float>(t.hits) / static_cast<float>(t.age + 1);
                if (s > best_score) { best_score = s; best = &t; }
            }
            if (best && best->history.size() >= static_cast<size_t>(stationarity_window_)) {
                float sum_x = 0, sum_y = 0, sum_x2 = 0, sum_y2 = 0;
                size_t n = std::min(best->history.size(), static_cast<size_t>(stationarity_window_));
                auto it = best->history.end();
                for (size_t i = 0; i < n; ++i) {
                    --it;
                    sum_x += it->x; sum_y += it->y;
                    sum_x2 += it->x * it->x; sum_y2 += it->y * it->y;
                }
                float var_x = sum_x2 / n - (sum_x / n) * (sum_x / n);
                float var_y = sum_y2 / n - (sum_y / n) * (sum_y / n);
                float std = std::sqrt(std::max(0.0f, var_x + var_y));
                if (std < stationarity_threshold_)
                    target_mode_ = TrackerMode::STATIONARY;
                else
                    target_mode_ = TrackerMode::MOVING;
            }
        }
    }

    MatrixZZ buildR() const {
        MatrixZZ R = MatrixZZ::Zero();
        R(0, 0) = current_r_pos_;
        R(1, 1) = current_r_pos_;
        R(2, 2) = current_r_size_;
        R(3, 3) = current_r_size_;
        return R;
    }

    // Fixed H (observation picks cx,cy,w,h from state)
    Eigen::Matrix<double, Z_N, X_N> H_mat_ = []() {
        Eigen::Matrix<double, Z_N, X_N> H = Eigen::Matrix<double, Z_N, X_N>::Zero();
        H(0, 0) = 1;
        H(1, 1) = 1;
        H(2, 4) = 1;
        H(3, 5) = 1;
        return H;
    }();

    ModeParams stationary_, moving_;
    TrackerMode current_mode_ = TrackerMode::STATIONARY;
    TrackerMode target_mode_ = TrackerMode::STATIONARY;
    TrackerMode initial_mode_ = TrackerMode::STATIONARY;

    float current_q_v_ = 0.5f;
    float current_q_size_ = 1.0f;
    float current_r_pos_ = 4.0f;
    float current_r_size_ = 9.0f;
    float current_dt_ = 0.005f;

    int confirm_frames_ = 3;
    int max_tentative_frames_ = 5;
    int iteration_num_ = 2;
    double gate_threshold_ = 9.49;
    double transition_alpha_ = 0.3;
    double v_max_ = 200.0;
    int next_id_ = 0;
    bool enable_auto_reset_ = false;
    double jump_distance_px_ = 50.0;
    int fast_confirm_frames_ = 1;
    int fast_confirm_duration_ = 10;
    int frames_since_reset_ = -1;
    bool enable_auto_maneuver_ = false;
    int stationarity_window_ = 20;
    double stationarity_threshold_ = 3.0;
    std::string last_preset_ = "fixed";

    std::vector<Track> tracks_;
};

} // namespace dart_vision
