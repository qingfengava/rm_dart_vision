#pragma once
#include <string>
#include <yaml-cpp/yaml.h>

namespace dart_vision {

enum class TrackerMode { OUTPOST, BASE_STATIONARY, BASE_MOVING };

struct ModeParams {
    float q_v;
    float q_size;
    float r_pos;
    float r_size;
    int max_lost_frames;

    void load(const YAML::Node& node) {
        q_v = node["q_v"].as<float>();
        q_size = node["q_size"].as<float>();
        r_pos = node["r_pos"].as<float>();
        r_size = node["r_size"].as<float>();
        max_lost_frames = node["max_lost_frames"].as<int>();
    }
};

TrackerMode modeFromString(const std::string& s) {
    if (s == "outpost") return TrackerMode::OUTPOST;
    if (s == "base_moving") return TrackerMode::BASE_MOVING;
    return TrackerMode::BASE_STATIONARY;
}

} // namespace dart_vision
