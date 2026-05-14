#pragma once
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace dart_vision {

class RunLogger {
public:
    explicit RunLogger(const std::string& dir): dir_(dir) {
        start_ = std::chrono::steady_clock::now();
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        set("start", oss.str());
    }

    void set(const std::string& key, const std::string& val) { map_[key] = val; }
    void set(const std::string& key, int val) { map_[key] = std::to_string(val); }
    void set(const std::string& key, float val, int prec = 1) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(prec) << val;
        map_[key] = oss.str();
    }
    void inc(const std::string& key) {  // thread-safe enough for single writer
        counters_[key]++;
    }

    void save() {
        if (saved_) return;
        saved_ = true;

        auto end = std::chrono::steady_clock::now();
        float dur = std::chrono::duration<float>(end - start_).count();
        set("duration_s", dur);

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream end_oss;
        end_oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        set("end", end_oss.str());

        for (auto& p : counters_)
            map_[p.first] = std::to_string(p.second);

        mkdir(dir_.c_str(), 0755);

        std::ostringstream name;
        auto tt = std::time(nullptr);
        auto ttm = *std::localtime(&tt);
        name << dir_ << "/" << std::put_time(&ttm, "%Y%m%d_%H%M%S") << ".log";

        std::ofstream f(name.str());
        f << "=== dart_vision Log ===\n";
        for (const auto& p : map_)
            f << p.first << ": " << p.second << "\n";
    }

private:
    std::string dir_;
    std::chrono::steady_clock::time_point start_;
    std::unordered_map<std::string, std::string> map_;
    std::unordered_map<std::string, int> counters_;
    bool saved_ = false;
};

} // namespace dart_vision
