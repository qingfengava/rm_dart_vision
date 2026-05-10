#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace dart_vision {

template<typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size = 2): max_size_(max_size) {}

    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_) {
            queue_.pop();
        }
        queue_.push(std::move(item));
        cv_.notify_one();
    }

    bool pop(T& item, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            if (!running_) return false;
            cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms));
            if (queue_.empty()) return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void stop() {
        running_ = false;
        cv_.notify_all();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) queue_.pop();
    }

    bool running() const { return running_; }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_size_;
    std::atomic<bool> running_{true};
};

} // namespace dart_vision
