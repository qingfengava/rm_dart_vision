#pragma once

#include <boost/asio.hpp>
#include <iostream>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace dart_vision {
template<typename T>
inline T from_vector(const std::vector<uint8_t>& data) {
    T packet {};
    std::memcpy(&packet, data.data(), sizeof(T));
    return packet;
}

template<typename T>
inline std::vector<uint8_t> to_vector(const T& data) {
    std::vector<uint8_t> packet(sizeof(T));
    std::memcpy(packet.data(), &data, sizeof(T));
    return packet;
}

class SerialDriver {
public:
    struct Params {
        unsigned int baud_rate = 115200;
        unsigned int char_size = 8;
        boost::asio::serial_port_base::parity::type parity =
            boost::asio::serial_port_base::parity::none;
        boost::asio::serial_port_base::stop_bits::type stop_bits =
            boost::asio::serial_port_base::stop_bits::one;
        boost::asio::serial_port_base::flow_control::type flow_control =
            boost::asio::serial_port_base::flow_control::none;

        std::string device_name;
        size_t read_buf_size = 4096;

        void load(const YAML::Node& config) {
            device_name = config["device_name"].as<std::string>();
            baud_rate = config["baud_rate"].as<unsigned int>();
            char_size = config["char_size"].as<unsigned int>();
            read_buf_size = config["read_buf_size"].as<size_t>();
        }
    } params_;

    SerialDriver(const YAML::Node& config): io_(), port_(io_), running_(false) {
        params_.load(config);
        read_buf_.resize(params_.read_buf_size);

        // Boost 1.58：io_service 工作守护
        work_ = std::make_unique<boost::asio::io_service::work>(io_);
    }

    ~SerialDriver() {
        stop();
    }

    void error_handler(const boost::system::error_code& ec) {
        if (ec && ec != boost::asio::error::operation_aborted) {
            std::cout << "serial error: " << ec.message() << std::endl;
        }
    }

    void start(std::function<void(const std::vector<uint8_t>&)> cb) {
        cb_ = cb;
        running_ = true;

        io_thread_ = std::thread([this]() { run_io(); });
    }

    void stop() {
        if (!running_)
            return;

        running_ = false;

        boost::system::error_code ec;
        port_.cancel(ec);
        port_.close(ec);

        // 释放工作守护，让 io_.run() 退出
        work_.reset();
        io_.stop();

        if (io_thread_.joinable())
            io_thread_.join();

        error_handler(ec);
    }

    bool write(const std::vector<uint8_t>& data) {
        uint64_t gen = generation_;

        io_.post([this, data, gen]() mutable { // ✔ 改这里
            if (gen != generation_)
                return;

            bool idle = write_queue_.empty();
            write_queue_.push_back(data);

            if (idle) {
                do_write(gen);
            }
        });

        return true;
    }

private:
    void run_io() {
        while (running_) {
            if (!open_port()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

            uint64_t gen = ++generation_;

            clear_buffers();
            start_read(gen);

            io_.run(); // 阻塞运行
            io_.reset(); // Boost 1.58 支持 io_service::reset()

            close_port();

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    bool open_port() {
        boost::system::error_code ec;

        port_.open(params_.device_name, ec);
        if (ec) {
            error_handler(ec);
            return false;
        }

        port_.set_option(boost::asio::serial_port_base::baud_rate(params_.baud_rate), ec);
        port_.set_option(boost::asio::serial_port_base::character_size(params_.char_size), ec);
        port_.set_option(boost::asio::serial_port_base::parity(params_.parity), ec);
        port_.set_option(boost::asio::serial_port_base::stop_bits(params_.stop_bits), ec);
        port_.set_option(boost::asio::serial_port_base::flow_control(params_.flow_control), ec);

        if (ec) {
            error_handler(ec);
            return false;
        }

        std::cout << "serial open success: " << params_.device_name << std::endl;
        return true;
    }

    void close_port() {
        boost::system::error_code ec;
        port_.cancel(ec);
        port_.close(ec);

        write_queue_.clear();
        error_handler(ec);
    }

    void clear_buffers() {
        std::fill(read_buf_.begin(), read_buf_.end(), 0);
        write_queue_.clear();
    }

    void start_read(uint64_t gen) {
        port_.async_read_some(
            boost::asio::buffer(read_buf_),
            [this, gen](boost::system::error_code ec, size_t n) {
                if (gen != generation_)
                    return;

                if (ec) {
                    if (ec != boost::asio::error::operation_aborted) {
                        error_handler(ec);
                    }
                    io_.stop();
                    return;
                }

                if (n > 0) {
                    std::vector<uint8_t> buf(read_buf_.begin(), read_buf_.begin() + n);
                    cb_(buf);
                }

                start_read(gen);
            }
        );
    }

    void do_write(uint64_t gen) {
        if (write_queue_.empty())
            return;

        auto data = write_queue_.front(); // copy，保证生命周期

        boost::asio::async_write(
            port_,
            boost::asio::buffer(data),
            [this, gen, data](boost::system::error_code ec, size_t) mutable {
                if (gen != generation_)
                    return;

                if (ec) {
                    if (ec != boost::asio::error::operation_aborted) {
                        error_handler(ec);
                    }
                    io_.stop();
                    return;
                }

                write_queue_.pop_front();

                if (!write_queue_.empty()) {
                    do_write(gen);
                }
            }
        );
    }

private:
    boost::asio::io_service io_;
    boost::asio::serial_port port_;

    std::unique_ptr<boost::asio::io_service::work> work_;

    std::vector<uint8_t> read_buf_;
    std::deque<std::vector<uint8_t>> write_queue_;

    std::atomic<bool> running_;
    std::atomic<uint64_t> generation_ { 0 };

    std::function<void(const std::vector<uint8_t>&)> cb_;
    std::thread io_thread_;
};

} // namespace dart_vision