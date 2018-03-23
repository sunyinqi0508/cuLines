#pragma once

#include <chrono>

class Stopwatch {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_, end_;
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_ = std::chrono::high_resolution_clock::now();
    }

    double elapsedSeconds() const {
        std::chrono::duration<double> dur = end_ - start_;
        return dur.count();
    }
};