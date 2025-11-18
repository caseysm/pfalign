#pragma once

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace pfalign::perf {

class PerfTimer {
    using clock = std::chrono::steady_clock;

public:
    explicit PerfTimer(std::string label) : enabled_(should_report()), label_(std::move(label)) {
        if (enabled_) {
            start_ = clock::now();
        }
    }

    ~PerfTimer() {
        if (!enabled_) {
            return;
        }
        const double elapsed = std::chrono::duration<double>(clock::now() - start_).count();
        std::cout << "[PF_PERF] " << label_ << " " << std::fixed << std::setprecision(3) << elapsed
                  << "s" << std::endl;
    }

private:
    static bool should_report() {
        const char* env = std::getenv("PF_PERF_REPORT");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }

    bool enabled_ = false;
    std::string label_;
    clock::time_point start_{};
};

}  // namespace pfalign::perf
