#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace pfalign {

/**
 * Track CPU utilization for a phase
 * Compares CPU time (sum of all threads) vs wall clock time
 */
struct CPUUtilizationRecord {
    std::string name;
    double cpu_time_s = 0.0;   // Total CPU time across all threads
    double wall_time_s = 0.0;  // Wall clock time
    int expected_threads = 1;  // Number of threads we expected to use

    // Metrics
    double utilization_percent() const {
        return wall_time_s > 0.0 ? (cpu_time_s / wall_time_s) * 100.0 : 0.0;
    }

    double effective_cores() const {
        return wall_time_s > 0.0 ? cpu_time_s / wall_time_s : 0.0;
    }

    double efficiency_percent() const {
        return expected_threads > 0 ? (effective_cores() / expected_threads) * 100.0 : 0.0;
    }
};

/**
 * Global CPU utilization tracking store
 */
class CPUUtilizationData {
public:
    static CPUUtilizationData& instance() {
        static CPUUtilizationData instance;
        return instance;
    }

    void record(const std::string& name, double cpu_time_s, double wall_time_s,
                int expected_threads = 1) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto& rec = records_[name];
        rec.name = name;
        rec.cpu_time_s += cpu_time_s;
        rec.wall_time_s += wall_time_s;
        rec.expected_threads = expected_threads;
    }

    // IMPORTANT: This method is NOT thread-safe with concurrent record() calls.
    // Only call after all tracking has completed (e.g., at program end).
    const std::unordered_map<std::string, CPUUtilizationRecord>& records() const {
        return records_;
    }

    // Thread-safe snapshot for reading while tracking is active
    std::unordered_map<std::string, CPUUtilizationRecord> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return records_;  // Returns a copy
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.clear();
    }

    // Thread-safe: takes snapshot under lock before printing
    void print_report(std::ostream& os = std::cout) const {
        auto records_snapshot = snapshot();

        if (records_snapshot.empty()) {
            os << "No CPU utilization data recorded.\n";
            return;
        }

        os << "\n========================================\n";
        os << "  CPU Utilization Report\n";
        os << "========================================\n\n";

        os << std::left << std::setw(30) << "Phase" << std::right << std::setw(12) << "CPU Time (s)"
           << std::setw(12) << "Wall Time (s)" << std::setw(12) << "Util %" << std::setw(15)
           << "Eff Cores" << std::setw(10) << "Expected" << std::setw(12) << "Efficiency %" << "\n";
        os << std::string(103, '-') << "\n";

        for (const auto& [name, rec] : records_snapshot) {
            os << std::left << std::setw(30) << name << std::right << std::fixed
               << std::setprecision(2) << std::setw(12) << rec.cpu_time_s << std::setw(12)
               << rec.wall_time_s << std::setw(12) << rec.utilization_percent() << std::setw(15)
               << rec.effective_cores() << std::setw(10) << rec.expected_threads << std::setw(12)
               << rec.efficiency_percent() << "\n";
        }
    }

private:
    CPUUtilizationData() = default;

    std::unordered_map<std::string, CPUUtilizationRecord> records_;
    mutable std::mutex mutex_;
};

/**
 * RAII CPU utilization tracker
 * Measures both CPU time and wall time for a phase
 *
 * NOTE: This measures thread CPU time using thread-specific clock.
 * For accurate multi-threaded measurements, each thread should create
 * its own tracker and results should be aggregated.
 */
class CPUUtilizationTracker {
public:
    explicit CPUUtilizationTracker(const std::string& name, int expected_threads = 1)
        : name_(name),
          expected_threads_(expected_threads),
          wall_start_(std::chrono::high_resolution_clock::now()) {
#ifdef __linux__
        // Get thread CPU time (Linux-specific)
        struct timespec ts;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
        cpu_start_ns_ = ts.tv_sec * 1000000000LL + ts.tv_nsec;
#else
        // Fallback: use wall time as approximation
        cpu_start_ns_ =
            std::chrono::duration_cast<std::chrono::nanoseconds>(wall_start_.time_since_epoch())
                .count();
#endif
    }

    ~CPUUtilizationTracker() {
        stop();
    }

    void stop() {
        if (stopped_)
            return;
        stopped_ = true;

        auto wall_end = std::chrono::high_resolution_clock::now();

#ifdef __linux__
        struct timespec ts;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
        int64_t cpu_end_ns = ts.tv_sec * 1000000000LL + ts.tv_nsec;
#else
        int64_t cpu_end_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end.time_since_epoch())
                .count();
#endif

        double cpu_time_s = (cpu_end_ns - cpu_start_ns_) / 1e9;
        double wall_time_s = std::chrono::duration<double>(wall_end - wall_start_).count();

        CPUUtilizationData::instance().record(name_, cpu_time_s, wall_time_s, expected_threads_);
    }

    // Get current utilization without stopping
    double current_utilization() const {
        auto wall_now = std::chrono::high_resolution_clock::now();

#ifdef __linux__
        struct timespec ts;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
        int64_t cpu_now_ns = ts.tv_sec * 1000000000LL + ts.tv_nsec;
#else
        int64_t cpu_now_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(wall_now.time_since_epoch())
                .count();
#endif

        double cpu_time_s = (cpu_now_ns - cpu_start_ns_) / 1e9;
        double wall_time_s = std::chrono::duration<double>(wall_now - wall_start_).count();

        return wall_time_s > 0.0 ? (cpu_time_s / wall_time_s) * 100.0 : 0.0;
    }

    // Non-copyable, non-movable
    CPUUtilizationTracker(const CPUUtilizationTracker&) = delete;
    CPUUtilizationTracker& operator=(const CPUUtilizationTracker&) = delete;
    CPUUtilizationTracker(CPUUtilizationTracker&&) = delete;
    CPUUtilizationTracker& operator=(CPUUtilizationTracker&&) = delete;

private:
    std::string name_;
    int expected_threads_;
    std::chrono::high_resolution_clock::time_point wall_start_;
    int64_t cpu_start_ns_;
    bool stopped_ = false;
};

/**
 * Aggregate CPU utilization across multiple threads
 *
 * Usage:
 *   MultiThreadCPUTracker tracker("parallel_phase", num_threads);
 *   #pragma omp parallel
 *   {
 *       tracker.record_thread_time(cpu_time_s);
 *   }
 *   tracker.finalize();  // Records to CPUUtilizationData
 */
class MultiThreadCPUTracker {
public:
    explicit MultiThreadCPUTracker(const std::string& name, int expected_threads)
        : name_(name),
          expected_threads_(expected_threads),
          wall_start_(std::chrono::high_resolution_clock::now()) {
    }

    ~MultiThreadCPUTracker() {
        finalize();
    }

    void record_thread_time(double cpu_time_s) {
        std::lock_guard<std::mutex> lock(mutex_);
        total_cpu_time_s_ += cpu_time_s;
    }

    void finalize() {
        if (finalized_)
            return;
        finalized_ = true;

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_time_s = std::chrono::duration<double>(wall_end - wall_start_).count();

        CPUUtilizationData::instance().record(name_, total_cpu_time_s_, wall_time_s,
                                              expected_threads_);
    }

private:
    std::string name_;
    int expected_threads_;
    std::chrono::high_resolution_clock::time_point wall_start_;
    double total_cpu_time_s_ = 0.0;
    bool finalized_ = false;
    std::mutex mutex_;
};

// Convenience function to print CPU utilization report
inline void print_cpu_utilization_report(std::ostream& os = std::cout) {
    CPUUtilizationData::instance().print_report(os);
}

}  // namespace pfalign
