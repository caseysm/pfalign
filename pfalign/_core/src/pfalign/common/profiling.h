#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <limits>
#include <ctime>
#include <mutex>

namespace pfalign {
namespace profiling {

#ifdef ENABLE_PROFILING

/**
 * Timing record for a single profiled section.
 * Tracks count, total time, min, max, and average.
 */
struct TimingRecord {
    size_t count = 0;
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;

    double avg_ms() const {
        return (count > 0) ? (total_ms / count) : 0.0;
    }

    void add_sample(double elapsed_ms) {
        count++;
        total_ms += elapsed_ms;
        min_ms = std::min(min_ms, elapsed_ms);
        max_ms = std::max(max_ms, elapsed_ms);
    }
};

/**
 * Global profiler singleton.
 * Collects timing data from all instrumented sections.
 *
 * Thread safety: NOT thread-safe in current implementation.
 * Each thread should have its own profiler or use external synchronization.
 */
class Profiler {
public:
    /**
     * Get singleton instance.
     */
    static Profiler& instance() {
        static Profiler profiler;
        return profiler;
    }

    /**
     * Start timing a section.
     * @param section_name Name of the section (must be string literal or stable pointer)
     */
    void start(const char* section_name) {
        active_stack_.push_back({section_name, std::chrono::high_resolution_clock::now()});
    }

    /**
     * Stop timing a section and record the elapsed time.
     * @param section_name Name of the section (must match start() call)
     */
    void stop(const char* section_name) {
        auto end = std::chrono::high_resolution_clock::now();

        if (active_stack_.empty()) {
            std::cerr << "Warning: Profiler::stop() called for unknown section: " << section_name
                      << "\n";
            return;
        }

        const auto& entry = active_stack_.back();
        if (entry.name != section_name) {
            std::cerr << "Warning: Profiler::stop() mismatch (expected " << entry.name << ", got "
                      << section_name << ")\n";
            return;
        }

        auto elapsed = std::chrono::duration<double, std::milli>(end - entry.start).count();
        active_stack_.pop_back();

        std::lock_guard<std::mutex> lock(mutex_);
        records_[section_name].add_sample(elapsed);
    }

    /**
     * Reset all timing data.
     */
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.clear();
        active_stack_.clear();
    }

    /**
     * Get all timing records.
     */
    const std::map<std::string, TimingRecord>& get_records() const {
        return records_;
    }

    std::map<std::string, TimingRecord> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return records_;
    }

    /**
     * Print human-readable report to output stream.
     * @param os Output stream (default: std::cout)
     */
    void print_report(std::ostream& os = std::cout) const {
        os << "========================================\n";
        os << "  PROFILING REPORT\n";
        os << "========================================\n";

        auto records_copy = snapshot();

        if (records_copy.empty()) {
            os << "No profiling data collected.\n";
            os << "========================================\n";
            return;
        }

        // Calculate total time
        double total_time_ms = 0.0;
        for (const auto& pair : records_copy) {
            total_time_ms += pair.second.total_ms;
        }

        // Sort by total time (descending)
        std::vector<std::pair<std::string, TimingRecord>> sorted;
        for (const auto& pair : records_copy) {
            sorted.push_back(pair);
        }
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.second.total_ms > b.second.total_ms;
        });

        // Print header
        os << std::left << std::setw(25) << "Section" << std::right << std::setw(8) << "Count"
           << std::setw(12) << "Total(ms)" << std::setw(12) << "Avg(ms)" << std::setw(12)
           << "Min(ms)" << std::setw(12) << "Max(ms)" << std::setw(8) << "%"
           << "\n";
        os << std::string(89, '-') << "\n";

        // Print rows
        for (const auto& pair : sorted) {
            const auto& name = pair.first;
            const auto& record = pair.second;

            double percent =
                (total_time_ms > 0.0) ? (record.total_ms / total_time_ms * 100.0) : 0.0;

            os << std::left << std::setw(25) << name << std::right << std::fixed
               << std::setprecision(2) << std::setw(8) << record.count << std::setw(12)
               << record.total_ms << std::setw(12) << record.avg_ms() << std::setw(12)
               << record.min_ms << std::setw(12) << record.max_ms << std::setw(7) << percent << "%"
               << "\n";
        }

        os << std::string(89, '-') << "\n";
        os << "TOTAL" << std::string(20, ' ') << std::fixed << std::setprecision(2) << total_time_ms
           << " ms\n";
        os << "========================================\n";
    }

    /**
     * Write profiling data to JSON file.
     * @param filepath Path to output JSON file
     * @return true if successful, false otherwise
     */
    bool write_json(const std::string& filepath) const {
        auto records_copy = snapshot();

        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << filepath << "\n";
            return false;
        }

        // Calculate total time
        double total_time_ms = 0.0;
        for (const auto& pair : records_copy) {
            total_time_ms += pair.second.total_ms;
        }

        // Get timestamp
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        char timestamp[100];
        std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now_time_t));

        // Write JSON
        file << "{\n";
        file << "  \"timestamp\": \"" << timestamp << "\",\n";
        file << std::fixed << std::setprecision(2);
        file << "  \"total_ms\": " << total_time_ms << ",\n";
        file << "  \"sections\": [\n";

        size_t idx = 0;
        for (const auto& pair : records_copy) {
            const auto& name = pair.first;
            const auto& record = pair.second;
            double percent =
                (total_time_ms > 0.0) ? (record.total_ms / total_time_ms * 100.0) : 0.0;

            file << "    {\n";
            file << "      \"name\": \"" << name << "\",\n";
            file << "      \"count\": " << record.count << ",\n";
            file << "      \"total_ms\": " << record.total_ms << ",\n";
            file << "      \"avg_ms\": " << record.avg_ms() << ",\n";
            file << "      \"min_ms\": " << record.min_ms << ",\n";
            file << "      \"max_ms\": " << record.max_ms << ",\n";
            file << "      \"percent\": " << percent << "\n";
            file << "    }";

            if (++idx < records_.size()) {
                file << ",";
            }
            file << "\n";
        }

        file << "  ]\n";
        file << "}\n";

        return true;
    }

    /**
     * Write profiling data to CSV file.
     * @param filepath Path to output CSV file
     * @return true if successful, false otherwise
     */
    bool write_csv(const std::string& filepath) const {
        auto records_copy = snapshot();

        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << filepath << "\n";
            return false;
        }

        // Calculate total time for percentage
        double total_time_ms = 0.0;
        for (const auto& pair : records_copy) {
            total_time_ms += pair.second.total_ms;
        }

        // Write header
        file << "Section,Count,Total_ms,Avg_ms,Min_ms,Max_ms,Percent\n";

        // Write rows
        file << std::fixed << std::setprecision(2);
        for (const auto& pair : records_copy) {
            const auto& name = pair.first;
            const auto& record = pair.second;
            double percent =
                (total_time_ms > 0.0) ? (record.total_ms / total_time_ms * 100.0) : 0.0;

            file << name << "," << record.count << "," << record.total_ms << "," << record.avg_ms()
                 << "," << record.min_ms << "," << record.max_ms << "," << percent << "\n";
        }

        return true;
    }

private:
    Profiler() = default;
    ~Profiler() = default;

    // Prevent copying
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    struct ActiveTimer {
        const char* name;
        std::chrono::high_resolution_clock::time_point start;
    };

    std::map<std::string, TimingRecord> records_;
    mutable std::mutex mutex_;
    inline static thread_local std::vector<ActiveTimer> active_stack_;
};

/**
 * RAII-style scoped timer.
 * Automatically starts timing on construction and stops on destruction.
 *
 * Usage:
 *   {
 *       ScopedTimer timer("my_section");
 *       // ... code to profile ...
 *   } // Timer stops automatically
 */
class ScopedTimer {
public:
    /**
     * Start timing a section.
     * @param section_name Name of the section (must be string literal or stable pointer)
     */
    explicit ScopedTimer(const char* section_name) : section_name_(section_name) {
        Profiler::instance().start(section_name_);
    }

    /**
     * Stop timing on destruction.
     */
    ~ScopedTimer() {
        Profiler::instance().stop(section_name_);
    }

    // Prevent copying and moving
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

private:
    const char* section_name_;
};

    // Convenience macros for profiling
    #define PROFILE_SCOPE(name) pfalign::profiling::ScopedTimer _profile_timer_##__LINE__(name)

    #define PROFILE_START(name) pfalign::profiling::Profiler::instance().start(name)

    #define PROFILE_STOP(name) pfalign::profiling::Profiler::instance().stop(name)

    #define PROFILE_RESET() pfalign::profiling::Profiler::instance().reset()

    #define PROFILE_PRINT() pfalign::profiling::Profiler::instance().print_report()

    #define PROFILE_WRITE_JSON(path) pfalign::profiling::Profiler::instance().write_json(path)

    #define PROFILE_WRITE_CSV(path) pfalign::profiling::Profiler::instance().write_csv(path)

#else  // ENABLE_PROFILING not defined

    // When profiling is disabled, all macros expand to nothing (zero overhead)
    #define PROFILE_SCOPE(name) ((void)0)
    #define PROFILE_START(name) ((void)0)
    #define PROFILE_STOP(name) ((void)0)
    #define PROFILE_RESET() ((void)0)
    #define PROFILE_PRINT() ((void)0)
    #define PROFILE_WRITE_JSON(path) (true)
    #define PROFILE_WRITE_CSV(path) (true)

#endif  // ENABLE_PROFILING

}  // namespace profiling
}  // namespace pfalign
