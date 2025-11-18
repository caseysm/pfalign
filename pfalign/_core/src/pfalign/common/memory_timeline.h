/**
 * Memory Timeline Tracker - Temporal Memory Usage Analysis
 *
 * IMPORTANT: This profiler is NOT currently integrated into the MSA pipeline.
 * It is kept for potential future memory timeline profiling needs.
 *
 * For current memory tracking, use:
 * - GrowableArena::peak() - peak memory per thread
 * - ThreadPool::total_peak_mb() - total peak across all threads
 *
 * To enable this profiler, instantiate MemoryTimeline in your code and
 * call snapshot() at regular intervals:
 *
 * Usage:
 *   MemoryTimeline timeline("msa_run");
 *
 *   timeline.snapshot("start");
 *   // ... distance matrix computation ...
 *   timeline.snapshot("distance_matrix_complete");
 *
 *   // ... progressive alignment ...
 *   timeline.snapshot("alignment_complete");
 *
 *   timeline.print_timeline();
 *   timeline.export_csv("memory_timeline.csv");
 */

#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <mutex>
#include <fstream>
#include <algorithm>
#include <set>
#include <cstdio>

namespace pfalign {
namespace memory {

/**
 * Single timeline snapshot with timestamp and memory breakdown
 */
struct TimelineSnapshot {
    double timestamp_ms;     // Time since timeline start
    std::string phase;       // Phase name (e.g., "distance_matrix")
    size_t total_allocated;  // Total memory allocated at this point

    // Component breakdown (optional)
    std::unordered_map<std::string, size_t> component_usage;

    TimelineSnapshot(double ts, const std::string& p, size_t total)
        : timestamp_ms(ts), phase(p), total_allocated(total) {
    }

    double total_mb() const {
        return total_allocated / (1024.0 * 1024.0);
    }
};

/**
 * Memory usage timeline tracker
 *
 * Records memory snapshots at key points in execution to analyze
 * temporal memory usage patterns.
 */
class MemoryTimeline {
public:
    explicit MemoryTimeline(const std::string& session_name)
        : session_name_(session_name), start_time_(std::chrono::steady_clock::now()) {
    }

    /**
     * Record a memory snapshot at current time
     *
     * @param phase         Phase name for this snapshot
     * @param total_bytes   Total memory allocated (pass 0 to calculate from components)
     */
    void snapshot(const std::string& phase, size_t total_bytes = 0) {
        std::lock_guard<std::mutex> lock(mutex_);

        double elapsed_ms = get_elapsed_ms();

        if (total_bytes == 0) {
            // Calculate total from component registry if available
            total_bytes = calculate_total_from_components();
        }

        snapshots_.emplace_back(elapsed_ms, phase, total_bytes);
    }

    /**
     * Record snapshot with component breakdown
     */
    void snapshot_with_components(const std::string& phase,
                                  const std::unordered_map<std::string, size_t>& components) {
        std::lock_guard<std::mutex> lock(mutex_);

        double elapsed_ms = get_elapsed_ms();

        // Calculate total
        size_t total = 0;
        for (const auto& [name, size] : components) {
            total += size;
        }

        TimelineSnapshot snapshot(elapsed_ms, phase, total);
        snapshot.component_usage = components;

        snapshots_.push_back(snapshot);
    }

    /**
     * Register a component for automatic tracking
     *
     * Components registered here will be included in snapshot calculations
     */
    void register_component(const std::string& name, size_t* size_ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        component_registry_[name] = size_ptr;
    }

    /**
     * Get all snapshots
     */
    const std::vector<TimelineSnapshot>& snapshots() const {
        return snapshots_;
    }

    /**
     * Get peak memory usage across timeline
     */
    size_t peak_memory() const {
        std::lock_guard<std::mutex> lock(mutex_);

        if (snapshots_.empty())
            return 0;

        return std::max_element(snapshots_.begin(), snapshots_.end(),
                                [](const TimelineSnapshot& a, const TimelineSnapshot& b) {
                                    return a.total_allocated < b.total_allocated;
                                })
            ->total_allocated;
    }

    /**
     * Get peak memory in MB
     */
    double peak_memory_mb() const {
        return peak_memory() / (1024.0 * 1024.0);
    }

    /**
     * Find phase where peak memory occurred
     */
    std::string peak_phase() const {
        std::lock_guard<std::mutex> lock(mutex_);

        if (snapshots_.empty())
            return "";

        return std::max_element(snapshots_.begin(), snapshots_.end(),
                                [](const TimelineSnapshot& a, const TimelineSnapshot& b) {
                                    return a.total_allocated < b.total_allocated;
                                })
            ->phase;
    }

    /**
     * Print timeline to console
     */
    void print_timeline() const {
        std::lock_guard<std::mutex> lock(mutex_);

        if (snapshots_.empty()) {
            printf("No timeline snapshots recorded.\n");
            return;
        }

        printf("\n");
        printf("================================================================\n");
        printf("Memory Timeline: %s\n", session_name_.c_str());
        printf("================================================================\n");
        printf("\n");
        printf("%-40s %12s %15s\n", "Phase", "Time (s)", "Memory (MB)");
        printf("----------------------------------------------------------------\n");

        for (const auto& snapshot : snapshots_) {
            printf("%-40s %12.3f %15.2f", snapshot.phase.c_str(), snapshot.timestamp_ms / 1000.0,
                   snapshot.total_mb());

            // Mark peak
            if (snapshot.total_allocated == peak_memory()) {
                printf(" <- PEAK");
            }
            printf("\n");

            // Print component breakdown if available
            if (!snapshot.component_usage.empty()) {
                std::vector<std::pair<std::string, size_t>> sorted_components(
                    snapshot.component_usage.begin(), snapshot.component_usage.end());
                std::sort(sorted_components.begin(), sorted_components.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });

                for (const auto& [name, size] : sorted_components) {
                    double mb = size / (1024.0 * 1024.0);
                    if (mb > 0.1) {  // Only show components > 0.1 MB
                        printf("  +- %-35s %15.2f\n", name.c_str(), mb);
                    }
                }
            }
        }

        printf("================================================================\n");
        printf("Peak Memory: %.2f MB at phase '%s'\n", peak_memory_mb(), peak_phase().c_str());
        printf("================================================================\n");
        printf("\n");
    }

    /**
     * Export timeline to CSV
     */
    void export_csv(const std::string& filename) const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            fprintf(stderr, "Failed to open %s for writing\n", filename.c_str());
            return;
        }

        // Header
        ofs << "phase,timestamp_ms,timestamp_s,memory_bytes,memory_mb\n";

        // Data
        for (const auto& snapshot : snapshots_) {
            ofs << snapshot.phase << "," << snapshot.timestamp_ms << ","
                << (snapshot.timestamp_ms / 1000.0) << "," << snapshot.total_allocated << ","
                << snapshot.total_mb() << "\n";
        }

        ofs.close();
    }

    /**
     * Export timeline with component breakdown to CSV
     */
    void export_detailed_csv(const std::string& filename) const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            fprintf(stderr, "Failed to open %s for writing\n", filename.c_str());
            return;
        }

        // Collect all component names
        std::set<std::string> all_components;
        for (const auto& snapshot : snapshots_) {
            for (const auto& [name, _] : snapshot.component_usage) {
                all_components.insert(name);
            }
        }

        // Header
        ofs << "phase,timestamp_s,total_mb";
        for (const auto& comp : all_components) {
            ofs << "," << comp << "_mb";
        }
        ofs << "\n";

        // Data
        for (const auto& snapshot : snapshots_) {
            ofs << snapshot.phase << "," << (snapshot.timestamp_ms / 1000.0) << ","
                << snapshot.total_mb();

            for (const auto& comp : all_components) {
                auto it = snapshot.component_usage.find(comp);
                if (it != snapshot.component_usage.end()) {
                    ofs << "," << (it->second / (1024.0 * 1024.0));
                } else {
                    ofs << ",0";
                }
            }
            ofs << "\n";
        }

        ofs.close();
    }

private:
    double get_elapsed_ms() const {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);
        return duration.count() / 1000.0;
    }

    size_t calculate_total_from_components() const {
        size_t total = 0;
        for (const auto& [name, size_ptr] : component_registry_) {
            if (size_ptr) {
                total += *size_ptr;
            }
        }
        return total;
    }

    std::string session_name_;
    std::chrono::steady_clock::time_point start_time_;
    std::vector<TimelineSnapshot> snapshots_;
    std::unordered_map<std::string, size_t*> component_registry_;
    mutable std::mutex mutex_;
};

/**
 * RAII helper for automatic snapshot on scope entry/exit
 */
class TimelineScope {
public:
    TimelineScope(MemoryTimeline& timeline, const std::string& phase_name)
        : timeline_(timeline), phase_name_(phase_name) {
        timeline_.snapshot(phase_name_ + "_start");
    }

    ~TimelineScope() {
        timeline_.snapshot(phase_name_ + "_end");
    }

private:
    MemoryTimeline& timeline_;
    std::string phase_name_;
};

}  // namespace memory
}  // namespace pfalign
