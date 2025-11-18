/**
 * MemoryProfiler - Hierarchical Memory Usage Tracking
 *
 * Tracks memory allocations across the MSA pipeline with zero runtime
 * overhead when profiling is disabled.
 *
 * Features:
 * - Per-component tracking (arenas, workspaces, thread pools)
 * - Peak usage detection
 * - Hierarchical reporting (total -> component -> thread)
 * - CSV export for analysis
 * - Thread-safe (lock-free reads, mutex for writes)
 *
 * Usage:
 *   MemoryProfiler profiler("msa_run");
 *   profiler.record_arena("main_arena", arena.peak(), arena.capacity());
 *   profiler.record_arena("thread_0", thread_arena.peak(), thread_arena.capacity());
 *   profiler.print_summary();
 *   profiler.export_csv("memory_profile.csv");
 *
 * Design:
 * - Compile-time enable/disable via MEMORY_PROFILING macro
 * - Hierarchical structure: ProfileSession -> ComponentRecord -> Snapshot
 * - Minimal overhead: Only stores peaks, not full history
 *
 * Integration Points:
 * - GrowableArena::peak() - track arena usage
 * - ThreadPool::total_peak_mb() - track per-thread arena usage
 * - MSAWorkspace growth events - track workspace expansions
 * - SequenceCache::size() - track cached embeddings
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <cstdio>

namespace pfalign {
namespace memory {

/**
 * Snapshot of memory usage at a point in time.
 */
struct MemorySnapshot {
    size_t used_bytes;      // Current usage
    size_t capacity_bytes;  // Total capacity
    size_t peak_bytes;      // Peak usage

    MemorySnapshot(size_t used = 0, size_t capacity = 0, size_t peak = 0)
        : used_bytes(used), capacity_bytes(capacity), peak_bytes(peak) {
    }

    double used_mb() const {
        return used_bytes / (1024.0 * 1024.0);
    }
    double capacity_mb() const {
        return capacity_bytes / (1024.0 * 1024.0);
    }
    double peak_mb() const {
        return peak_bytes / (1024.0 * 1024.0);
    }
    double utilization() const {
        return capacity_bytes > 0 ? (100.0 * peak_bytes / capacity_bytes) : 0.0;
    }
};

/**
 * Component record - tracks memory for a named component (e.g., "main_arena", "thread_0").
 */
struct ComponentRecord {
    std::string name;
    std::string type;  // "arena", "workspace", "cache", "threadpool"
    MemorySnapshot snapshot;

    ComponentRecord(const std::string& n, const std::string& t, const MemorySnapshot& s)
        : name(n), type(t), snapshot(s) {
    }
};

/**
 * MemoryProfiler - Hierarchical memory tracking for MSA pipeline.
 *
 * Records peak memory usage across all components and provides
 * detailed reporting and CSV export.
 *
 * Thread-safe: Multiple threads can record data concurrently.
 */
class MemoryProfiler {
public:
    /**
     * Create profiler for a named session (e.g., "msa_n100_upgma").
     */
    explicit MemoryProfiler(const std::string& session_name)
        : session_name_(session_name), start_time_(std::chrono::steady_clock::now()) {
    }

    /**
     * Record arena usage.
     *
     * @param name          Component name (e.g., "main_arena", "thread_0")
     * @param peak_bytes    Peak usage in bytes
     * @param capacity      Total capacity in bytes
     */
    void record_arena(const std::string& name, size_t peak_bytes, size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        MemorySnapshot snapshot(peak_bytes, capacity, peak_bytes);
        records_.emplace_back(name, "arena", snapshot);
    }

    /**
     * Record workspace usage.
     *
     * @param name          Component name (e.g., "msa_workspace")
     * @param peak_bytes    Peak usage in bytes
     * @param capacity      Total capacity in bytes
     */
    void record_workspace(const std::string& name, size_t peak_bytes, size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        MemorySnapshot snapshot(peak_bytes, capacity, peak_bytes);
        records_.emplace_back(name, "workspace", snapshot);
    }

    /**
     * Record cache usage.
     *
     * @param name          Component name (e.g., "sequence_cache")
     * @param size_bytes    Current size in bytes
     * @param num_entries   Number of cached entries
     */
    void record_cache(const std::string& name, size_t size_bytes, int num_entries) {
        std::lock_guard<std::mutex> lock(mutex_);
        MemorySnapshot snapshot(size_bytes, size_bytes, size_bytes);
        records_.emplace_back(name, "cache", snapshot);
        cache_entries_[name] = num_entries;
    }

    /**
     * Record thread pool arena usage.
     *
     * @param pool_name     Pool name (e.g., "distance_matrix_pool")
     * @param num_threads   Number of threads
     * @param total_peak_mb Total peak usage across all threads (MB)
     */
    void record_threadpool(const std::string& pool_name, int num_threads, double total_peak_mb) {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t total_peak_bytes = static_cast<size_t>(total_peak_mb * 1024 * 1024);
        MemorySnapshot snapshot(total_peak_bytes, total_peak_bytes, total_peak_bytes);
        records_.emplace_back(pool_name, "threadpool", snapshot);
        threadpool_threads_[pool_name] = num_threads;
    }

    /**
     * Get total peak memory usage (sum of all component peaks).
     *
     * @return Total peak usage in bytes
     */
    size_t total_peak_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t total = 0;
        for (const auto& record : records_) {
            total += record.snapshot.peak_bytes;
        }
        return total;
    }

    /**
     * Get total peak memory usage in MB.
     */
    double total_peak_mb() const {
        return total_peak_bytes() / (1024.0 * 1024.0);
    }

    /**
     * Get elapsed time since profiler creation (seconds).
     */
    double elapsed_seconds() const {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);
        return duration.count() / 1e6;
    }

    /**
     * Print hierarchical summary to stdout.
     *
     * Format:
     *   Memory Profile: msa_n100_upgma (duration: 45.2s)
     *   ================================================================
     *   Total Peak Memory: 1234.5 MB
     *
     *   Component Breakdown:
     *   ------------------------------------------------------------------
     *   Type         Name                   Peak (MB)  Capacity (MB)  Util%
     *   ------------------------------------------------------------------
     *   arena        main_arena                 512.0        1024.0  50.0%
     *   threadpool   distance_matrix_pool       256.0         256.0 100.0%
     *   workspace    msa_workspace              128.0         256.0  50.0%
     *   cache        sequence_cache              64.0          64.0 100.0%
     *   ------------------------------------------------------------------
     */
    void print_summary() const {
        std::lock_guard<std::mutex> lock(mutex_);

        // Compute values while holding lock (don't call other methods that lock)
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);
        double elapsed = duration.count() / 1e6;

        size_t total_peak = 0;
        for (const auto& record : records_) {
            total_peak += record.snapshot.peak_bytes;
        }
        double total_peak_mb_val = total_peak / (1024.0 * 1024.0);

        printf("\n");
        printf("================================================================\n");
        printf("Memory Profile: %s (duration: %.2fs)\n", session_name_.c_str(), elapsed);
        printf("================================================================\n");
        printf("Total Peak Memory: %.2f MB\n", total_peak_mb_val);
        printf("\n");
        printf("Component Breakdown:\n");
        printf("------------------------------------------------------------------\n");
        printf("%-12s %-25s %10s %12s %8s\n", "Type", "Name", "Peak (MB)", "Capacity (MB)",
               "Util%");
        printf("------------------------------------------------------------------\n");

        // Sort by peak descending
        std::vector<ComponentRecord> sorted = records_;
        std::sort(sorted.begin(), sorted.end(),
                  [](const ComponentRecord& a, const ComponentRecord& b) {
                      return a.snapshot.peak_bytes > b.snapshot.peak_bytes;
                  });

        for (const auto& record : sorted) {
            printf("%-12s %-25s %10.2f %12.2f %7.1f%%\n", record.type.c_str(), record.name.c_str(),
                   record.snapshot.peak_mb(), record.snapshot.capacity_mb(),
                   record.snapshot.utilization());

            // Print extra info for threadpools and caches
            if (record.type == "threadpool") {
                auto it = threadpool_threads_.find(record.name);
                if (it != threadpool_threads_.end()) {
                    printf("             +- threads: %d, avg per-thread: %.2f MB\n", it->second,
                           record.snapshot.peak_mb() / it->second);
                }
            } else if (record.type == "cache") {
                auto it = cache_entries_.find(record.name);
                if (it != cache_entries_.end()) {
                    printf("             +- entries: %d, avg per-entry: %.2f MB\n", it->second,
                           record.snapshot.peak_mb() / it->second);
                }
            }
        }

        printf("------------------------------------------------------------------\n");
        printf("\n");
    }

    /**
     * Export profiling data to CSV.
     *
     * CSV format:
     *   session,type,name,peak_mb,capacity_mb,utilization_pct,elapsed_sec
     *
     * @param filepath Path to CSV file (will be created/overwritten)
     * @return true on success, false on error
     */
    bool export_csv(const std::string& filepath) const {
        std::lock_guard<std::mutex> lock(mutex_);

        FILE* fp = fopen(filepath.c_str(), "w");
        if (!fp) {
            return false;
        }

        // Header
        fprintf(fp, "session,type,name,peak_mb,capacity_mb,utilization_pct,elapsed_sec\n");

        // Data rows
        for (const auto& record : records_) {
            fprintf(fp, "%s,%s,%s,%.2f,%.2f,%.2f,%.2f\n", session_name_.c_str(),
                    record.type.c_str(), record.name.c_str(), record.snapshot.peak_mb(),
                    record.snapshot.capacity_mb(), record.snapshot.utilization(),
                    elapsed_seconds());
        }

        fclose(fp);
        return true;
    }

    /**
     * Get session name.
     */
    const std::string& session_name() const {
        return session_name_;
    }

    /**
     * Get number of recorded components.
     */
    size_t num_components() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return records_.size();
    }

private:
    std::string session_name_;
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    std::vector<ComponentRecord> records_;
    std::unordered_map<std::string, int> threadpool_threads_;
    std::unordered_map<std::string, int> cache_entries_;
    mutable std::mutex mutex_;
};

/**
 * No-op profiler for zero overhead when profiling is disabled.
 *
 * All methods are empty inline functions that the compiler will optimize away.
 */
class NullMemoryProfiler {
public:
    explicit NullMemoryProfiler(const std::string&) {
    }
    void record_arena(const std::string&, size_t, size_t) {
    }
    void record_workspace(const std::string&, size_t, size_t) {
    }
    void record_cache(const std::string&, size_t, int) {
    }
    void record_threadpool(const std::string&, int, double) {
    }
    size_t total_peak_bytes() const {
        return 0;
    }
    double total_peak_mb() const {
        return 0.0;
    }
    double elapsed_seconds() const {
        return 0.0;
    }
    void print_summary() const {
    }
    bool export_csv(const std::string&) const {
        return false;
    }
    const char* session_name() const {
        return "";
    }
    size_t num_components() const {
        return 0;
    }
};

// Compile-time switch: Use real profiler if MEMORY_PROFILING defined, else use null profiler
#ifdef MEMORY_PROFILING
using ActiveMemoryProfiler = MemoryProfiler;
#else
using ActiveMemoryProfiler = NullMemoryProfiler;
#endif

}  // namespace memory
}  // namespace pfalign
