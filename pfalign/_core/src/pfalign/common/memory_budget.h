/**
 * Global memory budget tracker with system memory auto-detection.
 *
 * MemoryBudget provides:
 * - Global tracking of memory allocated across all GrowableArenas
 * - System RAM detection (Linux /proc/meminfo, macOS sysctl, Windows GlobalMemoryStatusEx)
 * - Intelligent thread count suggestions based on available memory
 * - Thread-safe atomic operations for concurrent updates
 *
 * Usage:
 *   MemoryBudget& budget = MemoryBudget::global();
 *   size_t system_ram = budget.system_ram_mb();
 *   size_t available = budget.available_mb();
 *   size_t threads = budget.suggest_thread_count(75);  // 75 MB per thread
 *
 * Thread safety: All operations are thread-safe (atomic operations).
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <algorithm>
#include <thread>

// Platform-specific headers for memory detection
#ifdef __linux__
    #include <sys/sysinfo.h>
    #include <unistd.h>
#endif

#ifdef __APPLE__
    #include <sys/sysctl.h>
    #include <sys/types.h>
#endif

#ifdef _WIN32
    #include <windows.h>
#endif


namespace pfalign::memory {

class MemoryBudget {
private:
    std::atomic<size_t> allocated_bytes_;  // Total allocated across all arenas
    std::atomic<size_t> peak_bytes_;       // Peak allocation since construction
    size_t system_ram_bytes_;              // Total system RAM (detected at construction)

    // Singleton instance
    static MemoryBudget& instance() {
        static MemoryBudget budget;
        return budget;
    }

    // Detect system RAM at construction
    MemoryBudget() : allocated_bytes_(0), peak_bytes_(0), system_ram_bytes_(detect_system_ram()) {
    }

    // Detect total system RAM (platform-specific)
    static size_t detect_system_ram() {
#ifdef __linux__
        // Linux: Use sysinfo() to get total RAM
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
            return static_cast<size_t>(info.totalram) * info.mem_unit;
        }

        // Fallback: Parse /proc/meminfo
        FILE* meminfo = fopen("/proc/meminfo", "r");
        if (meminfo) {
            char line[256];
            while (fgets(line, sizeof(line), meminfo)) {
                size_t kb = 0;
                if (sscanf(line, "MemTotal: %zu kB", &kb) == 1) {
                    fclose(meminfo);
                    return kb * 1024;  // Convert KB -> bytes
                }
            }
            fclose(meminfo);
        }
#endif

#ifdef __APPLE__
        // macOS: Use sysctl to get hw.memsize
        int mib[2] = {CTL_HW, HW_MEMSIZE};
        uint64_t memsize = 0;
        size_t len = sizeof(memsize);
        if (sysctl(mib, 2, &memsize, &len, NULL, 0) == 0) {
            return static_cast<size_t>(memsize);
        }
#endif

#ifdef _WIN32
        // Windows: Use GlobalMemoryStatusEx
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status)) {
            return static_cast<size_t>(status.ullTotalPhys);
        }
#endif

        // Fallback: Assume 8 GB if detection fails
        fprintf(stderr, "Warning: Failed to detect system RAM, assuming 8 GB\n");
        return 8ULL * 1024 * 1024 * 1024;
    }

public:
    // Non-copyable, non-movable (singleton)
    MemoryBudget(const MemoryBudget&) = delete;
    MemoryBudget& operator=(const MemoryBudget&) = delete;
    MemoryBudget(MemoryBudget&&) = delete;
    MemoryBudget& operator=(MemoryBudget&&) = delete;

    /**
     * Get global singleton instance.
     */
    static MemoryBudget& global() {
        return instance();
    }

    /**
     * Record allocation (called by GrowableArena when allocating new blocks).
     *
     * @param bytes Number of bytes allocated
     */
    void allocate(size_t bytes) {
        size_t current = allocated_bytes_.fetch_add(bytes, std::memory_order_relaxed) + bytes;

        // Update peak (thread-safe CAS loop)
        size_t expected_peak = peak_bytes_.load(std::memory_order_relaxed);
        while (current > expected_peak &&
               !peak_bytes_.compare_exchange_weak(expected_peak, current, std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            // Loop until successful CAS
        }
    }

    /**
     * Record deallocation (called by GrowableArena when deallocating blocks).
     *
     * @param bytes Number of bytes deallocated
     */
    void deallocate(size_t bytes) {
        allocated_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
    }

    /**
     * Get total system RAM in bytes.
     */
    [[nodiscard]] size_t system_ram_bytes() const {
        return system_ram_bytes_;
    }

    /**
     * Get total system RAM in MB.
     */
    [[nodiscard]] size_t system_ram_mb() const {
        return system_ram_bytes_ / (1024 * 1024);
    }

    /**
     * Get currently allocated memory in bytes.
     */
    [[nodiscard]] size_t allocated_bytes() const {
        return allocated_bytes_.load(std::memory_order_relaxed);
    }

    /**
     * Get currently allocated memory in MB.
     */
    [[nodiscard]] size_t allocated_mb() const {
        return allocated_bytes() / (1024 * 1024);
    }

    /**
     * Get peak allocated memory in bytes.
     */
    [[nodiscard]] size_t peak_bytes() const {
        return peak_bytes_.load(std::memory_order_relaxed);
    }

    /**
     * Get peak allocated memory in MB.
     */
    [[nodiscard]] size_t peak_mb() const {
        return peak_bytes() / (1024 * 1024);
    }

    /**
     * Get available memory in bytes (system RAM - allocated).
     */
    [[nodiscard]] size_t available_bytes() const {
        size_t allocated = allocated_bytes();
        return (allocated < system_ram_bytes_) ? (system_ram_bytes_ - allocated) : 0;
    }

    /**
     * Get available memory in MB.
     */
    [[nodiscard]] size_t available_mb() const {
        return available_bytes() / (1024 * 1024);
    }

    /**
     * Suggest optimal thread count based on memory constraints.
     *
     * Formula:
     *   threads = min(hardware_concurrency,
     *                 available_ram / arena_size_per_thread,
     *                 max_threads_cap)
     *
     * Ensures we don't over-commit memory by spawning too many threads.
     *
     * @param arena_mb Arena size per thread in MB (e.g., 75 MB)
     * @param safety_factor Fraction of available RAM to use (default: 0.8 = 80%)
     * @param max_threads_cap Hard cap on thread count (default: 64)
     * @return Suggested thread count
     */
    [[nodiscard]] size_t suggest_thread_count(size_t arena_mb, float safety_factor = 0.8f,
                                               size_t max_threads_cap = 64) const {
        // Get hardware concurrency
        size_t hw_threads = std::max(1u, std::thread::hardware_concurrency());

        // Calculate memory-limited threads
        size_t available = static_cast<size_t>(available_mb() * safety_factor);
        size_t memory_limited_threads = (arena_mb > 0) ? (available / arena_mb) : hw_threads;

        // Return minimum of (hw_threads, memory_limited_threads, max_cap)
        size_t threads = std::min({hw_threads, memory_limited_threads, max_threads_cap});

        // Ensure at least 1 thread
        return std::max<size_t>(1, threads);
    }

    /**
     * Reset peak tracking (useful for profiling experiments).
     */
    void reset_peak() {
        peak_bytes_.store(allocated_bytes_.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
    }

    /**
     * Print budget statistics (for debugging).
     */
    void print_stats(FILE* out = stderr) const {
        fprintf(out, "MemoryBudget:\n");
        fprintf(out, "  System RAM: %zu MB\n", system_ram_mb());
        fprintf(out, "  Allocated: %zu MB (%.1f%%)\n", allocated_mb(),
                100.0 * allocated_bytes() / std::max<size_t>(1, system_ram_bytes_));
        fprintf(out, "  Peak: %zu MB (%.1f%%)\n", peak_mb(),
                100.0 * peak_bytes() / std::max<size_t>(1, system_ram_bytes_));
        fprintf(out, "  Available: %zu MB\n", available_mb());
    }
};

} // namespace pfalign::memory

