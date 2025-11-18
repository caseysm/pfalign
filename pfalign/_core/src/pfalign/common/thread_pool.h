#pragma once

#include <thread>
#include <vector>
#include <functional>
#include <algorithm>
#include <string>
#include "pfalign/common/growable_arena.h"
#include "pfalign/common/memory_budget.h"

#if ENABLE_PROFILING
    #include "pfalign/common/cpu_utilization.h"
    #include "pfalign/common/profile_scope.h"
#endif

namespace pfalign {
namespace threading {

/**
 * Thread pool with per-thread growable arenas.
 *
 * Each worker thread gets a dedicated GrowableArena that grows automatically.
 * Uses std::thread (not OpenMP).
 *
 * Key improvements:
 * - No hard capacity limits (arenas grow via 1.5* linked blocks)
 * - MemoryBudget integration for intelligent thread scaling
 * - Tracks peak memory usage for profiling
 *
 * Thread Safety:
 * - Each thread has its own arena (no shared state)
 * - Multiple parallel_for calls can run sequentially (not concurrently)
 * - Arenas are reset between calls for reuse
 *
 * Usage:
 *   ThreadPool pool(4, 32);  // 4 threads, 32MB initial per arena
 *   pool.parallel_for(1000, [](int tid, size_t begin, size_t end, GrowableArena& arena) {
 *       for (size_t i = begin; i < end; i++) {
 *           float* buf = arena.allocate<float>(100);
 *           // ... use buf ...
 *       }
 *   });
 */
class ThreadPool {
public:
    /**
     * Create thread pool with growable arenas.
     *
     * @param num_threads Number of worker threads (0 = auto-detect from MemoryBudget)
     * @param initial_arena_mb Initial arena size per thread in MB (default: 32MB)
     */
    ThreadPool(size_t num_threads = 0, size_t initial_arena_mb = 32)
        : initial_arena_mb_(initial_arena_mb) {
        // Auto-detect threads using MemoryBudget if num_threads == 0
        if (num_threads == 0) {
            num_threads_ = memory::MemoryBudget::global().suggest_thread_count(initial_arena_mb);
        } else {
            num_threads_ = num_threads;
        }

        // Create per-thread growable arenas (move semantics)
        arenas_.reserve(num_threads_);
        for (size_t i = 0; i < num_threads_; i++) {
            std::string arena_name = "thread_" + std::to_string(i);
            arenas_.emplace_back(initial_arena_mb, arena_name.c_str());
        }
    }

    /**
     * Destructor - ensures clean shutdown.
     */
    ~ThreadPool() = default;

    /**
     * Execute parallel_for over range [0, count) using multiple threads.
     *
     * Divides work into chunks and spawns worker threads.
     * Each thread receives its own dedicated growable arena (thread-safe).
     *
     * The function blocks until all workers complete.
     *
     * @param count Total number of iterations
     * @param func  Lambda with signature: void(int thread_id, size_t begin, size_t end,
     * GrowableArena& arena)
     *              - thread_id: Thread index (0 to num_threads-1)
     *              - begin: Start index (inclusive)
     *              - end: End index (exclusive)
     *              - arena: Thread-local growable arena allocator
     *
     * Example:
     *   pool.parallel_for(1000, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
     *       for (size_t i = begin; i < end; i++) {
     *           // Process item i using thread tid and arena
     *       }
     *   });
     */
    template <typename Func>
    void parallel_for(size_t count, Func&& func) {
        if (count == 0)
            return;

        std::vector<std::thread> workers;
        workers.reserve(num_threads_);

        size_t chunk_size = (count + num_threads_ - 1) / num_threads_;

#if ENABLE_PROFILING
        // Capture phase name from PARENT thread's scope stack (before spawning workers)
        // Worker threads have their own thread-local stacks, so we must capture here
        // Must copy as std::string (not string_view) to avoid dangling reference
        std::string parent_scope_label = "unknown";
        {
            const auto& scope_stack = pfalign::get_scope_stack();
            if (!scope_stack.empty()) {
                parent_scope_label = std::string(scope_stack.back());
            }
        }
#endif

        for (size_t tid = 0; tid < num_threads_; tid++) {
            size_t begin = tid * chunk_size;
            size_t end = std::min(begin + chunk_size, count);

            if (begin >= count)
                break;

#if ENABLE_PROFILING
            workers.emplace_back([this, tid, begin, end, func, parent_scope_label]() {
#else
            workers.emplace_back([this, tid, begin, end, func]() {
#endif
                // Each thread gets its own growable arena (thread-safe!)
                memory::GrowableArena& thread_arena = arenas_[tid];
                thread_arena.reset();  // Clear before use

#if ENABLE_PROFILING
                // Per-thread CPU tracking with phase context (captured from parent thread)
                std::string tracker_name = parent_scope_label + "_worker_" + std::to_string(tid);
                pfalign::CPUUtilizationTracker thread_tracker(tracker_name, 1);
#endif

                func(static_cast<int>(tid), begin, end, thread_arena);
            });
        }

        // Wait for all workers to finish
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    /**
     * Reset all arenas for reuse.
     *
     * Call this between independent parallel_for operations to
     * reclaim arena memory without deallocation.
     */
    void reset_arenas() {
        for (auto& arena : arenas_) {
            arena.reset();
        }
    }

    /**
     * Get number of worker threads.
     */
    size_t num_threads() const {
        return num_threads_;
    }

    /**
     * Get total peak memory usage across all arenas (MB).
     */
    size_t total_peak_mb() const {
        size_t total = 0;
        for (const auto& arena : arenas_) {
            total += arena.peak();
        }
        return total / (1024 * 1024);
    }

    /**
     * Print arena statistics for all threads (for debugging).
     */
    void print_arena_stats(FILE* out = stderr) const {
        fprintf(out, "ThreadPool Arena Statistics:\n");
        fprintf(out, "  Threads: %zu\n", num_threads_);
        fprintf(out, "  Initial arena size: %zu MB per thread\n", initial_arena_mb_);
        fprintf(out, "  Total peak usage: %zu MB\n", total_peak_mb());
        for (size_t i = 0; i < arenas_.size(); i++) {
            fprintf(out, "  Thread %zu: ", i);
            fprintf(out, "peak=%zu MB, capacity=%zu MB, blocks=%zu\n",
                    arenas_[i].peak() / (1024 * 1024), arenas_[i].capacity() / (1024 * 1024),
                    arenas_[i].num_blocks());
        }
    }

private:
    size_t num_threads_;
    size_t initial_arena_mb_;
    std::vector<memory::GrowableArena> arenas_;

    // Non-copyable, non-movable (manages thread state)
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
};

}  // namespace threading
}  // namespace pfalign
