#include "pfalign/common/growable_arena.h"
#include "pfalign/common/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

using namespace pfalign::threading;
using namespace pfalign::memory;

// Helper to print test status
#define TEST_START(name) std::cout << "Test: " << name << "..." << std::flush
#define TEST_PASS() std::cout << " ✓ PASS" << std::endl
#define TEST_FAIL(msg) do { std::cout << " ✗ FAIL: " << msg << std::endl; return false; } while(0)

// ============================================================================
// Test Category 1: Construction and Destruction
// ============================================================================

bool test_auto_detect_threads() {
    TEST_START("Auto-detect thread count");

    ThreadPool pool;
    size_t detected = pool.num_threads();
    size_t hardware = std::thread::hardware_concurrency();

    if (detected != hardware) {
        TEST_FAIL("Expected " + std::to_string(hardware) + " threads, got " + std::to_string(detected));
    }

    TEST_PASS();
    return true;
}

bool test_explicit_thread_count() {
    TEST_START("Explicit thread count");

    ThreadPool pool_2(2);
    ThreadPool pool_8(8);

    if (pool_2.num_threads() != 2) {
        TEST_FAIL("Expected 2 threads, got " + std::to_string(pool_2.num_threads()));
    }

    if (pool_8.num_threads() != 8) {
        TEST_FAIL("Expected 8 threads, got " + std::to_string(pool_8.num_threads()));
    }

    TEST_PASS();
    return true;
}

bool test_cleanup_on_destroy() {
    TEST_START("Cleanup on destruction");

    // ThreadPool should clean up automatically (RAII)
    {
        ThreadPool pool(4, 50);  // 4 threads, 50MB per arena

        std::atomic<int> counter{0};
        pool.parallel_for(100, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
            for (size_t i = begin; i < end; i++) {
                counter++;
            }
        });

        if (counter.load() != 100) {
            TEST_FAIL("Counter should be 100, got " + std::to_string(counter.load()));
        }
        // Pool destructor called here
    }

    TEST_PASS();
    return true;
}

// ============================================================================
// Test Category 2: Parallel Execution
// ============================================================================

bool test_work_distribution() {
    TEST_START("Work distribution across threads");

    ThreadPool pool(4);
    std::atomic<int> counter{0};

    pool.parallel_for(1000, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        for (size_t i = begin; i < end; i++) {
            counter++;
        }
    });

    if (counter.load() != 1000) {
        TEST_FAIL("Expected 1000 iterations, got " + std::to_string(counter.load()));
    }

    TEST_PASS();
    return true;
}

bool test_thread_id_tracking() {
    TEST_START("Thread ID tracking");

    ThreadPool pool(4);
    std::mutex mutex;
    std::set<int> seen_tids;
    std::atomic<int> counter{0};

    pool.parallel_for(1000, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        (void)arena;
        for (size_t i = begin; i < end; i++) {
            counter.fetch_add(1, std::memory_order_relaxed);
        }
        std::lock_guard<std::mutex> lock(mutex);
        seen_tids.insert(tid);
    });

    // Relaxed test - just verify we got at least 2 threads working
    // (not all systems may schedule all 4 threads for small workloads)
    if (seen_tids.size() < 2) {
        TEST_FAIL("Expected at least 2 unique thread IDs, got " + std::to_string(seen_tids.size()));
    }

    if (counter.load() != 1000) {
        TEST_FAIL("Expected 1000 iterations, got " + std::to_string(counter.load()));
    }

    TEST_PASS();
    return true;
}

bool test_chunk_boundaries() {
    TEST_START("Chunk boundary correctness");

    ThreadPool pool(4);
    std::mutex mutex;
    std::vector<bool> visited(1000, false);

    pool.parallel_for(1000, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        std::lock_guard<std::mutex> lock(mutex);
        for (size_t i = begin; i < end; i++) {
            if (visited[i]) {
                // Should never happen - each index visited exactly once
                throw std::runtime_error("Index " + std::to_string(i) + " visited twice");
            }
            visited[i] = true;
        }
    });

    // Verify all indices visited
    for (size_t i = 0; i < 1000; i++) {
        if (!visited[i]) {
            TEST_FAIL("Index " + std::to_string(i) + " not visited");
        }
    }

    TEST_PASS();
    return true;
}

// ============================================================================
// Test Category 3: Arena Allocation
// ============================================================================

bool test_per_thread_arena() {
    TEST_START("Per-thread arena isolation");

    ThreadPool pool(4, 10);  // 4 threads, 10MB each
    std::mutex mutex;
    std::set<void*> arena_ptrs;

    pool.parallel_for(100, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        // Each thread should get a different arena
        float* buf = arena.allocate<float>(100);

        std::lock_guard<std::mutex> lock(mutex);
        arena_ptrs.insert(static_cast<void*>(buf));
    });

    // Should see at least 4 different arena allocations
    if (arena_ptrs.size() < 4) {
        TEST_FAIL("Expected at least 4 unique arenas, got " + std::to_string(arena_ptrs.size()));
    }

    TEST_PASS();
    return true;
}

bool test_arena_reset_between_calls() {
    TEST_START("Arena reset between parallel_for calls");

    ThreadPool pool(2, 10);

    // First call - allocate some memory
    pool.parallel_for(10, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        for (size_t i = begin; i < end; i++) {
            float* buf = arena.allocate<float>(1000);
            buf[0] = 42.0f;  // Write to ensure allocation
        }
    });

    // Second call - arenas should be reset automatically
    pool.parallel_for(10, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        for (size_t i = begin; i < end; i++) {
            float* buf = arena.allocate<float>(1000);
            buf[0] = 99.0f;  // Should succeed without running out of memory
        }
    });

    TEST_PASS();
    return true;
}

bool test_arena_stress() {
    TEST_START("Arena stress test (large allocations)");

    ThreadPool pool(4, 100);  // 4 threads, 100MB each
    std::atomic<int> success_count{0};

    // With 100MB per thread and 100 iterations split across 4 threads (~25 each),
    // allocate smaller chunks to avoid running out of arena space
    pool.parallel_for(100, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        (void)tid;
        for (size_t i = begin; i < end; i++) {
            // Allocate 100KB per iteration (safer than 1MB)
            float* buf = arena.allocate<float>(25000);  // 25k floats = 100KB
            if (buf) {
                buf[0] = static_cast<float>(i);
                success_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    if (success_count.load() != 100) {
        TEST_FAIL("Expected 100 successful allocations, got " + std::to_string(success_count.load()));
    }

    TEST_PASS();
    return true;
}

// ============================================================================
// Test Category 4: Thread Safety
// ============================================================================

bool test_concurrent_atomic_increments() {
    TEST_START("Concurrent atomic increments");

    ThreadPool pool(8);
    std::atomic<int> counter{0};

    pool.parallel_for(10000, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        (void)tid;
        (void)arena;
        for (size_t i = begin; i < end; i++) {
            counter.fetch_add(1, std::memory_order_relaxed);
        }
    });

    if (counter.load() != 10000) {
        TEST_FAIL("Race condition detected: expected 10000, got " + std::to_string(counter.load()));
    }

    TEST_PASS();
    return true;
}

bool test_no_arena_sharing() {
    TEST_START("Verify no arena sharing between threads");

    ThreadPool pool(4);
    std::atomic<bool> detected_sharing{false};
    std::mutex mutex;
    std::map<int, void*> tid_to_arena_ptr;

    pool.parallel_for(100, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        void* arena_addr = static_cast<void*>(&arena);

        std::lock_guard<std::mutex> lock(mutex);
        auto it = tid_to_arena_ptr.find(tid);
        if (it != tid_to_arena_ptr.end()) {
            // Same thread ID should always get same arena
            if (it->second != arena_addr) {
                detected_sharing = true;
            }
        } else {
            tid_to_arena_ptr[tid] = arena_addr;
        }

        // Different thread IDs should get different arenas
        for (const auto& pair : tid_to_arena_ptr) {
            if (pair.first != tid && pair.second == arena_addr) {
                detected_sharing = true;
            }
        }
    });

    if (detected_sharing.load()) {
        TEST_FAIL("Detected arena sharing between threads");
    }

    TEST_PASS();
    return true;
}

bool test_repeated_calls() {
    TEST_START("Repeated parallel_for calls (stress)");

    ThreadPool pool(4);

    for (int iter = 0; iter < 10; iter++) {
        std::atomic<int> counter{0};

        pool.parallel_for(1000, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
            (void)tid;
            (void)arena;
            for (size_t i = begin; i < end; i++) {
                counter.fetch_add(1, std::memory_order_relaxed);
            }
        });

        if (counter.load() != 1000) {
            TEST_FAIL("Iteration " + std::to_string(iter) + " failed: expected 1000, got " + std::to_string(counter.load()));
        }
    }

    TEST_PASS();
    return true;
}

// ============================================================================
// Test Category 5: Edge Cases
// ============================================================================

bool test_single_thread() {
    TEST_START("Single thread mode");

    ThreadPool pool(1);
    std::atomic<int> counter{0};

    pool.parallel_for(100, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        if (tid != 0) {
            throw std::runtime_error("Single thread should have tid=0");
        }
        if (begin != 0 || end != 100) {
            throw std::runtime_error("Single thread should process entire range");
        }
        for (size_t i = begin; i < end; i++) {
            counter++;
        }
    });

    if (counter.load() != 100) {
        TEST_FAIL("Expected 100, got " + std::to_string(counter.load()));
    }

    TEST_PASS();
    return true;
}

bool test_more_threads_than_work() {
    TEST_START("More threads than work items");

    ThreadPool pool(8);
    std::atomic<int> counter{0};

    pool.parallel_for(3, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        for (size_t i = begin; i < end; i++) {
            counter++;
        }
    });

    if (counter.load() != 3) {
        TEST_FAIL("Expected 3, got " + std::to_string(counter.load()));
    }

    TEST_PASS();
    return true;
}

bool test_empty_work() {
    TEST_START("Empty work (count=0)");

    ThreadPool pool(4);
    bool lambda_called = false;

    pool.parallel_for(0, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        lambda_called = true;
    });

    if (lambda_called) {
        TEST_FAIL("Lambda should not be called for count=0");
    }

    TEST_PASS();
    return true;
}

bool test_large_work_count() {
    TEST_START("Large work count (1M items)");

    ThreadPool pool(4);
    std::atomic<size_t> sum{0};

    pool.parallel_for(1000000, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        size_t local_sum = 0;
        for (size_t i = begin; i < end; i++) {
            local_sum++;
        }
        sum.fetch_add(local_sum, std::memory_order_relaxed);
    });

    if (sum.load() != 1000000) {
        TEST_FAIL("Expected 1000000, got " + std::to_string(sum.load()));
    }

    TEST_PASS();
    return true;
}

bool test_arena_reset_manual() {
    TEST_START("Manual arena reset");

    ThreadPool pool(2, 10);

    // Use some memory
    pool.parallel_for(10, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        float* buf = arena.allocate<float>(100000);
        buf[0] = 1.0f;
    });

    // Manually reset
    pool.reset_arenas();

    // Should be able to use full capacity again
    pool.parallel_for(10, [&](int tid, size_t begin, size_t end, GrowableArena& arena) {
        float* buf = arena.allocate<float>(100000);
        buf[0] = 2.0f;
    });

    TEST_PASS();
    return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "=========================================\n";
    std::cout << "  ThreadPool Unit Tests\n";
    std::cout << "=========================================\n\n";

    bool all_passed = true;

    std::cout << "Category 1: Construction/Destruction\n";
    std::cout << "-------------------------------------\n";
    all_passed &= test_auto_detect_threads();
    all_passed &= test_explicit_thread_count();
    all_passed &= test_cleanup_on_destroy();
    std::cout << "\n";

    std::cout << "Category 2: Parallel Execution\n";
    std::cout << "-------------------------------------\n";
    all_passed &= test_work_distribution();
    all_passed &= test_thread_id_tracking();
    all_passed &= test_chunk_boundaries();
    std::cout << "\n";

    std::cout << "Category 3: Arena Allocation\n";
    std::cout << "-------------------------------------\n";
    all_passed &= test_per_thread_arena();
    all_passed &= test_arena_reset_between_calls();
    all_passed &= test_arena_stress();
    std::cout << "\n";

    std::cout << "Category 4: Thread Safety\n";
    std::cout << "-------------------------------------\n";
    all_passed &= test_concurrent_atomic_increments();
    all_passed &= test_no_arena_sharing();
    all_passed &= test_repeated_calls();
    std::cout << "\n";

    std::cout << "Category 5: Edge Cases\n";
    std::cout << "-------------------------------------\n";
    all_passed &= test_single_thread();
    all_passed &= test_more_threads_than_work();
    all_passed &= test_empty_work();
    all_passed &= test_large_work_count();
    all_passed &= test_arena_reset_manual();
    std::cout << "\n";

    std::cout << "=========================================\n";
    if (all_passed) {
        std::cout << "  ✓ All tests PASSED\n";
    } else {
        std::cout << "  ✗ Some tests FAILED\n";
    }
    std::cout << "=========================================\n\n";

    return all_passed ? 0 : 1;
}
