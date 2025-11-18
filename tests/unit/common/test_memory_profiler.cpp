/**
 * Unit tests for MemoryProfiler.
 *
 * Tests:
 * - Basic recording (arena, workspace, cache, threadpool)
 * - Peak calculation
 * - Summary printing
 * - CSV export
 * - Thread safety (concurrent recording)
 * - Null profiler (zero overhead)
 */

#include "pfalign/common/memory_profiler.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>
#include <cassert>
#include <cmath>

using namespace pfalign::memory;

// Helper to check approximate equality
bool approx_equal(double a, double b, double eps = 0.1) {
    return std::fabs(a - b) < eps;
}

int main() {
    int test_num = 0;

    // Test 1: Basic arena recording
    {
        test_num++;
        std::cout << "Test " << test_num << ": Record arena" << std::endl;

        MemoryProfiler profiler("test_session");

        profiler.record_arena("main_arena", 100 * 1024 * 1024, 200 * 1024 * 1024);
        profiler.record_arena("thread_0", 50 * 1024 * 1024, 100 * 1024 * 1024);

        assert(profiler.num_components() == 2);
        assert(approx_equal(profiler.total_peak_mb(), 150.0));

        std::cout << "  ✓ Arena recording works" << std::endl;
    }

    // Test 2: Workspace recording
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Record workspace" << std::endl;

        MemoryProfiler profiler("test_session");

        profiler.record_workspace("msa_workspace", 128 * 1024 * 1024, 256 * 1024 * 1024);

        assert(profiler.num_components() == 1);
        assert(approx_equal(profiler.total_peak_mb(), 128.0));

        std::cout << "  ✓ Workspace recording works" << std::endl;
    }

    // Test 3: Cache recording
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Record cache" << std::endl;

        MemoryProfiler profiler("test_session");

        profiler.record_cache("sequence_cache", 64 * 1024 * 1024, 10);

        assert(profiler.num_components() == 1);
        assert(approx_equal(profiler.total_peak_mb(), 64.0));

        std::cout << "  ✓ Cache recording works" << std::endl;
    }

    // Test 4: Threadpool recording
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Record threadpool" << std::endl;

        MemoryProfiler profiler("test_session");

        profiler.record_threadpool("distance_matrix_pool", 8, 256.0);

        assert(profiler.num_components() == 1);
        assert(approx_equal(profiler.total_peak_mb(), 256.0));

        std::cout << "  ✓ Threadpool recording works" << std::endl;
    }

    // Test 5: Multiple component types
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Multiple components" << std::endl;

        MemoryProfiler profiler("msa_n100");

        profiler.record_arena("main_arena", 512 * 1024 * 1024, 1024 * 1024 * 1024);
        profiler.record_workspace("msa_workspace", 128 * 1024 * 1024, 256 * 1024 * 1024);
        profiler.record_cache("sequence_cache", 64 * 1024 * 1024, 100);
        profiler.record_threadpool("distance_matrix_pool", 8, 256.0);

        assert(profiler.num_components() == 4);
        // Total: 512 + 128 + 64 + 256 = 960 MB
        assert(approx_equal(profiler.total_peak_mb(), 960.0));

        std::cout << "  ✓ Multiple components work" << std::endl;
    }

    // Test 6: Elapsed time
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Elapsed time" << std::endl;

        MemoryProfiler profiler("test_session");

        // Sleep for a short time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        double elapsed = profiler.elapsed_seconds();
        assert(elapsed >= 0.1);  // At least 100ms
        assert(elapsed < 1.0);   // Less than 1 second

        std::cout << "  ✓ Elapsed time tracking works (" << elapsed << "s)" << std::endl;
    }

    // Test 7: CSV export
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": CSV export" << std::endl;

        MemoryProfiler profiler("test_session");

        profiler.record_arena("main_arena", 100 * 1024 * 1024, 200 * 1024 * 1024);
        profiler.record_workspace("msa_workspace", 50 * 1024 * 1024, 100 * 1024 * 1024);

        const char* filepath = "/tmp/test_memory_profile.csv";
        assert(profiler.export_csv(filepath));

        // Read CSV and verify format
        std::ifstream file(filepath);
        assert(file.is_open());

        std::string header;
        std::getline(file, header);
        assert(header == "session,type,name,peak_mb,capacity_mb,utilization_pct,elapsed_sec");

        // Count data rows
        int row_count = 0;
        std::string line;
        while (std::getline(file, line)) {
            row_count++;
            assert(!line.empty());
            // Check that line contains "test_session"
            assert(line.find("test_session") != std::string::npos);
        }

        assert(row_count == 2);  // 2 components recorded

        file.close();

        std::cout << "  ✓ CSV export works" << std::endl;
    }

    // Test 8: Print summary (just verify it doesn't crash)
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Print summary" << std::endl;

        MemoryProfiler profiler("test_session");

        profiler.record_arena("main_arena", 100 * 1024 * 1024, 200 * 1024 * 1024);
        profiler.record_workspace("msa_workspace", 50 * 1024 * 1024, 100 * 1024 * 1024);
        profiler.record_cache("sequence_cache", 32 * 1024 * 1024, 50);
        profiler.record_threadpool("distance_matrix_pool", 8, 128.0);

        // Print summary (verify it doesn't crash)
        profiler.print_summary();

        std::cout << "  ✓ Print summary works" << std::endl;
    }

    // Test 9: Thread safety (concurrent recording)
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Thread safety" << std::endl;

        MemoryProfiler profiler("concurrent_test");

        const int num_threads = 10;
        std::vector<std::thread> threads;

        for (int i = 0; i < num_threads; i++) {
            threads.emplace_back([&profiler, i]() {
                std::string name = "thread_" + std::to_string(i);
                profiler.record_arena(name, 10 * 1024 * 1024, 20 * 1024 * 1024);
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        assert(profiler.num_components() == num_threads);
        // Total: 10 threads * 10 MB = 100 MB
        assert(approx_equal(profiler.total_peak_mb(), 100.0));

        std::cout << "  ✓ Thread safety works" << std::endl;
    }

    // Test 10: MemorySnapshot utility methods
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": MemorySnapshot" << std::endl;

        MemorySnapshot snapshot(100 * 1024 * 1024, 200 * 1024 * 1024, 150 * 1024 * 1024);

        assert(approx_equal(snapshot.used_mb(), 100.0));
        assert(approx_equal(snapshot.capacity_mb(), 200.0));
        assert(approx_equal(snapshot.peak_mb(), 150.0));
        assert(approx_equal(snapshot.utilization(), 75.0));  // 150/200 = 75%

        std::cout << "  ✓ MemorySnapshot works" << std::endl;
    }

    // Test 11: NullMemoryProfiler (zero overhead)
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": NullProfiler" << std::endl;

        NullMemoryProfiler profiler("test_session");

        // All methods should be no-ops
        profiler.record_arena("main_arena", 100 * 1024 * 1024, 200 * 1024 * 1024);
        profiler.record_workspace("msa_workspace", 50 * 1024 * 1024, 100 * 1024 * 1024);
        profiler.record_cache("sequence_cache", 32 * 1024 * 1024, 50);
        profiler.record_threadpool("distance_matrix_pool", 8, 128.0);

        assert(profiler.num_components() == 0);
        assert(profiler.total_peak_mb() == 0.0);
        assert(profiler.elapsed_seconds() == 0.0);

        // These should not crash
        profiler.print_summary();
        assert(!profiler.export_csv("/tmp/null_profile.csv"));

        std::cout << "  ✓ NullProfiler works" << std::endl;
    }

    // Test 12: Empty profiler
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Empty profiler" << std::endl;

        MemoryProfiler profiler("empty_session");

        assert(profiler.num_components() == 0);
        assert(profiler.total_peak_bytes() == 0);
        assert(profiler.total_peak_mb() == 0.0);

        // Should not crash
        profiler.print_summary();

        const char* filepath = "/tmp/empty_profile.csv";
        assert(profiler.export_csv(filepath));

        // Verify CSV has header but no data rows
        std::ifstream file(filepath);
        assert(file.is_open());

        std::string header;
        std::getline(file, header);
        assert(header == "session,type,name,peak_mb,capacity_mb,utilization_pct,elapsed_sec");

        // No data rows
        std::string line;
        assert(!std::getline(file, line));

        file.close();

        std::cout << "  ✓ Empty profiler works" << std::endl;
    }

    // Test 13: Integration test - Simulate MSA profiling workflow
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": MSA workflow simulation" << std::endl;

        MemoryProfiler profiler("msa_n50_upgma");

        // Phase 1: MPNN encoding (cache + thread pool)
        profiler.record_cache("sequence_cache", 128 * 1024 * 1024, 50);
        profiler.record_threadpool("mpnn_pool", 8, 256.0);

        // Phase 2: Distance matrix (thread pool)
        profiler.record_threadpool("distance_matrix_pool", 8, 128.0);

        // Phase 3: Guide tree (arena)
        profiler.record_arena("guide_tree_arena", 32 * 1024 * 1024, 64 * 1024 * 1024);

        // Phase 4: Progressive MSA (workspace + arena + thread pool)
        profiler.record_workspace("msa_workspace", 512 * 1024 * 1024, 1024 * 1024 * 1024);
        profiler.record_arena("main_arena", 256 * 1024 * 1024, 512 * 1024 * 1024);
        profiler.record_threadpool("profile_align_pool", 4, 64.0);

        assert(profiler.num_components() == 7);

        // Total: 128 + 256 + 128 + 32 + 512 + 256 + 64 = 1376 MB
        assert(approx_equal(profiler.total_peak_mb(), 1376.0));

        // Export CSV for analysis
        const char* filepath = "/tmp/msa_workflow_profile.csv";
        assert(profiler.export_csv(filepath));

        // Print summary
        profiler.print_summary();

        std::cout << "  ✓ MSA workflow simulation works" << std::endl;
    }

    // Test 14: Session name retrieval
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Session name" << std::endl;

        MemoryProfiler profiler("my_test_session");
        assert(profiler.session_name() == "my_test_session");

        std::cout << "  ✓ Session name retrieval works" << std::endl;
    }

    // Test 15: Benchmark - Verify low overhead of recording
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Recording overhead benchmark" << std::endl;

        MemoryProfiler profiler("benchmark");

        auto start = std::chrono::steady_clock::now();

        // Record 1000 components
        for (int i = 0; i < 1000; i++) {
            std::string name = "component_" + std::to_string(i);
            profiler.record_arena(name, 1024 * 1024, 2 * 1024 * 1024);
        }

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Recording 1000 components should take < 10ms on modern hardware
        assert(duration.count() < 10000);  // 10ms = 10,000 microseconds

        assert(profiler.num_components() == 1000);

        std::cout << "  ✓ Recording overhead is low (" << duration.count() << " µs for 1000 records)" << std::endl;
    }

    std::cout << "\n✅ All " << test_num << " tests passed!\n" << std::endl;

    return 0;
}
