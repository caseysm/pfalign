/**
 * Test suite for profiling infrastructure
 *
 * Tests the profiling.h header including:
 * - ScopedTimer RAII timing
 * - Manual start/stop timing
 * - TimingRecord statistics
 * - Output formatters (console, JSON, CSV)
 */

#include "pfalign/common/profiling.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <thread>
#include <chrono>
#include <sstream>
#include <fstream>

using namespace pfalign::profiling;

#ifdef ENABLE_PROFILING

// Helper to simulate work
void simulate_work(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

// Test 1: Basic ScopedTimer functionality
void test_scoped_timer() {
    std::cout << "Test 1: ScopedTimer basic functionality...\n";

    Profiler::instance().reset();

    {
        ScopedTimer timer("test_section");
        simulate_work(10);
    }

    const auto& records = Profiler::instance().get_records();
    assert(records.size() == 1);
    assert(records.count("test_section") == 1);

    const auto& record = records.at("test_section");
    assert(record.count == 1);
    assert(record.total_ms >= 9.0 && record.total_ms <= 20.0);  // Allow some slack
    assert(record.min_ms == record.max_ms);  // Single sample
    assert(std::abs(record.avg_ms() - record.total_ms) < 0.001);

    std::cout << "  ✓ ScopedTimer test passed\n";
}

// Test 2: Manual start/stop timing
void test_manual_timing() {
    std::cout << "Test 2: Manual start/stop timing...\n";

    Profiler::instance().reset();

    Profiler::instance().start("manual_section");
    simulate_work(15);
    Profiler::instance().stop("manual_section");

    const auto& records = Profiler::instance().get_records();
    assert(records.size() == 1);

    const auto& record = records.at("manual_section");
    assert(record.count == 1);
    assert(record.total_ms >= 14.0 && record.total_ms <= 25.0);

    std::cout << "  ✓ Manual timing test passed\n";
}

// Test 3: Multiple samples for same section
void test_multiple_samples() {
    std::cout << "Test 3: Multiple samples...\n";

    Profiler::instance().reset();

    for (int i = 0; i < 5; i++) {
        ScopedTimer timer("repeated_section");
        simulate_work(5);
    }

    const auto& records = Profiler::instance().get_records();
    assert(records.size() == 1);

    const auto& record = records.at("repeated_section");
    assert(record.count == 5);
    assert(record.total_ms >= 24.0 && record.total_ms <= 40.0);  // ~25ms total
    assert(record.avg_ms() >= 4.0 && record.avg_ms() <= 10.0);   // ~5ms avg
    assert(record.min_ms <= record.avg_ms());
    assert(record.max_ms >= record.avg_ms());

    std::cout << "  ✓ Multiple samples test passed\n";
}

// Test 4: Multiple sections
void test_multiple_sections() {
    std::cout << "Test 4: Multiple sections...\n";

    Profiler::instance().reset();

    {
        ScopedTimer timer("section_a");
        simulate_work(10);
    }

    {
        ScopedTimer timer("section_b");
        simulate_work(5);
    }

    {
        ScopedTimer timer("section_c");
        simulate_work(8);
    }

    const auto& records = Profiler::instance().get_records();
    assert(records.size() == 3);
    assert(records.count("section_a") == 1);
    assert(records.count("section_b") == 1);
    assert(records.count("section_c") == 1);

    std::cout << "  ✓ Multiple sections test passed\n";
}

// Test 5: Nested timing (only outer timer should be recorded)
void test_nested_timing() {
    std::cout << "Test 5: Nested timing...\n";

    Profiler::instance().reset();

    {
        ScopedTimer outer("outer");
        simulate_work(5);
        {
            ScopedTimer inner("inner");
            simulate_work(10);
        }
        simulate_work(5);
    }

    const auto& records = Profiler::instance().get_records();
    assert(records.size() == 2);

    const auto& outer_rec = records.at("outer");
    const auto& inner_rec = records.at("inner");

    // Outer should include inner time
    assert(outer_rec.total_ms >= 19.0 && outer_rec.total_ms <= 35.0);  // ~20ms
    assert(inner_rec.total_ms >= 9.0 && inner_rec.total_ms <= 20.0);   // ~10ms

    std::cout << "  ✓ Nested timing test passed\n";
}

// Test 6: Console output format
void test_console_output() {
    std::cout << "Test 6: Console output format...\n";

    Profiler::instance().reset();

    {
        ScopedTimer timer("fast_section");
        simulate_work(5);
    }

    {
        ScopedTimer timer("slow_section");
        simulate_work(15);
    }

    // Capture output to string
    std::ostringstream oss;
    Profiler::instance().print_report(oss);
    std::string output = oss.str();

    // Check for expected content
    assert(output.find("PROFILING REPORT") != std::string::npos);
    assert(output.find("fast_section") != std::string::npos);
    assert(output.find("slow_section") != std::string::npos);
    assert(output.find("Count") != std::string::npos);
    assert(output.find("Total(ms)") != std::string::npos);
    assert(output.find("TOTAL") != std::string::npos);

    std::cout << "  ✓ Console output test passed\n";
}

// Test 7: JSON output format
void test_json_output() {
    std::cout << "Test 7: JSON output format...\n";

    Profiler::instance().reset();

    {
        ScopedTimer timer("json_test_section");
        simulate_work(10);
    }

    const char* json_path = "/tmp/test_profiling.json";
    bool success = Profiler::instance().write_json(json_path);
    assert(success);

    // Read and validate JSON content
    std::ifstream file(json_path);
    assert(file.is_open());

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    assert(content.find("\"timestamp\"") != std::string::npos);
    assert(content.find("\"total_ms\"") != std::string::npos);
    assert(content.find("\"sections\"") != std::string::npos);
    assert(content.find("\"json_test_section\"") != std::string::npos);
    assert(content.find("\"count\"") != std::string::npos);
    assert(content.find("\"avg_ms\"") != std::string::npos);

    std::cout << "  ✓ JSON output test passed\n";
}

// Test 8: CSV output format
void test_csv_output() {
    std::cout << "Test 8: CSV output format...\n";

    Profiler::instance().reset();

    {
        ScopedTimer timer("csv_test_section");
        simulate_work(10);
    }

    const char* csv_path = "/tmp/test_profiling.csv";
    bool success = Profiler::instance().write_csv(csv_path);
    assert(success);

    // Read and validate CSV content
    std::ifstream file(csv_path);
    assert(file.is_open());

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    assert(content.find("Section,Count,Total_ms,Avg_ms") != std::string::npos);
    assert(content.find("csv_test_section") != std::string::npos);

    // Count lines (should be header + 1 data row)
    int line_count = 0;
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
        line_count++;
    }
    assert(line_count == 2);  // Header + 1 section

    std::cout << "  ✓ CSV output test passed\n";
}

// Test 9: Reset functionality
void test_reset() {
    std::cout << "Test 9: Reset functionality...\n";

    Profiler::instance().reset();

    {
        ScopedTimer timer("test_section");
        simulate_work(5);
    }

    assert(Profiler::instance().get_records().size() == 1);

    Profiler::instance().reset();

    assert(Profiler::instance().get_records().size() == 0);

    std::cout << "  ✓ Reset test passed\n";
}

// Test 10: Macro convenience functions
void test_macros() {
    std::cout << "Test 10: Macro convenience functions...\n";

    PROFILE_RESET();

    {
        PROFILE_SCOPE("macro_scoped");
        simulate_work(5);
    }

    PROFILE_START("macro_manual");
    simulate_work(10);
    PROFILE_STOP("macro_manual");

    const auto& records = Profiler::instance().get_records();
    assert(records.size() == 2);
    assert(records.count("macro_scoped") == 1);
    assert(records.count("macro_manual") == 1);

    std::cout << "  ✓ Macro test passed\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Profiling Infrastructure Tests\n";
    std::cout << "========================================\n\n";

    test_scoped_timer();
    test_manual_timing();
    test_multiple_samples();
    test_multiple_sections();
    test_nested_timing();
    test_console_output();
    test_json_output();
    test_csv_output();
    test_reset();
    test_macros();

    std::cout << "\n========================================\n";
    std::cout << "  All tests passed! ✓\n";
    std::cout << "========================================\n";

    return 0;
}

#else  // ENABLE_PROFILING not defined

int main() {
    std::cout << "========================================\n";
    std::cout << "  Profiling Infrastructure Tests\n";
    std::cout << "========================================\n\n";
    std::cout << "⚠️  ENABLE_PROFILING not defined\n";
    std::cout << "Profiling is disabled. Rebuild with -DENABLE_PROFILING=ON\n";
    std::cout << "\n========================================\n";

    return 0;
}

#endif  // ENABLE_PROFILING
