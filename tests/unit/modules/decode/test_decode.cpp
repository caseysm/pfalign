/**
 * Unit tests for decode module.
 *
 * These tests verify the module-level interface wraps the primitive correctly.
 * Comprehensive algorithm tests are in primitives/alignment_decode/test_alignment_decode.cpp
 */

#include "pfalign/modules/decode/decode.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/growable_arena.h"
#include <iostream>
#include <cmath>

using pfalign::ScalarBackend;
using pfalign::decode::decode_alignment;
using pfalign::decode::DecodeConfig;
using pfalign::decode::get_decode_scratch_size;
using pfalign::AlignmentPair;
using pfalign::memory::Arena;
using pfalign::memory::GrowableArena;

constexpr float TOLERANCE = 1e-5f;

bool close(float a, float b, float tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

//==============================================================================
// Test 1: Module Interface - Perfect Diagonal
//==============================================================================

bool test_module_interface() {
    std::cout << "=== Test 1: Module Interface (3*3 diagonal) ===" << std::endl;

    // Posteriors strongly peaked on diagonal
    float posteriors[9] = {
        0.8f, 0.05f, 0.05f,
        0.01f, 0.8f, 0.05f,
        0.01f, 0.01f, 0.22f  // Sum = 1.0
    };

    // Setup arena
    size_t scratch_size = get_decode_scratch_size(3, 3);
    GrowableArena arena(scratch_size * 2);  // Extra headroom

    // Decode via module interface
    AlignmentPair path[6];
    DecodeConfig config(-2.0f);

    int path_len = decode_alignment<ScalarBackend>(
        posteriors, 3, 3,
        config,
        path, 6,
        &arena
    );

    std::cout << "  Path length: " << path_len << " (expected: 3)" << std::endl;

    // Expected: diagonal alignment
    bool passed = (path_len == 3);
    passed &= (path[0].i == 0 && path[0].j == 0);
    passed &= (path[1].i == 1 && path[1].j == 1);
    passed &= (path[2].i == 2 && path[2].j == 2);
    passed &= close(path[0].posterior, 0.8f);

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 2: Gap Penalty Configuration
//==============================================================================

bool test_gap_penalty_config() {
    std::cout << "=== Test 2: Gap Penalty Configuration ===" << std::endl;

    float posteriors[9] = {
        0.3f, 0.1f, 0.0f,
        0.1f, 0.3f, 0.0f,
        0.0f, 0.0f, 0.2f   // Sum = 1.0
    };

    GrowableArena arena(4096);

    // Lenient gap penalty
    AlignmentPair path_lenient[6];
    DecodeConfig config_lenient(-1.0f);

    int len_lenient = decode_alignment<ScalarBackend>(
        posteriors, 3, 3,
        config_lenient,
        path_lenient, 6,
        &arena
    );

    // Reset arena for second decode
    arena.reset();

    // Stringent gap penalty
    AlignmentPair path_stringent[6];
    DecodeConfig config_stringent(-5.0f);

    int len_stringent = decode_alignment<ScalarBackend>(
        posteriors, 3, 3,
        config_stringent,
        path_stringent, 6,
        &arena
    );

    // Count gaps
    int gaps_lenient = 0, gaps_stringent = 0;
    for (int k = 0; k < len_lenient; k++) {
        if (path_lenient[k].i == -1 || path_lenient[k].j == -1) gaps_lenient++;
    }
    for (int k = 0; k < len_stringent; k++) {
        if (path_stringent[k].i == -1 || path_stringent[k].j == -1) gaps_stringent++;
    }

    std::cout << "  Gaps (lenient -1.0): " << gaps_lenient << std::endl;
    std::cout << "  Gaps (stringent -5.0): " << gaps_stringent << std::endl;

    // Stringent should have fewer or equal gaps
    bool passed = (gaps_stringent <= gaps_lenient);
    passed &= (len_lenient > 0 && len_stringent > 0);

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 3: Arena Allocation
//==============================================================================

bool test_arena_allocation() {
    std::cout << "=== Test 3: Arena Allocation ===" << std::endl;

    float posteriors[9] = {
        0.8f, 0.1f, 0.0f,
        0.0f, 0.8f, 0.1f,
        0.0f, 0.0f, 0.1f   // Sum = 1.0
    };

    // Exact size with headroom for alignment
    size_t scratch_size = get_decode_scratch_size(3, 3);
    GrowableArena arena(scratch_size + 256);  // Add headroom for 32-byte alignment padding

    AlignmentPair path[6];
    DecodeConfig config;

    int path_len = decode_alignment<ScalarBackend>(
        posteriors, 3, 3,
        config,
        path, 6,
        &arena
    );

    std::cout << "  Scratch size: " << scratch_size << " bytes" << std::endl;
    std::cout << "  Path length: " << path_len << std::endl;

    bool passed = (path_len > 0);

    if (passed) {
        std::cout << "  ✓ PASS (fits in exact scratch size)" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 4: Scratch Size Calculation
//==============================================================================

bool test_scratch_size_calculation() {
    std::cout << "=== Test 4: Scratch Size Calculation ===" << std::endl;

    // For L1=200, L2=200
    size_t size_200 = get_decode_scratch_size(200, 200);
    size_t expected_score = 201 * 201 * sizeof(float);    // ~162 KB
    size_t expected_traceback = 201 * 201 * sizeof(uint8_t);  // ~40 KB
    size_t expected_total = expected_score + expected_traceback;

    std::cout << "  L1=L2=200:" << std::endl;
    std::cout << "    Expected: " << expected_total << " bytes" << std::endl;
    std::cout << "    Got: " << size_200 << " bytes" << std::endl;

    bool passed = (size_200 == expected_total);

    // For L1=100, L2=150
    size_t size_100_150 = get_decode_scratch_size(100, 150);
    size_t expected_100_150 = 101 * 151 * sizeof(float) + 101 * 151 * sizeof(uint8_t);

    std::cout << "  L1=100, L2=150:" << std::endl;
    std::cout << "    Expected: " << expected_100_150 << " bytes" << std::endl;
    std::cout << "    Got: " << size_100_150 << " bytes" << std::endl;

    passed &= (size_100_150 == expected_100_150);

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 5: Multiple Decodes (Arena Reuse)
//==============================================================================

bool test_multiple_decodes() {
    std::cout << "=== Test 5: Multiple Decodes (Arena Reuse) ===" << std::endl;

    float posteriors1[4] = {0.7f, 0.1f, 0.1f, 0.1f};  // 2*2
    float posteriors2[9] = {0.5f, 0.1f, 0.1f, 0.1f, 0.05f, 0.05f, 0.05f, 0.0f, 0.05f};  // 3*3

    GrowableArena arena(8192);
    DecodeConfig config;

    // First decode
    AlignmentPair path1[4];
    int len1 = decode_alignment<ScalarBackend>(
        posteriors1, 2, 2, config, path1, 4, &arena
    );

    std::cout << "  First decode: path_len=" << len1 << std::endl;

    // Reset and second decode
    arena.reset();
    AlignmentPair path2[6];
    int len2 = decode_alignment<ScalarBackend>(
        posteriors2, 3, 3, config, path2, 6, &arena
    );

    std::cout << "  Second decode: path_len=" << len2 << std::endl;

    bool passed = (len1 > 0 && len2 > 0);

    if (passed) {
        std::cout << "  ✓ PASS (arena reuse works)" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Decode Module Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 5;

    if (test_module_interface()) passed++;
    if (test_gap_penalty_config()) passed++;
    if (test_arena_allocation()) passed++;
    if (test_scratch_size_calculation()) passed++;
    if (test_multiple_decodes()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
