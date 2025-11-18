#pragma once

#include <string>
#include <filesystem>

/**
 * Test utilities for accessing test data
 *
 * All test data is organized under tests/data/:
 * - fixtures/       : Minimal test data (4 essential structures + weights)
 * - integration/    : Integration test data (MSA families, parity tests, structures)
 * - validation/     : Golden reference data (end-to-end, metrics)
 * - benchmarks/     : Performance test data (small/medium/large/huge)
 */

namespace pfalign::test {

/**
 * Get absolute path to test data directory
 *
 * Uses TEST_DATA_DIR from CMake if available, otherwise computes from __FILE__
 */
inline std::filesystem::path get_test_data_dir() {
#ifdef TEST_DATA_DIR
    return std::filesystem::path(TEST_DATA_DIR);
#else
    // Fallback: compute from source location
    auto path = std::filesystem::path(__FILE__).parent_path();
    return path / "data";
#endif
}

/**
 * Get path to fixture file (minimal test data)
 *
 * Examples:
 *   get_fixture_path("structures/crambin.pdb")
 *   get_fixture_path("weights/v1_mpnn_weights.safetensors")
 */
inline std::string get_fixture_path(const std::string& relative_path) {
    return (get_test_data_dir() / "fixtures" / relative_path).string();
}

/**
 * Get path to integration test file
 *
 * Examples:
 *   get_integration_path("msa_families/globin/1HBS.pdb")
 *   get_integration_path("structures/medium/1MBO.pdb")
 *   get_integration_path("structures/predicted/P00519.pdb")
 *   get_integration_path("parity/test_dhfr.npy")
 */
inline std::string get_integration_path(const std::string& relative_path) {
    return (get_test_data_dir() / "integration" / relative_path).string();
}

/**
 * Get path to validation golden data
 *
 * Examples:
 *   get_validation_path("end_to_end/case_001")
 *   get_validation_path("metrics/lddt_test_1.npy")
 */
inline std::string get_validation_path(const std::string& relative_path) {
    return (get_test_data_dir() / "validation" / relative_path).string();
}

/**
 * Get path to benchmark file
 *
 * Examples:
 *   get_benchmark_path("small/small_30res_d2hy6a1.ent")
 *   get_benchmark_path("huge/huge_1008res_d2hqia_.ent")
 */
inline std::string get_benchmark_path(const std::string& relative_path) {
    return (get_test_data_dir() / "benchmarks" / relative_path).string();
}

/**
 * Legacy path mapping for backwards compatibility
 *
 * Maps old data/structures/pdb paths to new test data organization
 */
inline std::string map_legacy_path(const std::string& old_path) {
    // Common fixture files (renamed for clarity)
    if (old_path.find("1CRN.pdb") != std::string::npos) {
        return get_fixture_path("structures/crambin.pdb");
    }
    if (old_path.find("1UBQ.pdb") != std::string::npos) {
        return get_fixture_path("structures/ubiquitin.pdb");
    }
    if (old_path.find("1LYZ.pdb") != std::string::npos) {
        return get_fixture_path("structures/lysozyme.pdb");
    }
    if (old_path.find("2VIL.pdb") != std::string::npos) {
        return get_fixture_path("structures/villin.pdb");
    }

    // Medium structures (now in integration)
    if (old_path.find("1MBO.pdb") != std::string::npos) {
        return get_integration_path("structures/medium/1MBO.pdb");
    }
    if (old_path.find("1HBS.pdb") != std::string::npos) {
        return get_integration_path("structures/medium/1HBS.pdb");
    }
    if (old_path.find("1MBA.pdb") != std::string::npos) {
        return get_integration_path("structures/medium/1MBA.pdb");
    }
    if (old_path.find("1MYT.pdb") != std::string::npos) {
        return get_integration_path("structures/medium/1MYT.pdb");
    }
    if (old_path.find("1IGY.pdb") != std::string::npos) {
        return get_integration_path("structures/medium/1IGY.pdb");
    }
    if (old_path.find("1RNH.pdb") != std::string::npos) {
        return get_integration_path("structures/medium/1RNH.pdb");
    }
    if (old_path.find("1REX.pdb") != std::string::npos) {
        return get_integration_path("structures/medium/1REX.pdb");
    }
    if (old_path.find("2LYZ.pdb") != std::string::npos) {
        return get_integration_path("structures/medium/2LYZ.pdb");
    }
    if (old_path.find("2NRL.pdb") != std::string::npos) {
        return get_integration_path("structures/medium/2NRL.pdb");
    }

    // Large structures
    if (old_path.find("1FBI.pdb") != std::string::npos) {
        return get_integration_path("structures/large/1FBI.pdb");
    }
    if (old_path.find("1IGT.pdb") != std::string::npos) {
        return get_integration_path("structures/large/1IGT.pdb");
    }
    if (old_path.find("4CHA.pdb") != std::string::npos) {
        return get_integration_path("structures/large/4CHA.pdb");
    }

    // Predicted structures (AFDB and ESM Atlas)
    if (old_path.find("P00519.pdb") != std::string::npos) {
        return get_integration_path("structures/predicted/P00519.pdb");
    }
    if (old_path.find("P69905.pdb") != std::string::npos) {
        return get_integration_path("structures/predicted/P69905.pdb");
    }
    if (old_path.find("MGYP003592128331.pdb") != std::string::npos) {
        return get_integration_path("structures/predicted/MGYP003592128331.pdb");
    }
    if (old_path.find("MGYP002802792805.pdb") != std::string::npos) {
        return get_integration_path("structures/predicted/MGYP002802792805.pdb");
    }
    if (old_path.find("MGYP001105483357.pdb") != std::string::npos) {
        return get_integration_path("structures/predicted/MGYP001105483357.pdb");
    }

    // If no mapping found, return original path (file may not exist)
    return old_path;
}

}  // namespace pfalign::test
