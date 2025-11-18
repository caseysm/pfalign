/**
 * Similarity computation validation against JAX reference.
 *
 * Reads embeddings from files and computes similarity matrix.
 */

#include "pfalign/modules/similarity/similarity.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/common/golden_data_test.h"
#include "pfalign/common/perf_timer.h"
#include "../test_utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <algorithm>

using pfalign::ScalarBackend;
using pfalign::similarity::compute_similarity;

namespace {

std::filesystem::path similarity_golden_root() {
    return std::filesystem::path(pfalign::test::get_validation_path("similarity"));
}

bool run_similarity_case(const std::filesystem::path& dir) {
    pfalign::testing::GoldenDataTest test(dir.string());
    auto [emb1, emb1_shape] = test.load_with_shape("input_emb1.npy");
    auto [emb2, emb2_shape] = test.load_with_shape("input_emb2.npy");
    auto [expected, sim_shape] = test.load_with_shape("output_similarity.npy");

    if (emb1_shape.size() != 2 || emb2_shape.size() != 2 || sim_shape.size() != 2) {
        std::cerr << "  ✗ Invalid array shapes" << std::endl;
        return false;
    }

    const int L1 = static_cast<int>(emb1_shape[0]);
    const int D1 = static_cast<int>(emb1_shape[1]);
    const int L2 = static_cast<int>(emb2_shape[0]);
    const int D2 = static_cast<int>(emb2_shape[1]);

    if (D1 != D2) {
        std::cerr << "  ✗ Embedding dims mismatch: " << D1 << " vs " << D2 << std::endl;
        return false;
    }
    if (sim_shape[0] != static_cast<size_t>(L1) || sim_shape[1] != static_cast<size_t>(L2)) {
        std::cerr << "  ✗ Similarity shape mismatch" << std::endl;
        return false;
    }

    std::vector<float> actual(static_cast<size_t>(L1) * L2, 0.0f);
    compute_similarity<ScalarBackend>(
        emb1.data(),
        emb2.data(),
        actual.data(),
        L1,
        L2,
        D1
    );

    float max_diff = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(actual[i] - expected[i]));
    }

    std::cout << "  Max abs diff: " << max_diff << std::endl;
    return max_diff <= 1e-5f;
}

int run_golden_similarity_suite() {
    auto root = similarity_golden_root();
    if (!std::filesystem::exists(root)) {
        std::cerr << "No similarity golden data at " << root << std::endl;
        return 1;
    }

    std::vector<std::filesystem::path> cases;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (!entry.is_directory()) {
            continue;
        }
        if (std::filesystem::exists(entry.path() / "input_emb1.npy")) {
            cases.push_back(entry.path());
        }
    }

    if (cases.empty()) {
        std::cerr << "No similarity golden cases found in " << root << std::endl;
        return 1;
    }

    std::sort(cases.begin(), cases.end());
    bool all_passed = true;
    for (const auto& dir : cases) {
        std::cout << "\n=== Similarity Case: " << dir.filename().string() << " ===" << std::endl;
        if (!run_similarity_case(dir)) {
            all_passed = false;
            std::cout << "  ✗ FAIL" << std::endl;
        } else {
            std::cout << "  ✓ PASS" << std::endl;
        }
    }

    return all_passed ? 0 : 1;
}

}  // namespace

/**
 * Read embeddings from text file.
 * Format: L D
 *         embedding data (L * D values)
 */
bool read_embeddings(const char* filename, std::vector<float>& data, int& L, int& D) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }

    file >> L >> D;
    data.resize(L * D);

    for (int i = 0; i < L * D; i++) {
        file >> data[i];
    }

    return true;
}

/**
 * Write similarity matrix to file.
 */
bool write_matrix(const char* filename, const float* data, int L1, int L2) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return false;
    }

    file << std::setprecision(10);
    file << L1 << " " << L2 << "\n";
    for (int i = 0; i < L1; i++) {
        for (int j = 0; j < L2; j++) {
            file << data[i * L2 + j] << " ";
        }
        file << "\n";
    }

    return true;
}

int main(int argc, char** argv) {
    pfalign::perf::PerfTimer perf_timer("validate_similarity");
    if (argc == 1) {
        return run_golden_similarity_suite();
    }

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <emb1.txt> <emb2.txt> <output.txt>" << std::endl;
        return 1;
    }

    const char* emb1_file = argv[1];
    const char* emb2_file = argv[2];
    const char* output_file = argv[3];

    // Read embeddings
    std::vector<float> emb1, emb2;
    int L1, D1, L2, D2;

    if (!read_embeddings(emb1_file, emb1, L1, D1)) return 1;
    if (!read_embeddings(emb2_file, emb2, L2, D2)) return 1;

    if (D1 != D2) {
        std::cerr << "Embedding dimensions don't match: " << D1 << " vs " << D2 << std::endl;
        return 1;
    }

    std::cout << "Embedding 1: " << L1 << " * " << D1 << std::endl;
    std::cout << "Embedding 2: " << L2 << " * " << D2 << std::endl;

    // Compute similarity
    std::vector<float> similarity(L1 * L2);
    compute_similarity<ScalarBackend>(
        emb1.data(),
        emb2.data(),
        similarity.data(),
        L1,
        L2,
        D1
    );

    std::cout << "Similarity matrix: " << L1 << " * " << L2 << std::endl;

    // Write result
    if (!write_matrix(output_file, similarity.data(), L1, L2)) {
        return 1;
    }

    std::cout << "✓ Similarity matrix written to " << output_file << std::endl;

    return 0;
}
