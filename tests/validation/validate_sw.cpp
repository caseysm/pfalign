/**
 * Smith-Waterman validation against JAX reference.
 *
 * Reads similarity matrices from files and outputs partition functions
 * for comparison with JAX implementation.
 */

#include "pfalign/primitives/smith_waterman/smith_waterman.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/common/golden_data_test.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/perf_timer.h"
#include "pfalign/tools/weights/save_npy.h"
#include "../test_utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <cmath>

using pfalign::ScalarBackend;
using pfalign::smith_waterman::smith_waterman_jax_regular;
using pfalign::smith_waterman::smith_waterman_jax_affine;
using pfalign::smith_waterman::smith_waterman_jax_affine_flexible;
using pfalign::smith_waterman::smith_waterman_jax_regular_backward;
using pfalign::smith_waterman::smith_waterman_jax_affine_flexible_backward;
using pfalign::smith_waterman::smith_waterman_direct_regular;
using pfalign::smith_waterman::smith_waterman_direct_affine;
using pfalign::smith_waterman::smith_waterman_direct_affine_flexible;
using pfalign::smith_waterman::smith_waterman_direct_regular_backward;
using pfalign::smith_waterman::smith_waterman_direct_affine_backward;
using pfalign::smith_waterman::smith_waterman_direct_affine_flexible_backward;
using pfalign::smith_waterman::SWConfig;

namespace {

enum class SWMode {
    JaxRegular,
    JaxAffine,
    JaxAffineFlexible,
    DirectRegular,
    DirectAffine,
    DirectAffineFlexible
};

struct GoldenCase {
    std::string name;
    SWMode mode;
};

SWConfig make_config(SWMode mode) {
    SWConfig config;
    config.temperature = 1.0f;
    switch (mode) {
    case SWMode::DirectRegular:
    case SWMode::JaxRegular:
        config.affine = false;
        config.gap = -0.1f;
        break;
    default:
        config.affine = true;
        config.gap_open = -0.5f;
        config.gap_extend = -0.1f;
        break;
    }
    return config;
}

std::filesystem::path smith_waterman_golden_root() {
    return std::filesystem::path(pfalign::test::get_validation_path("smith_waterman"));
}

bool approximately_equal(float a, float b, float atol = 1e-4f) {
    return std::fabs(a - b) <= atol;
}

bool is_direct_mode(SWMode mode) {
    return mode == SWMode::DirectRegular ||
           mode == SWMode::DirectAffine ||
           mode == SWMode::DirectAffineFlexible;
}

bool run_golden_case(
    const GoldenCase& info,
    const std::filesystem::path& dir,
    bool rewrite_direct
) {
    pfalign::testing::GoldenDataTest test(dir.string());
    auto [similarity, sim_shape] = test.load_with_shape("input_similarity.npy");
    auto [posterior_ref, posterior_shape] = test.load_with_shape("output_posterior.npy");
    auto partition_ref = test.load("output_partition.npy");

    if (sim_shape.size() != 2) {
        std::cerr << "  ✗ Invalid similarity shape\n";
        return false;
    }

    const int L1 = static_cast<int>(sim_shape[0]);
    const int L2 = static_cast<int>(sim_shape[1]);
    if (posterior_ref.size() != static_cast<size_t>(L1) * L2) {
        std::cerr << "  ✗ Posterior shape mismatch\n";
        return false;
    }
    if (partition_ref.empty()) {
        std::cerr << "  ✗ Partition reference missing\n";
        return false;
    }

    SWConfig config = make_config(info.mode);
    float partition = 0.0f;
    pfalign::memory::Arena arena(4);

    auto compare_outputs = [&](const std::vector<float>& posterior) {
        float max_diff = 0.0f;
        for (size_t i = 0; i < posterior.size(); ++i) {
            max_diff = std::max(max_diff, std::fabs(posterior[i] - posterior_ref[i]));
        }
        std::cout << "  partition diff: " << std::fabs(partition - partition_ref[0]) << "\n";
        std::cout << "  posterior max abs diff: " << max_diff << "\n";
        return approximately_equal(partition, partition_ref[0]) && max_diff <= 1e-4f;
    };

    switch (info.mode) {
    case SWMode::JaxRegular: {
        std::vector<float> hij(static_cast<size_t>(L1) * L2);
        smith_waterman_jax_regular<ScalarBackend>(
            similarity.data(), L1, L2, config, hij.data(), &partition
        );

        std::vector<float> posterior(static_cast<size_t>(L1) * L2, 0.0f);
        smith_waterman_jax_regular_backward<ScalarBackend>(
            hij.data(),
            similarity.data(),
            L1,
            L2,
            config,
            partition,
            posterior.data(),
            &arena
        );
        return compare_outputs(posterior);
    }
    case SWMode::JaxAffine: {
        std::vector<float> hij(static_cast<size_t>(L1) * L2 * 3);
        smith_waterman_jax_affine<ScalarBackend>(
            similarity.data(), L1, L2, config, hij.data(), &partition
        );

        std::vector<float> posterior(static_cast<size_t>(L1) * L2, 0.0f);
        smith_waterman_jax_affine_flexible_backward<ScalarBackend>(
            hij.data(),
            similarity.data(),
            L1,
            L2,
            config,
            partition,
            posterior.data(),
            &arena
        );
        return compare_outputs(posterior);
    }
    case SWMode::JaxAffineFlexible: {
        std::vector<float> hij(static_cast<size_t>(L1) * L2 * 3);
        smith_waterman_jax_affine_flexible<ScalarBackend>(
            similarity.data(), L1, L2, config, hij.data(), &partition
        );

        std::vector<float> posterior(static_cast<size_t>(L1) * L2, 0.0f);
        smith_waterman_jax_affine_flexible_backward<ScalarBackend>(
            hij.data(),
            similarity.data(),
            L1,
            L2,
            config,
            partition,
            posterior.data(),
            &arena
        );
        return compare_outputs(posterior);
    }
    case SWMode::DirectRegular: {
        std::vector<float> alpha(static_cast<size_t>(L1 + 1) * (L2 + 1));
        smith_waterman_direct_regular<ScalarBackend>(
            similarity.data(), L1, L2, config, alpha.data(), &partition
        );
        std::vector<float> posterior(static_cast<size_t>(L1) * L2, 0.0f);
        smith_waterman_direct_regular_backward<ScalarBackend>(
            alpha.data(),
            similarity.data(),
            L1,
            L2,
            config,
            partition,
            posterior.data(),
            &arena
        );
        if (rewrite_direct) {
            save_npy_float32((dir / "output_posterior.npy").string(), posterior.data(), {static_cast<size_t>(L1), static_cast<size_t>(L2)});
            save_npy_float32((dir / "output_partition.npy").string(), &partition, {});
            std::cout << "  ✎ Updated direct golden tensors\n";
            return true;
        }
        return compare_outputs(posterior);
    }
    case SWMode::DirectAffine: {
        std::vector<float> alpha(static_cast<size_t>(L1 + 1) * (L2 + 1) * 3);
        smith_waterman_direct_affine<ScalarBackend>(
            similarity.data(), L1, L2, config, alpha.data(), &partition
        );
        std::vector<float> posterior(static_cast<size_t>(L1) * L2, 0.0f);
        smith_waterman_direct_affine_backward<ScalarBackend>(
            alpha.data(),
            similarity.data(),
            L1,
            L2,
            config,
            partition,
            posterior.data(),
            &arena
        );
        if (rewrite_direct) {
            save_npy_float32((dir / "output_posterior.npy").string(), posterior.data(), {static_cast<size_t>(L1), static_cast<size_t>(L2)});
            save_npy_float32((dir / "output_partition.npy").string(), &partition, {});
            std::cout << "  ✎ Updated direct golden tensors\n";
            return true;
        }
        return compare_outputs(posterior);
    }
    case SWMode::DirectAffineFlexible: {
        std::vector<float> alpha(static_cast<size_t>(L1 + 1) * (L2 + 1) * 3);
        smith_waterman_direct_affine_flexible<ScalarBackend>(
            similarity.data(), L1, L2, config, alpha.data(), &partition
        );
        std::vector<float> posterior(static_cast<size_t>(L1) * L2, 0.0f);
        smith_waterman_direct_affine_flexible_backward<ScalarBackend>(
            alpha.data(),
            similarity.data(),
            L1,
            L2,
            config,
            partition,
            posterior.data(),
            &arena
        );
        if (rewrite_direct) {
            save_npy_float32((dir / "output_posterior.npy").string(), posterior.data(), {static_cast<size_t>(L1), static_cast<size_t>(L2)});
            save_npy_float32((dir / "output_partition.npy").string(), &partition, {});
            std::cout << "  ✎ Updated direct golden tensors\n";
            return true;
        }
        return compare_outputs(posterior);
    }
    }
    return false;
}

int run_golden_suite(bool rewrite_direct) {
    const std::vector<GoldenCase> cases = {
        {"regular_jax_small_8x10", SWMode::JaxRegular},
        {"regular_jax_medium_20x25", SWMode::JaxRegular},
        {"regular_jax_large_50x60", SWMode::JaxRegular},
        {"affine_jax_small_8x10", SWMode::JaxAffineFlexible},
        {"affine_jax_medium_20x25", SWMode::JaxAffineFlexible},
        {"affine_jax_large_50x60", SWMode::JaxAffineFlexible},
        {"regular_direct_small_8x10", SWMode::DirectRegular},
        {"regular_direct_medium_20x25", SWMode::DirectRegular},
        {"regular_direct_large_50x60", SWMode::DirectRegular},
        {"affine_direct_small_8x10", SWMode::DirectAffine},
        {"affine_direct_medium_20x25", SWMode::DirectAffine},
        {"affine_direct_large_50x60", SWMode::DirectAffine},
    };

    auto root = smith_waterman_golden_root();
    bool any_case = false;
    bool all_passed = true;

    for (const auto& info : cases) {
        auto dir = root / info.name;
        if (!std::filesystem::exists(dir / "input_similarity.npy")) {
            std::cout << "SKIP: Missing " << dir << std::endl;
            continue;
        }

        any_case = true;
        std::cout << "\n=== Golden Case: " << info.name << " ===" << std::endl;
        if (!run_golden_case(info, dir, rewrite_direct)) {
            all_passed = false;
            std::cout << "  ✗ FAIL" << std::endl;
        } else {
            std::cout << "  ✓ PASS" << std::endl;
        }
    }

    if (!any_case) {
        std::cerr << "No Smith-Waterman golden data found at " << root << std::endl;
        return 1;
    }

    return all_passed ? 0 : 1;
}

}  // namespace

/**
 * Read matrix from text file (space-separated values).
 */
bool read_matrix(const char* filename, std::vector<float>& data, int& rows, int& cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }

    // Read dimensions
    file >> rows >> cols;
    data.resize(rows * cols);

    // Read data
    for (int i = 0; i < rows * cols; i++) {
        file >> data[i];
    }

    return true;
}

/**
 * Write partition function to file.
 */
void write_result(const char* filename, float partition) {
    std::ofstream file(filename);
    file << std::setprecision(10) << partition << std::endl;
}

int main(int argc, char** argv) {
    pfalign::perf::PerfTimer perf_timer("validate_sw");
    bool rewrite_direct = false;
    int arg_index = 1;
    while (arg_index < argc && std::string(argv[arg_index]) == "--rewrite-direct") {
        rewrite_direct = true;
        ++arg_index;
    }

    if (arg_index == argc) {
        return run_golden_suite(rewrite_direct);
    }

    if (argc - arg_index < 3) {
        std::cerr << "Usage: " << argv[0] << " <similarity_matrix.txt> <output.txt> <mode>" << std::endl;
        std::cerr << "  mode: jax_regular | jax_affine | jax_affine_flexible" << std::endl;
        return 1;
    }

    const char* input_file = argv[arg_index];
    const char* output_file = argv[arg_index + 1];
    const char* mode = argv[arg_index + 2];

    // Read similarity matrix
    std::vector<float> similarity;
    int L1, L2;
    if (!read_matrix(input_file, similarity, L1, L2)) {
        return 1;
    }

    std::cout << "Loaded similarity matrix: " << L1 << " * " << L2 << std::endl;

    // Configure Smith-Waterman
    SWConfig config;
    config.temperature = 1.0f;

    if (std::string(mode).find("affine") != std::string::npos) {
        config.gap_open = -0.5f;
        config.gap_extend = -0.1f;
    } else {
        config.gap = -0.2f;
    }

    // Allocate workspace
    float partition = 0.0f;

    if (std::string(mode) == "jax_regular") {
        std::vector<float> hij(L1 * L2);
        smith_waterman_jax_regular<ScalarBackend>(
            similarity.data(), L1, L2, config, hij.data(), &partition
        );
    } else if (std::string(mode) == "jax_affine") {
        std::vector<float> hij(L1 * L2 * 3);
        smith_waterman_jax_affine<ScalarBackend>(
            similarity.data(), L1, L2, config, hij.data(), &partition
        );
    } else if (std::string(mode) == "jax_affine_flexible") {
        std::vector<float> hij(L1 * L2 * 3);
        smith_waterman_jax_affine_flexible<ScalarBackend>(
            similarity.data(), L1, L2, config, hij.data(), &partition
        );
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }

    std::cout << "Partition function: " << std::setprecision(10) << partition << std::endl;

    // Write result
    write_result(output_file, partition);

    return 0;
}
