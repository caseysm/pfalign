/**
 * Edge Case Validation Suite
 *
 * Exercises pathological alignment scenarios to ensure the full pipeline
 * (MPNN → Similarity → Smith-Waterman) remains numerically stable.
 *
 * Cases covered:
 * 1. Very short sequences (L <= 10)
 * 2. Very long sequences (L ~= 1000)
 * 3. Extreme length ratios (50:1)
 * 4. Identical sequences (should resemble self-match behaviour)
 * 5. Minimal sequences (L = 1–2)
 * 6. Numerical edge cases (temperature and gap extremes)
 */

#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/modules/similarity/similarity.h"
#include "pfalign/primitives/smith_waterman/smith_waterman.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/common/arena_allocator.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

using namespace pfalign;
using namespace pfalign::mpnn;
using namespace pfalign::smith_waterman;
using namespace pfalign::memory;
using namespace pfalign::weights;

namespace {

constexpr size_t kArenaBytes = size_t(1024) * 1024 * 1024;  // 1 GB for large tests

MPNNWeights g_weights(3);
MPNNConfig g_config;
SWConfig g_sw_config;
std::unique_ptr<Arena> g_arena;
bool g_weights_loaded = false;

struct AlignmentResult {
    float partition = 0.0f;
    float diagonal_similarity_mean = 0.0f;
    float diagonal_posterior_mean = 0.0f;
    bool similarity_valid = false;
    bool posteriors_valid = false;
    int L1 = 0;
    int L2 = 0;
};

std::vector<float> generate_synthetic_coords(int L, float phase = 0.0f) {
    std::vector<float> coords(L * 12, 0.0f);  // 4 atoms * 3 coords
    const float radius = 5.0f;

    for (int i = 0; i < L; ++i) {
        float t = phase + static_cast<float>(i) * 0.45f;
        float x = radius * std::cos(t);
        float y = radius * std::sin(t);
        float z = 1.5f * static_cast<float>(i);

        for (int atom = 0; atom < 4; ++atom) {
            float offset = 0.25f * static_cast<float>(atom);
            coords[i * 12 + atom * 3 + 0] = x + offset;
            coords[i * 12 + atom * 3 + 1] = y - offset * 0.5f;
            coords[i * 12 + atom * 3 + 2] = z + offset * 0.1f;
        }
    }

    return coords;
}

bool has_invalid_values(const float* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (!std::isfinite(data[i])) {
            return true;
        }
    }
    return false;
}

float* encode_sequence(
    const std::vector<float>& coords,
    const MPNNWeights& weights,
    const MPNNConfig& config,
    Arena* arena
) {
    int L = static_cast<int>(coords.size() / 12);
    float* embeddings = arena->allocate<float>(L * config.hidden_dim);

    MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim);
    for (int i = 0; i < L; ++i) {
        workspace.residue_idx[i] = i;
        workspace.chain_labels[i] = 0;
    }

    mpnn_forward<ScalarBackend>(
        coords.data(),
        L,
        weights,
        config,
        embeddings,
        &workspace
    );

    return embeddings;
}

AlignmentResult run_alignment_case(
    const std::vector<float>& coords_a,
    const std::vector<float>& coords_b,
    const SWConfig& sw_override
) {
    AlignmentResult result;

    if (!g_weights_loaded || !g_arena) {
        return result;
    }

    Arena* arena = g_arena.get();
    arena->reset();

    const int L1 = static_cast<int>(coords_a.size() / 12);
    const int L2 = static_cast<int>(coords_b.size() / 12);

    result.L1 = L1;
    result.L2 = L2;

    float* embed_a = encode_sequence(coords_a, g_weights, g_config, arena);
    float* embed_b = encode_sequence(coords_b, g_weights, g_config, arena);

    float* similarity = arena->allocate<float>(static_cast<size_t>(L1) * static_cast<size_t>(L2));

    pfalign::similarity::compute_similarity<ScalarBackend>(
        embed_a,
        embed_b,
        similarity,
        L1,
        L2,
        g_config.hidden_dim
    );

    result.similarity_valid = !has_invalid_values(
        similarity,
        static_cast<size_t>(L1) * static_cast<size_t>(L2)
    );

    const int diag = std::min(L1, L2);
    if (diag > 0) {
        double diag_sum = 0.0;
        for (int i = 0; i < diag; ++i) {
            diag_sum += similarity[i * L2 + i];
        }
        result.diagonal_similarity_mean = static_cast<float>(diag_sum / diag);
    }

    if (L1 < 2 || L2 < 2) {
        SWConfig regular_config = sw_override;
        regular_config.affine = false;
        if (regular_config.gap >= 0.0f) {
            regular_config.gap = regular_config.gap_open != 0.0f
                ? std::min(regular_config.gap_open, -0.01f)
                : -0.1f;
        }

        float* alpha = arena->allocate<float>(static_cast<size_t>(L1 + 1) * static_cast<size_t>(L2 + 1));
        float partition = 0.0f;
        smith_waterman_direct_regular<ScalarBackend>(
            similarity,
            L1,
            L2,
            regular_config,
            alpha,
            &partition
        );

        result.partition = partition;

        float* posteriors = arena->allocate<float>(static_cast<size_t>(L1) * static_cast<size_t>(L2));
        smith_waterman_direct_regular_backward<ScalarBackend>(
            alpha,
            similarity,
            L1,
            L2,
            regular_config,
            partition,
            posteriors,
            arena
        );

        result.posteriors_valid = !has_invalid_values(
            posteriors,
            static_cast<size_t>(L1) * static_cast<size_t>(L2)
        );

        if (diag > 0) {
            double diag_post = 0.0;
            for (int i = 0; i < diag; ++i) {
                diag_post += posteriors[i * L2 + i];
            }
            result.diagonal_posterior_mean = static_cast<float>(diag_post / diag);
        }

        return result;
    }

    float* dp = arena->allocate<float>(
        static_cast<size_t>(L1) * static_cast<size_t>(L2) * 3
    );

    float partition = 0.0f;
    smith_waterman_jax_affine_flexible<ScalarBackend>(
        similarity,
        L1,
        L2,
        sw_override,
        dp,
        &partition
    );
    result.partition = partition;

    float* posteriors = arena->allocate<float>(static_cast<size_t>(L1) * static_cast<size_t>(L2));
    smith_waterman_jax_affine_flexible_backward<ScalarBackend>(
        dp,
        similarity,
        L1,
        L2,
        sw_override,
        partition,
        posteriors,
        arena
    );

    result.posteriors_valid = !has_invalid_values(
        posteriors,
        static_cast<size_t>(L1) * static_cast<size_t>(L2)
    );

    if (diag > 0) {
        double diag_post = 0.0;
        for (int i = 0; i < diag; ++i) {
            diag_post += posteriors[i * L2 + i];
        }
        result.diagonal_posterior_mean = static_cast<float>(diag_post / diag);
    }

    return result;
}

bool check_finite(float value) {
    return std::isfinite(value) && !std::isnan(value);
}

bool test_very_short_sequences() {
    std::cout << "\n[EdgeCase] Very Short Sequences (L=5 vs L=10)" << std::endl;
    auto coords_short = generate_synthetic_coords(5);
    auto coords_long = generate_synthetic_coords(10, 0.35f);

    AlignmentResult result = run_alignment_case(coords_short, coords_long, g_sw_config);

    std::cout << "  Partition: " << result.partition << std::endl;
    std::cout << "  Diagonal similarity mean: " << result.diagonal_similarity_mean << std::endl;
    std::cout << "  Diagonal posterior mean: " << result.diagonal_posterior_mean << std::endl;

    bool passed = result.similarity_valid &&
                  result.posteriors_valid &&
                  check_finite(result.partition) &&
                  result.partition > 0.01f;

    std::cout << (passed ? "  ✓ PASS" : "  ✗ FAIL") << std::endl;
    return passed;
}

bool test_very_long_sequences() {
    std::cout << "\n[EdgeCase] Very Long Sequences (L=1000 vs L=1200)" << std::endl;
    auto coords_a = generate_synthetic_coords(1000);
    auto coords_b = generate_synthetic_coords(1200, 0.7f);

    AlignmentResult result = run_alignment_case(coords_a, coords_b, g_sw_config);

    std::cout << "  Partition: " << result.partition << std::endl;
    std::cout << "  Diagonal similarity mean: " << result.diagonal_similarity_mean << std::endl;
    std::cout << "  Diagonal posterior mean: " << result.diagonal_posterior_mean << std::endl;

    bool passed = result.similarity_valid &&
                  result.posteriors_valid &&
                  check_finite(result.partition) &&
                  result.partition > 0.0f;

    std::cout << (passed ? "  ✓ PASS" : "  ✗ FAIL") << std::endl;
    return passed;
}

bool test_extreme_length_ratios() {
    std::cout << "\n[EdgeCase] Extreme Length Ratios (L=10 vs L=500)" << std::endl;
    auto short_coords = generate_synthetic_coords(10, 0.1f);
    auto long_coords = generate_synthetic_coords(500, 0.45f);

    AlignmentResult result = run_alignment_case(short_coords, long_coords, g_sw_config);

    std::cout << "  Partition: " << result.partition << std::endl;
    std::cout << "  Diagonal similarity mean: " << result.diagonal_similarity_mean << std::endl;

    bool passed = result.similarity_valid &&
                  result.posteriors_valid &&
                  check_finite(result.partition) &&
                  result.partition > 0.0f &&
                  result.diagonal_similarity_mean > 0.1f;

    std::cout << (passed ? "  ✓ PASS" : "  ✗ FAIL") << std::endl;
    return passed;
}

bool test_identical_sequences() {
    std::cout << "\n[EdgeCase] Identical Synthetic Sequences (L=150)" << std::endl;
    auto coords = generate_synthetic_coords(150, 0.2f);

    AlignmentResult result = run_alignment_case(coords, coords, g_sw_config);

    std::cout << "  Partition: " << result.partition << std::endl;
    std::cout << "  Diagonal posterior mean: " << result.diagonal_posterior_mean << std::endl;

    bool passed = result.similarity_valid &&
                  result.posteriors_valid &&
                  check_finite(result.partition) &&
                  result.diagonal_posterior_mean > 0.05f;

    std::cout << (passed ? "  ✓ PASS" : "  ✗ FAIL") << std::endl;
    return passed;
}

bool test_minimal_sequences() {
    std::cout << "\n[EdgeCase] Minimal Sequences (L=1 vs L=2)" << std::endl;
    auto coords_a = generate_synthetic_coords(1);
    auto coords_b = generate_synthetic_coords(2, 0.5f);

    AlignmentResult result = run_alignment_case(coords_a, coords_b, g_sw_config);

    std::cout << "  Partition: " << result.partition << std::endl;
    std::cout << "  Valid similarity: " << (result.similarity_valid ? "yes" : "no") << std::endl;

    bool passed = result.similarity_valid &&
                  result.posteriors_valid &&
                  check_finite(result.partition);

    std::cout << (passed ? "  ✓ PASS" : "  ✗ FAIL") << std::endl;
    return passed;
}

bool test_numerical_edge_cases() {
    std::cout << "\n[EdgeCase] Numerical Parameters (temperature/gap extremes)" << std::endl;

    auto coords_a = generate_synthetic_coords(64, 0.15f);
    auto coords_b = generate_synthetic_coords(64, 0.65f);

    std::vector<SWConfig> configs;

    SWConfig low_temp = g_sw_config;
    low_temp.temperature = 0.01f;
    configs.push_back(low_temp);

    SWConfig high_temp = g_sw_config;
    high_temp.temperature = 100.0f;
    configs.push_back(high_temp);

    SWConfig aggressive_gaps = g_sw_config;
    aggressive_gaps.gap_open = -100.0f;
    aggressive_gaps.gap_extend = -50.0f;
    configs.push_back(aggressive_gaps);

    SWConfig shallow_gaps = g_sw_config;
    shallow_gaps.gap_open = -0.01f;
    shallow_gaps.gap_extend = -0.001f;
    configs.push_back(shallow_gaps);

    bool passed = true;
    int idx = 0;

    for (const auto& cfg : configs) {
        AlignmentResult result = run_alignment_case(coords_a, coords_b, cfg);
        std::cout << "  Config " << idx++
                  << " (temp=" << cfg.temperature
                  << ", gap_open=" << cfg.gap_open
                  << ", gap_extend=" << cfg.gap_extend << "): "
                  << result.partition << std::endl;

        bool ok = result.similarity_valid &&
                  result.posteriors_valid &&
                  check_finite(result.partition);

        if (!ok) {
            passed = false;
        }
    }

    std::cout << (passed ? "  ✓ PASS" : "  ✗ FAIL") << std::endl;
    return passed;
}

}  // namespace

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Edge Case Validation Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        auto [loaded_weights, loaded_config, sw_defaults] = load_embedded_mpnn_weights();
        g_weights = std::move(loaded_weights);
        g_config = loaded_config;
        auto make_negative = [](float value) {
            return value > 0.0f ? -value : value;
        };
        g_sw_config.gap = make_negative(sw_defaults.gap);
        g_sw_config.gap_open = make_negative(sw_defaults.gap_open);
        g_sw_config.gap_extend = make_negative(sw_defaults.gap);
        g_sw_config.temperature = sw_defaults.temperature;
        g_sw_config.affine = true;
        g_arena = std::make_unique<Arena>(kArenaBytes);
        g_weights_loaded = true;
        std::cout << "✅ Loaded embedded MPNN weights (hidden_dim="
                  << g_config.hidden_dim << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "SKIP: Could not load embedded MPNN weights (" << e.what() << ")" << std::endl;
        return 0;  // Not a failure; embedded weights may be optional
    }

    const bool results[] = {
        test_very_short_sequences(),
        test_very_long_sequences(),
        test_extreme_length_ratios(),
        test_identical_sequences(),
        test_minimal_sequences(),
        test_numerical_edge_cases()
    };

    const int total_tests = static_cast<int>(sizeof(results) / sizeof(results[0]));
    int passed = 0;
    for (bool ok : results) {
        if (ok) {
            passed++;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << " / " << total_tests << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total_tests) ? 0 : 1;
}
