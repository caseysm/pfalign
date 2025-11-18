/**
 * End-to-end pipeline golden test.
 *
 * Re-runs the full alignment pipeline (PDB → MPNN → similarity → SW)
 * for every directory under data/golden/end_to_end and compares the
 * posterior alignment matrices plus summary metrics against the saved
 * golden references.
 *
 * This keeps the CLI binary honest without depending on filesystem paths.
 */

#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/modules/similarity/similarity.h"
#include "pfalign/primitives/smith_waterman/smith_waterman.h"
#include "pfalign/io/pdb_parser.h"
#include "pfalign/common/golden_data_test.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/common/perf_timer.h"
#include "../test_utils.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <iomanip>

using namespace pfalign;
using namespace pfalign::mpnn;
using namespace pfalign::similarity;
using namespace pfalign::smith_waterman;
using namespace pfalign::io;
using namespace pfalign::testing;

namespace {

struct CaseMetadata {
    std::string name;
    std::string pdb1_rel;
    std::string pdb2_rel;
    std::string sw_mode = "jax_affine_flexible";
    float gap_extend = 0.194247f;
    float gap_open = -2.54418f;
    float temperature = 1.0f;
    float partition = 0.0f;
    float alignment_score = 0.0f;
    int L1 = 0;
    int L2 = 0;
};


std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open metadata: " + path.string());
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

std::string extract_string(const std::string& blob, const std::string& key) {
    std::string token = "\"" + key + "\"";
    auto pos = blob.find(token);
    if (pos == std::string::npos) {
        throw std::runtime_error("Missing key in metadata: " + key);
    }
    pos = blob.find(':', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("Malformed metadata for key: " + key);
    }
    auto first_quote = blob.find('"', pos + 1);
    auto second_quote = blob.find('"', first_quote + 1);
    if (first_quote == std::string::npos || second_quote == std::string::npos) {
        throw std::runtime_error("Expected quoted string for key: " + key);
    }
    return blob.substr(first_quote + 1, second_quote - first_quote - 1);
}

double extract_number(const std::string& blob, const std::string& key) {
    std::string token = "\"" + key + "\"";
    auto pos = blob.find(token);
    if (pos == std::string::npos) {
        throw std::runtime_error("Missing key in metadata: " + key);
    }
    pos = blob.find(':', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("Malformed metadata for key: " + key);
    }
    auto start = blob.find_first_of("-0123456789", pos + 1);
    if (start == std::string::npos) {
        throw std::runtime_error("Expected numeric value for key: " + key);
    }
    auto end = start;
    while (end < blob.size()) {
        char c = blob[end];
        if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+')) {
            break;
        }
        ++end;
    }
    return std::stod(blob.substr(start, end - start));
}

CaseMetadata load_case_metadata(const std::filesystem::path& metadata_path) {
    CaseMetadata meta;
    const std::string blob = read_file(metadata_path);
    meta.name = extract_string(blob, "case");
    meta.pdb1_rel = extract_string(blob, "pdb1");
    meta.pdb2_rel = extract_string(blob, "pdb2");
    meta.sw_mode = extract_string(blob, "sw_mode");
    meta.gap_extend = static_cast<float>(extract_number(blob, "gap_extend"));
    meta.gap_open = static_cast<float>(extract_number(blob, "gap_open"));
    meta.temperature = static_cast<float>(extract_number(blob, "temperature"));
    meta.partition = static_cast<float>(extract_number(blob, "partition"));
    meta.alignment_score = static_cast<float>(extract_number(blob, "alignment_score"));
    meta.L1 = static_cast<int>(extract_number(blob, "l1"));
    meta.L2 = static_cast<int>(extract_number(blob, "l2"));
    return meta;
}

struct PipelineResult {
    std::vector<float> posteriors;
    float partition = 0.0f;
    float alignment_score = 0.0f;
    int L1 = 0;
    int L2 = 0;
};

float compute_alignment_score(
    const std::vector<float>& emb1,
    const std::vector<float>& emb2,
    const std::vector<float>& similarity,
    const std::vector<float>& posteriors,
    int L1,
    int L2,
    int hidden_dim
) {
    std::vector<float> norms1(L1, 0.0f);
    std::vector<float> norms2(L2, 0.0f);

    for (int i = 0; i < L1; ++i) {
        float norm_sq = 0.0f;
        for (int d = 0; d < hidden_dim; ++d) {
            float val = emb1[i * hidden_dim + d];
            norm_sq += val * val;
        }
        norms1[i] = std::sqrt(norm_sq);
    }

    for (int j = 0; j < L2; ++j) {
        float norm_sq = 0.0f;
        for (int d = 0; d < hidden_dim; ++d) {
            float val = emb2[j * hidden_dim + d];
            norm_sq += val * val;
        }
        norms2[j] = std::sqrt(norm_sq);
    }

    float score = 0.0f;
    float posterior_sum = 0.0f;
    for (int i = 0; i < L1; ++i) {
        for (int j = 0; j < L2; ++j) {
            int idx = i * L2 + j;
            float norm_product = norms1[i] * norms2[j];
            float cosine = (norm_product > 1e-10f) ? (similarity[idx] / norm_product) : 0.0f;
            score += posteriors[idx] * cosine;
            posterior_sum += posteriors[idx];
        }
    }

    if (posterior_sum > 0.0f) {
        score /= posterior_sum;
    }
    return score;
}

PipelineResult run_pipeline(const CaseMetadata& meta, const std::filesystem::path& root) {
    auto pdb1_path = root / meta.pdb1_rel;
    auto pdb2_path = root / meta.pdb2_rel;

    PDBParser parser;
    Protein prot1 = parser.parse_file(pdb1_path.string());
    Protein prot2 = parser.parse_file(pdb2_path.string());

    int L1 = prot1.get_chain(0).size();
    int L2 = prot2.get_chain(0).size();

    auto coords1 = prot1.get_backbone_coords(0);
    auto coords2 = prot2.get_backbone_coords(0);

    auto [weights, config, sw_params] = weights::load_embedded_mpnn_weights();
    sw_params.gap = meta.gap_extend;
    sw_params.gap_open = meta.gap_open;
    sw_params.temperature = meta.temperature;

    std::vector<float> emb1(L1 * config.hidden_dim);
    std::vector<float> emb2(L2 * config.hidden_dim);
    MPNNWorkspace workspace1(L1, config.k_neighbors, config.hidden_dim);
    MPNNWorkspace workspace2(L2, config.k_neighbors, config.hidden_dim);

    mpnn_forward<ScalarBackend>(
        coords1.data(), L1, weights, config, emb1.data(), &workspace1
    );
    mpnn_forward<ScalarBackend>(
        coords2.data(), L2, weights, config, emb2.data(), &workspace2
    );

    std::vector<float> similarity_matrix(L1 * L2);
    similarity::compute_similarity<ScalarBackend>(
        emb1.data(), emb2.data(), similarity_matrix.data(), L1, L2, config.hidden_dim
    );

    SWConfig sw_config;
    sw_config.gap = sw_params.gap;
    sw_config.gap_open = sw_params.gap_open;
    sw_config.gap_extend = sw_params.gap;
    sw_config.temperature = sw_params.temperature;

    std::vector<float> posteriors(L1 * L2);
    float partition = 0.0f;

    if (meta.sw_mode == "jax_affine_flexible") {
        std::vector<float> forward(L1 * L2 * 3);
        smith_waterman_jax_affine_flexible<ScalarBackend>(
            similarity_matrix.data(), L1, L2, sw_config, forward.data(), &partition
        );
        smith_waterman_jax_affine_flexible_backward<ScalarBackend>(
            forward.data(), similarity_matrix.data(), L1, L2, sw_config, partition, posteriors.data()
        );
    } else if (meta.sw_mode == "jax_regular") {
        std::vector<float> forward(L1 * L2);
        smith_waterman_jax_regular<ScalarBackend>(
            similarity_matrix.data(), L1, L2, sw_config, forward.data(), &partition
        );
        smith_waterman_jax_regular_backward<ScalarBackend>(
            forward.data(), similarity_matrix.data(), L1, L2, sw_config, partition, posteriors.data()
        );
    } else if (meta.sw_mode == "jax_affine") {
        std::vector<float> forward(L1 * L2 * 3);
        smith_waterman_jax_affine<ScalarBackend>(
            similarity_matrix.data(), L1, L2, sw_config, forward.data(), &partition
        );
        smith_waterman_jax_affine_backward<ScalarBackend>(
            forward.data(), similarity_matrix.data(), L1, L2, sw_config, partition, posteriors.data()
        );
    } else if (meta.sw_mode == "direct_regular") {
        std::vector<float> forward((L1 + 1) * (L2 + 1));
        smith_waterman_direct_regular<ScalarBackend>(
            similarity_matrix.data(), L1, L2, sw_config, forward.data(), &partition
        );
        smith_waterman_direct_regular_backward<ScalarBackend>(
            forward.data(),
            similarity_matrix.data(),
            L1,
            L2,
            sw_config,
            partition,
            posteriors.data(),
            nullptr
        );
    } else if (meta.sw_mode == "direct_affine") {
        std::vector<float> forward((L1 + 1) * (L2 + 1) * 3);
        smith_waterman_direct_affine<ScalarBackend>(
            similarity_matrix.data(), L1, L2, sw_config, forward.data(), &partition
        );
        smith_waterman_direct_affine_backward<ScalarBackend>(
            forward.data(),
            similarity_matrix.data(),
            L1,
            L2,
            sw_config,
            partition,
            posteriors.data(),
            nullptr
        );
    } else if (meta.sw_mode == "direct_affine_flexible") {
        std::vector<float> forward((L1 + 1) * (L2 + 1) * 3);
        smith_waterman_direct_affine_flexible<ScalarBackend>(
            similarity_matrix.data(), L1, L2, sw_config, forward.data(), &partition
        );
        smith_waterman_direct_affine_flexible_backward<ScalarBackend>(
            forward.data(),
            similarity_matrix.data(),
            L1,
            L2,
            sw_config,
            partition,
            posteriors.data(),
            nullptr
        );
    } else {
        throw std::runtime_error("Unsupported sw_mode in metadata: " + meta.sw_mode);
    }

    PipelineResult result;
    result.L1 = L1;
    result.L2 = L2;
    result.partition = partition;
    result.posteriors = std::move(posteriors);
    result.alignment_score = compute_alignment_score(
        emb1, emb2, similarity_matrix, result.posteriors, L1, L2, config.hidden_dim
    );
    return result;
}

bool compare_metadata(const CaseMetadata& meta, const PipelineResult& result) {
    constexpr float partition_tol = 5e-3f;
    constexpr float score_tol = 5e-4f;

    bool l_match = (meta.L1 == result.L1) && (meta.L2 == result.L2);
    bool partition_match = std::abs(result.partition - meta.partition) <= partition_tol;
    bool score_match = std::abs(result.alignment_score - meta.alignment_score) <= score_tol;

    if (!l_match) {
        std::cout << "  ✗ FAIL: Length mismatch (expected "
                  << meta.L1 << "x" << meta.L2 << ", got "
                  << result.L1 << "x" << result.L2 << ")\n";
    }
    if (!partition_match) {
        std::cout << "  ✗ FAIL: Partition mismatch (expected "
                  << meta.partition << ", got " << result.partition << ")\n";
    }
    if (!score_match) {
        std::cout << "  ✗ FAIL: Alignment score mismatch (expected "
                  << meta.alignment_score << ", got " << result.alignment_score << ")\n";
    }

    return l_match && partition_match && score_match;
}

bool run_case(const std::filesystem::path& case_dir, const std::filesystem::path& root) {
    std::cout << "\n========================================\n";
    std::cout << "Case: " << case_dir.filename().string() << "\n";
    std::cout << "========================================\n";

    CaseMetadata meta = load_case_metadata(case_dir / "metadata.json");
    PipelineResult result = run_pipeline(meta, root);

    GoldenDataTest golden(case_dir.string());
    auto [expected_posteriors, shape] = golden.load_with_shape("posteriors.npy");
    if (shape.size() != 2 || shape[0] != static_cast<size_t>(result.L1) || shape[1] != static_cast<size_t>(result.L2)) {
        std::cerr << "  ✗ FAIL: Unexpected posterior shape in golden data\n";
        return false;
    }

    golden.compare(
        "posteriors",
        expected_posteriors,
        result.posteriors,
        5e-4,
        5e-4
    );
    golden.print_summary();

    bool metadata_ok = compare_metadata(meta, result);
    bool tensors_ok = golden.all_passed();

    if (metadata_ok && tensors_ok) {
        std::cout << "  ✓ PASS: " << meta.name << "\n";
    } else {
        std::cout << "  ✗ FAIL: " << meta.name << "\n";
    }
    return metadata_ok && tensors_ok;
}

int run_benchmark_suite(
    const std::vector<std::filesystem::path>& cases,
    const std::filesystem::path& root
) {
    using clock = std::chrono::steady_clock;
    double total_time = 0.0;
    int processed = 0;
    int failures = 0;

    std::cout << "\n========================================\n";
    std::cout << "  End-to-End Benchmark Mode\n";
    std::cout << "========================================\n";

    for (const auto& case_dir : cases) {
        try {
            CaseMetadata meta = load_case_metadata(case_dir / "metadata.json");
            auto start = clock::now();
            PipelineResult result = run_pipeline(meta, root);
            auto elapsed = std::chrono::duration<double>(clock::now() - start).count();
            total_time += elapsed;
            processed++;
            std::cout << "Benchmark: " << meta.name
                      << " | L1=" << result.L1
                      << " L2=" << result.L2
                      << " | time=" << std::fixed << std::setprecision(3) << elapsed << "s\n";
        } catch (const std::exception& e) {
            failures++;
            std::cerr << "Benchmark FAIL for " << case_dir.filename().string()
                      << " (" << e.what() << ")\n";
        }
    }

    if (processed > 0) {
        std::cout << "----------------------------------------\n";
        std::cout << "Average time: "
                  << std::fixed << std::setprecision(3)
                  << (total_time / processed) << "s over "
                  << processed << " cases\n";
    }

    if (failures > 0) {
        std::cerr << failures << " benchmark case(s) failed.\n";
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    pfalign::perf::PerfTimer perf_timer("test_end_to_end_golden");
    bool benchmark_mode = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--benchmark") {
            benchmark_mode = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    std::cout << "========================================\n";
    std::cout << "  End-to-End Golden Validation\n";
    std::cout << "========================================\n";

    const std::filesystem::path root = pfalign::test::get_test_data_dir();
    const std::filesystem::path golden_dir = std::filesystem::path(pfalign::test::get_validation_path("end_to_end"));

    if (!std::filesystem::exists(golden_dir)) {
        std::cout << "SKIP: Missing directory " << golden_dir << "\n";
        return 0;
    }

    std::vector<std::filesystem::path> cases;
    for (const auto& entry : std::filesystem::directory_iterator(golden_dir)) {
        if (entry.is_directory()) {
            cases.push_back(entry.path());
        }
    }
    std::sort(cases.begin(), cases.end());

    if (cases.empty()) {
        std::cout << "SKIP: No golden cases found in " << golden_dir << "\n";
        return 0;
    }

    if (benchmark_mode) {
        return run_benchmark_suite(cases, root);
    }

    int passed = 0;
    int total = 0;
    for (const auto& case_dir : cases) {
        total++;
        try {
            if (run_case(case_dir, root)) {
                passed++;
            }
        } catch (const std::exception& e) {
            std::cerr << "  ✗ FAIL: Exception while running " << case_dir.filename().string()
                      << " (" << e.what() << ")\n";
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << " / " << total << " cases passed\n";
    std::cout << "========================================\n";

    return (passed == total) ? 0 : 1;
}
