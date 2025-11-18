/**
 * Test backward pass parity: C++ vs JAX reference.
 * 
 * Compares posterior alignment matrices from C++ backward passes
 * against JAX jax.value_and_grad() gradients.
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <filesystem>
#include <stdexcept>

#include "pfalign/primitives/smith_waterman/smith_waterman.h"
#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/tools/weights/save_npy.h"

using pfalign::ScalarBackend;
namespace sw = pfalign::smith_waterman;

// Helper to load .npy file (simple format for our case)
bool load_npy_simple(const std::filesystem::path& path, float* data, int size) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    
    // Skip numpy header (find the data array start)
    // For our simple case, data starts after header (around byte 128)
    char header[256];
    f.read(header, 128);
    
    // Read float data
    f.read(reinterpret_cast<char*>(data), size * sizeof(float));
    f.close();
    return true;
}

namespace {

std::filesystem::path golden_dir() {
    auto path = std::filesystem::path(__FILE__).parent_path();
    for (int i = 0; i < 5; ++i) {
        path = path.parent_path();
    }
    return path / "data" / "golden" / "smith_waterman";
}

float load_partition_text(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open partition file: " + path.string());
    }
    float value = 0.0f;
    f >> value;
    if (!f) {
        throw std::runtime_error("Failed to parse partition value from: " + path.string());
    }
    return value;
}

} // namespace

int main() {
    std::cout << "================================================================\n";
    std::cout << "C++ Backward Pass Parity Test\n";
    std::cout << "================================================================\n\n";

    // Load similarity matrix
    const int L1 = 12, L2 = 12;
    float* similarity = new float[L1 * L2];
    
    auto data_dir = golden_dir();
    auto similarity_path = data_dir / "backward_similarity_12x12.npy";
    std::cout << "Loading similarity matrix from " << similarity_path << "...\n";
    if (!load_npy_simple(similarity_path, similarity, L1 * L2)) {
        std::cerr << "ERROR: Failed to load similarity matrix\n";
        return 1;
    }
    std::cout << "  Loaded [" << L1 << " * " << L2 << "]\n\n";

    // Parameters
    float gap = 0.194247f;
    float gap_open = -2.54418f;
    float temp = 1.0f;

    // ========================================================================
    // Test 1: JAX Regular
    // ========================================================================
    std::cout << "================================================================\n";
    std::cout << "Test 1: C++ jax_regular backward\n";
    std::cout << "================================================================\n\n";

    sw::SWConfig config_reg;
    config_reg.gap = gap;
    config_reg.temperature = temp;

    // Forward pass
    float* hij_reg = new float[L1 * L2];
    float partition_reg = 0.0f;
    
    sw::smith_waterman_jax_regular<ScalarBackend>(
        similarity, L1, L2, config_reg, hij_reg, &partition_reg
    );

    // Backward pass
    float* posteriors_reg = new float[L1 * L2];
    sw::smith_waterman_jax_regular_backward<ScalarBackend>(
        hij_reg, similarity, L1, L2, config_reg, partition_reg, posteriors_reg
    );

    std::cout << "  C++ partition:  " << partition_reg << "\n";
    std::cout << "  C++ post sum:   " << std::accumulate(posteriors_reg, posteriors_reg + L1*L2, 0.0f) << "\n";
    std::cout << "  C++ post[0,0]:  " << posteriors_reg[0] << "\n";
    std::cout << "  C++ post[0,:5]: ";
    for (int i = 0; i < 5; i++) std::cout << posteriors_reg[i] << " ";
    std::cout << "\n\n";

    // Load JAX reference
    float* jax_posteriors_reg = new float[L1 * L2];
    auto reg_path = data_dir / "backward_posteriors_regular_12x12.npy";
    if (!load_npy_simple(reg_path, jax_posteriors_reg, L1 * L2)) {
        std::cerr << "ERROR: Failed to load JAX reference\n";
        return 1;
    }

    // Compare
    float max_error_reg = 0.0f;
    float sum_error_reg = 0.0f;
    for (int i = 0; i < L1 * L2; i++) {
        float err = std::abs(posteriors_reg[i] - jax_posteriors_reg[i]);
        max_error_reg = std::max(max_error_reg, err);
        sum_error_reg += err;
    }
    float mean_error_reg = sum_error_reg / (L1 * L2);

    std::cout << "Comparison vs JAX:\n";
    std::cout << "  Max absolute error:  " << max_error_reg << "\n";
    std::cout << "  Mean absolute error: " << mean_error_reg << "\n";
    
    float golden_partition_reg = load_partition_text(data_dir / "backward_partition_regular.txt");
    float partition_error_reg = std::abs(partition_reg - golden_partition_reg);
    std::cout << "  Golden partition: " << golden_partition_reg << "\n";
    std::cout << "  Partition error:  " << partition_error_reg << "\n";

    bool test1_pass = max_error_reg < 1e-5f && partition_error_reg < 1e-6f;
    std::cout << "  Status: " << (test1_pass ? "✓ PASS" : "✗ FAIL") << "\n\n";

    // ========================================================================
    // Test 2: JAX Affine Flexible
    // ========================================================================
    std::cout << "================================================================\n";
    std::cout << "Test 2: C++ jax_affine_flexible backward\n";
    std::cout << "================================================================\n\n";

    sw::SWConfig config_aff;
    config_aff.gap = gap;
    config_aff.gap_open = gap_open;
    config_aff.gap_extend = gap;
    config_aff.temperature = temp;

    // Forward pass
    float* hij_aff = new float[L1 * L2 * 3];
    float partition_aff = 0.0f;
    
    sw::smith_waterman_jax_affine_flexible<ScalarBackend>(
        similarity, L1, L2, config_aff, hij_aff, &partition_aff
    );

    // Backward pass
    float* posteriors_aff = new float[L1 * L2];
    sw::smith_waterman_jax_affine_flexible_backward<ScalarBackend>(
        hij_aff, similarity, L1, L2, config_aff, partition_aff, posteriors_aff
    );

    std::cout << "  C++ partition:  " << partition_aff << "\n";
    std::cout << "  C++ post sum:   " << std::accumulate(posteriors_aff, posteriors_aff + L1*L2, 0.0f) << "\n";
    std::cout << "  C++ post[0,0]:  " << posteriors_aff[0] << "\n";
    std::cout << "  C++ post[0,:5]: ";
    for (int i = 0; i < 5; i++) std::cout << posteriors_aff[i] << " ";
    std::cout << "\n\n";

    // Load JAX reference
    float* jax_posteriors_aff = new float[L1 * L2];
    auto aff_path = data_dir / "backward_posteriors_affine_12x12.npy";
    if (!load_npy_simple(aff_path, jax_posteriors_aff, L1 * L2)) {
        std::cerr << "ERROR: Failed to load JAX affine reference\n";
        return 1;
    }

    // Compare
    float max_error_aff = 0.0f;
    float sum_error_aff = 0.0f;
    for (int i = 0; i < L1 * L2; i++) {
        float err = std::abs(posteriors_aff[i] - jax_posteriors_aff[i]);
        max_error_aff = std::max(max_error_aff, err);
        sum_error_aff += err;
    }
    float mean_error_aff = sum_error_aff / (L1 * L2);

    std::cout << "Comparison vs JAX:\n";
    std::cout << "  Max absolute error:  " << max_error_aff << "\n";
    std::cout << "  Mean absolute error: " << mean_error_aff << "\n";
    
    float golden_partition_aff = load_partition_text(data_dir / "backward_partition_affine.txt");
    float partition_error_aff = std::abs(partition_aff - golden_partition_aff);
    std::cout << "  Golden partition: " << golden_partition_aff << "\n";
    std::cout << "  Partition error:  " << partition_error_aff << "\n";

    bool test2_pass = max_error_aff < 1e-5f && partition_error_aff < 1e-6f;
    std::cout << "  Status: " << (test2_pass ? "✓ PASS" : "✗ FAIL") << "\n\n";

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "================================================================\n";
    std::cout << "Summary\n";
    std::cout << "================================================================\n";
    std::cout << "Test 1 (Regular):         " << (test1_pass ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "Test 2 (Affine Flexible): " << (test2_pass ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "\nOverall: " << ((test1_pass && test2_pass) ? "✓ ALL TESTS PASSED" : "✗ SOME TESTS FAILED") << "\n";
    std::cout << "================================================================\n";

    // Cleanup
    delete[] similarity;
    delete[] hij_reg;
    delete[] posteriors_reg;
    delete[] jax_posteriors_reg;
    delete[] hij_aff;
    delete[] posteriors_aff;
    delete[] jax_posteriors_aff;

    return (test1_pass && test2_pass) ? 0 : 1;
}
