/**
 * Golden data testing utilities for validating C++ implementations against JAX reference.
 *
 * This header provides:
 * - .npy file loading (NumPy format)
 * - Numerical comparison with configurable tolerances
 * - Test result reporting
 *
 * Usage:
 *   GoldenDataTest test("data/golden/mpnn/small_10res");
 *   auto coords = test.load<float>("input_coords.npy");
 *   auto expected = test.load<float>("output_embeddings.npy");
 *
 *   // ... run your implementation ...
 *
 *   test.compare("embeddings", expected, actual, 1e-4, 1e-5);
 *   test.print_summary();
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

namespace pfalign {
namespace testing {

/**
 * Simple .npy file loader (supports float32 only, C-contiguous arrays)
 */
struct NumpyArray {
    std::vector<size_t> shape;
    std::vector<float> data;

    size_t size() const {
        size_t s = 1;
        for (auto dim : shape)
            s *= dim;
        return s;
    }

    size_t ndim() const {
        return shape.size();
    }
};

/**
 * Load a .npy file (simplified implementation for float32 only)
 */
inline NumpyArray load_npy(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open .npy file: " + filepath);
    }

    // Read magic string (6 bytes)
    char magic[7] = {0};
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("Invalid .npy file (bad magic): " + filepath);
    }

    // Read version (2 bytes)
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    // Read header length (2 or 4 bytes depending on version)
    uint32_t header_len;
    if (major == 1) {
        uint16_t len16;
        file.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else if (major == 2 || major == 3) {
        file.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("Unsupported .npy version: " + std::to_string(major));
    }

    // Read header (Python dict as string)
    std::vector<char> header(header_len);
    file.read(header.data(), header_len);
    std::string header_str(header.begin(), header.end());

    // Parse shape from header
    NumpyArray array;
    size_t shape_start = header_str.find("'shape': (");
    if (shape_start == std::string::npos) {
        shape_start = header_str.find("\"shape\": (");
    }
    if (shape_start == std::string::npos) {
        throw std::runtime_error("Failed to parse shape from .npy header");
    }

    size_t shape_end = header_str.find(")", shape_start);
    std::string shape_str = header_str.substr(shape_start + 10, shape_end - shape_start - 10);

    // Parse shape dimensions
    std::istringstream shape_stream(shape_str);
    std::string dim_str;
    while (std::getline(shape_stream, dim_str, ',')) {
        // Trim whitespace
        dim_str.erase(0, dim_str.find_first_not_of(" \t"));
        dim_str.erase(dim_str.find_last_not_of(" \t") + 1);
        if (!dim_str.empty()) {
            array.shape.push_back(std::stoul(dim_str));
        }
    }

    // Check dtype - support both float32 and int32
    bool is_float32 = (header_str.find("'<f4'") != std::string::npos ||
                       header_str.find("\"<f4\"") != std::string::npos ||
                       header_str.find("'f4'") != std::string::npos ||
                       header_str.find("\"f4\"") != std::string::npos);

    bool is_int32 = (header_str.find("'<i4'") != std::string::npos ||
                     header_str.find("\"<i4\"") != std::string::npos ||
                     header_str.find("'i4'") != std::string::npos ||
                     header_str.find("\"i4\"") != std::string::npos);

    if (!is_float32 && !is_int32) {
        throw std::runtime_error("Only float32 and int32 .npy files are supported");
    }

    // Read data
    size_t data_size = array.size();
    array.data.resize(data_size);

    if (is_float32) {
        file.read(reinterpret_cast<char*>(array.data.data()), data_size * sizeof(float));
    } else {
        // Read int32 and convert to float
        std::vector<int32_t> int_data(data_size);
        file.read(reinterpret_cast<char*>(int_data.data()), data_size * sizeof(int32_t));
        for (size_t i = 0; i < data_size; ++i) {
            array.data[i] = static_cast<float>(int_data[i]);
        }
    }

    if (!file) {
        throw std::runtime_error("Failed to read .npy data");
    }

    return array;
}

/**
 * Comparison result for a single test
 */
struct ComparisonResult {
    std::string name;
    bool passed;
    double max_abs_error;
    double mean_abs_error;
    double max_rel_error;
    size_t num_elements;
    double atol;
    double rtol;

    void print() const {
        std::cout << "  " << name << ": ";
        if (passed) {
            std::cout << "[OK] PASS" << std::endl;
        } else {
            std::cout << "[FAIL] FAIL" << std::endl;
        }
        std::cout << "    Max abs error: " << std::scientific << std::setprecision(6)
                  << max_abs_error << " (tol: " << atol << ")" << std::endl;
        std::cout << "    Max rel error: " << std::scientific << std::setprecision(6)
                  << max_rel_error << " (tol: " << rtol << ")" << std::endl;
        std::cout << "    Mean abs error: " << std::scientific << std::setprecision(6)
                  << mean_abs_error << std::endl;
        std::cout << "    Elements: " << num_elements << std::endl;
    }
};

/**
 * Golden data test harness
 */
class GoldenDataTest {
public:
    explicit GoldenDataTest(const std::string& data_dir)
        : data_dir_(data_dir), num_passed_(0), num_failed_(0) {
        std::cout << "Golden Data Test: " << data_dir << std::endl;
        std::cout << std::string(60, '-') << std::endl;
    }

    /**
     * Load a .npy file from the data directory
     */
    template <typename T = float>
    std::vector<T> load(const std::string& filename) {
        std::string path = data_dir_ + "/" + filename;
        auto array = load_npy(path);

        std::cout << "  Loaded " << filename << ": shape=[";
        for (size_t i = 0; i < array.shape.size(); i++) {
            std::cout << array.shape[i];
            if (i < array.shape.size() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Convert to target type if needed
        std::vector<T> result(array.data.begin(), array.data.end());
        return result;
    }

    /**
     * Load array and return shape information
     */
    std::pair<std::vector<float>, std::vector<size_t>>
    load_with_shape(const std::string& filename) {
        std::string path = data_dir_ + "/" + filename;
        auto array = load_npy(path);

        std::cout << "  Loaded " << filename << ": shape=[";
        for (size_t i = 0; i < array.shape.size(); i++) {
            std::cout << array.shape[i];
            if (i < array.shape.size() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        return {array.data, array.shape};
    }

    /**
     * Compare two arrays with absolute and relative tolerance
     */
    ComparisonResult compare(const std::string& name, const std::vector<float>& expected,
                             const std::vector<float>& actual, double atol = 1e-5,
                             double rtol = 1e-5) {
        ComparisonResult result;
        result.name = name;
        result.atol = atol;
        result.rtol = rtol;
        result.num_elements = expected.size();
        result.max_abs_error = 0.0;
        result.mean_abs_error = 0.0;
        result.max_rel_error = 0.0;
        result.passed = true;

        if (expected.size() != actual.size()) {
            std::cerr << "ERROR: Size mismatch for " << name << ": expected " << expected.size()
                      << ", got " << actual.size() << std::endl;
            result.passed = false;
            num_failed_++;
            results_.push_back(result);
            return result;
        }

        double sum_abs_error = 0.0;

        for (size_t i = 0; i < expected.size(); i++) {
            double exp = expected[i];
            double act = actual[i];
            double abs_err = std::abs(exp - act);
            double rel_err = (std::abs(exp) > 1e-10) ? abs_err / std::abs(exp) : 0.0;

            sum_abs_error += abs_err;
            result.max_abs_error = std::max(result.max_abs_error, abs_err);
            result.max_rel_error = std::max(result.max_rel_error, rel_err);

            // Check tolerance
            if (abs_err > atol && rel_err > rtol) {
                result.passed = false;
            }
        }

        result.mean_abs_error = sum_abs_error / expected.size();

        if (result.passed) {
            num_passed_++;
        } else {
            num_failed_++;
        }

        results_.push_back(result);
        result.print();

        return result;
    }

    /**
     * Compare single scalar value
     */
    ComparisonResult compare_scalar(const std::string& name, float expected, float actual,
                                    double atol = 1e-5, double rtol = 1e-5) {
        return compare(name, {expected}, {actual}, atol, rtol);
    }

    /**
     * Print test summary
     */
    void print_summary() const {
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "Summary: " << num_passed_ << " passed, " << num_failed_ << " failed"
                  << std::endl;

        if (num_failed_ > 0) {
            std::cout << "\nFailed tests:" << std::endl;
            for (const auto& result : results_) {
                if (!result.passed) {
                    std::cout << "  - " << result.name << std::endl;
                }
            }
        }

        std::cout << std::string(60, '=') << std::endl;
    }

    /**
     * Return true if all tests passed
     */
    bool all_passed() const {
        return num_failed_ == 0;
    }

    /**
     * Get number of passed/failed tests
     */
    int num_passed() const {
        return num_passed_;
    }
    int num_failed() const {
        return num_failed_;
    }

    /**
     * Get test directory path
     */
    std::string test_dir() const {
        return data_dir_;
    }

    /**
     * Load a .npz file (ZIP archive of .npy files)
     * Returns a map of array names to (data, shape) pairs
     */
    std::map<std::string, std::pair<std::vector<float>, std::vector<size_t>>>
    load_npz(const std::string& filename) {
        std::string path = data_dir_.empty() ? filename : data_dir_ + "/" + filename;

        // Open the .npz file as a ZIP archive
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open .npz file: " + path);
        }

        std::map<std::string, std::pair<std::vector<float>, std::vector<size_t>>> arrays;

        // Parse ZIP format
        // ZIP local file header signature: 0x04034b50
        while (file) {
            uint32_t signature;
            file.read(reinterpret_cast<char*>(&signature), 4);

            if (!file)
                break;  // End of file

            if (signature == 0x04034b50) {  // Local file header
                // Read version needed (2)
                uint16_t version_needed;
                file.read(reinterpret_cast<char*>(&version_needed), 2);

                // Read flags (2)
                uint16_t flags;
                file.read(reinterpret_cast<char*>(&flags), 2);

                // Read compression method (2)
                uint16_t compression_method;
                file.read(reinterpret_cast<char*>(&compression_method), 2);

                // Read modification time (2) and date (2) - skip
                file.seekg(4, std::ios::cur);

                // Read CRC-32, compressed size, uncompressed size
                uint32_t crc32, compressed_size, uncompressed_size;
                file.read(reinterpret_cast<char*>(&crc32), 4);
                file.read(reinterpret_cast<char*>(&compressed_size), 4);
                file.read(reinterpret_cast<char*>(&uncompressed_size), 4);

                // Read filename length and extra field length
                uint16_t filename_len, extra_len;
                file.read(reinterpret_cast<char*>(&filename_len), 2);
                file.read(reinterpret_cast<char*>(&extra_len), 2);

                // Read filename
                std::vector<char> filename_buf(filename_len);
                file.read(filename_buf.data(), filename_len);
                std::string array_name(filename_buf.begin(), filename_buf.end());

                // Remove .npy extension if present
                if (array_name.size() > 4 && array_name.substr(array_name.size() - 4) == ".npy") {
                    array_name = array_name.substr(0, array_name.size() - 4);
                }

                // Skip extra field
                file.seekg(extra_len, std::ios::cur);

                // Read the .npy data
                if (compressed_size > 0 && compression_method == 0) {
                    // Method 0 = stored (uncompressed)
                    std::vector<char> npy_data(compressed_size);
                    file.read(npy_data.data(), compressed_size);

                    // Parse .npy data from memory
                    auto array = parse_npy_from_memory(npy_data);
                    arrays[array_name] = {array.data, array.shape};
                } else if (compressed_size > 0) {
                    // Compressed data - skip it for now
                    file.seekg(compressed_size, std::ios::cur);
                    std::cerr << "WARNING: Skipping compressed file " << array_name
                              << " (compression not supported)" << std::endl;
                }
            } else if (signature == 0x02014b50) {
                // Central directory header - we're done with file entries
                break;
            } else {
                // Unknown signature - might be corrupted or at end of file
                break;
            }
        }

        std::cout << "  Loaded " << filename << " (" << arrays.size() << " arrays)" << std::endl;

        return arrays;
    }

private:
    /**
     * Parse .npy data from memory buffer
     */
    NumpyArray parse_npy_from_memory(const std::vector<char>& data) {
        size_t pos = 0;

        // Check magic string
        if (data.size() < 10 || std::string(data.data(), 6) != "\x93NUMPY") {
            throw std::runtime_error("Invalid .npy data (bad magic)");
        }
        pos += 6;

        // Read version
        uint8_t major = data[pos++];
        uint8_t minor = data[pos++];

        // Read header length
        uint32_t header_len;
        if (major == 1) {
            header_len = *reinterpret_cast<const uint16_t*>(&data[pos]);
            pos += 2;
        } else if (major == 2 || major == 3) {
            header_len = *reinterpret_cast<const uint32_t*>(&data[pos]);
            pos += 4;
        } else {
            throw std::runtime_error("Unsupported .npy version");
        }

        // Read header
        std::string header_str(data.begin() + pos, data.begin() + pos + header_len);
        pos += header_len;

        // Parse shape
        NumpyArray array;
        size_t shape_start = header_str.find("'shape': (");
        if (shape_start == std::string::npos) {
            shape_start = header_str.find("\"shape\": (");
        }
        if (shape_start == std::string::npos) {
            throw std::runtime_error("Failed to parse shape");
        }

        size_t shape_end = header_str.find(")", shape_start);
        std::string shape_str = header_str.substr(shape_start + 10, shape_end - shape_start - 10);

        std::istringstream shape_stream(shape_str);
        std::string dim_str;
        while (std::getline(shape_stream, dim_str, ',')) {
            dim_str.erase(0, dim_str.find_first_not_of(" \t"));
            dim_str.erase(dim_str.find_last_not_of(" \t") + 1);
            if (!dim_str.empty()) {
                array.shape.push_back(std::stoul(dim_str));
            }
        }

        // Read data
        size_t data_size = array.size();
        array.data.resize(data_size);
        std::memcpy(array.data.data(), &data[pos], data_size * sizeof(float));

        return array;
    }

    std::string data_dir_;
    int num_passed_;
    int num_failed_;
    std::vector<ComparisonResult> results_;
};

}  // namespace testing
}  // namespace pfalign
