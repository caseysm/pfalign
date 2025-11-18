/**
 * Simple .npy loader/saver for CLI commands.
 * Supports float32, C-order arrays only.
 */

#pragma once

#include <fstream>
#include <vector>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pfalign {
namespace cli {

/**
 * NPY file header information.
 */
struct NpyHeader {
    std::vector<size_t> shape;
    std::string dtype;
    bool fortran_order;
};

/**
 * Parse NPY file header to extract shape and dtype information.
 */
inline NpyHeader parse_npy_header(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Cannot open NPY file: " + path);
    }

    // Read magic number
    char magic[6];
    f.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("Not a valid NPY file: " + path);
    }

    // Read version
    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    // Read header length
    uint16_t header_len;
    f.read(reinterpret_cast<char*>(&header_len), 2);

    // Read header dictionary
    std::string header_str(header_len, ' ');
    f.read(&header_str[0], header_len);

    // Parse shape from header
    NpyHeader header;
    header.fortran_order = false;

    // Find shape tuple: 'shape': (d1, d2, ...)
    size_t shape_pos = header_str.find("'shape':");
    if (shape_pos == std::string::npos) {
        throw std::runtime_error("Cannot find 'shape' in NPY header");
    }

    size_t tuple_start = header_str.find('(', shape_pos);
    size_t tuple_end = header_str.find(')', tuple_start);
    if (tuple_start == std::string::npos || tuple_end == std::string::npos) {
        throw std::runtime_error("Cannot parse shape tuple in NPY header");
    }

    std::string shape_str = header_str.substr(tuple_start + 1, tuple_end - tuple_start - 1);

    // Parse comma-separated dimensions
    std::istringstream iss(shape_str);
    std::string dim_str;
    while (std::getline(iss, dim_str, ',')) {
        // Trim whitespace
        dim_str.erase(0, dim_str.find_first_not_of(" \t"));
        dim_str.erase(dim_str.find_last_not_of(" \t") + 1);
        if (!dim_str.empty()) {
            header.shape.push_back(std::stoull(dim_str));
        }
    }

    // Extract dtype
    size_t descr_pos = header_str.find("'descr':");
    if (descr_pos != std::string::npos) {
        size_t quote1 = header_str.find('\'', descr_pos + 8);
        size_t quote2 = header_str.find('\'', quote1 + 1);
        if (quote1 != std::string::npos && quote2 != std::string::npos) {
            header.dtype = header_str.substr(quote1 + 1, quote2 - quote1 - 1);
        }
    }

    return header;
}

/**
 * Load .npy file (simple format - skips header, assumes float32).
 */
inline bool load_npy_simple(const std::string& path, float* data, size_t size) {
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

/**
 * Save 1D array to .npy format.
 */
inline void save_npy_1d(const std::string& filepath, const float* data, int n) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open: " + filepath);

    // Magic + version
    file.write("\x93NUMPY", 6);
    file.write("\x01\x00", 2);  // version 1.0

    // Header
    std::stringstream header;
    header << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << n << ",), }";
    std::string header_str = header.str();

    // Pad to multiple of 64 bytes (including 10-byte prefix)
    int total_header_len = header_str.size();
    int padding = (64 - (10 + total_header_len) % 64) % 64;
    for (int i = 0; i < padding; i++)
        header_str += ' ';
    header_str += '\n';

    uint16_t header_len = header_str.size();
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header_str.c_str(), header_len);

    // Data
    file.write(reinterpret_cast<const char*>(data), n * sizeof(float));
}

/**
 * Save 2D array to .npy format.
 */
inline void save_npy_2d(const std::string& filepath, const float* data, int rows, int cols) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open: " + filepath);

    // Magic + version
    file.write("\x93NUMPY", 6);
    file.write("\x01\x00", 2);

    // Header
    std::stringstream header;
    header << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << rows << ", " << cols
           << "), }";
    std::string header_str = header.str();

    int padding = (64 - (10 + header_str.size()) % 64) % 64;
    for (int i = 0; i < padding; i++)
        header_str += ' ';
    header_str += '\n';

    uint16_t header_len = header_str.size();
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header_str.c_str(), header_len);

    // Data
    file.write(reinterpret_cast<const char*>(data), rows * cols * sizeof(float));
}

/**
 * Save 3D array to .npy format.
 */
inline void save_npy_3d(const std::string& filepath, const float* data, int d1, int d2, int d3) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open: " + filepath);

    file.write("\x93NUMPY", 6);
    file.write("\x01\x00", 2);

    std::stringstream header;
    header << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << d1 << ", " << d2 << ", "
           << d3 << "), }";
    std::string header_str = header.str();

    int padding = (64 - (10 + header_str.size()) % 64) % 64;
    for (int i = 0; i < padding; i++)
        header_str += ' ';
    header_str += '\n';

    uint16_t header_len = header_str.size();
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header_str.c_str(), header_len);

    file.write(reinterpret_cast<const char*>(data), d1 * d2 * d3 * sizeof(float));
}

}  // namespace cli
}  // namespace pfalign
