/**
 * Simple .npy saver for debugging V2 intermediates.
 *
 * Only supports float32, C-order arrays.
 */

#pragma once

#include <fstream>
#include <vector>
#include <cstdint>
#include <sstream>

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

// Generic version that accepts any shape
inline bool save_npy_float32(const std::string& filepath, const float* data,
                             const std::vector<size_t>& shape) {
    try {
        std::ofstream file(filepath, std::ios::binary);
        if (!file)
            return false;

        file.write("\x93NUMPY", 6);
        file.write("\x01\x00", 2);

        // Build shape string
        std::stringstream shape_str;
        shape_str << "(";
        for (size_t i = 0; i < shape.size(); i++) {
            shape_str << shape[i];
            if (i + 1 < shape.size())
                shape_str << ", ";
            else
                shape_str << ",";  // Trailing comma for tuples
        }
        shape_str << ")";

        std::stringstream header;
        header << "{'descr': '<f4', 'fortran_order': False, 'shape': " << shape_str.str() << ", }";
        std::string header_str = header.str();

        int padding = (64 - (10 + header_str.size()) % 64) % 64;
        for (int i = 0; i < padding; i++)
            header_str += ' ';
        header_str += '\n';

        uint16_t header_len = header_str.size();
        file.write(reinterpret_cast<const char*>(&header_len), 2);
        file.write(header_str.c_str(), header_len);

        // Compute total size
        size_t total_size = 1;
        for (size_t dim : shape)
            total_size *= dim;

        file.write(reinterpret_cast<const char*>(data), total_size * sizeof(float));
        return true;
    } catch (...) {
        return false;
    }
}
