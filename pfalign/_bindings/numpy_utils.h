/**
 * NumPy buffer protocol utilities for PyBind11.
 *
 * Provides helper functions for converting between NumPy arrays and C++ data structures.
 */

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <sstream>
#include <vector>

namespace py = pybind11;

namespace pfalign {
namespace bindings {

/**
 * Validate and extract NumPy array pointer.
 *
 * Checks that the array is C-contiguous and has the expected dtype.
 * Throws std::invalid_argument if validation fails.
 */
template<typename T>
const T* get_array_ptr(py::array_t<T> arr, const char* name) {
    // Check C-contiguous
    if (!(arr.flags() & py::array::c_style)) {
        std::ostringstream msg;
        msg << name << " must be C-contiguous (row-major)";
        throw std::invalid_argument(msg.str());
    }

    // Return raw pointer
    return arr.data();
}

/**
 * Create NumPy array from C++ data.
 *
 * Copies data from C++ into a new NumPy array with the specified shape.
 */
template<typename T>
py::array_t<T> make_array(const std::vector<size_t> &shape, const T* data) {
    // Calculate total size
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }

    // Create array (automatically copies data)
    py::array_t<T> arr(shape);
    T* arr_ptr = arr.mutable_data();
    std::copy(data, data + total_size, arr_ptr);

    return arr;
}

/**
 * Create NumPy array from C++ vector.
 *
 * More convenient version that takes a std::vector for shape.
 */
template<typename T>
py::array_t<T> make_array_from_vector(const std::vector<size_t> &shape, const std::vector<T> &data) {
    // Verify size matches
    size_t expected_size = 1;
    for (size_t dim : shape) {
        expected_size *= dim;
    }

    if (data.size() != expected_size) {
        std::ostringstream msg;
        msg << "Data size (" << data.size() << ") doesn't match shape size (" << expected_size << ")";
        throw std::invalid_argument(msg.str());
    }

    return make_array(shape, data.data());
}

/**
 * Validate coordinate array shape [L, 14, 3].
 *
 * Throws std::invalid_argument if shape is invalid.
 */
inline void validate_coords_array(py::array_t<float> arr, const char* name) {
    // Check dimensions
    if (arr.ndim() != 3) {
        std::ostringstream msg;
        msg << name << " must be 3D array, got " << arr.ndim() << "D";
        throw std::invalid_argument(msg.str());
    }

    // Check shape [L, 14, 3]
    auto shape = arr.shape();
    if (shape[1] != 14 || shape[2] != 3) {
        std::ostringstream msg;
        msg << name << " must have shape (L, 14, 3), got ("
            << shape[0] << ", " << shape[1] << ", " << shape[2] << ")";
        throw std::invalid_argument(msg.str());
    }

    // Check length > 0
    if (shape[0] == 0) {
        std::ostringstream msg;
        msg << name << " must have at least 1 residue";
        throw std::invalid_argument(msg.str());
    }
}

/**
 * Validate 2D array shape.
 *
 * Throws std::invalid_argument if shape is invalid.
 */
inline void validate_2d_array(py::array_t<float> arr, const char* name) {
    if (arr.ndim() != 2) {
        std::ostringstream msg;
        msg << name << " must be 2D array, got " << arr.ndim() << "D";
        throw std::invalid_argument(msg.str());
    }

    auto shape = arr.shape();
    if (shape[0] == 0 || shape[1] == 0) {
        std::ostringstream msg;
        msg << name << " must have non-zero dimensions, got ("
            << shape[0] << ", " << shape[1] << ")";
        throw std::invalid_argument(msg.str());
    }
}

} // namespace bindings
} // namespace pfalign
