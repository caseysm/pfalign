#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace pfalign {

/**
 * Lightweight non-owning view over contiguous data.
 *
 * Similar to std::span (C++20) but works with C++17.
 * Provides bounds checking and iterator interface.
 *
 * Usage:
 *   float data[100];
 *   Span<float> view(data, 100);
 *   view[10] = 3.14f;  // Bounds-checked in debug
 *   for (float& x : view) { x *= 2; }
 */
template <typename T>
class Span {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;
    using size_type = size_t;

    // Default constructor (empty span)
    constexpr Span() noexcept : data_(nullptr), size_(0) {
    }

    // Construct from pointer + size
    constexpr Span(T* data, size_type size) noexcept : data_(data), size_(size) {
    }

    // Construct from C array
    template <size_t N>
    constexpr Span(T (&arr)[N]) noexcept : data_(arr), size_(N) {
    }

    // Element access (bounds-checked in debug)
    constexpr reference operator[](size_type idx) noexcept {
#ifndef NDEBUG
        if (idx >= size_) {
            throw std::out_of_range("Span index out of bounds");
        }
#endif
        return data_[idx];
    }

    constexpr const_reference operator[](size_type idx) const noexcept {
#ifndef NDEBUG
        if (idx >= size_) {
            throw std::out_of_range("Span index out of bounds");
        }
#endif
        return data_[idx];
    }

    // Iterator interface
    constexpr iterator begin() noexcept {
        return data_;
    }
    constexpr const_iterator begin() const noexcept {
        return data_;
    }
    constexpr iterator end() noexcept {
        return data_ + size_;
    }
    constexpr const_iterator end() const noexcept {
        return data_ + size_;
    }

    // Size and data access
    constexpr size_type size() const noexcept {
        return size_;
    }
    constexpr bool empty() const noexcept {
        return size_ == 0;
    }
    constexpr pointer data() noexcept {
        return data_;
    }
    constexpr const_pointer data() const noexcept {
        return data_;
    }

    // Subspan
    constexpr Span<T> subspan(size_type offset, size_type count) const {
#ifndef NDEBUG
        if (offset + count > size_) {
            throw std::out_of_range("Subspan out of bounds");
        }
#endif
        return Span<T>(data_ + offset, count);
    }

    // Convert to const view
    constexpr operator Span<const T>() const noexcept {
        return Span<const T>(data_, size_);
    }

private:
    T* data_;
    size_type size_;
};

/**
 * 2D span (row-major matrix view).
 *
 * Usage:
 *   float* data = arena.allocate<float>(M * N);
 *   Span2D<float> mat(data, M, N);
 *   mat(i, j) = 3.14f;  // Bounds-checked
 */
template <typename T>
class Span2D {
public:
    constexpr Span2D() noexcept : data_(nullptr), rows_(0), cols_(0), stride_(0) {
    }

    constexpr Span2D(T* data, size_t rows, size_t cols, size_t stride = 0) noexcept
        : data_(data), rows_(rows), cols_(cols), stride_(stride ? stride : cols) {
    }

    // Element access
    constexpr T& operator()(size_t i, size_t j) noexcept {
#ifndef NDEBUG
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("Span2D index out of bounds");
        }
#endif
        return data_[i * stride_ + j];
    }

    constexpr const T& operator()(size_t i, size_t j) const noexcept {
#ifndef NDEBUG
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("Span2D index out of bounds");
        }
#endif
        return data_[i * stride_ + j];
    }

    // Row access (returns Span)
    constexpr Span<T> row(size_t i) noexcept {
#ifndef NDEBUG
        if (i >= rows_) {
            throw std::out_of_range("Row index out of bounds");
        }
#endif
        return Span<T>(data_ + i * stride_, cols_);
    }

    constexpr size_t rows() const noexcept {
        return rows_;
    }
    constexpr size_t cols() const noexcept {
        return cols_;
    }
    constexpr size_t stride() const noexcept {
        return stride_;
    }
    constexpr T* data() noexcept {
        return data_;
    }
    constexpr const T* data() const noexcept {
        return data_;
    }

private:
    T* data_;
    size_t rows_;
    size_t cols_;
    size_t stride_;  // Leading dimension
};

// Deduction guides (C++17)
template <typename T, size_t N>
Span(T (&)[N]) -> Span<T>;

}  // namespace pfalign
