#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace pfalign {
namespace memory {

/**
 * RAII Arena Allocator with proper ownership semantics.
 *
 * Design principles:
 * - RAII: Memory is automatically freed when arena goes out of scope
 * - Alignment: All allocations are 64-byte aligned (cache line size for optimal performance)
 *   - Includes 16-byte SIMD alignment for ARM NEON (128-bit vectors)
 * - Thread-safe: Each thread should have its own arena (not internally synchronized)
 * - Move-only: Prevent accidental copies
 * - Clear ownership: Arena owns all allocated memory
 *
 * Usage:
 *   Arena arena(1024 * 1024);  // 1 MB
 *   float* buf = arena.allocate<float>(256);
 *   // ... use buf ...
 *   arena.reset();  // Reuse for next batch
 *   // ~Arena() frees all memory automatically
 */
class Arena {
public:
    // Alignment requirements
    static constexpr size_t SIMD_ALIGNMENT = 16;               // 128-bit NEON vectors (ARM)
    static constexpr size_t CACHE_LINE_ALIGNMENT = 64;         // Cache line size
    static constexpr size_t ALIGNMENT = CACHE_LINE_ALIGNMENT;  // Use cache line alignment

    // Verify that cache line alignment includes SIMD alignment
    static_assert(CACHE_LINE_ALIGNMENT >= SIMD_ALIGNMENT,
                  "Cache line alignment must be at least SIMD alignment");
    /**
     * Create arena with specified capacity.
     * @param capacity_bytes Total capacity in bytes
     */
    explicit Arena(size_t capacity_bytes) : capacity_(capacity_bytes), offset_(0) {
// Allocate aligned memory (ALIGNMENT bytes for cache line + SIMD alignment)
#ifdef _WIN32
        buffer_ = static_cast<char*>(_aligned_malloc(capacity_bytes, ALIGNMENT));
        if (!buffer_) {
            throw std::bad_alloc();
        }
#else
        if (posix_memalign(reinterpret_cast<void**>(&buffer_), ALIGNMENT, capacity_bytes) != 0) {
            throw std::bad_alloc();
        }
#endif
    }

    ~Arena() {
#ifdef _WIN32
        _aligned_free(buffer_);
#else
        free(buffer_);
#endif
    }

    // Move-only (no copy)
    Arena(Arena&& other) noexcept
        : buffer_(other.buffer_), capacity_(other.capacity_), offset_(other.offset_) {
        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
    }

    Arena& operator=(Arena&& other) noexcept {
        if (this != &other) {
// Free our memory
#ifdef _WIN32
            _aligned_free(buffer_);
#else
            free(buffer_);
#endif

            // Take ownership of other's memory
            buffer_ = other.buffer_;
            capacity_ = other.capacity_;
            offset_ = other.offset_;

            // Nullify other
            other.buffer_ = nullptr;
            other.capacity_ = 0;
            other.offset_ = 0;
        }
        return *this;
    }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Allow ScopedArena to restore offset on destruction
    friend class ScopedArena;

    /**
     * Allocate typed array from arena.
     * @param count Number of elements
     * @return Pointer to allocated memory (ALIGNMENT-byte aligned for cache lines + SIMD)
     * @throws std::bad_alloc if arena is full
     */
    template <typename T>
    T* allocate(size_t count) {
        size_t bytes = count * sizeof(T);
        size_t aligned_bytes = align_up(bytes, ALIGNMENT);

        if (offset_ + aligned_bytes > capacity_) {
            throw std::bad_alloc();
        }

        T* ptr = reinterpret_cast<T*>(buffer_ + offset_);
        offset_ += aligned_bytes;
        return ptr;
    }

    /**
     * Reset arena for reuse (zero-cost).
     * Does NOT free memory, just resets the allocation pointer.
     * All previous pointers become invalid.
     */
    void reset() {
        offset_ = 0;
    }

    /**
     * Get current usage in bytes.
     */
    size_t used() const {
        return offset_;
    }

    /**
     * Get total capacity in bytes.
     */
    size_t capacity() const {
        return capacity_;
    }

    /**
     * Check if arena has space for N bytes.
     */
    bool can_allocate(size_t bytes) const {
        return offset_ + align_up(bytes, ALIGNMENT) <= capacity_;
    }

private:
    char* buffer_;
    size_t capacity_;
    size_t offset_;

    static inline size_t align_up(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }
};

/**
 * Scoped arena allocation with automatic reset.
 *
 * Usage:
 *   Arena arena(1MB);
 *   {
 *       ScopedArena scope(arena);
 *       float* temp = scope.allocate<float>(100);
 *       // ... use temp ...
 *   }  // Automatic reset here
 */
class ScopedArena {
public:
    explicit ScopedArena(Arena& arena) : arena_(arena), saved_offset_(arena.used()) {
    }

    ~ScopedArena() {
        // Reset to saved offset (releases allocations made in this scope)
        arena_.offset_ = saved_offset_;
    }

    template <typename T>
    T* allocate(size_t count) {
        return arena_.allocate<T>(count);
    }

    // Non-copyable, non-movable (tied to stack lifetime)
    ScopedArena(const ScopedArena&) = delete;
    ScopedArena& operator=(const ScopedArena&) = delete;
    ScopedArena(ScopedArena&&) = delete;
    ScopedArena& operator=(ScopedArena&&) = delete;

private:
    Arena& arena_;
    size_t saved_offset_;
};

/**
 * Per-backend arena manager.
 *
 * Ensures CUDA streams don't contend with CPU buffers.
 * Each backend gets its own arena instance.
 */
class BackendArenas {
public:
    BackendArenas(size_t cpu_size, size_t gpu_size) : cpu_arena_(cpu_size), gpu_arena_(gpu_size) {
    }

    Arena& cpu() {
        return cpu_arena_;
    }
    Arena& gpu() {
        return gpu_arena_;
    }

    void reset_all() {
        cpu_arena_.reset();
        gpu_arena_.reset();
    }

private:
    Arena cpu_arena_;
    Arena gpu_arena_;
};

}  // namespace memory
}  // namespace pfalign
