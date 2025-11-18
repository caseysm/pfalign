/**
 * Growable arena allocator with linked blocks.
 *
 * Unlike fixed-size Arena, GrowableArena automatically allocates additional
 * blocks when capacity is exceeded, allowing unbounded growth (limited only
 * by system RAM).
 *
 * Key features:
 * - No hard capacity limits (grows as needed)
 * - Fast O(1) reset (rewinds all blocks)
 * - Tracks peak usage for profiling
 * - 1.5* growth strategy (balances waste vs reallocations)
 * - Cache-aligned allocations (64-byte alignment)
 *
 * Usage:
 *   GrowableArena arena(32, "my_arena");  // Start with 32 MB
 *   float* data = arena.allocate<float>(1000000);  // Allocates from current block
 *   // ... if block full, automatically allocates new block ...
 *   arena.reset();  // Rewinds all blocks, ready for reuse
 *
 * Thread safety: NOT thread-safe. Use one GrowableArena per thread.
 */

#pragma once

#include <cstdlib>
#include <cstddef>
#include <cstdio>
#include <string>
#include <algorithm>
#include <stdexcept>
#include "memory_budget.h"


namespace pfalign::memory {

class GrowableArena {
private:
    // Individual memory block in linked list
    struct Block {
        char* buffer;
        size_t capacity;
        size_t offset;
        Block* next;

        Block(size_t cap) : capacity(cap), offset(0), next(nullptr) {
// Allocate cache-line aligned memory
#ifdef _WIN32
            buffer = static_cast<char*>(_aligned_malloc(cap, CACHE_LINE_ALIGNMENT));
            if (!buffer) {
                throw std::bad_alloc();
            }
#else
            if (posix_memalign(reinterpret_cast<void**>(&buffer), CACHE_LINE_ALIGNMENT, cap) != 0) {
                throw std::bad_alloc();
            }
#endif

            // Track allocation in global budget
            MemoryBudget::global().allocate(cap);
        }

        ~Block() {
            // Track deallocation in global budget
            MemoryBudget::global().deallocate(capacity);

#ifdef _WIN32
            _aligned_free(buffer);
#else
            free(buffer);
#endif

            // Don't recursively delete - we do it iteratively in GrowableArena destructor
            next = nullptr;
        }

        [[nodiscard]] bool can_fit(size_t bytes) const {
            return offset + bytes <= capacity;
        }

        char* allocate_raw(size_t bytes) {
            if (!can_fit(bytes)) {
                return nullptr;
            }
            char* ptr = buffer + offset;
            offset += bytes;
            return ptr;
        }

        void reset() {
            offset = 0;
        }
    };

public:
    // Alignment requirements (same as fixed Arena)
    static constexpr size_t SIMD_ALIGNMENT = 16;        // 128-bit NEON vectors (ARM)
    static constexpr size_t CACHE_LINE_ALIGNMENT = 64;  // Cache line size
    static constexpr size_t ALIGNMENT = CACHE_LINE_ALIGNMENT;

    // Growth strategy constants
    static constexpr size_t MAX_BLOCK_SIZE_MB = 256;  // Cap individual block size
    static constexpr size_t MAX_BLOCK_SIZE = MAX_BLOCK_SIZE_MB * 1024 * 1024;
    static constexpr float GROWTH_FACTOR = 1.5f;

private:
    Block* first_block_;
    Block* current_block_;
    size_t default_block_size_;
    size_t total_capacity_;
    size_t peak_usage_;
    std::string debug_name_;

    // Align size up to ALIGNMENT boundary
    static size_t align_up(size_t bytes) {
        return (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }

    // Allocate new block (called when current block is full)
    Block* allocate_new_block(size_t min_size) {
        // New block size = max(min_size, previous_block_size * 1.5, MAX_BLOCK_SIZE)
        size_t new_size =
            std::max(min_size, static_cast<size_t>(current_block_->capacity * GROWTH_FACTOR));
        new_size = std::min(new_size, MAX_BLOCK_SIZE);

        // If request exceeds MAX_BLOCK_SIZE, allocate exact-fit block
        if (min_size > MAX_BLOCK_SIZE) {
            new_size = min_size;
        }

        Block* new_block = new Block(new_size);
        total_capacity_ += new_size;

        return new_block;
    }

public:
    /**
     * Create growable arena with initial block size.
     *
     * @param initial_mb Initial block size in MB (default: 32 MB)
     * @param name Debug name for profiling (default: "unnamed")
     */
    explicit GrowableArena(size_t initial_mb = 32, const char* name = "unnamed")
        : default_block_size_(initial_mb * 1024 * 1024),
          total_capacity_(default_block_size_),
          peak_usage_(0),
          debug_name_(name) {
        first_block_ = new Block(default_block_size_);
        current_block_ = first_block_;
    }

    /**
     * Destructor - frees all blocks iteratively.
     */
    ~GrowableArena() {
        // Iteratively delete blocks to avoid stack overflow from deep recursion
        Block* block = first_block_;
        while (block != nullptr) {
            Block* next = block->next;
            delete block;
            block = next;
        }
    }

    // Move-only (no copy)
    GrowableArena(GrowableArena&& other) noexcept
        : first_block_(other.first_block_),
          current_block_(other.current_block_),
          default_block_size_(other.default_block_size_),
          total_capacity_(other.total_capacity_),
          peak_usage_(other.peak_usage_),
          debug_name_(std::move(other.debug_name_)) {
        other.first_block_ = nullptr;
        other.current_block_ = nullptr;
        other.total_capacity_ = 0;
        other.peak_usage_ = 0;
    }

    GrowableArena& operator=(GrowableArena&& other) noexcept {
        if (this != &other) {
            // Free our blocks
            Block* block = first_block_;
            while (block != nullptr) {
                Block* next = block->next;
                delete block;
                block = next;
            }

            // Take ownership of other's blocks
            first_block_ = other.first_block_;
            current_block_ = other.current_block_;
            default_block_size_ = other.default_block_size_;
            total_capacity_ = other.total_capacity_;
            peak_usage_ = other.peak_usage_;
            debug_name_ = std::move(other.debug_name_);

            other.first_block_ = nullptr;
            other.current_block_ = nullptr;
            other.total_capacity_ = 0;
            other.peak_usage_ = 0;
        }
        return *this;
    }

    GrowableArena(const GrowableArena&) = delete;
    GrowableArena& operator=(const GrowableArena&) = delete;

    /**
     * Allocate array of T from arena (auto-grows if needed).
     *
     * @param count Number of elements to allocate
     * @return Pointer to aligned memory
     * @throws std::bad_alloc if system runs out of memory
     */
    template <typename T>
    T* allocate(size_t count) {
        if (count == 0) {
            return nullptr;
        }

        size_t bytes = count * sizeof(T);
        size_t aligned_bytes = align_up(bytes);

        // Try current block first
        char* ptr = current_block_->allocate_raw(aligned_bytes);
        if (ptr) {
            // Update peak usage
            size_t current_usage = used();
            if (current_usage > peak_usage_) {
                peak_usage_ = current_usage;
            }
            return reinterpret_cast<T*>(ptr);
        }

        // Current block full - allocate new block
        Block* new_block = allocate_new_block(aligned_bytes);
        current_block_->next = new_block;
        current_block_ = new_block;

        // Allocate from new block (guaranteed to succeed)
        ptr = current_block_->allocate_raw(aligned_bytes);
        if (!ptr) {
            throw std::bad_alloc();  // Shouldn't happen
        }

        // Update peak usage
        size_t current_usage = used();
        if (current_usage > peak_usage_) {
            peak_usage_ = current_usage;
        }

        return reinterpret_cast<T*>(ptr);
    }

    /**
     * Reset arena to beginning (rewinds all blocks, doesn't deallocate).
     *
     * O(num_blocks) operation. Typically < 10 blocks, so very fast.
     */
    void reset() {
        Block* block = first_block_;
        while (block != nullptr) {
            block->reset();
            block = block->next;
        }
        current_block_ = first_block_;
    }

    /**
     * Checkpoint: Save current state for later restoration.
     *
     * Used by ScopedGrowableArena for RAII-style temporary allocations.
     * Stores the current block and its offset.
     *
     * @return Opaque checkpoint value (block pointer + offset)
     */
    struct Checkpoint {
        Block* block;
        size_t offset;
    };

    [[nodiscard]] Checkpoint get_checkpoint() const {
        return {current_block_, current_block_->offset};
    }

    /**
     * Restore arena to a previous checkpoint.
     *
     * Rewinds current block to checkpoint offset and resets all later blocks.
     * Does not deallocate memory.
     *
     * @param checkpoint Checkpoint from get_checkpoint()
     */
    void restore_checkpoint(const Checkpoint& checkpoint) {
        // Restore current block offset
        checkpoint.block->offset = checkpoint.offset;
        current_block_ = checkpoint.block;

        // Reset all blocks after checkpoint
        Block* block = checkpoint.block->next;
        while (block != nullptr) {
            block->reset();
            block = block->next;
        }
    }

    // Allow ScopedGrowableArena to access checkpoint methods
    friend class ScopedGrowableArena;

    /**
     * Shrink to fit: deallocate unused blocks beyond first block.
     *
     * Useful after peak usage has passed to reclaim memory.
     */
    void shrink_to_fit() {
        if ((first_block_ == nullptr) || (first_block_->next == nullptr)) {
            return;  // Only one block or no blocks
        }

        // Keep first block, deallocate rest
        Block* block = first_block_->next;
        while (block != nullptr) {
            Block* next = block->next;
            total_capacity_ -= block->capacity;
            delete block;
            block = next;
        }

        first_block_->next = nullptr;
        current_block_ = first_block_;
        first_block_->reset();
    }

    /**
     * Get total capacity (sum of all block capacities).
     */
    [[nodiscard]] size_t capacity() const {
        return total_capacity_;
    }

    /**
     * Get current usage (sum of offsets in all blocks).
     */
    [[nodiscard]] size_t used() const {
        size_t total = 0;
        Block* block = first_block_;
        while (block != nullptr) {
            total += block->offset;
            block = block->next;
        }
        return total;
    }

    /**
     * Get peak usage since construction or last reset.
     */
    [[nodiscard]] size_t peak() const {
        return peak_usage_;
    }

    /**
     * Get debug name.
     */
    [[nodiscard]] const char* name() const {
        return debug_name_.c_str();
    }

    /**
     * Get number of blocks currently allocated.
     */
    [[nodiscard]] size_t num_blocks() const {
        size_t count = 0;
        Block* block = first_block_;
        while (block != nullptr) {
            count++;
            block = block->next;
        }
        return count;
    }

    /**
     * Print usage statistics (for debugging).
     */
    void print_stats(FILE* out = stderr) const {
        fprintf(out, "GrowableArena '%s':\n", debug_name_.c_str());
        fprintf(out, "  Capacity: %zu MB (%zu blocks)\n", capacity() / (1024 * 1024), num_blocks());
        fprintf(out, "  Used: %zu MB (%.1f%%)\n", used() / (1024 * 1024),
                100.0 * used() / std::max<size_t>(1, capacity()));
        fprintf(out, "  Peak: %zu MB (%.1f%%)\n", peak() / (1024 * 1024),
                100.0 * peak() / std::max<size_t>(1, capacity()));
    }

    /**
     * TEMPORARY: Cast to Arena* for backward compatibility.
     *
     * This is unsafe but allows existing code to compile during migration.
     * Only works if GrowableArena has exactly one block.
     * Will be removed in Step 5 when all code is migrated.
     *
     * @return Pointer to this arena cast as Arena* (UNSAFE!)
     */
    void* as_arena_ptr() {
        return static_cast<void*>(this);
    }
};

/**
 * RAII wrapper for temporary arena allocations (like std::lock_guard).
 *
 * Saves a checkpoint on construction and restores it on destruction.
 * Useful for loop-local allocations that should be freed each iteration.
 *
 * Example:
 *   for (int i = 0; i < N; i++) {
 *       ScopedGrowableArena scope(arena);
 *       float* temp = scope.allocate<float>(1000);  // or arena.allocate<float>(1000)
 *       // ... use temp ...
 *       // Automatic cleanup on scope exit
 *   }
 */
class ScopedGrowableArena {
public:
    explicit ScopedGrowableArena(GrowableArena& arena)
        : arena_(arena), checkpoint_(arena.get_checkpoint()) {
    }

    ~ScopedGrowableArena() {
        arena_.restore_checkpoint(checkpoint_);
    }

    /**
     * Allocate from the underlying arena (convenience forwarding method).
     *
     * Equivalent to arena.allocate<T>(count).
     */
    template <typename T>
    T* allocate(size_t count) {
        return arena_.allocate<T>(count);
    }

    // Non-copyable, non-movable
    ScopedGrowableArena(const ScopedGrowableArena&) = delete;
    ScopedGrowableArena& operator=(const ScopedGrowableArena&) = delete;
    ScopedGrowableArena(ScopedGrowableArena&&) = delete;
    ScopedGrowableArena& operator=(ScopedGrowableArena&&) = delete;

private:
    GrowableArena& arena_;
    GrowableArena::Checkpoint checkpoint_;
};

} // namespace pfalign::memory

