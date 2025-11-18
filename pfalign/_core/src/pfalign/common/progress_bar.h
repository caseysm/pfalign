#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>
#include <chrono>

namespace pfalign {
namespace common {

/**
 * APT-style progress bar for CLI operations
 *
 * Renders progress like:
 *   [########........] 40% (20/50) Processing embeddings...
 *
 * Thread-safe for single-threaded use. For multi-threaded scenarios,
 * external synchronization is required.
 */
class ProgressBar {
public:
    /**
     * Create a progress bar
     *
     * @param total Total number of items to process
     * @param description Optional description shown after percentage
     * @param width Width of the progress bar in characters (default: 20)
     * @param show_percent Show percentage (default: true)
     * @param show_count Show count like (20/50) (default: true)
     */
    explicit ProgressBar(int total,
                        const std::string& description = "",
                        int width = 20,
                        bool show_percent = true,
                        bool show_count = true)
        : total_(total)
        , current_(0)
        , description_(description)
        , width_(width)
        , show_percent_(show_percent)
        , show_count_(show_count)
        , finished_(false)
        , start_time_(std::chrono::steady_clock::now())
    {
        if (total_ <= 0) {
            total_ = 1;  // Prevent division by zero
        }
    }

    /**
     * Update progress to current count
     *
     * @param current Current progress count
     */
    void update(int current) {
        if (finished_) return;

        current_ = current;
        if (current_ > total_) {
            current_ = total_;
        }

        render();
    }

    /**
     * Increment progress by 1
     */
    void tick() {
        update(current_ + 1);
    }

    /**
     * Increment progress by specified amount
     *
     * @param delta Amount to increment
     */
    void tick(int delta) {
        update(current_ + delta);
    }

    /**
     * Mark progress as complete and clear the bar
     */
    void finish() {
        if (finished_) return;

        current_ = total_;
        render();
        std::cerr << "\n";  // Move to next line
        finished_ = true;
    }

    /**
     * Get current progress (0 to total)
     */
    int current() const { return current_; }

    /**
     * Get total count
     */
    int total() const { return total_; }

    /**
     * Get completion fraction (0.0 to 1.0)
     */
    double fraction() const {
        return static_cast<double>(current_) / static_cast<double>(total_);
    }

    /**
     * Check if finished
     */
    bool is_finished() const { return finished_; }

    /**
     * Update the description text
     *
     * @param description New description to display
     */
    void set_description(const std::string& description) {
        description_ = description;
    }

    /**
     * Reset progress for a new phase with different total
     *
     * @param new_total New total count
     * @param description New description
     */
    void reset(int new_total, const std::string& description) {
        total_ = new_total > 0 ? new_total : 1;
        current_ = 0;
        description_ = description;
        finished_ = false;
        start_time_ = std::chrono::steady_clock::now();
    }

private:
    void render() {
        // Calculate progress fraction
        double frac = fraction();
        int filled = static_cast<int>(std::round(frac * width_));

        // Build progress bar string
        std::string bar = "[";
        for (int i = 0; i < width_; ++i) {
            bar += (i < filled) ? "#" : ".";
        }
        bar += "]";

        // Build complete line
        std::ostringstream line;
        line << "\r" << bar;

        if (show_percent_) {
            line << " " << std::fixed << std::setprecision(0)
                 << (frac * 100.0) << "%";
        }

        if (show_count_) {
            line << " (" << current_ << "/" << total_ << ")";
        }

        if (!description_.empty()) {
            line << " " << description_;
        }

        // Add elapsed time for long operations
        auto elapsed = std::chrono::steady_clock::now() - start_time_;
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
        if (seconds > 5 && current_ > 0) {
            // Estimate remaining time
            double rate = static_cast<double>(current_) / seconds;
            int remaining_items = total_ - current_;
            int eta_seconds = static_cast<int>(remaining_items / rate);

            if (eta_seconds < 60) {
                line << " [ETA: " << eta_seconds << "s]";
            } else if (eta_seconds < 3600) {
                int minutes = eta_seconds / 60;
                line << " [ETA: " << minutes << "m]";
            } else {
                int hours = eta_seconds / 3600;
                int minutes = (eta_seconds % 3600) / 60;
                line << " [ETA: " << hours << "h " << minutes << "m]";
            }
        }

        // Clear to end of line and write
        line << "   ";  // Extra spaces to clear previous content
        std::cerr << line.str() << std::flush;
    }

    int total_;
    int current_;
    std::string description_;
    int width_;
    bool show_percent_;
    bool show_count_;
    bool finished_;
    std::chrono::steady_clock::time_point start_time_;
};

/**
 * RAII wrapper for progress bar that auto-finishes on destruction
 */
class ScopedProgressBar {
public:
    template<typename... Args>
    explicit ScopedProgressBar(Args&&... args)
        : bar_(std::forward<Args>(args)...)
    {}

    ~ScopedProgressBar() {
        if (!bar_.is_finished()) {
            bar_.finish();
        }
    }

    ProgressBar& get() { return bar_; }
    const ProgressBar& get() const { return bar_; }

    // Forward common methods
    void update(int current) { bar_.update(current); }
    void tick() { bar_.tick(); }
    void tick(int delta) { bar_.tick(delta); }
    void finish() { bar_.finish(); }

private:
    ProgressBar bar_;
};

}  // namespace common
}  // namespace pfalign
