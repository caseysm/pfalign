#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pfalign {

// Compile-time enable/disable profiling
#ifndef ENABLE_PROFILING
    #define ENABLE_PROFILING 0
#endif

#if ENABLE_PROFILING
// Thread-local scope stack for automatic hierarchy tracking
namespace {
thread_local std::vector<std::string_view> scope_stack;
}

// Export accessor for external use (e.g., thread_pool.h CPU tracking)
inline const std::vector<std::string_view>& get_scope_stack() {
    return scope_stack;
}
#endif

/**
 * Hierarchical timing data for a named scope
 */
struct TimingRecord {
    std::string_view name;  // Points to canonical storage in ProfilingData
    double total_time_ms = 0.0;
    int64_t call_count = 0;
    double min_time_ms = 1e9;
    double max_time_ms = 0.0;

    // Hierarchical structure - tracks ALL parents this scope has been called from
    std::set<std::string_view> parents;   // All unique parents (point to canonical storage)
    std::set<std::string_view> children;  // All unique children (point to canonical storage)

    // Statistics
    double mean_time_ms() const {
        return call_count > 0 ? total_time_ms / call_count : 0.0;
    }

    void update(double elapsed_ms) {
        total_time_ms += elapsed_ms;
        call_count++;
        min_time_ms = std::min(min_time_ms, elapsed_ms);
        max_time_ms = std::max(max_time_ms, elapsed_ms);
    }
};

/**
 * Global profiling data store
 * Thread-safe accumulation of timing records
 */
class ProfilingData {
public:
    static ProfilingData& instance() {
        static ProfilingData instance;
        return instance;
    }

    void record(std::string_view name, double elapsed_ms, std::string_view parent = "") {
        std::lock_guard<std::mutex> lock(mutex_);

        // Canonicalize names (allocate once per unique string)
        auto canonical_name = canonicalize(name);
        auto canonical_parent = parent.empty() ? std::string_view{} : canonicalize(parent);

        // Update flat timing record
        auto& record = records_[canonical_name];
        record.name = canonical_name;
        record.update(elapsed_ms);

        // Track parent-child relationship (allows multiple parents)
        if (!canonical_parent.empty()) {
            // Add parent to this scope's parent set
            record.parents.insert(canonical_parent);

            // Add this scope to parent's children set
            auto& parent_record = records_[canonical_parent];
            parent_record.children.insert(canonical_name);
        }
    }

    // IMPORTANT: This method is NOT thread-safe with concurrent record() calls.
    // Only call after all profiling scopes have completed (e.g., at program end).
    const std::unordered_map<std::string_view, TimingRecord>& records() const {
        return records_;
    }

    // Thread-safe snapshot for reading while profiling is active
    std::unordered_map<std::string_view, TimingRecord> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return records_;  // Returns a copy
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.clear();
        canonical_names_.clear();
    }

    // Print timing report to console
    // Thread-safe: takes snapshot under lock before printing
    void print_report(std::ostream& os = std::cout, bool hierarchical = true) const {
        auto records_snapshot = snapshot();

        if (records_snapshot.empty()) {
            os << "No profiling data recorded.\n";
            return;
        }

        os << "\n========================================\n";
        os << "  Profiling Report\n";
        os << "========================================\n\n";

        if (hierarchical) {
            print_hierarchical(os, records_snapshot);
        } else {
            print_flat(os, records_snapshot);
        }
    }

    // Export to JSON
    // Thread-safe: takes snapshot under lock before exporting
    std::string to_json() const {
        auto records_snapshot = snapshot();

        std::ostringstream oss;
        oss << "{\n  \"timing_records\": [\n";

        bool first = true;
        for (const auto& [name, record] : records_snapshot) {
            if (!first)
                oss << ",\n";
            first = false;

            oss << "    {\n";
            oss << "      \"name\": \"" << name << "\",\n";
            oss << "      \"total_time_ms\": " << record.total_time_ms << ",\n";
            oss << "      \"call_count\": " << record.call_count << ",\n";
            oss << "      \"mean_time_ms\": " << record.mean_time_ms() << ",\n";
            oss << "      \"min_time_ms\": " << record.min_time_ms << ",\n";
            oss << "      \"max_time_ms\": " << record.max_time_ms << ",\n";
            oss << "      \"parents\": [";
            bool first_parent = true;
            for (const auto& parent : record.parents) {
                if (!first_parent)
                    oss << ", ";
                first_parent = false;
                oss << "\"" << parent << "\"";
            }
            oss << "],\n";
            oss << "      \"children\": [";
            bool first_child = true;
            for (const auto& child : record.children) {
                if (!first_child)
                    oss << ", ";
                first_child = false;
                oss << "\"" << child << "\"";
            }
            oss << "]\n";
            oss << "    }";
        }

        oss << "\n  ]\n}\n";
        return oss.str();
    }

    // Export to CSV
    // Thread-safe: takes snapshot under lock before exporting
    std::string to_csv() const {
        auto records_snapshot = snapshot();

        std::ostringstream oss;
        oss << "name,total_time_ms,call_count,mean_time_ms,min_time_ms,max_time_ms,parents\n";

        for (const auto& [name, record] : records_snapshot) {
            oss << name << "," << record.total_time_ms << "," << record.call_count << ","
                << record.mean_time_ms() << "," << record.min_time_ms << "," << record.max_time_ms
                << ",\"";

            // Export parents as semicolon-separated list in quotes
            bool first_parent = true;
            for (const auto& parent : record.parents) {
                if (!first_parent)
                    oss << ";";
                first_parent = false;
                oss << parent;
            }
            oss << "\"\n";
        }

        return oss.str();
    }

private:
    ProfilingData() = default;

    // Canonicalize a name - get or create canonical string_view
    // All string_views point into canonical_names_ set for stable lifetime
    std::string_view canonicalize(std::string_view name) {
        // Convert to string for lookup (C++17 doesn't have heterogeneous lookup for unordered_set
        // by default)
        std::string name_str(name);

        // Try to find or insert
        auto [insert_it, inserted] = canonical_names_.insert(name_str);

        // Return string_view pointing to the stable string in the set
        return std::string_view(*insert_it);
    }

    void
    print_flat(std::ostream& os,
               const std::unordered_map<std::string_view, TimingRecord>& records_snapshot) const {
        os << std::left << std::setw(40) << "Scope" << std::right << std::setw(12) << "Total (ms)"
           << std::setw(10) << "Calls" << std::setw(12) << "Mean (ms)" << std::setw(12)
           << "Min (ms)" << std::setw(12) << "Max (ms)" << "\n";
        os << std::string(98, '-') << "\n";

        // Sort by total time
        std::vector<std::pair<std::string_view, TimingRecord>> sorted;
        for (const auto& [name, record] : records_snapshot) {
            sorted.push_back({name, record});
        }
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.second.total_time_ms > b.second.total_time_ms;
        });

        for (const auto& [name, record] : sorted) {
            os << std::left << std::setw(40) << name << std::right << std::fixed
               << std::setprecision(2) << std::setw(12) << record.total_time_ms << std::setw(10)
               << record.call_count << std::setw(12) << record.mean_time_ms() << std::setw(12)
               << record.min_time_ms << std::setw(12) << record.max_time_ms << "\n";
        }
    }

    void print_hierarchical(
        std::ostream& os,
        const std::unordered_map<std::string_view, TimingRecord>& records_snapshot) const {
        // Find root nodes (scopes with no parents)
        std::vector<std::string_view> roots;
        for (const auto& [name, record] : records_snapshot) {
            if (record.parents.empty()) {
                roots.push_back(name);
            }
        }

        // Sort roots by total time
        std::sort(roots.begin(), roots.end(),
                  [&records_snapshot](const std::string_view& a, const std::string_view& b) {
                      return records_snapshot.at(a).total_time_ms >
                             records_snapshot.at(b).total_time_ms;
                  });

        for (const auto& root : roots) {
            print_node(os, root, 0, records_snapshot);
        }
    }

    void
    print_node(std::ostream& os, const std::string_view& name, int depth,
               const std::unordered_map<std::string_view, TimingRecord>& records_snapshot) const {
        auto it = records_snapshot.find(name);
        if (it == records_snapshot.end())
            return;

        const auto& record = it->second;

        std::string indent(depth * 2, ' ');
        os << indent;
        if (depth > 0)
            os << "+- ";

        os << std::left << std::setw(40 - depth * 2) << name << std::right << std::fixed
           << std::setprecision(2) << std::setw(12) << record.total_time_ms << " ms"
           << " (" << record.call_count << " calls, "
           << "mean=" << record.mean_time_ms() << " ms)\n";

        // Print children
        for (const auto& child : record.children) {
            print_node(os, child, depth + 1, records_snapshot);
        }
    }

    // Canonical string storage - all string_views point here
    std::unordered_set<std::string> canonical_names_;

    // Records keyed by canonical string_view
    std::unordered_map<std::string_view, TimingRecord> records_;

    mutable std::mutex mutex_;
};

/**
 * RAII scope timer
 * Automatically records timing on destruction
 *
 * IMPORTANT: Hierarchy is now tracked automatically via thread-local scope stack.
 * The explicit parent parameter is deprecated but kept for backward compatibility.
 */
class ProfileScope {
public:
    // New constructor: uses string_view + automatic hierarchy from scope stack
    explicit ProfileScope(std::string_view name)
        : name_(name), start_(std::chrono::high_resolution_clock::now()), use_auto_parent_(true) {
#if ENABLE_PROFILING
        // Push onto thread-local scope stack for automatic hierarchy
        scope_stack.push_back(name_);
#else
        // When profiling is disabled, mark as unused to avoid warnings
        (void)name_;
        (void)start_;
        (void)use_auto_parent_;
#endif
    }

    // Legacy constructor: explicit parent (deprecated, for backward compatibility)
    explicit ProfileScope(const std::string& name, const std::string& parent)
        : name_(name),
          explicit_parent_(parent),
          start_(std::chrono::high_resolution_clock::now()),
          use_auto_parent_(false) {
#if ENABLE_PROFILING
        // Still push onto stack for nested scope tracking
        scope_stack.push_back(name_);
#else
        (void)name_;
        (void)explicit_parent_;
        (void)start_;
        (void)use_auto_parent_;
#endif
    }

    ~ProfileScope() {
#if ENABLE_PROFILING
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(end - start_).count();

        // Pop from scope stack
        if (!scope_stack.empty()) {
            scope_stack.pop_back();
        }

        // Determine parent: either from stack (automatic) or explicit
        std::string_view parent;
        if (use_auto_parent_) {
            // Get parent from remaining scope stack (automatic hierarchy)
            parent = scope_stack.empty() ? std::string_view{} : scope_stack.back();
        } else {
            // Use explicit parent (legacy path)
            parent = explicit_parent_;
        }

        // Record with computed parent (string_view - no allocation!)
        ProfilingData::instance().record(name_, elapsed, parent);
#endif
    }

    // Non-copyable, non-movable
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
    ProfileScope(ProfileScope&&) = delete;
    ProfileScope& operator=(ProfileScope&&) = delete;

private:
    std::string_view name_;
    std::string explicit_parent_;  // Only used in legacy constructor
    std::chrono::high_resolution_clock::time_point start_;
    bool use_auto_parent_;
};

// Convenience macro for creating scoped timers
#if ENABLE_PROFILING
    #define PROFILE_SCOPE(name) pfalign::ProfileScope _profile_scope_##__LINE__(name)
    #define PROFILE_SCOPE_WITH_PARENT(name, parent) \
        pfalign::ProfileScope _profile_scope_##__LINE__(name, parent)
#else
    #define PROFILE_SCOPE(name) ((void)0)
    #define PROFILE_SCOPE_WITH_PARENT(name, parent) ((void)0)
#endif

// Convenience function to print report
inline void print_profiling_report(std::ostream& os = std::cout, bool hierarchical = true) {
#if ENABLE_PROFILING
    ProfilingData::instance().print_report(os, hierarchical);
#else
    os << "Profiling is disabled. Compile with -DENABLE_PROFILING=1 to enable.\n";
#endif
}

// Convenience function to export profiling data
inline void export_profiling_json(const std::string& filename) {
#if ENABLE_PROFILING
    std::ofstream ofs(filename);
    if (ofs.is_open()) {
        ofs << ProfilingData::instance().to_json();
        ofs.close();
    }
#else
    (void)filename;
#endif
}

inline void export_profiling_csv(const std::string& filename) {
#if ENABLE_PROFILING
    std::ofstream ofs(filename);
    if (ofs.is_open()) {
        ofs << ProfilingData::instance().to_csv();
        ofs.close();
    }
#else
    (void)filename;
#endif
}

}  // namespace pfalign
