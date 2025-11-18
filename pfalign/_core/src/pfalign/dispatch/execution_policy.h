#pragma once

namespace pfalign {

// Runtime backend selection (scalar-only on this branch)
enum class ExecutionPolicy {
    Auto,   // Automatic: resolves to Scalar
    Scalar  // Portable C++ scalar implementation
};

// Query available backends at runtime
struct DeviceInfo {
    ExecutionPolicy best_cpu_backend() const;  // Always returns Scalar
};

DeviceInfo get_device_info();

}  // namespace pfalign
