#include "pfalign/dispatch/execution_policy.h"

namespace pfalign {

ExecutionPolicy DeviceInfo::best_cpu_backend() const {
    // Scalar-only branch: always return Scalar
    return ExecutionPolicy::Scalar;
}

DeviceInfo get_device_info() {
    static DeviceInfo info;  // Cached
    return info;
}

}  // namespace pfalign
