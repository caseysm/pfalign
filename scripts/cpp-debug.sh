#!/bin/bash
# Development C++ build script using Meson (debug mode)

set -e

# Build directory depends on whether profiling is enabled
# Will be updated after argument parsing
BUILD_DIR="build-debug"
# Detect number of CPUs (works on Linux and macOS)
if command -v nproc &> /dev/null; then
    DEFAULT_JOBS=$(nproc)
elif command -v sysctl &> /dev/null; then
    DEFAULT_JOBS=$(sysctl -n hw.ncpu)
else
    DEFAULT_JOBS=4
fi
NUM_JOBS="${NUM_JOBS:-$DEFAULT_JOBS}"
ENABLE_PROFILING=false

# Parse arguments
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Fast incremental build for development (debug mode).

OPTIONS:
    --profiling         Enable profiling instrumentation (ENABLE_PROFILING=1)
                        See docs/profiling/PROFILING_GUIDE.md for usage
    --help, -h          Show this help message

EXAMPLES:
    $0                  # Standard debug build
    $0 --profiling      # Debug build with profiling enabled

ENVIRONMENT:
    NUM_JOBS            Number of parallel jobs (default: nproc)

EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --profiling)
            ENABLE_PROFILING=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

# Update build directory based on profiling flag
if [ "${ENABLE_PROFILING}" = true ]; then
    BUILD_DIR="build-debug-profiling"
fi

# Setup build directory on first run or reconfigure if profiling changes
NEEDS_RECONFIGURE=false
if [ ! -d "${BUILD_DIR}" ]; then
    echo "=== First-time setup ==="
    NEEDS_RECONFIGURE=true
elif [ "${ENABLE_PROFILING}" = true ]; then
    # Check if profiling is currently disabled
    if ! grep -q '"name": "profiling"' "${BUILD_DIR}/meson-info/intro-buildoptions.json" 2>/dev/null || \
       ! grep -A1 '"name": "profiling"' "${BUILD_DIR}/meson-info/intro-buildoptions.json" | grep -q '"value": true'; then
        echo "=== Reconfiguring with profiling enabled ==="
        NEEDS_RECONFIGURE=true
    fi
elif [ "${ENABLE_PROFILING}" = false ]; then
    # Check if profiling is currently enabled
    if grep -A1 '"name": "profiling"' "${BUILD_DIR}/meson-info/intro-buildoptions.json" 2>/dev/null | grep -q '"value": true'; then
        echo "=== Reconfiguring with profiling disabled ==="
        NEEDS_RECONFIGURE=true
    fi
fi

if [ "${NEEDS_RECONFIGURE}" = true ]; then
    if [ ! -d "${BUILD_DIR}" ]; then
        # First-time setup
        if [ "${ENABLE_PROFILING}" = true ]; then
            meson setup "${BUILD_DIR}" --buildtype=debug -Dprofiling=true
        else
            meson setup "${BUILD_DIR}" --buildtype=debug -Dprofiling=false
        fi
    else
        # Reconfigure existing build
        if [ "${ENABLE_PROFILING}" = true ]; then
            meson configure "${BUILD_DIR}" -Dprofiling=true
        else
            meson configure "${BUILD_DIR}" -Dprofiling=false
        fi
    fi

    if [ "${ENABLE_PROFILING}" = true ]; then
        echo "✓ Profiling enabled (ENABLE_PROFILING=1)"
        echo "  See docs/profiling/PROFILING_GUIDE.md for usage"
    fi
fi

# Incremental build
echo "Building (${NUM_JOBS} jobs)..."
time meson compile -C "${BUILD_DIR}" -j "${NUM_JOBS}"

echo ""
echo "Build complete. Run tests:"
echo "  meson test -C ${BUILD_DIR}"
echo ""
echo "CLI binary:"
echo "  ${BUILD_DIR}/cli/pfalign"
echo ""
echo "Python extension:"
echo "  Build wheel: python -m build --wheel"
echo "  Install: pip install dist/*.whl"
echo ""
