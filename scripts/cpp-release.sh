#!/bin/bash
# Production C++ build script using Meson (release mode)

set -e

# Configuration
BUILD_DIR="build-release"
BUILDTYPE="${BUILDTYPE:-release}"
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

Production build script using Meson (release mode by default).

OPTIONS:
    --profiling         Enable profiling instrumentation (ENABLE_PROFILING=1)
                        See docs/profiling/PROFILING_GUIDE.md for usage
    --help, -h          Show this help message

EXAMPLES:
    $0                  # Standard release build
    $0 --profiling      # Release build with profiling enabled

ENVIRONMENT:
    BUILDTYPE           Build type: release, debug (default: release)
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
    BUILD_DIR="build-release-profiling"
fi

echo "=== PFalign Build Script (Meson) ==="
echo "Build type: ${BUILDTYPE}"
echo "Jobs: ${NUM_JOBS}"
if [ "${ENABLE_PROFILING}" = true ]; then
    echo "Profiling: ENABLED"
fi
echo ""

# Check if Meson is installed
if ! command -v meson &> /dev/null; then
    echo "Error: Meson not found. Install with:"
    echo "  pip install meson ninja"
    exit 1
fi

# Check if Ninja is installed
if ! command -v ninja &> /dev/null; then
    echo "Error: Ninja not found. Install with:"
    echo "  Ubuntu/Debian: sudo apt-get install ninja-build"
    echo "  Fedora/RHEL:   sudo dnf install ninja-build"
    echo "  macOS:         brew install ninja"
    echo "  Or:            pip install ninja"
    exit 1
fi

# Setup build directory on first run or reconfigure if profiling changes
NEEDS_RECONFIGURE=false
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Setting up build directory..."
    NEEDS_RECONFIGURE=true
elif [ "${ENABLE_PROFILING}" = true ]; then
    # Check if profiling is currently disabled
    if ! grep -q '"name": "profiling"' "${BUILD_DIR}/meson-info/intro-buildoptions.json" 2>/dev/null || \
       ! grep -A1 '"name": "profiling"' "${BUILD_DIR}/meson-info/intro-buildoptions.json" | grep -q '"value": true'; then
        echo "Reconfiguring with profiling enabled..."
        NEEDS_RECONFIGURE=true
    fi
elif [ "${ENABLE_PROFILING}" = false ]; then
    # Check if profiling is currently enabled
    if grep -A1 '"name": "profiling"' "${BUILD_DIR}/meson-info/intro-buildoptions.json" 2>/dev/null | grep -q '"value": true'; then
        echo "Reconfiguring with profiling disabled..."
        NEEDS_RECONFIGURE=true
    fi
fi

if [ "${NEEDS_RECONFIGURE}" = true ]; then
    if [ ! -d "${BUILD_DIR}" ]; then
        # First-time setup
        if [ "${ENABLE_PROFILING}" = true ]; then
            meson setup "${BUILD_DIR}" --buildtype="${BUILDTYPE}" -Dprofiling=true
        else
            meson setup "${BUILD_DIR}" --buildtype="${BUILDTYPE}" -Dprofiling=false
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

# Build with Meson
echo "Building with Meson (${NUM_JOBS} jobs)..."
meson compile -C "${BUILD_DIR}" -j "${NUM_JOBS}"

# Run tests
echo "Running C++ tests..."
meson test -C "${BUILD_DIR}"

echo "Build complete!"
echo ""
echo "Executables:"
echo "  - CLI binary: ${BUILD_DIR}/cli/pfalign"
echo ""
echo "Python wheel:"
echo "  - Build: python -m build --wheel"
echo "  - Install: pip install dist/*.whl"
echo ""
echo "Test:"
echo "  meson test -C ${BUILD_DIR}"
echo ""
