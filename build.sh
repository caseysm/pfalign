#!/bin/bash
# PFalign Unified Build Script
#
# Single entry point for building PFalign in various configurations.
# Use this script to build C++ CLI, Python package, or both.

set -e

show_help() {
    cat << EOF
PFalign Build Script

USAGE:
    $0 [TARGET] [OPTIONS]

TARGETS:
    cpp          Build C++ CLI (release mode, optimized)
    cpp-dev      Build C++ CLI (debug mode, for development)
    python       Build Python package (.whl)
    all          Build both C++ and Python (default)

OPTIONS:
    --profiling    Enable profiling instrumentation (C++ only)
    --help, -h     Show this help message

EXAMPLES:
    # Build everything (C++ + Python)
    $0
    $0 all

    # Build only C++ CLI (optimized)
    $0 cpp

    # Build C++ for development (debug, faster compilation)
    $0 cpp-dev

    # Build only Python package
    $0 python

    # Build C++ with profiling enabled
    $0 cpp --profiling
    $0 cpp-dev --profiling

OUTPUT LOCATIONS:
    C++ CLI (release):       build-release/cli/pfalign
    C++ CLI (debug):         build-debug/cli/pfalign
    Python wheel:            dist/pfalign-*.whl

EOF
    exit 0
}

# Parse arguments
TARGET="all"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        cpp|cpp-dev|python|all)
            TARGET=$1
            shift
            ;;
        --profiling)
            EXTRA_ARGS="$EXTRA_ARGS --profiling"
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

echo "=========================================="
echo "  PFalign Build System"
echo "=========================================="
echo ""

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Execute the appropriate build script
case $TARGET in
    cpp)
        echo "Building: C++ CLI (release mode)"
        echo ""
        exec "${SCRIPT_DIR}/scripts/cpp-release.sh" $EXTRA_ARGS
        ;;

    cpp-dev)
        echo "Building: C++ CLI (debug mode)"
        echo ""
        exec "${SCRIPT_DIR}/scripts/cpp-debug.sh" $EXTRA_ARGS
        ;;

    python)
        echo "Building: Python package"
        echo ""
        if [[ -n "$EXTRA_ARGS" ]]; then
            echo "Warning: --profiling is ignored for Python builds"
            echo ""
        fi
        exec "${SCRIPT_DIR}/scripts/python.sh"
        ;;

    all)
        echo "Building: C++ + Python (full build)"
        echo ""
        if [[ -n "$EXTRA_ARGS" ]]; then
            echo "Note: Profiling options apply only to C++ build"
            echo ""
        fi
        exec "${SCRIPT_DIR}/scripts/all.sh" $EXTRA_ARGS
        ;;

    *)
        echo "Error: Unknown target: $TARGET"
        echo "Run '$0 --help' for available targets."
        exit 1
        ;;
esac
