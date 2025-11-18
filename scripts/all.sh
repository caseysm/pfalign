#!/bin/bash
# Unified build script - builds both C++ and Python packages
#
# This script provides a single entry point for building the entire project.
# It runs both the C++ production build and Python package build in sequence.
#
# Usage:
#   ./build-all.sh           # Build everything
#   ./build-all.sh --help    # Show help

set -e

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Unified build script that builds both C++ (release mode) and Python packages.

This is equivalent to running:
  ./scripts/cpp-release.sh && ./scripts/python.sh

OPTIONS:
    --cpp-only      Build only C++ (skip Python)
    --python-only   Build only Python (skip C++)
    --help, -h      Show this help message

EXAMPLES:
    $0                  # Build both C++ and Python
    $0 --cpp-only       # Build only C++
    $0 --python-only    # Build only Python

For more control, use the individual build scripts:
  - ./scripts/cpp-release.sh    Production C++ build (release)
  - ./scripts/cpp-debug.sh      Development C++ build (debug)
  - ./scripts/python.sh         Python package build

See BUILD_SYSTEM.md for complete build system documentation.
EOF
    exit 0
}

# Parse arguments
BUILD_CPP=true
BUILD_PYTHON=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpp-only)
            BUILD_PYTHON=false
            shift
            ;;
        --python-only)
            BUILD_CPP=false
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
echo "  PFalign Unified Build"
echo "=========================================="
echo ""

# Build C++
if [ "$BUILD_CPP" = true ]; then
    echo "[1/2] Building C++ (release mode)..."
    ./scripts/cpp-release.sh
    echo ""
fi

# Build Python
if [ "$BUILD_PYTHON" = true ]; then
    if [ "$BUILD_CPP" = true ]; then
        echo "[2/2] Building Python package..."
    else
        echo "[1/1] Building Python package..."
    fi
    ./scripts/python.sh
    echo ""
fi

echo "=========================================="
echo "  Build Complete!"
echo "=========================================="
echo ""

if [ "$BUILD_CPP" = true ]; then
    echo "C++ outputs:"
    echo "  - CLI binary: build-release/cli/pfalign"
    echo "  - Run tests: meson test -C build-release"
    echo ""
fi

if [ "$BUILD_PYTHON" = true ]; then
    echo "Python outputs:"
    echo "  - Wheel: dist/pfalign-*.whl"
    echo "  - Run tests: cd /tmp && python -m pytest <path-to-pfalign>/pfalign/tests/"
    echo ""
fi
