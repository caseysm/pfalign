#!/bin/bash
set -e

# Convenience script for building the PFalign Python wheel via meson-python

echo "=========================================="
echo "  Building PFalign (meson-python)"
echo "=========================================="
echo ""

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider activating a venv or conda environment first"
    echo ""
fi

WORKDIR=$(pwd)
DIST_DIR="${WORKDIR}/dist"
rm -rf "${DIST_DIR}"

echo "📦 Building wheel with meson-python..."
python -m build --wheel --no-isolation

WHEEL=$(ls dist/pfalign-*.whl | head -n 1)
if [[ -z "$WHEEL" ]]; then
    echo "❌ No wheel found in dist/"
    exit 1
fi

echo "📥 Installing ${WHEEL}"
pip install "${WHEEL}"

echo ""
echo "✅ PFalign built and installed!"
echo ""
echo "Smoke tests:"
echo "  python -c 'import pfalign; print(pfalign.__version__)'"
echo "  python -m pytest pfalign/tests -v"
echo ""
