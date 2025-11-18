# PFalign

Fast pairwise and multiple protein structure alignment using ProteinMPNN embeddings.

The standalone CLI has no runtime dependencies, and the Python package only needs NumPy.

[![Build Status](https://github.com/caseysm/pfalign/actions/workflows/test.yml/badge.svg)](https://github.com/caseysm/pfalign/actions/workflows/test.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

### Core Capabilities
- **Pairwise alignment** using Smith-Waterman with forward-backward algorithm and optimized kernels
- **Multiple sequence alignment (MSA)** via progressive alignment with multiple guide tree options (UPGMA, NJ, BioNJ, MST)
- **ProteinMPNN embeddings** for structural representation, with parallel batch processing
- **Trained gap parameters** optimized for soft alignment

### Usability
- Automatically detects input types (structure files vs. embeddings)
- Accepts single files, directories, file lists, or Python iterables
- Supports chain specification by ID ("A", "B") or index (0, 1)
- Shows progress bars for long-running operations
- Converts between common alignment formats (FASTA, A2M, A3M, Stockholm, PSI-BLAST, Clustal)
- Clear error messages with helpful diagnostics

### Specialized Modules
- **`pfalign.alignment`**: Low-level alignment primitives (forward, backward, viterbi_decode, etc.)
- **`pfalign.metrics`**: Quality metrics (RMSD, TM-score, GDT, lDDT, DALI, sequence identity, coverage, conservation)
- **`pfalign.structure`**: Structural superposition (Kabsch algorithm, transformation matrices)
- **`pfalign.tree`**: Phylogenetic tree building (UPGMA, Neighbor-Joining, BioNJ, MST)

### Two Interfaces
- **Python API** with native NumPy integration
- **Standalone CLI** (C++ binary, no Python needed at runtime)
- Both interfaces have the same functionality

### Minimal Dependencies
- **Standalone CLI**: Single binary, no dependencies
- **Python package**: Just NumPy (no PyTorch, TensorFlow, or other ML frameworks)
- **Embedded weights**: ProteinMPNN weights are built into the binary

## Installation

### From PyPI (Coming Soon)

```bash
pip install pfalign
```

Requires Python 3.9+ and NumPy 1.20+.

### From GitHub

Install directly from the repo:

```bash
# Install build dependencies first
pip install meson-python meson ninja pybind11 numpy

# Then install pfalign
pip install git+https://github.com/caseysm/pfalign.git
```

### From Source

Clone and build locally:

```bash
# Install build dependencies
pip install meson-python meson ninja pybind11 numpy

# Clone and build
git clone https://github.com/caseysm/pfalign.git
cd pfalign
./build.sh python
pip install dist/*.whl
```

### Development

For editable installs:

```bash
pip install meson-python meson ninja pybind11 numpy
git clone https://github.com/caseysm/pfalign.git
cd pfalign
pip install -e . --no-build-isolation
```

### Standalone CLI

Build the C++ binary (no Python needed):

```bash
# Install build dependencies (meson, ninja, C++ compiler)
pip install meson ninja

# Clone and build
git clone https://github.com/caseysm/pfalign.git
cd pfalign
./build.sh cpp
```

The binary will be at `build-release/cli/pfalign` (or `build-debug/cli/pfalign` for debug builds).

You can copy it anywhere:

```bash
cp build-release/cli/pfalign /usr/local/bin/

# Or just use it directly
./build-release/cli/pfalign pairwise protein1.pdb protein2.pdb
```

### Requirements

**Runtime:**
- Python 3.9-3.13
- NumPy 1.20+
- That's it (no PyTorch, TensorFlow, etc.)

**Building from source:**
- Python 3.9+ and NumPy 1.20+
- C++17 compiler (GCC 9+ or Clang 10+)
- Build tools: `meson`, `ninja`, `meson-python`, `pybind11`
- Works on Linux and macOS

**Standalone CLI:**
- No runtime dependencies
- Build needs: C++17 compiler, `meson`, `ninja`
- Works on Linux and macOS

## Quick Start

### Python API

```python
import pfalign
import numpy as np

# Pairwise alignment from structure files
result = pfalign.pairwise(
    "protein1.pdb",
    "protein2.pdb",
    chain1="A",  # Can use chain ID or index
    chain2=0,
    k_neighbors=30
)

print(f"Alignment score: {result.score()}")
print(f"Sequence identity: {result.compute_identity():.2%}")
print(f"Coverage: {result.compute_coverage():.2%}")

# Access alignment posteriors
posteriors = result.posteriors  # NumPy array (L1 x L2)

# Multiple sequence alignment
msa_result = pfalign.msa(
    ["protein1.pdb", "protein2.pdb", "protein3.pdb"],
    method="upgma",  # or "nj", "bionj", "mst"
    k_neighbors=30,
    ecs_temperature=5.0
)

print(f"Number of sequences: {msa_result.num_sequences()}")
print(f"Alignment length: {msa_result.alignment_length()}")

# Get consensus sequence
consensus = msa_result.get_consensus(threshold=0.5)

# Export to FASTA
fasta_str = msa_result.to_fasta()

# Encode a single structure
embedding = pfalign.encode(
    "protein.pdb",
    chain="A",
    k_neighbors=30
)
print(f"Embedding shape: {embedding.embeddings.shape}")  # (L, 64)

# Batch encode from a directory
output_paths = pfalign.encode_batch(
    "/path/to/structures/",  # Directory of PDB files
    output_dir="/path/to/embeddings/",
    k_neighbors=30
)
# Returns list of .npy file paths
```

### CLI Usage

```bash
# Pairwise alignment
pfalign pairwise protein1.pdb protein2.pdb \
    --chain1 A --chain2 A \
    -k 30 \
    --output alignment.txt

# Multiple sequence alignment
pfalign msa protein1.pdb protein2.pdb protein3.pdb \
    --method upgma \
    -k 30 \
    --output msa.fasta

# Encode structure to embeddings (positional args: input output)
pfalign encode protein.pdb embeddings.npy \
    --chain A \
    -k 30

# Compute similarity between two embeddings
pfalign similarity emb1.npy emb2.npy

# Compute distance matrix for multiple structures
pfalign compute-distances dir/ distances.npy -k 30

# Build guide tree from distance matrix
pfalign tree build distances.npy tree.newick --method upgma
# Or use specific method directly:
# pfalign tree upgma distances.npy tree.newick

# Structural superposition
pfalign structure superpose protein1.pdb protein2.pdb \
    --output superposed.pdb

# Format conversion
pfalign reformat input.a3m output.fasta \
    --input-format a3m \
    --output-format fasta

# Get structure info
pfalign info protein.pdb --chains

# Show version and build information
pfalign --version
```

### Advanced Commands

#### Batch Processing

```bash
# Batch encoding - encode multiple structures
pfalign batch encode input_dir/ output_dir/ -k 30

# Batch pairwise alignments
pfalign batch pairwise queries_dir/ targets_dir/ output_dir/ --mode all-vs-all
```

#### Metrics Commands

```bash
# RMSD between coordinate sets
pfalign metrics rmsd coords1.npy coords2.npy

# TM-score for structural similarity
pfalign metrics tm-score coords1.npy coords2.npy

# GDT-TS and GDT-HA (returns both scores)
pfalign metrics gdt coords1.npy coords2.npy

# GDT-TS only
pfalign metrics gdt-ts coords1.npy coords2.npy

# GDT-HA only
pfalign metrics gdt-ha coords1.npy coords2.npy

# lDDT (Local Distance Difference Test) - superposition-free
pfalign metrics lddt coords1.npy coords2.npy alignment.txt

# DALI score with Z-score
pfalign metrics dali-score coords1.npy coords2.npy alignment.txt

# Sequence identity from alignment
pfalign metrics identity seq1.txt seq2.txt alignment.txt

# Coverage from alignment
pfalign metrics coverage alignment.txt L1 L2

# Alignment statistics (comprehensive)
pfalign metrics stats alignment.txt

# Pairwise identity matrix for MSA
pfalign metrics pairwise-identity msa.fasta

# Conservation scores
pfalign metrics conservation msa.fasta

# ECS (Embedding Coherence Score)
pfalign metrics ecs emb1.npy emb2.npy
```

#### Structure Commands

```bash
# Superpose structures
pfalign structure superpose protein1.pdb protein2.pdb --output out.pdb

# Kabsch algorithm (get transformation matrix)
pfalign structure kabsch coords1.npy coords2.npy --output transform.txt

# Extract coordinates from structure
pfalign structure get-coords protein.pdb coords.npy --chain A

# Superpose using pre-computed alignment
pfalign structure superpose-from-alignment coords1.npy coords2.npy alignment.txt --output out.pdb
```

#### Alignment Commands

```bash
# Forward pass (Smith-Waterman)
pfalign alignment forward emb1.npy emb2.npy forward.npy

# Backward pass
pfalign alignment backward emb1.npy emb2.npy backward.npy

# Decode posteriors to alignment
pfalign alignment decode posteriors.npy alignment.txt
```

#### Tree Commands

```bash
# Build tree with specific method
pfalign tree build distances.npy tree.newick --method upgma

# Or use method-specific commands
pfalign tree upgma distances.npy tree.newick
pfalign tree nj distances.npy tree.newick
pfalign tree bionj distances.npy tree.newick
pfalign tree mst distances.npy tree.newick
```

## API Reference

### Core Functions

#### `pairwise(seq1, seq2, **kwargs) -> PairwiseResult`
Align two protein structures or embeddings.

**Parameters:**
- `seq1`, `seq2`: Structure files, embedding arrays, or paths
- `chain1`, `chain2`: Which chain to use (string ID or integer index)
- `k_neighbors`: MPNN k-neighbors (default: 30)
- `gap_open`, `gap_extend`: Gap penalties (defaults to trained values: -2.544, 0.194)
- `temperature`: MPNN temperature (default: 1.0)

**Returns:** `PairwiseResult` with score, posteriors, and metrics

#### `msa(sequences, method="upgma", **kwargs) -> MSAResult`
Align multiple structures.

**Parameters:**
- `sequences`: List of structure files or embeddings
- `method`: Guide tree method - "upgma", "nj", "bionj", or "mst" (default: "upgma")
- `k_neighbors`: MPNN k-neighbors (default: 30)
- `ecs_temperature`: Temperature for similarity calculation (default: 5.0)
- `arena_size_mb`: Memory limit for distance matrix (default: 200)
- `threads`: Parallel threads to use (default: all CPUs)

**Returns:** `MSAResult` with sequences, consensus, and export options

#### `encode(structure, chain=None, k_neighbors=30) -> EmbeddingResult`
Convert a structure to MPNN embeddings.

**Parameters:**
- `structure`: Structure file (.pdb or .cif)
- `chain`: Which chain (optional)
- `k_neighbors`: MPNN k-neighbors (default: 30)

**Returns:** `EmbeddingResult` with shape (L, 64)

#### `similarity(embeddings1, embeddings2) -> SimilarityResult`
Compute similarity between two embeddings.

**Parameters:**
- `embeddings1`, `embeddings2`: NumPy arrays (L1 × D) and (L2 × D)

**Returns:** `SimilarityResult` with similarity score

**Note:** For computing distance matrices between multiple sequences, use `compute_distances()` instead.

### Helper Functions

#### `encode_batch(inputs, *, output_dir, k_neighbors=30, chain=0) -> List[str]`
Encode multiple structures in parallel.

**Parameters:**
- `inputs`: Directory, list of files, or text file listing structures
- `output_dir`: Where to save .npy files (required)
- `k_neighbors`: MPNN k-neighbors (default: 30)
- `chain`: Which chain to use (default: 0)

**Returns:** List of .npy file paths

#### `batch_pairwise(queries, targets, *, output_dir, mode="all-vs-all", **kwargs) -> List[dict]`
Run many pairwise alignments at once.

**Parameters:**
- `queries`: Directory or list of query structures
- `targets`: Directory or list of targets
- `output_dir`: Where to save results (required)
- `mode`: "all-vs-all" or "one-to-one" (default: "all-vs-all")
- `k_neighbors`: MPNN k-neighbors (default: 30)

**Returns:** List of dicts with 'query', 'target', 'output', 'score'

#### `compute_distances(sequences, **kwargs) -> np.ndarray`
Get pairwise distances for building a guide tree.

**Parameters:**
- `sequences`: List of structures or embeddings
- `k_neighbors`: MPNN k-neighbors (default: 30)
- `ecs_temperature`: Temperature (default: 5.0)

**Returns:** Symmetric distance matrix (N × N)

#### `progressive_align(sequences, tree, **kwargs) -> MSAResult`
Run MSA with a custom guide tree.

**Parameters:**
- `sequences`: List of structures or embeddings
- `tree`: Pre-built tree (from `pfalign.tree.*`)
- `k_neighbors`: MPNN k-neighbors (default: 30)
- `ecs_temperature`: Temperature (default: 5.0)

**Returns:** `MSAResult`

#### `reformat(input_path, output_path, **kwargs)`
Convert between alignment formats (FASTA, A2M, A3M, Stockholm, PSI-BLAST, Clustal).

#### `info(path, chain=None) -> dict`
Get detailed information about a structure or directory.

#### `version_info() -> dict`
Get build information, version, and compiler details.

### Result Objects

All result objects give you:
- Direct access to output data
- Methods to compute quality metrics
- Export functions (to_fasta(), to_array(), etc.)
- Nice string representations for debugging

## Modules

### `pfalign.alignment`
Low-level alignment primitives for custom workflows:
- `forward()`: Forward pass (Smith-Waterman DP matrix)
- `backward()`: Backward pass for posterior computation
- `forward_backward()`: Combined forward-backward → (posteriors, score)
- `score()`: Fast scoring without posteriors
- `viterbi_decode()`: Hard alignment with gap characters
- `viterbi_path()`: Hard alignment path coordinates

### `pfalign.metrics`
Quality metrics for alignment evaluation:

**Sequence and alignment metrics:**
- `rmsd()`: Root-mean-square deviation
- `identity()`: Sequence identity percentage
- `coverage()`: Alignment coverage
- `alignment_stats()`: Comprehensive alignment statistics
- `ecs()`: Embedding Coherence Score (from MSAResult)
- `pairwise_identity()`: Identity matrix for MSA
- `conservation()`: Per-position conservation scores

**Structural similarity metrics:**
- `tm_score()`: TM-score for global fold similarity (range 0-1, >0.5 same fold)
- `gdt()`: Global Distance Test (returns both GDT-TS and GDT-HA)
- `gdt_ts()`: GDT Total Score (cutoffs: 1, 2, 4, 8 Å)
- `gdt_ha()`: GDT High Accuracy (cutoffs: 0.5, 1, 2, 4 Å)
- `lddt()`: Local Distance Difference Test (superposition-free, range 0-1)
- `dali_score()`: Distance Alignment score with Z-score (returns tuple of score and Z)

### `pfalign.structure`
Structural superposition and transformation:
- `kabsch()`: Kabsch algorithm (returns rotation, translation, RMSD)
- `transform()`: Apply rotation and translation to coordinates
- `get_coords()`: Extract Cα coordinates from structure file
- `superpose()`: Full superposition workflow
- `superpose_from_alignment()`: Superpose using pre-computed alignment

### `pfalign.tree`
Build phylogenetic trees from distance matrices:
- `upgma(distances)`: Unweighted Pair Group Method with Arithmetic Mean
- `nj(distances)`: Neighbor-Joining
- `bionj(distances)`: BioNJ (variance-weighted NJ)
- `mst(distances)`: Minimum Spanning Tree
- `build(distances, method)`: Generic interface

**Note:** These take distance matrices (N × N), not raw sequences. Use `compute_distances()` first.

## Examples

### Working with Pre-computed Embeddings

```python
import pfalign
import numpy as np

# Load pre-computed embeddings
emb1 = np.load("protein1_embeddings.npy")
emb2 = np.load("protein2_embeddings.npy")

# Align directly (no MPNN encoding needed)
result = pfalign.pairwise(emb1, emb2)
print(f"Score: {result.score()}")
```

### Custom Gap Parameters

```python
# Use custom gap penalties instead of trained defaults
result = pfalign.pairwise(
    "protein1.pdb",
    "protein2.pdb",
    chain1=0,
    chain2=0,
    gap_open=-1.5,
    gap_extend=-0.2,
    temperature=0.8
)
```

### Structural Quality Metrics

```python
import pfalign.metrics as metrics
import numpy as np

# Get aligned coordinates (from pairwise alignment)
coords1 = np.load("aligned_coords1.npy")  # Shape: (N, 3)
coords2 = np.load("aligned_coords2.npy")  # Shape: (N, 3)
alignment = np.load("alignment.npy")      # Shape: (M, 2)

# TM-score (global fold similarity)
tm = metrics.tm_score(coords1, coords2, len1=150, len2=145)
print(f"TM-score: {tm:.3f}")  # >0.5 = same fold

# GDT scores
gdt_ts, gdt_ha = metrics.gdt(coords1, coords2)
print(f"GDT-TS: {gdt_ts:.3f}, GDT-HA: {gdt_ha:.3f}")

# lDDT (superposition-free, local similarity)
lddt = metrics.lddt(coords1, coords2, alignment)
print(f"lDDT: {lddt:.3f}")  # >0.8 = high quality

# DALI score with Z-score
dali, z = metrics.dali_score(coords1, coords2, alignment, len1=150, len2=145)
print(f"DALI: {dali:.2f} (Z={z:.2f})")  # Z>5 = likely homology

# Traditional RMSD
rmsd = metrics.rmsd(coords1, coords2)
print(f"RMSD: {rmsd:.2f} Å")
```

### Progressive Alignment with Custom Tree

```python
import pfalign

# Step 1: Compute distance matrix from sequences
sequences = ["protein1.pdb", "protein2.pdb", "protein3.pdb"]
distances = pfalign.compute_distances(sequences, k_neighbors=30)

# Step 2: Build custom guide tree from distance matrix
tree = pfalign.tree.bionj(distances)

# Step 3: Perform MSA with this tree
result = pfalign.progressive_align(sequences, tree, k_neighbors=30)
```

### Batch Processing

```python
# Encode entire directory of structures
output_paths = pfalign.encode_batch(
    "/path/to/structures/",
    output_dir="/path/to/embeddings/",
    k_neighbors=30
)
# Returns list of .npy file paths

# Batch pairwise alignments
results = pfalign.batch_pairwise(
    queries="/path/to/queries/",
    targets="/path/to/targets/",
    output_dir="/path/to/alignments/",
    mode="all-vs-all",
    k_neighbors=30
)
# Returns list of dicts with 'query', 'target', 'output', 'score'
```

### Format Conversion

```python
# Convert HHsuite A3M to FASTA
pfalign.reformat(
    "input.a3m",
    "output.fasta",
    input_format="a3m",
    output_format="fasta"
)

# Supported formats: fasta, a2m, a3m, stockholm, psiblast, clustal
```

## Contributing

Contributions welcome! Here's the process:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes (follow existing code style)
4. Add tests for new functionality
5. Make sure tests pass (`./build.sh && meson test -C build-release`)
6. Commit with clear messages
7. Open a pull request

## Architecture

PFalign has a hybrid design:
- **C++ core** for performance-critical code (`pfalign/_core/src/pfalign/`)
- **PyBind11 bindings** to expose C++ to Python (`pfalign/_bindings/`)
- **Python API** with nice input handling and error messages
- **Standalone CLI** built from the same C++ code

Why this works well:
- Heavy computation stays in C++
- Python gives a nice scripting interface
- CLI has no Python overhead
- Both interfaces do the same things

## Citation

If you use PFalign in your research, please cite:

```
@software{pfalign2024,
  title = {PFalign: Fast pairwise and multiple protein structure alignment powered by ProteinMPNN embeddings},
  author = {{Casey mogilevsky}},
  year = {2024},
  url = {https://github.com/csm70/pfalign}
}
```

## License

PFalign is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

We would like to acknowledge Jeanne Trinquier, Samantha Petti, and Sergey Ovchinnikov for their Soft Smith-Waterman implementation and the SoftAlign model weights.

Built on:
- ProteinMPNN for structural embeddings
- SoftAlign for gap parameters and alignment approach

---

**Issues:** https://github.com/csm70/pfalign/issues
**Repository:** https://github.com/csm70/pfalign
