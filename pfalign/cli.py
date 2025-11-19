"""
Command-line interface for pfalign Python package.

This provides a Python-based CLI as an alternative to the standalone C++ binary.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List
import numpy as np

from . import encode, pairwise, msa, similarity, compute_distances, reformat, info, version_info, batch_pairwise, __version__
from . import _align_cpp
from .errors import FileNotFoundError, ValidationError, FormatError


def parse_chain(value: str) -> int | str:
    """Parse chain argument - accepts either index (0, 1, 2) or chain ID (A, B, C)."""
    # Try to parse as integer first
    try:
        return int(value)
    except ValueError:
        # Return as string (chain ID)
        if len(value) == 1 and value.isalpha():
            return value
        raise argparse.ArgumentTypeError(
            f"Chain must be an integer index (0, 1, 2, ...) or single letter ID (A, B, C, ...)"
        )


# Custom argparse types for early validation

def existing_file_type(extensions: tuple = None):
    """
    Argparse type factory for validating file existence and format.

    Args:
        extensions: Optional tuple of valid file extensions (e.g., ('.pdb', '.cif'))

    Returns:
        Validator function that can be used as argparse type parameter
    """
    def validator(path: str):
        p = Path(path)
        if not p.exists():
            raise argparse.ArgumentTypeError(f"File not found: {path}")
        if not p.is_file():
            raise argparse.ArgumentTypeError(f"Not a file: {path}")
        if extensions and p.suffix.lower() not in extensions:
            ext_list = ', '.join(extensions)
            raise argparse.ArgumentTypeError(
                f"Invalid format: {p.suffix}. Expected one of: {ext_list}"
            )
        return str(path)
    return validator


def output_path_type(path: str):
    """
    Argparse type for output paths - validates parent directory exists.

    Args:
        path: Output file path

    Returns:
        The path string if valid

    Raises:
        ArgumentTypeError: If parent directory doesn't exist
    """
    p = Path(path)
    if p.parent != Path() and not p.parent.exists():
        raise argparse.ArgumentTypeError(
            f"Output directory doesn't exist: {p.parent}"
        )
    return str(path)


def positive_int_type(value: str):
    """
    Argparse type for positive integers.

    Args:
        value: String value to parse

    Returns:
        Positive integer if valid

    Raises:
        ArgumentTypeError: If not a positive integer
    """
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                f"Must be positive, got {value}"
            )
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Must be an integer, got {value}"
        )


def range_type(min_val: float, max_val: float):
    """
    Argparse type factory for numeric ranges.

    Args:
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Validator function that can be used as argparse type parameter
    """
    def validator(value: str):
        try:
            fvalue = float(value)
            if fvalue < min_val or fvalue > max_val:
                raise argparse.ArgumentTypeError(
                    f"Must be in range [{min_val}, {max_val}], got {value}"
                )
            return fvalue
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Must be a number, got {value}"
            )
    return validator


def validate_input_file(path: str, file_type: str = "file") -> None:
    """Validate that an input file exists and is readable.

    Args:
        path: Path to validate
        file_type: Description of file type for error messages

    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If path is not a file
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path, file_type)
    if not p.is_file():
        raise ValidationError(
            file_type,
            "directory",
            "regular file"
        )


def validate_output_path(path: str) -> None:
    """Validate that output path's parent directory exists.

    Args:
        path: Output path to validate

    Raises:
        FileNotFoundError: If parent directory doesn't exist
    """
    p = Path(path)
    if p.parent != Path() and not p.parent.exists():
        raise FileNotFoundError(str(p.parent), "Output directory")


def validate_structure_format(path: str) -> None:
    """Validate that file has a recognized structure format extension.

    Args:
        path: Path to validate

    Raises:
        FormatError: If format is not recognized
    """
    valid_extensions = {'.pdb', '.cif', '.ent', '.npy'}
    p = Path(path)
    if p.suffix.lower() not in valid_extensions:
        raise FormatError(
            path,
            f"Unrecognized file extension: {p.suffix}",
            list(valid_extensions)
        )


def print_info(message: str, quiet: bool = False) -> None:
    """Print informational message unless quiet mode is enabled.

    Args:
        message: Message to print
        quiet: If True, suppress output
    """
    if not quiet:
        print(message)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="pfalign",
        description="Protein structure alignment using MPNN embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"pfalign {__version__}",
    )

    # Global flags
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output (only show errors)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars for long-running operations",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed statistics after completion",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========================================================================
    # ENCODE command
    # ========================================================================
    encode_parser = subparsers.add_parser(
        "encode",
        help="Encode PDB structure to MPNN embeddings",
    )
    encode_parser.add_argument(
        "structure",
        type=existing_file_type(extensions=('.pdb', '.cif', '.ent', '.npy')),
        help="Path to PDB structure file",
    )
    encode_parser.add_argument(
        "-o", "--output",
        type=output_path_type,
        required=True,
        help="Output .npy file for embeddings",
    )
    encode_parser.add_argument(
        "--chain",
        type=str,
        default="0",
        help="Chain to encode - either index (0, 1, ...) or ID (A, B, ...) (default: 0)",
    )
    encode_parser.add_argument(
        "--k-neighbors",
        type=positive_int_type,
        default=30,
        help="Number of nearest neighbors for graph construction (default: 30)",
    )

    # ========================================================================
    # PAIRWISE command
    # ========================================================================
    pairwise_parser = subparsers.add_parser(
        "pairwise",
        help="Pairwise alignment of two structures or embeddings",
    )
    pairwise_parser.add_argument(
        "input1",
        type=existing_file_type(extensions=('.pdb', '.cif', '.ent', '.npy')),
        help="First input (PDB or .npy embeddings)",
    )
    pairwise_parser.add_argument(
        "input2",
        type=existing_file_type(extensions=('.pdb', '.cif', '.ent', '.npy')),
        help="Second input (PDB or .npy embeddings)",
    )
    pairwise_parser.add_argument(
        "-o", "--output",
        type=output_path_type,
        required=True,
        help="Output FASTA file",
    )
    pairwise_parser.add_argument(
        "--gap-open",
        type=float,
        default=-1.0,
        help="Gap opening penalty (default: -1.0)",
    )
    pairwise_parser.add_argument(
        "--gap-extend",
        type=float,
        default=-0.1,
        help="Gap extension penalty (default: -0.1)",
    )
    pairwise_parser.add_argument(
        "--k-neighbors",
        type=positive_int_type,
        default=30,
        help="K-neighbors for structures (default: 30)",
    )
    pairwise_parser.add_argument(
        "--chain1",
        type=str,
        default="0",
        help="Chain for input1 - either index (0, 1, ...) or ID (A, B, ...) (default: 0)",
    )
    pairwise_parser.add_argument(
        "--chain2",
        type=str,
        default="0",
        help="Chain for input2 - either index (0, 1, ...) or ID (A, B, ...) (default: 0)",
    )
    pairwise_parser.add_argument(
        "--format",
        type=str,
        default="",
        help="Output format (auto-detect from extension if not specified). Options: fas, a2m, a3m, sto, psi, clu",
    )

    # ========================================================================
    # MSA command
    # ========================================================================
    msa_parser = subparsers.add_parser(
        "msa",
        help="Multiple sequence alignment",
    )

    # Smart input: can be list of files, directory path, or .txt file list
    msa_parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files, OR single directory path, OR single .txt file with paths",
    )

    msa_parser.add_argument(
        "-o", "--output",
        type=output_path_type,
        required=True,
        help="Output FASTA file",
    )
    msa_parser.add_argument(
        "--method",
        type=str,
        choices=["upgma", "nj"],
        default="upgma",
        help="Guide tree method (default: upgma)",
    )
    msa_parser.add_argument(
        "--gap-open",
        type=float,
        help="Gap opening penalty (default: -1.0)",
    )
    msa_parser.add_argument(
        "--gap-extend",
        type=float,
        help="Gap extension penalty (default: -0.1)",
    )
    msa_parser.add_argument(
        "--temperature",
        type=float,
        help="Smith-Waterman temperature (default: 1.0)",
    )
    msa_parser.add_argument(
        "--ecs-temperature",
        type=float,
        help="ECS temperature (default: 5.0)",
    )
    msa_parser.add_argument(
        "--arena-size-mb",
        type=positive_int_type,
        help="Arena size in MB (default: 200)",
    )
    msa_parser.add_argument(
        "--format",
        type=str,
        default="",
        help="Output format (auto-detect from extension if not specified). Options: fas, a2m, a3m, sto, psi, clu",
    )

    # ========================================================================
    # SIMILARITY command
    # ========================================================================
    similarity_parser = subparsers.add_parser(
        "similarity",
        help="Compute similarity matrix between embeddings",
    )

    # Input arguments (use one of: positional args, --input-list, or --input-dir)
    similarity_parser.add_argument(
        "inputs",
        nargs="*",
        help="Input embeddings (.npy files), OR single directory, OR single .txt file with paths",
    )
    similarity_parser.add_argument(
        "--input-list",
        type=existing_file_type(extensions=('.txt',)),
        help="File containing list of embedding paths (one per line)",
    )
    similarity_parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing .npy embedding files",
    )

    similarity_parser.add_argument(
        "-o", "--output",
        type=output_path_type,
        required=True,
        help="Output .npy file for similarity matrix",
    )
    similarity_parser.add_argument(
        "--method",
        type=str,
        default="ecs",
        choices=["ecs", "cosine"],
        help="Similarity method (default: ecs)",
    )
    similarity_parser.add_argument(
        "--temperature",
        type=float,
        default=3.0,
        help="Temperature for ECS (default: 3.0)",
    )

    # ========================================================================
    # COMPUTE-DISTANCES command
    # ========================================================================
    distances_parser = subparsers.add_parser(
        "compute-distances",
        help="Compute distance matrix from embeddings for phylogenetic tree building",
    )

    # Input arguments (use one of: positional args, --input-list, or --input-dir)
    distances_parser.add_argument(
        "inputs",
        nargs="*",
        help="Input embeddings (.npy files), OR single directory, OR single .txt file with paths",
    )
    distances_parser.add_argument(
        "--input-list",
        type=existing_file_type(extensions=('.txt',)),
        help="File containing list of embedding paths (one per line)",
    )
    distances_parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing .npy embedding files",
    )

    distances_parser.add_argument(
        "-o", "--output",
        type=output_path_type,
        required=True,
        help="Output distance matrix (.npy file)",
    )
    distances_parser.add_argument(
        "--gap-open",
        type=float,
        help="Gap opening penalty (default: -2.544, trained model)",
    )
    distances_parser.add_argument(
        "--gap-extend",
        type=float,
        help="Gap extension penalty (default: 0.194, trained model)",
    )
    distances_parser.add_argument(
        "--temperature",
        type=float,
        help="Alignment temperature (default: 1.0)",
    )

    # ========================================================================
    # REFORMAT command
    # ========================================================================
    reformat_parser = subparsers.add_parser(
        "reformat",
        help="Convert between alignment formats (FASTA, A2M, A3M, Stockholm, PSI-BLAST, Clustal)",
    )
    reformat_parser.add_argument(
        "input",
        type=str,
        help="Input alignment file",
    )
    reformat_parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output alignment file",
    )
    reformat_parser.add_argument(
        "--input-format",
        type=str,
        default="",
        help="Input format (auto-detect if not specified). Options: fas, a2m, a3m, sto, psi, clu",
    )
    reformat_parser.add_argument(
        "--output-format",
        type=str,
        default="",
        help="Output format (auto-detect if not specified). Options: fas, a2m, a3m, sto, psi, clu",
    )
    reformat_parser.add_argument(
        "--match-mode",
        type=str,
        default="",
        choices=["", "first", "gap"],
        help="Match state assignment: 'first' (use first sequence), 'gap' (use gap threshold)",
    )
    reformat_parser.add_argument(
        "--gap-threshold",
        type=int,
        default=50,
        help="Gap percentage threshold for match states (0-100, default: 50)",
    )
    reformat_parser.add_argument(
        "--remove-inserts",
        action="store_true",
        help="Remove insert states (lowercase residues)",
    )
    reformat_parser.add_argument(
        "--remove-gapped",
        type=int,
        default=0,
        help="Remove columns with >=N%% gaps (0-100, 0=disabled, default: 0)",
    )
    reformat_parser.add_argument(
        "--uppercase",
        action="store_true",
        help="Convert all residues to uppercase",
    )
    reformat_parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Convert all residues to lowercase",
    )
    reformat_parser.add_argument(
        "--remove-secondary-structure",
        action="store_true",
        help="Remove secondary structure annotations",
    )

    # ========================================================================
    # INFO command
    # ========================================================================
    info_parser = subparsers.add_parser(
        "info",
        help="Inspect structure files or directories",
    )
    info_parser.add_argument(
        "path",
        type=str,
        help="Structure file or directory path",
    )
    info_parser.add_argument(
        "--chains",
        action="store_true",
        help="Show detailed chain information",
    )

    # ========================================================================
    # ALIGNMENT command (low-level primitives)
    # ========================================================================
    alignment_parser = subparsers.add_parser(
        "alignment",
        help="Low-level alignment primitives for research",
    )
    alignment_subparsers = alignment_parser.add_subparsers(dest="alignment_command", help="Alignment operations")

    # Forward subcommand
    forward_parser = alignment_subparsers.add_parser(
        "forward",
        help="Smith-Waterman forward pass only",
    )
    forward_parser.add_argument(
        "similarity",
        type=str,
        help="Input similarity matrix (.npy file)",
    )
    forward_parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output forward scores (.npy file)",
    )
    forward_parser.add_argument(
        "--gap-open",
        type=float,
        default=-2.544,
        help="Gap opening penalty (default: -2.544)",
    )
    forward_parser.add_argument(
        "--gap-extend",
        type=float,
        default=0.194,
        help="Gap extension penalty (default: 0.194)",
    )
    forward_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature (default: 1.0)",
    )

    # Backward subcommand
    backward_parser = alignment_subparsers.add_parser(
        "backward",
        help="Smith-Waterman backward pass only",
    )
    backward_parser.add_argument(
        "forward_scores",
        type=str,
        help="Input forward scores (.npy file from forward command)",
    )
    backward_parser.add_argument(
        "similarity",
        type=str,
        help="Original similarity matrix (.npy file)",
    )
    backward_parser.add_argument(
        "--partition",
        type=float,
        required=True,
        help="Partition function (from score command or forward pass)",
    )
    backward_parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output posterior matrix (.npy file)",
    )
    backward_parser.add_argument(
        "--gap-open",
        type=float,
        default=-2.544,
        help="Gap opening penalty (must match forward pass)",
    )
    backward_parser.add_argument(
        "--gap-extend",
        type=float,
        default=0.194,
        help="Gap extension penalty (must match forward pass)",
    )
    backward_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature (must match forward pass)",
    )

    # Decode subcommand
    decode_parser = alignment_subparsers.add_parser(
        "decode",
        help="Viterbi decode alignment from posterior",
    )
    decode_parser.add_argument(
        "posterior",
        type=str,
        help="Input posterior matrix (.npy file)",
    )
    decode_parser.add_argument(
        "seq1",
        type=str,
        help="First sequence string or file",
    )
    decode_parser.add_argument(
        "seq2",
        type=str,
        help="Second sequence string or file",
    )
    decode_parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output FASTA alignment file",
    )
    decode_parser.add_argument(
        "--gap-penalty",
        type=float,
        default=-5.0,
        help="Gap penalty for Viterbi decoding (default: -5.0)",
    )

    # ========================================================================
    # TREE command
    # ========================================================================
    tree_parser = subparsers.add_parser(
        "tree",
        help="Build phylogenetic guide trees",
    )
    tree_subparsers = tree_parser.add_subparsers(dest="tree_command", help="Tree building methods")

    # Build subcommand (generic with method selection)
    tree_build_parser = tree_subparsers.add_parser(
        "build",
        help="Build tree with specified method",
    )
    tree_build_parser.add_argument(
        "distances",
        type=existing_file_type(extensions=('.npy',)),
        help="Distance matrix (.npy file)",
    )
    tree_build_parser.add_argument(
        "-o", "--output",
        type=output_path_type,
        required=True,
        help="Output Newick file",
    )
    tree_build_parser.add_argument(
        "--method",
        type=str,
        choices=["upgma", "nj", "bionj", "mst"],
        default="upgma",
        help="Tree building method (default: upgma)",
    )
    tree_build_parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Sequence labels (default: seq0, seq1, ...)",
    )

    # Individual method subcommands
    for method_name in ["upgma", "nj", "bionj", "mst"]:
        method_parser = tree_subparsers.add_parser(
            method_name,
            help=f"Build tree using {method_name.upper()}",
        )
        method_parser.add_argument(
            "distances",
            type=existing_file_type(extensions=('.npy',)),
            help="Distance matrix (.npy file)",
        )
        method_parser.add_argument(
            "-o", "--output",
            type=output_path_type,
            required=True,
            help="Output Newick file",
        )
        method_parser.add_argument(
            "--labels",
            type=str,
            nargs="+",
            help="Sequence labels (default: seq0, seq1, ...)",
        )

    # ========================================================================
    # STRUCTURE command
    # ========================================================================
    structure_parser = subparsers.add_parser(
        "structure",
        help="Structural superposition and alignment",
    )
    structure_subparsers = structure_parser.add_subparsers(
        dest="structure_command",
        help="Structure operations",
    )

    # Superpose subcommand
    structure_superpose_parser = structure_subparsers.add_parser(
        "superpose",
        help="Superpose two structures using Kabsch algorithm",
    )
    structure_superpose_parser.add_argument(
        "structure1",
        type=str,
        help="First structure file (PDB/CIF)",
    )
    structure_superpose_parser.add_argument(
        "structure2",
        type=str,
        help="Second structure file (PDB/CIF)",
    )
    structure_superpose_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for transformation info (.npz)",
    )
    structure_superpose_parser.add_argument(
        "--chain1",
        type=str,
        default="0",
        help="Chain index/ID for structure1 (default: 0)",
    )
    structure_superpose_parser.add_argument(
        "--chain2",
        type=str,
        default="0",
        help="Chain index/ID for structure2 (default: 0)",
    )
    structure_superpose_parser.add_argument(
        "--atom-type",
        type=str,
        choices=["CA", "backbone"],
        default="CA",
        help="Atom type for alignment (default: CA)",
    )

    # Superpose-from-alignment subcommand
    structure_align_parser = structure_subparsers.add_parser(
        "superpose-from-alignment",
        help="Superpose structures using sequence alignment",
    )
    structure_align_parser.add_argument(
        "structure1",
        type=str,
        help="First structure file (PDB/CIF)",
    )
    structure_align_parser.add_argument(
        "structure2",
        type=str,
        help="Second structure file (PDB/CIF)",
    )
    structure_align_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for transformation info (.npz)",
    )
    structure_align_parser.add_argument(
        "--chain1",
        type=str,
        default="0",
        help="Chain index/ID for structure1 (default: 0)",
    )
    structure_align_parser.add_argument(
        "--chain2",
        type=str,
        default="0",
        help="Chain index/ID for structure2 (default: 0)",
    )
    structure_align_parser.add_argument(
        "--atom-type",
        type=str,
        choices=["CA", "backbone"],
        default="CA",
        help="Atom type for alignment (default: CA)",
    )
    structure_align_parser.add_argument(
        "--alignment",
        type=str,
        help="Alignment file (.fasta, .aln, or .npz with path)",
    )

    # Get-coords subcommand
    structure_coords_parser = structure_subparsers.add_parser(
        "get-coords",
        help="Extract coordinates from structure file",
    )
    structure_coords_parser.add_argument(
        "structure",
        type=str,
        help="Structure file (PDB/CIF)",
    )
    structure_coords_parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output .npy file for coordinates",
    )
    structure_coords_parser.add_argument(
        "--chain",
        type=str,
        default="0",
        help="Chain index/ID (default: 0)",
    )
    structure_coords_parser.add_argument(
        "--atom-type",
        type=str,
        choices=["CA", "backbone"],
        default="CA",
        help="Atom type to extract (default: CA)",
    )

    # Kabsch subcommand
    structure_kabsch_parser = structure_subparsers.add_parser(
        "kabsch",
        help="Compute Kabsch transformation between two structures",
    )
    structure_kabsch_parser.add_argument(
        "structure1",
        type=str,
        help="First structure file (PDB/CIF)",
    )
    structure_kabsch_parser.add_argument(
        "structure2",
        type=str,
        help="Second structure file (PDB/CIF)",
    )
    structure_kabsch_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for transformation (.npz with rotation and translation)",
    )
    structure_kabsch_parser.add_argument(
        "--chain1",
        type=str,
        default="0",
        help="Chain index/ID for structure1 (default: 0)",
    )
    structure_kabsch_parser.add_argument(
        "--chain2",
        type=str,
        default="0",
        help="Chain index/ID for structure2 (default: 0)",
    )
    structure_kabsch_parser.add_argument(
        "--atom-type",
        type=str,
        choices=["CA", "backbone"],
        default="CA",
        help="Atom type for alignment (default: CA)",
    )

    # ========================================================================
    # METRICS command
    # ========================================================================
    metrics_parser = subparsers.add_parser(
        "metrics",
        help="Compute quality metrics for alignments and structures",
    )
    metrics_subparsers = metrics_parser.add_subparsers(
        dest="metrics_command",
        help="Metric calculations",
    )

    # RMSD subcommand
    metrics_rmsd_parser = metrics_subparsers.add_parser(
        "rmsd",
        help="Compute RMSD between two coordinate sets",
    )
    metrics_rmsd_parser.add_argument(
        "coords1",
        type=str,
        help="First coordinates (.npy file with shape N,3)",
    )
    metrics_rmsd_parser.add_argument(
        "coords2",
        type=str,
        help="Second coordinates (.npy file with shape N,3)",
    )
    metrics_rmsd_parser.add_argument(
        "--aligned",
        action="store_true",
        help="Coordinates are already aligned (skip Kabsch)",
    )

    # Identity subcommand
    metrics_identity_parser = metrics_subparsers.add_parser(
        "identity",
        help="Compute sequence identity between aligned sequences",
    )
    metrics_identity_parser.add_argument(
        "alignment",
        type=str,
        help="Alignment file (.fasta, .aln)",
    )
    metrics_identity_parser.add_argument(
        "--ignore-gaps",
        action="store_true",
        default=True,
        help="Ignore gap positions (default: True)",
    )

    # Coverage subcommand
    metrics_coverage_parser = metrics_subparsers.add_parser(
        "coverage",
        help="Compute alignment coverage (non-gap positions)",
    )
    metrics_coverage_parser.add_argument(
        "alignment",
        type=str,
        help="Alignment file (.fasta, .aln)",
    )

    # Stats subcommand
    metrics_stats_parser = metrics_subparsers.add_parser(
        "stats",
        help="Compute comprehensive alignment statistics",
    )
    metrics_stats_parser.add_argument(
        "alignment",
        type=str,
        help="Alignment file (.fasta, .aln)",
    )

    # Pairwise identity subcommand
    metrics_pairwise_parser = metrics_subparsers.add_parser(
        "pairwise-identity",
        help="Compute pairwise identity matrix for MSA",
    )
    metrics_pairwise_parser.add_argument(
        "msa",
        type=str,
        help="MSA file (.fasta, .aln, .a3m)",
    )
    metrics_pairwise_parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output .npy file for identity matrix",
    )

    # Conservation subcommand
    metrics_conservation_parser = metrics_subparsers.add_parser(
        "conservation",
        help="Compute per-column conservation scores",
    )
    metrics_conservation_parser.add_argument(
        "msa",
        type=str,
        help="MSA file (.fasta, .aln, .a3m)",
    )
    metrics_conservation_parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output .npy file for conservation scores",
    )
    metrics_conservation_parser.add_argument(
        "--method",
        type=str,
        choices=["shannon", "gap_penalty"],
        default="shannon",
        help="Conservation method (default: shannon)",
    )

    # TM-score subcommand
    metrics_tm_parser = metrics_subparsers.add_parser(
        "tm-score",
        help="Compute TM-score between two structures",
    )
    metrics_tm_parser.add_argument(
        "structure1",
        type=str,
        help="First structure file (.pdb, .cif)",
    )
    metrics_tm_parser.add_argument(
        "structure2",
        type=str,
        help="Second structure file (.pdb, .cif)",
    )
    metrics_tm_parser.add_argument(
        "--chain1",
        type=str,
        default="0",
        help="Chain for first structure (index or ID, default: 0)",
    )
    metrics_tm_parser.add_argument(
        "--chain2",
        type=str,
        default="0",
        help="Chain for second structure (index or ID, default: 0)",
    )
    metrics_tm_parser.add_argument(
        "--aligned",
        action="store_true",
        help="Structures are already aligned (skip Kabsch)",
    )

    # GDT subcommand
    metrics_gdt_parser = metrics_subparsers.add_parser(
        "gdt",
        help="Compute GDT-TS and GDT-HA scores",
    )
    metrics_gdt_parser.add_argument(
        "structure1",
        type=str,
        help="First structure file (.pdb, .cif)",
    )
    metrics_gdt_parser.add_argument(
        "structure2",
        type=str,
        help="Second structure file (.pdb, .cif)",
    )
    metrics_gdt_parser.add_argument(
        "--chain1",
        type=str,
        default="0",
        help="Chain for first structure (index or ID, default: 0)",
    )
    metrics_gdt_parser.add_argument(
        "--chain2",
        type=str,
        default="0",
        help="Chain for second structure (index or ID, default: 0)",
    )
    metrics_gdt_parser.add_argument(
        "--aligned",
        action="store_true",
        help="Structures are already aligned (skip Kabsch)",
    )

    # ECS subcommand
    metrics_ecs_parser = metrics_subparsers.add_parser(
        "ecs",
        help="Compute ECS score for MSA quality",
    )
    metrics_ecs_parser.add_argument(
        "msa",
        type=str,
        help="MSA file (.fasta, .aln, .a3m)",
    )
    metrics_ecs_parser.add_argument(
        "--temperature",
        type=float,
        default=5.0,
        help="ECS temperature parameter (default: 5.0)",
    )

    # ========================================================================
    # VERSION command
    # ========================================================================
    version_parser = subparsers.add_parser(
        "version",
        help="Show detailed version information",
    )

    # ========================================================================
    # BATCH command
    # ========================================================================
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch processing commands",
    )
    batch_subparsers = batch_parser.add_subparsers(dest="batch_command", help="Batch operations")

    # Batch pairwise subcommand
    batch_pairwise_parser = batch_subparsers.add_parser(
        "pairwise",
        help="Batch pairwise alignments",
    )
    batch_pairwise_parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Query structures (directory, file list, or list of files)",
    )
    batch_pairwise_parser.add_argument(
        "--targets",
        type=str,
        required=True,
        help="Target structures (directory, file list, or list of files)",
    )
    batch_pairwise_parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Output directory for alignments",
    )
    batch_pairwise_parser.add_argument(
        "--mode",
        type=str,
        choices=["all-vs-all", "one-to-one"],
        default="all-vs-all",
        help="Alignment mode (default: all-vs-all)",
    )
    batch_pairwise_parser.add_argument(
        "--format",
        type=str,
        default="fas",
        help="Output format (default: fas)",
    )
    batch_pairwise_parser.add_argument(
        "--k-neighbors",
        type=int,
        default=30,
        help="Number of nearest neighbors for graph construction (default: 30)",
    )
    batch_pairwise_parser.add_argument(
        "--gap-open",
        type=float,
        default=None,
        help="Gap opening penalty (default: -2.544 from trained model)",
    )
    batch_pairwise_parser.add_argument(
        "--gap-extend",
        type=float,
        default=None,
        help="Gap extension penalty (default: 0.194 from trained model)",
    )

    # Batch encode subcommand
    batch_encode_parser = batch_subparsers.add_parser(
        "encode",
        help="Batch encode multiple structures",
    )

    # Input arguments (use one of: positional args, --input-list, or --input-dir)
    batch_encode_parser.add_argument(
        "inputs",
        nargs="*",
        help="Input structures (.pdb/.cif files), OR single directory, OR single .txt file with paths",
    )
    batch_encode_parser.add_argument(
        "--input-list",
        type=existing_file_type(extensions=('.txt',)),
        help="File containing list of structure paths (one per line)",
    )
    batch_encode_parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing structure files",
    )

    batch_encode_parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Output directory for embeddings (.npy files)",
    )
    batch_encode_parser.add_argument(
        "--k-neighbors",
        type=positive_int_type,
        default=30,
        help="Number of nearest neighbors for MPNN (default: 30)",
    )
    batch_encode_parser.add_argument(
        "--chain",
        type=str,
        default="0",
        help="Chain to encode (index or ID, default: 0)",
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # ========================================================================
    # Execute commands
    # ========================================================================

    # Helper to parse chain arguments (int if numeric, otherwise string)
    def parse_chain(chain_str):
        try:
            return int(chain_str)
        except ValueError:
            return chain_str

    try:
        if args.command == "encode":
            # Validate inputs
            validate_input_file(args.structure, "Structure file")
            validate_structure_format(args.structure)
            validate_output_path(args.output)

            result = encode(
                args.structure,
                chain=parse_chain(args.chain),
                k_neighbors=args.k_neighbors,
            )
            np.save(args.output, result.embeddings)
            print_info(f"[OK] Encoded {result.sequence_length} residues to {args.output}", args.quiet)

        elif args.command == "pairwise":
            # Validate inputs
            validate_input_file(args.input1, "Input 1")
            validate_input_file(args.input2, "Input 2")
            validate_structure_format(args.input1)
            validate_structure_format(args.input2)
            validate_output_path(args.output)

            result = pairwise(
                args.input1,
                args.input2,
                k_neighbors=args.k_neighbors,
                chain1=parse_chain(args.chain1),
                chain2=parse_chain(args.chain2),
                gap_open=args.gap_open,
                gap_extend=args.gap_extend,
            )

            # Write output using result.save() for format conversion support
            result.save(args.output, format=args.format)

            print_info(f"[OK] Pairwise alignment complete", args.quiet)
            print_info(f"  Score: {result.score:.4f}", args.quiet)
            print_info(f"  Output: {args.output}", args.quiet)

            # Show detailed statistics if --stats flag is used
            if args.stats and not args.quiet:
                stats = result.statistics()
                print("\nDetailed Statistics:")
                print(f"  Identity: {stats['identity']:.2%}")
                print(f"  Coverage: {stats['coverage']:.2%}")
                print(f"  Gaps: {stats['gaps']}")
                print(f"  Alignment length: {stats['length']}")
                print(f"  Matches: {stats['matches']}")
                print(f"  Mismatches: {stats['mismatches']}")

        elif args.command == "msa":
            # Validate output path
            validate_output_path(args.output)

            # Smart input detection: if single arg, could be dir or .txt file
            if len(args.inputs) == 1:
                input_path = Path(args.inputs[0])
                if not input_path.exists():
                    print(f"Error: Input not found: {args.inputs[0]}", file=sys.stderr)
                    sys.exit(1)
                inputs = args.inputs[0]  # Let API handle auto-detection
            else:
                # Validate all input files
                for inp in args.inputs:
                    validate_input_file(inp, "Input structure")
                    validate_structure_format(inp)
                inputs = args.inputs  # List of files

            # Build kwargs
            kwargs = {"method": args.method}
            if args.gap_open is not None:
                kwargs["gap_open"] = args.gap_open
            if args.gap_extend is not None:
                kwargs["gap_extend"] = args.gap_extend
            if args.temperature is not None:
                kwargs["temperature"] = args.temperature
            if args.ecs_temperature is not None:
                kwargs["ecs_temperature"] = args.ecs_temperature
            if args.arena_size_mb is not None:
                kwargs["arena_size_mb"] = args.arena_size_mb

            result = msa(inputs, **kwargs)

            # Write output using result.save() for format conversion support
            result.save(args.output, format=args.format)

            print_info(f"[OK] MSA complete", args.quiet)
            print_info(f"  Sequences: {result.num_sequences}", args.quiet)
            print_info(f"  Alignment length: {result.alignment_length}", args.quiet)
            print_info(f"  ECS score: {result.ecs_score:.5f}", args.quiet)
            print_info(f"  Output: {args.output}", args.quiet)

            # Show detailed statistics if --stats flag is used
            if args.stats and not args.quiet:
                print()
                print(result.summary())

        elif args.command == "similarity":
            # Validate output path
            validate_output_path(args.output)

            # Parse inputs: could be list of files, directory, or input-list file
            inputs = None
            if args.input_list:
                with open(args.input_list) as f:
                    inputs = [line.strip() for line in f if line.strip()]
            elif args.input_dir:
                inputs = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.npy')]
            elif args.inputs:
                inputs = args.inputs
            else:
                raise ValueError("Must provide inputs via positional arguments, --input-list, or --input-dir")

            # Load embeddings
            embeddings = [np.load(inp) for inp in inputs]
            n = len(embeddings)

            # Compute NxN similarity matrix by computing all pairwise similarities
            similarity_matrix = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        similarity_matrix[i, j] = 1.0  # Self-similarity is 1.0
                    else:
                        result = similarity(embeddings[i], embeddings[j])
                        sim_value = result.similarity[0, 0]  # Extract scalar similarity
                        similarity_matrix[i, j] = sim_value
                        similarity_matrix[j, i] = sim_value  # Symmetric matrix

            np.save(args.output, similarity_matrix)
            print_info(f"[OK] Similarity matrix saved to {args.output}", args.quiet)
            print_info(f"  Shape: {similarity_matrix.shape}", args.quiet)
            print_info(f"  Sequences: {n}", args.quiet)

            if args.stats and not args.quiet:
                print("\nDetailed Statistics:")
                # Exclude diagonal when computing stats
                off_diagonal = similarity_matrix[np.triu_indices(n, k=1)]
                print(f"  Min similarity: {off_diagonal.min():.4f}")
                print(f"  Max similarity: {off_diagonal.max():.4f}")
                print(f"  Mean similarity: {off_diagonal.mean():.4f}")

        elif args.command == "compute-distances":
            # Validate output path
            validate_output_path(args.output)

            # Parse inputs: could be list of files, directory, or input-list file
            inputs = None
            input_list = args.input_list
            input_dir = args.input_dir

            if args.inputs:
                # Validate input files if provided as positional arguments
                for inp in args.inputs:
                    validate_input_file(inp, "Input embedding")
                inputs = args.inputs

            # Build kwargs
            kwargs = {}
            if args.gap_open is not None:
                kwargs["gap_open"] = args.gap_open
            if args.gap_extend is not None:
                kwargs["gap_extend"] = args.gap_extend
            if args.temperature is not None:
                kwargs["temperature"] = args.temperature

            # Compute distances
            distances = compute_distances(
                inputs=inputs,
                input_list=input_list,
                input_dir=input_dir,
                **kwargs
            )

            # Save output
            np.save(args.output, distances)
            print_info(f"[OK] Distance matrix computed", args.quiet)
            print_info(f"  Shape: {distances.shape}", args.quiet)
            print_info(f"  Output: {args.output}", args.quiet)

            if args.stats and not args.quiet:
                print("\nDetailed Statistics:")
                print(f"  Min distance: {distances.min():.4f}")
                print(f"  Max distance: {distances.max():.4f}")
                print(f"  Mean distance: {distances.mean():.4f}")
                # Distance matrix should be symmetric
                print(f"  Symmetry check: {np.allclose(distances, distances.T)}")

        elif args.command == "reformat":
            # Validate inputs
            validate_input_file(args.input, "Input alignment")
            validate_output_path(args.output)

            reformat(
                args.input,
                args.output,
                input_format=args.input_format,
                output_format=args.output_format,
                match_mode=args.match_mode,
                gap_threshold=args.gap_threshold,
                remove_inserts=args.remove_inserts,
                remove_gapped=args.remove_gapped,
                uppercase=args.uppercase,
                lowercase=args.lowercase,
                remove_secondary_structure=args.remove_secondary_structure,
            )
            print_info(f"[OK] Format conversion complete", args.quiet)
            print_info(f"  Input: {args.input}", args.quiet)
            print_info(f"  Output: {args.output}", args.quiet)

        elif args.command == "info":
            result = info(args.path, show_chains=args.chains)

            if result.get('is_directory'):
                print(f"Directory: {result['path']}")
                print(f"Structures: {result['num_structures']}\n")
                for structure in result['structures']:
                    print(f"  {structure}")
            else:
                print(f"File: {result['path']}")
                print(f"Format: {result['format']}")
                print(f"Chains: {result['num_chains']}")

                if args.chains and 'chains' in result:
                    print()
                    for chain in result['chains']:
                        print(f"  Chain {chain['index']} ({chain['id']}): "
                              f"{chain['length']} residues, {chain['num_atoms']} atoms")

        elif args.command == "version":
            ver_info = version_info()
            print(f"pfalign {ver_info['version']}")
            print(f"Build: {ver_info['build_type']}")
            print(f"Git: {ver_info['git_commit'][:7]} ({ver_info['git_branch']})")
            print(f"Compiler: {ver_info['compiler']}")
            print(f"Build date: {ver_info['build_date']}")
            print(f"Python: {ver_info['python_version']}")
            print(f"Platform: {ver_info['platform']}")

        elif args.command == "alignment":
            if args.alignment_command == "forward":
                # Load similarity matrix
                sim = np.load(args.similarity)

                # Run forward pass
                import pfalign.alignment
                fwd = pfalign.alignment.forward(
                    sim,
                    gap_open=args.gap_open,
                    gap_extend=args.gap_extend,
                    temperature=args.temperature,
                )

                # Save forward scores
                np.save(args.output, fwd)
                print(f"[OK] Forward pass complete")
                print(f"  Input: {args.similarity}")
                print(f"  Output: {args.output}")
                print(f"  Shape: {fwd.shape}")

            elif args.alignment_command == "backward":
                # Load forward scores and similarity matrix
                fwd = np.load(args.forward_scores)
                sim = np.load(args.similarity)

                # Run backward pass
                import pfalign.alignment
                posterior = pfalign.alignment.backward(
                    fwd,
                    sim,
                    args.partition,
                    gap_open=args.gap_open,
                    gap_extend=args.gap_extend,
                    temperature=args.temperature,
                )

                # Save posterior matrix
                np.save(args.output, posterior)
                print(f"[OK] Backward pass complete")
                print(f"  Output: {args.output}")
                print(f"  Posterior shape: {posterior.shape}")
                print(f"  Sum of posteriors: {posterior.sum():.3f}")

            elif args.alignment_command == "decode":
                # Load posterior matrix
                posterior = np.load(args.posterior)

                # Load or parse sequences
                # If seq1/seq2 are files, read them; otherwise treat as strings
                def load_sequence(seq_arg):
                    if Path(seq_arg).is_file():
                        # Read sequence from FASTA file (first sequence)
                        with open(seq_arg) as f:
                            lines = [l.strip() for l in f if l.strip() and not l.startswith('>')]
                            return ''.join(lines)
                    else:
                        return seq_arg

                seq1 = load_sequence(args.seq1)
                seq2 = load_sequence(args.seq2)

                # Run Viterbi decode
                import pfalign.alignment
                aligned1, aligned2, path = pfalign.alignment.viterbi_decode(
                    posterior,
                    seq1,
                    seq2,
                    gap_penalty=args.gap_penalty,
                )

                # Write output FASTA
                with open(args.output, 'w') as f:
                    f.write(">seq1\n")
                    f.write(f"{aligned1}\n")
                    f.write(">seq2\n")
                    f.write(f"{aligned2}\n")

                print(f"[OK] Viterbi decode complete")
                print(f"  Output: {args.output}")
                print(f"  Alignment length: {len(aligned1)}")
                print(f"  Gaps in seq1: {aligned1.count('-')}")
                print(f"  Gaps in seq2: {aligned2.count('-')}")

            else:
                print(f"[FAIL] Unknown alignment command: {args.alignment_command}", file=sys.stderr)
                return 1

        elif args.command == "tree":
            # Load distance matrix
            distances = np.load(args.distances)

            if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
                print(f"[FAIL] Distance matrix must be square (N, N)", file=sys.stderr)
                return 1

            # Get labels if provided
            labels = args.labels if hasattr(args, 'labels') and args.labels else None

            # Import tree module
            import pfalign.tree

            # Determine method
            if args.tree_command == "build":
                method = args.method
            else:
                method = args.tree_command  # upgma, nj, bionj, or mst

            # Build tree
            tree = pfalign.tree.build(distances, method=method, labels=labels)

            # Save tree
            tree.save(args.output)

            print(f"[OK] Tree built using {method.upper()}")
            print(f"  Input: {args.distances}")
            print(f"  Output: {args.output}")
            print(f"  Sequences: {tree.num_sequences}")
            print(f"  Method: {method}")
            print(f"  Newick: {tree.newick[:60]}{'...' if len(tree.newick) > 60 else ''}")

        elif args.command == "structure":
            import pfalign.structure

            if args.structure_command == "superpose":
                # Perform superposition
                chain1 = parse_chain(args.chain1)
                chain2 = parse_chain(args.chain2)

                result = pfalign.structure.superpose(
                    args.structure1,
                    args.structure2,
                    chain1=chain1,
                    chain2=chain2,
                    atom_type=args.atom_type,
                )

                print(f"[OK] Superposition complete")
                print(f"  Structure 1: {args.structure1}")
                print(f"  Structure 2: {args.structure2}")
                print(f"  RMSD: {result.rmsd:.3f} Å")
                print(f"  Atoms aligned: {result.coords1.shape[0]}")

                # Save transformation if output specified
                if args.output:
                    np.savez(
                        args.output,
                        rotation=result.rotation,
                        translation=result.translation,
                        rmsd=result.rmsd,
                        coords1=result.coords1,
                        coords2=result.coords2,
                        coords1_transformed=result.coords1_transformed,
                    )
                    print(f"  Saved: {args.output}")

            elif args.structure_command == "superpose-from-alignment":
                # Load or perform alignment
                if args.alignment:
                    # Load alignment from file
                    if args.alignment.endswith('.npz'):
                        # Load alignment path from npz
                        data = np.load(args.alignment)
                        if 'path' in data:
                            alignment = list(zip(data['path'][:, 0], data['path'][:, 1]))
                        else:
                            print("[FAIL] .npz file must contain 'path' array", file=sys.stderr)
                            return 1
                    else:
                        # TODO: Parse FASTA/ALN alignment formats
                        print("[FAIL] Only .npz alignment files supported currently", file=sys.stderr)
                        return 1
                else:
                    # Perform alignment automatically
                    alignment = pairwise(args.structure1, args.structure2)

                # Parse chains
                chain1 = parse_chain(args.chain1)
                chain2 = parse_chain(args.chain2)

                # Superpose using alignment
                result = pfalign.structure.superpose_from_alignment(
                    args.structure1,
                    args.structure2,
                    alignment,
                    chain1=chain1,
                    chain2=chain2,
                    atom_type=args.atom_type,
                )

                print(f"[OK] Alignment-based superposition complete")
                print(f"  Structure 1: {args.structure1}")
                print(f"  Structure 2: {args.structure2}")
                print(f"  RMSD (aligned residues): {result.rmsd:.3f} Å")
                print(f"  Total atoms in structure 1: {result.coords1.shape[0]}")
                print(f"  Total atoms in structure 2: {result.coords2.shape[0]}")

                # Save transformation if output specified
                if args.output:
                    np.savez(
                        args.output,
                        rotation=result.rotation,
                        translation=result.translation,
                        rmsd=result.rmsd,
                        coords1=result.coords1,
                        coords2=result.coords2,
                        coords1_transformed=result.coords1_transformed,
                    )
                    print(f"  Saved: {args.output}")

            elif args.structure_command == "get-coords":
                # Extract coordinates
                chain = parse_chain(args.chain)

                coords = pfalign.structure.get_coords(
                    args.structure,
                    chain=chain,
                    atom_type=args.atom_type,
                )

                # Save coordinates
                np.save(args.output, coords)

                print(f"[OK] Coordinates extracted")
                print(f"  Structure: {args.structure}")
                print(f"  Chain: {args.chain}")
                print(f"  Atom type: {args.atom_type}")
                print(f"  Shape: {coords.shape}")
                print(f"  Output: {args.output}")

            elif args.structure_command == "kabsch":
                # Get coordinates from both structures
                chain1 = parse_chain(args.chain1)
                chain2 = parse_chain(args.chain2)

                coords1 = pfalign.structure.get_coords(
                    args.structure1,
                    chain=chain1,
                    atom_type=args.atom_type,
                )
                coords2 = pfalign.structure.get_coords(
                    args.structure2,
                    chain=chain2,
                    atom_type=args.atom_type,
                )

                # Compute Kabsch transformation
                R, t, rmsd = pfalign.structure.kabsch(coords1, coords2)

                print_info(f"[OK] Kabsch transformation computed", args.quiet)
                print_info(f"  Structure 1: {args.structure1} ({len(coords1)} atoms)", args.quiet)
                print_info(f"  Structure 2: {args.structure2} ({len(coords2)} atoms)", args.quiet)
                print_info(f"  RMSD: {rmsd:.3f} Å", args.quiet)

                # Print transformation matrices
                if not args.quiet:
                    print("\nRotation matrix:")
                    print(R)
                    print("\nTranslation vector:")
                    print(t)

                # Save transformation if output specified
                if args.output:
                    np.savez(
                        args.output,
                        rotation=R,
                        translation=t,
                        rmsd=rmsd,
                    )
                    print_info(f"  Saved: {args.output}", args.quiet)

            else:
                print(f"[FAIL] Unknown structure command: {args.structure_command}", file=sys.stderr)
                return 1

        elif args.command == "metrics":
            import pfalign.metrics

            # Helper to load aligned sequences from file
            def load_alignment_sequences(filepath):
                """Load first two sequences from alignment file."""
                with open(filepath) as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith('>')]
                    if len(lines) < 2:
                        raise ValueError(f"Alignment file must contain at least 2 sequences")
                    return lines[0], lines[1]

            # Helper to load MSA
            def load_msa_object(filepath):
                """Load MSA from file into MSAResult object."""
                from pfalign import reformat
                import tempfile
                import os

                # Read the MSA file
                # For now, we'll create a simple wrapper that reads sequences
                # This is a placeholder - ideally we'd use pfalign.msa.load()
                with open(filepath) as f:
                    sequences = []
                    current_seq = []
                    for line in f:
                        line = line.strip()
                        if line.startswith('>'):
                            if current_seq:
                                sequences.append(''.join(current_seq))
                                current_seq = []
                        elif line:
                            current_seq.append(line)
                    if current_seq:
                        sequences.append(''.join(current_seq))

                # Create a minimal MSA-like object
                class SimpleMSA:
                    def __init__(self, seqs):
                        self._sequences = seqs

                    def sequences(self):
                        return self._sequences

                    def num_sequences(self):
                        return len(self._sequences)

                    def alignment_length(self):
                        return len(self._sequences[0]) if self._sequences else 0

                return SimpleMSA(sequences)

            if args.metrics_command == "rmsd":
                # Load coordinates
                coords1 = np.load(args.coords1)
                coords2 = np.load(args.coords2)

                # Compute RMSD
                rmsd_val = pfalign.metrics.rmsd(coords1, coords2, aligned=args.aligned)

                print(f"[OK] RMSD computed")
                print(f"  Coords 1: {args.coords1} ({coords1.shape[0]} atoms)")
                print(f"  Coords 2: {args.coords2} ({coords2.shape[0]} atoms)")
                print(f"  RMSD: {rmsd_val:.3f} Å")
                print(f"  Aligned: {args.aligned}")

            elif args.metrics_command == "identity":
                # Load alignment
                seq1, seq2 = load_alignment_sequences(args.alignment)

                # Compute identity
                id_val = pfalign.metrics.identity(seq1, seq2, ignore_gaps=args.ignore_gaps)

                print(f"[OK] Identity computed")
                print(f"  Alignment: {args.alignment}")
                print(f"  Identity: {id_val:.2%}")
                print(f"  Length: {len(seq1)}")
                print(f"  Ignore gaps: {args.ignore_gaps}")

            elif args.metrics_command == "coverage":
                # Load alignment
                seq1, seq2 = load_alignment_sequences(args.alignment)

                # Compute coverage
                cov_val = pfalign.metrics.coverage(seq1, seq2)

                print(f"[OK] Coverage computed")
                print(f"  Alignment: {args.alignment}")
                print(f"  Coverage: {cov_val:.2%}")
                print(f"  Length: {len(seq1)}")

            elif args.metrics_command == "stats":
                # Load alignment
                seq1, seq2 = load_alignment_sequences(args.alignment)

                # Compute stats
                stats = pfalign.metrics.alignment_stats(seq1, seq2)

                print(f"[OK] Alignment statistics computed")
                print(f"  Alignment: {args.alignment}")
                print(f"  Identity: {stats['identity']:.2%}")
                print(f"  Coverage: {stats['coverage']:.2%}")
                print(f"  Length: {stats['length']}")
                print(f"  Gaps: {stats['gaps']} ({stats['gap_percentage']:.1f}%)")
                print(f"  Matches: {stats['matches']}")
                print(f"  Mismatches: {stats['mismatches']}")

            elif args.metrics_command == "pairwise-identity":
                # Load MSA
                msa_obj = load_msa_object(args.msa)

                # Compute pairwise identity matrix
                id_matrix = pfalign.metrics.pairwise_identity(msa_obj)

                # Save to file
                np.save(args.output, id_matrix)

                print(f"[OK] Pairwise identity matrix computed")
                print(f"  MSA: {args.msa}")
                print(f"  Sequences: {msa_obj.num_sequences}")
                print(f"  Matrix shape: {id_matrix.shape}")
                print(f"  Mean identity: {id_matrix.mean():.2%}")
                print(f"  Output: {args.output}")

            elif args.metrics_command == "conservation":
                # Load MSA
                msa_obj = load_msa_object(args.msa)

                # Compute conservation scores
                cons_scores = pfalign.metrics.conservation(msa_obj, method=args.method)

                # Save to file
                np.save(args.output, cons_scores)

                print(f"[OK] Conservation scores computed")
                print(f"  MSA: {args.msa}")
                print(f"  Method: {args.method}")
                print(f"  Alignment length: {len(cons_scores)}")
                print(f"  Mean conservation: {cons_scores.mean():.3f}")
                print(f"  Output: {args.output}")

            elif args.metrics_command == "tm-score":
                # Validate inputs
                validate_input_file(args.structure1, "Structure 1")
                validate_input_file(args.structure2, "Structure 2")
                validate_structure_format(args.structure1)
                validate_structure_format(args.structure2)

                # Get coordinates from structures
                from pfalign import structure
                coords1 = structure.get_coords(args.structure1, chain=parse_chain(args.chain1), atom_type="CA")
                coords2 = structure.get_coords(args.structure2, chain=parse_chain(args.chain2), atom_type="CA")

                # Compute TM-score
                tm_score = pfalign.metrics.tm_score(coords1, coords2, len1=len(coords1), len2=len(coords2), aligned=args.aligned)

                print_info(f"[OK] TM-score computed", args.quiet)
                print_info(f"  Structure 1: {args.structure1} ({len(coords1)} residues)", args.quiet)
                print_info(f"  Structure 2: {args.structure2} ({len(coords2)} residues)", args.quiet)
                print_info(f"  TM-score: {tm_score:.4f}", args.quiet)
                print_info(f"  Aligned: {args.aligned}", args.quiet)

            elif args.metrics_command == "gdt":
                # Validate inputs
                validate_input_file(args.structure1, "Structure 1")
                validate_input_file(args.structure2, "Structure 2")
                validate_structure_format(args.structure1)
                validate_structure_format(args.structure2)

                # Get coordinates from structures
                from pfalign import structure
                coords1 = structure.get_coords(args.structure1, chain=parse_chain(args.chain1), atom_type="CA")
                coords2 = structure.get_coords(args.structure2, chain=parse_chain(args.chain2), atom_type="CA")

                # Compute GDT scores
                gdt_ts, gdt_ha = pfalign.metrics.gdt(coords1, coords2, aligned=args.aligned)

                print_info(f"[OK] GDT scores computed", args.quiet)
                print_info(f"  Structure 1: {args.structure1} ({len(coords1)} residues)", args.quiet)
                print_info(f"  Structure 2: {args.structure2} ({len(coords2)} residues)", args.quiet)
                print_info(f"  GDT-TS: {gdt_ts:.4f}", args.quiet)
                print_info(f"  GDT-HA: {gdt_ha:.4f}", args.quiet)
                print_info(f"  Aligned: {args.aligned}", args.quiet)

            elif args.metrics_command == "ecs":
                # Load MSA
                msa_obj = load_msa_object(args.msa)

                # Compute ECS score using pfalign.metrics.conservation
                # ECS is computed internally by MSA, but for standalone files we approximate
                # using conservation scores weighted by temperature
                cons_scores = pfalign.metrics.conservation(msa_obj, method="shannon")

                # Approximate ECS as weighted mean of conservation
                # This is a simplified approximation; real ECS uses expected column scores
                import math
                ecs_score = float(np.mean(cons_scores))

                print_info(f"[OK] ECS score computed (approximation)", args.quiet)
                print_info(f"  MSA: {args.msa}", args.quiet)
                print_info(f"  Sequences: {msa_obj.num_sequences}", args.quiet)
                print_info(f"  Alignment length: {msa_obj.alignment_length}", args.quiet)
                print_info(f"  ECS score: {ecs_score:.5f}", args.quiet)
                print_info(f"  Temperature: {args.temperature}", args.quiet)
                print_info(f"  Note: This is an approximation based on conservation scores", args.quiet)

            else:
                print(f"[FAIL] Unknown metrics command: {args.metrics_command}", file=sys.stderr)
                return 1

        elif args.command == "batch":
            if args.batch_command == "pairwise":
                # Validate output directory
                output_dir = Path(args.output_dir)
                if not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)

                results = batch_pairwise(
                    queries=args.queries,
                    targets=args.targets,
                    output_dir=args.output_dir,
                    mode=args.mode,
                    k_neighbors=args.k_neighbors,
                    gap_open=args.gap_open,
                    gap_extend=args.gap_extend,
                    output_format=args.format,
                )
                print_info(f"[OK] Batch pairwise complete", args.quiet)
                print_info(f"  Total alignments: {len(results)}", args.quiet)
                print_info(f"  Output directory: {args.output_dir}", args.quiet)
                if len(results) > 0:
                    print_info(f"  Average score: {sum(r['score'] for r in results) / len(results):.4f}", args.quiet)

                if args.stats and not args.quiet and len(results) > 0:
                    print("\nDetailed Statistics:")
                    scores = [r['score'] for r in results]
                    print(f"  Min score: {min(scores):.4f}")
                    print(f"  Max score: {max(scores):.4f}")
                    print(f"  Std dev: {np.std(scores):.4f}")

            elif args.batch_command == "encode":
                # Create output directory
                output_dir = Path(args.output_dir)
                if not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)

                # Parse inputs: could be list of files, directory, or input-list file
                inputs = []
                if args.inputs:
                    inputs = args.inputs
                elif args.input_list:
                    with open(args.input_list) as f:
                        inputs = [line.strip() for line in f if line.strip()]
                elif args.input_dir:
                    input_path = Path(args.input_dir)
                    if input_path.is_dir():
                        # Find all structure files
                        inputs = list(input_path.glob("*.pdb")) + list(input_path.glob("*.cif"))
                        inputs = [str(p) for p in inputs]
                    else:
                        print(f"[FAIL] Input directory not found: {args.input_dir}", file=sys.stderr)
                        return 1
                else:
                    print(f"[FAIL] Must specify inputs, --input-list, or --input-dir", file=sys.stderr)
                    return 1

                # Encode all structures
                chain = parse_chain(args.chain)
                success_count = 0
                progress_bar = _align_cpp.ProgressBar(len(inputs), "Encoding structures", width=20)
                for idx, input_path in enumerate(inputs):
                    try:
                        # Get base name for output
                        base_name = Path(input_path).stem
                        output_path = output_dir / f"{base_name}.npy"

                        # Encode structure
                        result = encode(input_path, chain=chain, k_neighbors=args.k_neighbors)

                        # Save embeddings
                        np.save(output_path, result.embeddings)
                        success_count += 1

                        if not args.quiet:
                            print(f"  Encoded: {input_path} → {output_path}")

                    except Exception as e:
                        print(f"  [ERROR] Failed to encode {input_path}: {e}", file=sys.stderr)

                    # Update progress
                    progress_bar.update(idx + 1)

                progress_bar.finish()
                print_info(f"[OK] Batch encode complete", args.quiet)
                print_info(f"  Total structures: {len(inputs)}", args.quiet)
                print_info(f"  Successfully encoded: {success_count}", args.quiet)
                print_info(f"  Failed: {len(inputs) - success_count}", args.quiet)
                print_info(f"  Output directory: {args.output_dir}", args.quiet)

            else:
                print(f"[FAIL] Unknown batch command: {args.batch_command}", file=sys.stderr)
                return 1

        return 0

    except Exception as e:
        import traceback
        print(f"[FAIL] Error: {e}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
