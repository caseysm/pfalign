"""Custom exceptions for pfalign Python API.

This module provides structured error classes that match the C++ error
hierarchy for consistent error reporting across all components.
"""

from __future__ import annotations
from typing import Optional


class PFalignError(Exception):
    """Base exception for all pfalign errors.

    Provides structured error information with category, message,
    suggestion, and context for helpful error reporting.

    Args:
        message: Error message describing what went wrong
        suggestion: What the user should do to fix the error
        context: Additional context about the error

    Examples:
        >>> raise PFalignError(
        ...     "Operation failed",
        ...     suggestion="Try different parameters",
        ...     context="Input was empty"
        ... )
    """

    def __init__(
        self,
        message: str,
        suggestion: Optional[str] = None,
        context: Optional[str] = None,
    ):
        self.message = message
        self.suggestion = suggestion
        self.context = context
        super().__init__(self.formatted())

    def formatted(self) -> str:
        """Get fully formatted error message for display.

        Returns:
            Formatted error message with context and suggestion
        """
        msg = f"[ERROR] {self.message}"

        if self.context:
            msg += f"\n  Context: {self.context}"

        if self.suggestion:
            msg += f"\n  Suggestion: {self.suggestion}"

        return msg

    def __str__(self) -> str:
        return self.formatted()


class FileNotFoundError(PFalignError, FileNotFoundError):
    """File not found error.

    Raised when a required file does not exist or cannot be read.

    Args:
        path: Path to the missing file
        file_type: Type of file (e.g., "Structure", "Alignment")
    """

    def __init__(self, path: str, file_type: str = "file"):
        super().__init__(
            f"{file_type} not found: {path}",
            suggestion="Check that the file path exists and is readable",
        )
        self.path = path
        self.file_type = file_type


class FileWriteError(PFalignError, OSError):
    """File write error.

    Raised when a file cannot be written.

    Args:
        path: Path to the file that cannot be written
        reason: Optional reason for the failure
    """

    def __init__(self, path: str, reason: Optional[str] = None):
        super().__init__(
            f"Cannot write to file: {path}",
            suggestion="Check that the directory exists and you have write permissions",
            context=f"Reason: {reason}" if reason else None,
        )
        self.path = path


class ValidationError(PFalignError, ValueError):
    """Validation error for invalid parameters or values.

    Raised when a parameter has an invalid value or is out of range.

    Args:
        param_name: Name of the parameter
        value: Actual value provided
        expected: Expected value or range
    """

    def __init__(self, param_name: str, value: str, expected: str):
        super().__init__(
            f"Invalid value for {param_name}: {value}",
            suggestion=f"Expected: {expected}",
        )
        self.param_name = param_name
        self.value = value
        self.expected = expected


class FormatError(PFalignError, ValueError):
    """File format error.

    Raised when a file has an unsupported format or cannot be parsed.

    Args:
        path: Path to the file with format error
        reason: Reason for the format error
        supported_formats: List of supported formats
    """

    def __init__(
        self,
        path: str,
        reason: str,
        supported_formats: Optional[list[str]] = None,
    ):
        suggestion = None
        if supported_formats:
            suggestion = "Supported formats: " + ", ".join(supported_formats)

        super().__init__(
            f"Format error in {path}: {reason}",
            suggestion=suggestion,
        )
        self.path = path
        self.supported_formats = supported_formats or []


class ChainNotFoundError(PFalignError, KeyError):
    """Chain not found in structure.

    Raised when a requested chain ID or index does not exist in the structure.

    Args:
        chain_id: Requested chain ID or index
        structure_path: Path to the structure file
        available_chains: List of available chain IDs
    """

    def __init__(
        self,
        chain_id: str,
        structure_path: str,
        available_chains: list[str],
    ):
        if available_chains:
            suggestion = "Available chains: " + ", ".join(available_chains)
        else:
            suggestion = "Structure has no chains"

        super().__init__(
            f"Chain '{chain_id}' not found in {structure_path}",
            suggestion=suggestion,
            context="Use chain ID (A, B, C, ...) or index (0, 1, 2, ...)",
        )
        self.chain_id = chain_id
        self.structure_path = structure_path
        self.available_chains = available_chains


class DimensionError(PFalignError, ValueError):
    """Array or tensor dimension error.

    Raised when an array has incorrect shape or dimensions.

    Args:
        param_name: Name of the parameter
        actual_shape: Actual shape of the array
        expected_shape: Expected shape
    """

    def __init__(self, param_name: str, actual_shape: str, expected_shape: str):
        super().__init__(
            f"Invalid shape for {param_name}: {actual_shape}",
            suggestion=f"Expected shape: {expected_shape}",
        )
        self.param_name = param_name
        self.actual_shape = actual_shape
        self.expected_shape = expected_shape


class AlgorithmError(PFalignError, RuntimeError):
    """Algorithm error.

    Raised when an alignment or tree building algorithm fails.

    Args:
        algorithm_name: Name of the algorithm that failed
        error_message: Description of the failure
        suggestion: Optional suggestion for fixing the error
    """

    def __init__(
        self,
        algorithm_name: str,
        error_message: str,
        suggestion: Optional[str] = None,
    ):
        super().__init__(
            f"{algorithm_name} failed: {error_message}",
            suggestion=suggestion
            or "Try adjusting algorithm parameters or input data",
        )
        self.algorithm_name = algorithm_name


__all__ = [
    "PFalignError",
    "FileNotFoundError",
    "FileWriteError",
    "ValidationError",
    "FormatError",
    "ChainNotFoundError",
    "DimensionError",
    "AlgorithmError",
]
