"""Errors that end a run with a specific exit code (ARCHITECTURE §14)."""

from __future__ import annotations


class ScanError(Exception):
    exit_code = 1


class InputError(ScanError):
    """Unusable input: wrong layout, empty or unreadable capture. Message names the file/folder."""

    exit_code = 2


class WeightsMissingError(ScanError, FileNotFoundError):
    exit_code = 3
