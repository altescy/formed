"""Git repository interaction utilities.

This module provides utilities for interacting with Git repositories, extracting
repository metadata, and managing Git-related operations for workflow tracking.

Key Features:
    - Check Git availability and repository status
    - Extract commit and branch information
    - Get repository diffs and metadata
    - Capture Git state for reproducibility tracking

Example:
    >>> from formed.common.git import get_git_info, GitClient
    >>>
    >>> # Get current repository info
    >>> git_info = get_git_info()
    >>> if git_info:
    ...     print(f"Commit: {git_info.commit}")
    ...     print(f"Branch: {git_info.branch}")
    >>>
    >>> # Use GitClient for more operations
    >>> client = GitClient(".")
    >>> if client.is_initialized():
    ...     diff = client.diff()

"""

import dataclasses
import os
import subprocess
from os import PathLike
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class GitInfo:
    """Git repository metadata.

    Attributes:
        commit: The full commit hash (SHA-1).
        branch: The current branch name.

    """

    commit: str
    branch: str


class GitClient:
    """Client for interacting with Git repositories.

    GitClient provides methods for querying Git repository state, extracting
    metadata, and performing basic Git operations.

    Example:
        >>> client = GitClient(".")
        >>>
        >>> # Check if Git is available and initialized
        >>> if client.is_available() and client.is_initialized():
        ...     # Get repository info
        ...     info = client.get_info()
        ...     print(f"Current commit: {info.commit}")
        ...     print(f"Current branch: {info.branch}")
        ...
        ...     # Get diff
        ...     diff = client.diff()

    Note:
        - Requires Git to be installed and available in PATH
        - Raises RuntimeError if Git is not available during initialization

    """

    @staticmethod
    def is_available() -> bool:
        """Check if Git is installed and available.

        Returns:
            True if Git command is available, False otherwise.

        """
        return (
            subprocess.run(
                ["git", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )

    def __init__(self, directory: str | PathLike) -> None:
        """Initialize Git client for a directory.

        Args:
            directory: Path to the directory (need not be Git root).

        Raises:
            RuntimeError: If Git is not available.

        """
        if not self.is_available():
            raise RuntimeError("Git is not available")

        self._directory = Path(directory)

    def root(self) -> Path:
        return Path(os.popen(f"git -C {self._directory} rev-parse --show-toplevel").read().strip())

    def is_initialized(self) -> bool:
        return (
            subprocess.run(
                ["git", "status"],
                cwd=self._directory,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )

    def diff(self) -> str:
        return os.popen(f"git -C {self._directory} diff").read()

    def show(
        self,
        commit: str | None = None,
        format: str | None = None,
        no_patch: bool = False,
    ) -> str:
        command = f"git -C {self._directory} show"
        if commit:
            command += f" {commit}"
        if format:
            command += f" --format='{format}'"
        if no_patch:
            command += " --no-patch"
        return os.popen(command).read()

    def get_commit(self) -> str:
        return self.show(no_patch=True, format="%H").strip()

    def get_branch(self) -> str:
        return os.popen(f"git -C {self._directory} rev-parse --abbrev-ref HEAD").read().strip()

    def get_info(self) -> GitInfo:
        commit = self.get_commit()
        branch = self.get_branch()
        return GitInfo(commit, branch)


def get_git_info(directory: str | PathLike | None = None) -> GitInfo | None:
    """Get Git repository information for a directory.

    Convenience function to extract Git metadata (commit hash and branch name)
    from a directory. Returns None if Git is not available or the directory
    is not a Git repository.

    Args:
        directory: Directory to query. Defaults to current working directory.

    Returns:
        GitInfo with commit and branch, or None if not available/initialized.

    Example:
        >>> git_info = get_git_info()
        >>> if git_info:
        ...     print(f"Running on commit {git_info.commit[:8]}")
        ...     print(f"Branch: {git_info.branch}")

    """
    if not GitClient.is_available():
        return None
    directory = directory or Path.cwd()
    client = GitClient(directory)
    if not client.is_available() or not client.is_initialized():
        return None
    return client.get_info()
