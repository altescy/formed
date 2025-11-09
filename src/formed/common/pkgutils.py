"""Package information utilities for dependency tracking.

This module provides utilities for querying installed Python packages and their
versions. It's used for capturing environment metadata in workflow execution.

Example:
    >>> from formed.common.pkgutils import get_installed_packages, PackageInfo
    >>>
    >>> # Get all installed packages
    >>> packages = get_installed_packages()
    >>> for pkg in packages[:3]:
    ...     print(f"{pkg.name}: {pkg.version}")

"""

import dataclasses
import importlib.metadata


@dataclasses.dataclass(frozen=True)
class PackageInfo:
    """Information about an installed Python package.

    Attributes:
        name: Package name (e.g., "numpy", "torch").
        version: Package version string (e.g., "1.24.0").

    """

    name: str
    version: str


def get_installed_packages() -> list[PackageInfo]:
    """Get information about all installed Python packages.

    Queries the Python environment using importlib.metadata to retrieve
    the name and version of all installed distributions.

    Returns:
        List of PackageInfo objects sorted by package name.

    Example:
        >>> packages = get_installed_packages()
        >>> numpy_pkg = next(p for p in packages if p.name == "numpy")
        >>> print(f"numpy version: {numpy_pkg.version}")

    Note:
        - Includes all packages in the current environment
        - Sorted alphabetically by package name
        - Used for workflow metadata and reproducibility tracking

    """
    distributions = importlib.metadata.distributions()
    return sorted(
        [PackageInfo(d.metadata["Name"], d.version) for d in distributions],
        key=lambda p: p.name,
    )
