"""Workflow step result caching implementations.

This module provides caching backends for storing and retrieving workflow step
results. Multiple cache implementations are available for different use cases.

Available Caches:
    - EmptyWorkflowCache: No-op cache that never stores results
    - MemoryWorkflowCache: In-memory cache for development/testing
    - FilesystemWorkflowCache: Persistent file-based cache (default)

Examples:
    >>> from formed.workflow.cache import FilesystemWorkflowCache
    >>>
    >>> # Create filesystem cache
    >>> cache = FilesystemWorkflowCache(".formed/cache")
    >>>
    >>> # Check if step result is cached
    >>> if step_info in cache:
    ...     result = cache[step_info]
    ... else:
    ...     result = execute_step()
    ...     cache[step_info] = result

"""

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

from colt import Registrable
from filelock import BaseFileLock, FileLock

if TYPE_CHECKING:
    from .step import WorkflowStep, WorkflowStepInfo


T = TypeVar("T")
WorkflowCacheT = TypeVar("WorkflowCacheT", bound="WorkflowCache")


class WorkflowCache(Registrable):
    """Abstract base class for workflow step result caching.

    WorkflowCache provides a dict-like interface for storing and retrieving
    step execution results, keyed by WorkflowStepInfo (which includes the
    step's fingerprint).

    Subclasses implement different storage backends (memory, filesystem, etc.).

    Examples:
        >>> # Implement custom cache
        >>> class MyCache(WorkflowCache):
        ...     def __getitem__(self, step_info):
        ...         # Retrieve from custom backend
        ...         pass
        ...     def __setitem__(self, step_info, value):
        ...         # Store to custom backend
        ...         pass
        ...     def __contains__(self, step_info):
        ...         # Check if cached
        ...         pass
        ...     def __delitem__(self, step_info):
        ...         # Remove from cache
        ...         pass

    Note:
        - Cache keys are WorkflowStepInfo instances
        - Fingerprints uniquely identify step configurations
        - Thread-safety depends on implementation

    """

    def __getitem__(self, step_info: "WorkflowStepInfo[WorkflowStep[T]]") -> T:
        """Retrieve cached result for a step.

        Args:
            step_info: Metadata identifying the step.

        Returns:
            The cached result.

        Raises:
            KeyError: If step result is not cached.

        """
        raise NotImplementedError

    def __setitem__(self, step_info: "WorkflowStepInfo[WorkflowStep[T]]", value: T) -> None:
        """Store a step result in the cache.

        Args:
            step_info: Metadata identifying the step.
            value: The result to cache.

        """
        raise NotImplementedError

    def __delitem__(self, step_info: "WorkflowStepInfo") -> None:
        """Remove a cached step result.

        Args:
            step_info: Metadata identifying the step to remove.

        """
        raise NotImplementedError

    def __contains__(self, step_info: "WorkflowStepInfo") -> bool:
        """Check if a step result is cached.

        Args:
            step_info: Metadata identifying the step.

        Returns:
            True if the step result is cached, False otherwise.

        """
        raise NotImplementedError


@WorkflowCache.register("empty")
class EmptyWorkflowCache(WorkflowCache):
    """No-op cache that never stores results.

    EmptyWorkflowCache disables caching entirely. All __contains__ checks return False
    and all __getitem__ calls raise KeyError, forcing steps to always re-execute.

    This is useful for debugging or when caching is undesirable.

    Examples:
        >>> cache = EmptyWorkflowCache()
        >>> cache[step_info] = result  # Does nothing
        >>> step_info in cache  # Always returns False

    """

    def __getitem__(self, step_info: "WorkflowStepInfo[WorkflowStep[T]]") -> T:
        raise KeyError(step_info)

    def __setitem__(self, step_info: "WorkflowStepInfo[WorkflowStep[T]]", value: T) -> None:
        pass

    def __delitem__(self, step_info: "WorkflowStepInfo") -> None:
        pass

    def __contains__(self, step_info: "WorkflowStepInfo") -> bool:
        return False


@WorkflowCache.register("memory")
class MemoryWorkflowCache(WorkflowCache):
    """In-memory cache for workflow step results.

    MemoryWorkflowCache stores results in a Python dictionary, providing fast access
    but no persistence across process restarts. Useful for development, testing,
    or when results don't need to survive process boundaries.

    Examples:
        >>> cache = MemoryWorkflowCache()
        >>> cache[step_info] = result
        >>> if step_info in cache:
        ...     result = cache[step_info]
        >>> print(len(cache))  # Number of cached steps

    Note:
        - Cache is lost when process ends
        - Not suitable for production workflows
        - No size limit - can grow unbounded

    """

    def __init__(self) -> None:
        self._cache: dict["WorkflowStepInfo", Any] = {}

    def __getitem__(self, step_info: "WorkflowStepInfo[WorkflowStep[T]]") -> T:
        return cast(T, self._cache[step_info])

    def __setitem__(self, step_info: "WorkflowStepInfo[WorkflowStep[T]]", value: T) -> None:
        self._cache[step_info] = value

    def __delitem__(self, step_info: "WorkflowStepInfo") -> None:
        del self._cache[step_info]

    def __contains__(self, step_info: "WorkflowStepInfo") -> bool:
        return step_info in self._cache

    def __len__(self) -> int:
        return len(self._cache)


@WorkflowCache.register("filesystem")
class FilesystemWorkflowCache(WorkflowCache):
    """Persistent file-based cache for workflow step results.

    FilesystemWorkflowCache stores step results in a directory structure organized
    by fingerprint. Each step's result is serialized using its configured Format
    and written to a subdirectory. File locking ensures thread-safe concurrent access.

    Attributes:
        _directory: Root directory for cache storage.

    Examples:
        >>> cache = FilesystemWorkflowCache(".formed/cache")
        >>> cache[step_info] = result  # Writes to .formed/cache/<fingerprint>/
        >>> if step_info in cache:
        ...     result = cache[step_info]  # Reads from disk
        >>> del cache[step_info]  # Removes cached result

    Note:
        - Results persist across process restarts
        - Thread-safe via file locking
        - Cache directory structure: {cache_dir}/{fingerprint}/
        - Each step uses its Format for serialization
        - Suitable for production workflows

    """

    _LOCK_FILENAME: ClassVar[str] = "__lock__"

    def __init__(self, directory: str | PathLike) -> None:
        """Initialize filesystem cache.

        Args:
            directory: Root directory for cache storage. Created if doesn't exist.

        """
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _get_step_cache_dir(self, step_info: "WorkflowStepInfo") -> Path:
        return self._directory / step_info.fingerprint

    def _get_step_cache_lock(self, step_info: "WorkflowStepInfo") -> BaseFileLock:
        return FileLock(str(self._get_step_cache_dir(step_info) / self._LOCK_FILENAME))

    def __getitem__(self, step_info: "WorkflowStepInfo[WorkflowStep[T]]") -> T:
        with self._get_step_cache_lock(step_info):
            step_cache_dir = self._get_step_cache_dir(step_info)
            return cast(T, step_info.format.read(step_cache_dir))

    def __setitem__(self, step_info: "WorkflowStepInfo[WorkflowStep[T]]", value: T) -> None:
        with self._get_step_cache_lock(step_info):
            step_cache_dir = self._get_step_cache_dir(step_info)
            step_cache_dir.mkdir(parents=True, exist_ok=True)
            step_info.format.write(value, step_cache_dir)

    def __delitem__(self, step_info: "WorkflowStepInfo") -> None:
        with self._get_step_cache_lock(step_info):
            step_cache_dir = self._get_step_cache_dir(step_info)
            for path in step_cache_dir.glob("**/*"):
                path.unlink()
            step_cache_dir.rmdir()

    def __contains__(self, step_info: "WorkflowStepInfo") -> bool:
        return self._get_step_cache_dir(step_info).exists()
