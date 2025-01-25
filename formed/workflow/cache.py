from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, Union, cast

from colt import Registrable
from filelock import FileLock

if TYPE_CHECKING:
    from .step import WorkflowStep, WorkflowStepInfo


T = TypeVar("T")
T_WorkflowCache = TypeVar("T_WorkflowCache", bound="WorkflowCache")


class WorkflowCache(Registrable):
    def __getitem__(self, step_info: "WorkflowStepInfo[WorkflowStep[T]]") -> T:
        raise NotImplementedError

    def __setitem__(self, step_info: "WorkflowStepInfo[WorkflowStep[T]]", value: T) -> None:
        raise NotImplementedError

    def __delitem__(self, step_info: "WorkflowStepInfo") -> None:
        raise NotImplementedError

    def __contains__(self, step_info: "WorkflowStepInfo") -> bool:
        raise NotImplementedError


@WorkflowCache.register("empty")
class EmptyWorkflowCache(WorkflowCache):
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
    _LOCK_FILENAME: ClassVar[str] = "__lock__"

    def __init__(self, directory: Union[str, PathLike]) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _get_step_cache_dir(self, step_info: "WorkflowStepInfo") -> Path:
        return self._directory / step_info.fingerprint

    def _get_step_cache_lock(self, step_info: "WorkflowStepInfo") -> FileLock:
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
