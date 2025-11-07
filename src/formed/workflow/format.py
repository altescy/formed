import importlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import IO, Any, ClassVar, Generic, Optional, TypeVar, Union, cast

import cloudpickle
import colt
from colt import Registrable
from pydantic import BaseModel

from formed.types import DataContainer, IDataclass, INamedTuple, JsonValue

from .utils import WorkflowJSONDecoder, WorkflowJSONEncoder

_JsonFormattable = Union[DataContainer, JsonValue]

_S = TypeVar("_S")
_T = TypeVar("_T")
_JsonFormattableT = TypeVar("_JsonFormattableT", bound=Union[_JsonFormattable, Iterator[_JsonFormattable]])


class Format(Generic[_T], Registrable):
    @property
    def identifier(self) -> str:
        return f"{self.__class__.__module__}:{self.__class__.__name__}"

    def write(self, artifact: _T, directory: Path) -> None:
        raise NotImplementedError

    def read(self, directory: Path) -> _T:
        raise NotImplementedError

    @classmethod
    def is_default_of(cls, obj: Any) -> bool:
        return False


@Format.register("pickle")
class PickleFormat(Format[_T], Generic[_T]):
    class _IteratorWrapper(Generic[_S]):
        def __init__(self, path: Path) -> None:
            self._file: Optional[IO[Any]] = path.open("rb")
            assert cloudpickle.load(self._file)  # Check if it is an iterator

        def __iter__(self) -> Iterator[_S]:
            return self

        def __next__(self) -> _S:
            if self._file is None:
                raise StopIteration
            try:
                return cast(_S, cloudpickle.load(self._file))
            except EOFError:
                self._file.close()
                self._file = None
                raise StopIteration

    def _get_artifact_path(self, directory: Path) -> Path:
        return directory / "artifact.pkl"

    def write(self, artifact: _T, directory: Path) -> None:
        artifact_path = self._get_artifact_path(directory)
        with open(artifact_path, "wb") as f:
            if isinstance(artifact, Iterator):
                cloudpickle.dump(True, f)
                for item in artifact:
                    cloudpickle.dump(item, f)
            else:
                cloudpickle.dump(False, f)
                cloudpickle.dump(artifact, f)

    def read(self, directory: Path) -> _T:
        artifact_path = self._get_artifact_path(directory)
        with open(artifact_path, "rb") as f:
            is_iterator = cloudpickle.load(f)
            if is_iterator:
                return cast(_T, self._IteratorWrapper(artifact_path))
            return cast(_T, cloudpickle.load(f))


@Format.register("json")
class JsonFormat(Format[_JsonFormattableT], Generic[_JsonFormattableT]):
    class _IteratorWrapper(Generic[_S]):
        def __init__(self, path: Path, artifact_class: Optional[type[_S]]) -> None:
            self._file = path.open("r")
            self._artifact_class = artifact_class

        def __iter__(self) -> Iterator[_S]:
            return self

        def __next__(self) -> _S:
            line = self._file.readline()
            if not line:
                self._file.close()
                raise StopIteration
            data = json.loads(line, cls=WorkflowJSONDecoder)
            if self._artifact_class is not None:
                return colt.build(data, self._artifact_class)
            return cast(_S, data)

    def write(self, artifact: _JsonFormattableT, directory: Path) -> None:
        artifact_class: Optional[type[_JsonFormattableT]] = None
        if isinstance(artifact, Iterator):
            artifact_path = directory / "artifact.jsonl"
            with open(artifact_path, "w") as f:
                for item in artifact:
                    artifact_class = cast(
                        type[_JsonFormattableT],
                        artifact_class or type(item),
                    )
                    json.dump(item, f, cls=WorkflowJSONEncoder, ensure_ascii=False)
                    f.write("\n")
        else:
            artifact_class = type(artifact)
            artifact_path = directory / "artifact.json"
            with open(artifact_path, "w") as f:
                json.dump(artifact, f, cls=WorkflowJSONEncoder, ensure_ascii=False)
        if artifact_class is not None:
            metadata = {
                "module": artifact_class.__module__,
                "class": artifact_class.__name__,
            }
            metadata_path = directory / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, ensure_ascii=False)

    def read(self, directory: Path) -> _JsonFormattableT:
        metadata_path = directory / "metadata.json"
        artifact_class: Optional[type[_JsonFormattableT]] = None
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f, cls=WorkflowJSONDecoder)
            module = importlib.import_module(metadata["module"])
            artifact_class = getattr(module, metadata["class"])

        is_iterator = (directory / "artifact.jsonl").exists()
        if is_iterator:
            artifact_path = directory / "artifact.jsonl"
            return cast(_JsonFormattableT, self._IteratorWrapper(artifact_path, artifact_class))

        artifact_path = directory / "artifact.json"
        with open(artifact_path, "r") as f:
            data = json.load(f, cls=WorkflowJSONDecoder)
            if artifact_class is not None:
                return colt.build(data, artifact_class)
            return cast(_JsonFormattableT, data)

    @classmethod
    def is_default_of(cls, obj: Any) -> bool:
        return isinstance(
            obj,
            (
                int,
                float,
                str,
                bool,
                dict,
                list,
                tuple,
                IDataclass,
                INamedTuple,
                BaseModel,
            ),
        )


@Format.register("mapping")
class MappingFormat(Format[Mapping[str, _T]], Generic[_T]):
    def __init__(self, format: Format[_T]) -> None:
        self._format = format

    def write(self, artifact: Mapping[str, _T], directory: Path) -> None:
        for key, value in artifact.items():
            subdir = directory / key
            subdir.mkdir(parents=True)
            self._format.write(value, subdir)

    def read(self, directory: Path) -> Mapping[str, _T]:
        artifact: dict[str, _T] = {}
        for subdir in directory.glob("*"):
            artifact[subdir.name] = self._format.read(subdir)
        return artifact


@Format.register("auto")
class AutoFormat(Format[_T]):
    _DEFAULT_FORMAT: ClassVar[str] = "pickle"
    _FORMAT_FILENAME: ClassVar[str] = "__format__"

    @classmethod
    def _get_default_format_name(cls, obj: _T) -> str:
        registry = Format._registry[Format]
        # NOTE: `reversed` is a workaround to prioritize the last registered format
        # that may be more specific than the first registered format
        for name, (format_cls, _) in reversed(registry.items()):
            if format_cls.is_default_of(obj):
                return name
        return cls._DEFAULT_FORMAT

    def write(self, artifact: _T, directory: Path) -> None:
        format_name = self._get_default_format_name(artifact)
        format = cast(type[Format[_T]], Format.by_name(format_name))()
        format.write(artifact, directory)
        (directory / self._FORMAT_FILENAME).write_text(json.dumps({"name": format_name}))

    def read(self, directory: Path) -> _T:
        format_metadata = json.loads((directory / self._FORMAT_FILENAME).read_text())
        format_name = format_metadata["name"]
        format = cast(type[Format[_T]], Format.by_name(format_name))()
        return format.read(directory)
