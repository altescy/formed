"""Serialization formats for workflow step results.

This module provides format abstractions for serializing and deserializing
workflow step results. Multiple formats support different data types and
use cases.

Key Components:
    - Format: Abstract base class for all formats
    - PickleFormat: Universal format using cloudpickle
    - JsonFormat: JSON format for JSON-serializable data
    - MappingFormat: JSON format for mappings/dicts
    - AutoFormat: Automatically selects appropriate format

Features:
    - Support for iterators and streaming data
    - Type-safe serialization/deserialization
    - Format auto-detection based on data type
    - Extensible via registration system

Example:
    >>> from formed.workflow import JsonFormat, PickleFormat
    >>>
    >>> # JSON format for simple data
    >>> json_format = JsonFormat()
    >>> json_format.write({"key": "value"}, directory)
    >>> data = json_format.read(directory)
    >>>
    >>> # Pickle format for complex objects
    >>> pickle_format = PickleFormat()
    >>> pickle_format.write(my_model, directory)
    >>> model = pickle_format.read(directory)

"""

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
    """Abstract base class for serialization formats.

    Formats handle serialization and deserialization of workflow step
    results to/from disk. Each format is identified by a unique identifier
    and can indicate if it's the default format for a given type.

    Type Parameters:
        _T: Type of data this format can serialize/deserialize.

    """
    @property
    def identifier(self) -> str:
        """Get the unique identifier for this format.

        Returns:
            Format identifier string.

        """
        return f"{self.__class__.__module__}:{self.__class__.__name__}"

    def write(self, artifact: _T, directory: Path) -> None:
        """Write artifact to directory.

        Args:
            artifact: Data to serialize.
            directory: Directory to write to.

        """
        raise NotImplementedError

    def read(self, directory: Path) -> _T:
        """Read artifact from directory.

        Args:
            directory: Directory to read from.

        Returns:
            Deserialized data.

        """
        raise NotImplementedError

    @classmethod
    def is_default_of(cls, obj: Any) -> bool:
        """Check if this format is the default for the given object type.

        Args:
            obj: Object to check.

        Returns:
            True if this format should be used by default for this type.

        """
        return False


@Format.register("pickle")
class PickleFormat(Format[_T], Generic[_T]):
    """Universal serialization format using cloudpickle.

    This format can serialize almost any Python object, including
    functions, classes, and complex nested structures. It also
    supports streaming iterators.

    Example:
        >>> format = PickleFormat()
        >>> format.write(my_object, directory)
        >>> obj = format.read(directory)
        >>>
        >>> # For iterators
        >>> format.write(iter(range(1000)), directory)
        >>> iterator = format.read(directory)  # Returns iterator

    Note:
        This is the fallback format when no other format applies.

    """
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
    """JSON-based serialization format for JSON-compatible data.

    This format serializes data to JSON files (.json for single objects,
    .jsonl for iterators). It supports all JSON-serializable types plus
    dataclasses, named tuples, and Pydantic models.

    Features:
        - Human-readable format
        - Support for iterators via JSON Lines format
        - Automatic type reconstruction using metadata
        - Custom JSON encoder/decoder for extended types

    Type Parameters:
        _JsonFormattableT: JSON-compatible type (primitives, containers, dataclasses, etc.)

    Example:
        >>> format = JsonFormat()
        >>>
        >>> # Single object
        >>> format.write({"key": "value"}, directory)
        >>> data = format.read(directory)
        >>>
        >>> # Iterator (uses JSONL)
        >>> format.write(iter([{"a": 1}, {"a": 2}]), directory)
        >>> iterator = format.read(directory)

    Note:
        This is the default format for JSON-serializable types.
        Objects are reconstructed with their original type using metadata.

    """
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
        """Write JSON-serializable artifact to directory.

        Writes artifact.json for single objects or artifact.jsonl for
        iterators. Also writes metadata.json containing type information
        for reconstruction.

        Args:
            artifact: Data to serialize (single object or iterator).
            directory: Directory to write to.

        """
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
        """Read JSON artifact from directory.

        Reads artifact.json or artifact.jsonl and reconstructs the
        original type using metadata.json if available.

        Args:
            directory: Directory to read from.

        Returns:
            Deserialized data with original type.

        """
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
        """Check if JSON format is default for object type.

        Args:
            obj: Object to check.

        Returns:
            True for JSON-serializable types (primitives, containers,
            dataclasses, named tuples, Pydantic models).

        """
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
    """Format for mappings using subdirectories for values.

    This format stores each mapping entry as a subdirectory, with the
    key as the directory name and the value serialized using a nested
    format. This allows mappings of complex objects to be stored in
    an organized directory structure.

    Type Parameters:
        _T: Type of mapping values.

    Args:
        format: Format to use for serializing mapping values.

    Example:
        >>> # Mapping of strings to dataframes
        >>> inner_format = PickleFormat()
        >>> format = MappingFormat(inner_format)
        >>>
        >>> data = {
        ...     "train": train_df,
        ...     "test": test_df,
        ... }
        >>> format.write(data, directory)
        >>> # Creates: directory/train/artifact.pkl
        >>> #          directory/test/artifact.pkl
        >>>
        >>> loaded = format.read(directory)

    Note:
        Keys must be valid directory names (no special characters).

    """

    def __init__(self, format: Format[_T]) -> None:
        self._format = format

    def write(self, artifact: Mapping[str, _T], directory: Path) -> None:
        """Write mapping to subdirectories.

        Each mapping entry is written to a subdirectory named after
        the key, with the value serialized using the nested format.

        Args:
            artifact: Mapping to serialize.
            directory: Directory to write to.

        """
        for key, value in artifact.items():
            subdir = directory / key
            subdir.mkdir(parents=True)
            self._format.write(value, subdir)

    def read(self, directory: Path) -> Mapping[str, _T]:
        """Read mapping from subdirectories.

        Reconstructs the mapping by reading each subdirectory as a
        key-value pair.

        Args:
            directory: Directory to read from.

        Returns:
            Reconstructed mapping.

        """
        artifact: dict[str, _T] = {}
        for subdir in directory.glob("*"):
            artifact[subdir.name] = self._format.read(subdir)
        return artifact


@Format.register("auto")
class AutoFormat(Format[_T]):
    """Automatic format selection based on object type.

    This format automatically selects the most appropriate format for
    an object by checking each registered format's `is_default_of()`
    method. It stores the chosen format name in metadata for correct
    deserialization.

    Selection priority:
        1. Last registered format that claims the type (most specific)
        2. Falls back to pickle format if no format claims the type

    Example:
        >>> format = AutoFormat()
        >>>
        >>> # Automatically uses JsonFormat for dict
        >>> format.write({"key": "value"}, directory)
        >>>
        >>> # Automatically uses PickleFormat for custom objects
        >>> format.write(my_custom_object, directory)
        >>>
        >>> # Reads with the same format used during write
        >>> obj = format.read(directory)

    Note:
        This is the recommended format for most use cases as it
        provides optimal serialization for each type.

    """

    _DEFAULT_FORMAT: ClassVar[str] = "pickle"
    _FORMAT_FILENAME: ClassVar[str] = "__format__"

    @classmethod
    def _get_default_format_name(cls, obj: _T) -> str:
        """Determine the best format for an object.

        Args:
            obj: Object to find format for.

        Returns:
            Name of the format to use.

        """
        registry = Format._registry[Format]
        # NOTE: `reversed` is a workaround to prioritize the last registered format
        # that may be more specific than the first registered format
        for name, (format_cls, _) in reversed(registry.items()):
            if format_cls.is_default_of(obj):
                return name
        return cls._DEFAULT_FORMAT

    def write(self, artifact: _T, directory: Path) -> None:
        """Write artifact using automatically selected format.

        Selects the appropriate format, writes the artifact, and stores
        the format name in metadata for deserialization.

        Args:
            artifact: Data to serialize.
            directory: Directory to write to.

        """
        format_name = self._get_default_format_name(artifact)
        format = cast(type[Format[_T]], Format.by_name(format_name))()
        format.write(artifact, directory)
        (directory / self._FORMAT_FILENAME).write_text(json.dumps({"name": format_name}))

    def read(self, directory: Path) -> _T:
        """Read artifact using the format recorded in metadata.

        Reads the format metadata and uses the same format that was
        used during writing.

        Args:
            directory: Directory to read from.

        Returns:
            Deserialized data.

        """
        format_metadata = json.loads((directory / self._FORMAT_FILENAME).read_text())
        format_name = format_metadata["name"]
        format = cast(type[Format[_T]], Format.by_name(format_name))()
        return format.read(directory)
