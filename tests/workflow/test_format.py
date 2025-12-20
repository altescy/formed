"""Tests for formed.workflow.format module"""

import dataclasses
import tempfile
from collections.abc import Iterator
from pathlib import Path

from formed.common.dataset import Dataset
from formed.workflow.format import AutoFormat, DatasetFormat, Format, JsonFormat, PickleFormat


# Test data structures
@dataclasses.dataclass
class SimpleData:
    id: str
    value: int
    text: str


@dataclasses.dataclass
class NestedData:
    simple: SimpleData
    items: list[str]
    metadata: dict[str, int]


class TestPickleFormat:
    """Tests for PickleFormat"""

    def test_write_and_read_simple_object(self) -> None:
        """Test writing and reading a simple object"""
        data = SimpleData(id="test1", value=42, text="hello")
        format_handler = PickleFormat[SimpleData]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            # Write
            format_handler.write(data, directory)

            # Check artifact file exists
            assert (directory / "artifact.pkl").exists()

            # Read
            loaded = format_handler.read(directory)

            # Verify
            assert isinstance(loaded, SimpleData)
            assert loaded.id == "test1"
            assert loaded.value == 42
            assert loaded.text == "hello"

    def test_write_and_read_complex_object(self) -> None:
        """Test writing and reading a complex nested object"""
        simple = SimpleData(id="nested", value=100, text="world")
        data = NestedData(
            simple=simple,
            items=["a", "b", "c"],
            metadata={"x": 1, "y": 2},
        )
        format_handler = PickleFormat[NestedData]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            format_handler.write(data, directory)
            loaded = format_handler.read(directory)

            assert isinstance(loaded, NestedData)
            assert loaded.simple.id == "nested"
            assert loaded.items == ["a", "b", "c"]
            assert loaded.metadata == {"x": 1, "y": 2}

    def test_write_and_read_iterator(self) -> None:
        """Test writing and reading an iterator"""

        def create_iterator() -> Iterator[SimpleData]:
            for i in range(5):
                yield SimpleData(id=f"item{i}", value=i * 10, text=f"text{i}")

        format_handler = PickleFormat[Iterator[SimpleData]]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            # Write
            format_handler.write(create_iterator(), directory)

            # Read
            loaded = format_handler.read(directory)

            # Verify it's an iterator
            assert hasattr(loaded, "__iter__")
            assert hasattr(loaded, "__next__")

            # Consume and verify
            items = list(loaded)
            assert len(items) == 5
            assert items[0].id == "item0"
            assert items[0].value == 0
            assert items[4].id == "item4"
            assert items[4].value == 40

    def test_write_and_read_empty_iterator(self) -> None:
        """Test writing and reading an empty iterator"""

        def create_empty_iterator() -> Iterator[SimpleData]:
            return
            yield  # Make it a generator

        format_handler = PickleFormat[Iterator[SimpleData]]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            format_handler.write(create_empty_iterator(), directory)
            loaded = format_handler.read(directory)

            items = list(loaded)
            assert len(items) == 0

    def test_iterator_can_be_consumed_once(self) -> None:
        """Test that iterator can only be consumed once"""

        def create_iterator() -> Iterator[SimpleData]:
            for i in range(3):
                yield SimpleData(id=f"item{i}", value=i, text=f"text{i}")

        format_handler = PickleFormat[Iterator[SimpleData]]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            format_handler.write(create_iterator(), directory)
            loaded = format_handler.read(directory)

            # First consumption
            items1 = list(loaded)
            assert len(items1) == 3

            # Second consumption should be empty
            items2 = list(loaded)
            assert len(items2) == 0


class TestJsonFormat:
    """Tests for JsonFormat"""

    def test_write_and_read_simple_dataclass(self) -> None:
        """Test writing and reading a simple dataclass"""
        data = SimpleData(id="json1", value=99, text="json test")
        format_handler = JsonFormat[SimpleData]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            # Write
            format_handler.write(data, directory)

            # Check files exist
            assert (directory / "artifact.json").exists()
            assert (directory / "metadata.json").exists()

            # Read
            loaded = format_handler.read(directory)

            # Verify
            assert isinstance(loaded, SimpleData)
            assert loaded.id == "json1"
            assert loaded.value == 99
            assert loaded.text == "json test"

    def test_write_and_read_dict(self) -> None:
        """Test writing and reading a dict"""
        data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
        format_handler = JsonFormat[dict[str, int | str | list[int]]]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            format_handler.write(data, directory)
            loaded = format_handler.read(directory)

            assert isinstance(loaded, dict)
            assert loaded == data

    def test_write_and_read_iterator_jsonl(self) -> None:
        """Test writing and reading an iterator as JSONL"""

        def create_iterator() -> Iterator[SimpleData]:
            for i in range(4):
                yield SimpleData(id=f"jsonl{i}", value=i * 5, text=f"line{i}")

        format_handler = JsonFormat[Iterator[SimpleData]]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            # Write
            format_handler.write(create_iterator(), directory)

            # Check JSONL file exists
            assert (directory / "artifact.jsonl").exists()
            assert (directory / "metadata.json").exists()

            # Read
            loaded = format_handler.read(directory)

            # Verify it's an iterator
            items = list(loaded)
            assert len(items) == 4
            assert items[0].id == "jsonl0"
            assert items[3].value == 15

    def test_write_and_read_empty_iterator_jsonl(self) -> None:
        """Test writing and reading an empty iterator"""

        def create_empty_iterator() -> Iterator[SimpleData]:
            return
            yield

        format_handler = JsonFormat[Iterator[SimpleData]]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            format_handler.write(create_empty_iterator(), directory)

            # Empty JSONL should still create the file
            assert (directory / "artifact.jsonl").exists()

            loaded = format_handler.read(directory)
            items = list(loaded)
            assert len(items) == 0

    def test_metadata_preserves_class_info(self) -> None:
        """Test that metadata correctly preserves class information"""
        data = SimpleData(id="meta", value=123, text="metadata")
        format_handler = JsonFormat[SimpleData]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            format_handler.write(data, directory)

            # Check metadata content
            import json

            metadata_path = directory / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            assert "module" in metadata
            assert "class" in metadata
            assert metadata["class"] == "SimpleData"
            assert "test_format" in metadata["module"]

    def test_is_default_of(self) -> None:
        """Test is_default_of class method"""
        # Should return True for basic types
        assert JsonFormat.is_default_of(42)
        assert JsonFormat.is_default_of(3.14)
        assert JsonFormat.is_default_of("string")
        assert JsonFormat.is_default_of(True)
        assert JsonFormat.is_default_of({"key": "value"})
        assert JsonFormat.is_default_of([1, 2, 3])
        assert JsonFormat.is_default_of((1, 2, 3))

        # Should return True for dataclasses
        assert JsonFormat.is_default_of(SimpleData(id="x", value=1, text="y"))

        # Should return False for custom classes without proper protocols
        class CustomClass:
            pass

        assert not JsonFormat.is_default_of(CustomClass())


class TestDatasetFormat:
    """Tests for DatasetFormat"""

    def test_write_and_read_dataset(self) -> None:
        """Test writing and reading a Dataset"""
        # Create a dataset
        items = [SimpleData(id=f"ds{i}", value=i * 10, text=f"dataset{i}") for i in range(5)]
        dataset = Dataset.from_iterable(items)

        format_handler = DatasetFormat[SimpleData]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            # Write
            format_handler.write(dataset, directory)

            # Check dataset directory exists
            assert (directory / "dataset").exists()
            assert (directory / "dataset" / "metadata.db").exists()

            # Read
            loaded = format_handler.read(directory)

            # Verify
            assert isinstance(loaded, Dataset)
            assert len(loaded) == 5
            assert loaded[0].id == "ds0"
            assert loaded[4].value == 40

    def test_write_and_read_empty_dataset(self) -> None:
        """Test writing and reading an empty Dataset"""
        dataset = Dataset.from_iterable([])

        format_handler = DatasetFormat[SimpleData]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            format_handler.write(dataset, directory)
            loaded = format_handler.read(directory)

            assert isinstance(loaded, Dataset)
            assert len(loaded) == 0

    def test_is_default_of(self) -> None:
        """Test is_default_of class method"""
        dataset = Dataset.from_iterable([1, 2, 3])
        assert DatasetFormat.is_default_of(dataset)

        assert not DatasetFormat.is_default_of([1, 2, 3])
        assert not DatasetFormat.is_default_of("string")


class TestAutoFormat:
    """Tests for AutoFormat"""

    def test_auto_select_json_for_dataclass(self) -> None:
        """Test that AutoFormat selects JsonFormat for dataclass"""
        data = SimpleData(id="auto1", value=77, text="auto")
        format_handler = AutoFormat[SimpleData]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            format_handler.write(data, directory)

            # Check format metadata
            import json

            format_meta = json.loads((directory / "__format__").read_text())
            assert format_meta["name"] == "json"

            # Should have JSON files
            assert (directory / "artifact.json").exists()
            assert (directory / "metadata.json").exists()

            loaded = format_handler.read(directory)
            assert isinstance(loaded, SimpleData)
            assert loaded.id == "auto1"

    def test_auto_select_json_for_dict(self) -> None:
        """Test that AutoFormat selects JsonFormat for dict"""
        data = {"key": "value", "num": 42}
        format_handler = AutoFormat[dict[str, str | int]]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            format_handler.write(data, directory)

            import json

            format_meta = json.loads((directory / "__format__").read_text())
            assert format_meta["name"] == "json"

            loaded = format_handler.read(directory)
            assert loaded == data

    def test_auto_select_dataset_for_dataset(self) -> None:
        """Test that AutoFormat selects DatasetFormat for Dataset"""
        items = [SimpleData(id=f"d{i}", value=i, text=f"t{i}") for i in range(3)]
        dataset = Dataset.from_iterable(items)

        format_handler = AutoFormat[Dataset[SimpleData]]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            format_handler.write(dataset, directory)

            import json

            format_meta = json.loads((directory / "__format__").read_text())
            assert format_meta["name"] == "dataset"

            # Should have dataset directory
            assert (directory / "dataset").exists()

            loaded = format_handler.read(directory)
            assert isinstance(loaded, Dataset)
            assert len(loaded) == 3

    def test_auto_select_pickle_for_custom_class(self) -> None:
        """Test that AutoFormat falls back to PickleFormat for custom classes"""

        class CustomClass:
            def __init__(self, value: int) -> None:
                self.value = value

        data = CustomClass(value=999)
        format_handler = AutoFormat[CustomClass]()

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            format_handler.write(data, directory)

            import json

            format_meta = json.loads((directory / "__format__").read_text())
            assert format_meta["name"] == "pickle"

            # Should have pickle file
            assert (directory / "artifact.pkl").exists()

            loaded = format_handler.read(directory)
            assert isinstance(loaded, CustomClass)
            assert loaded.value == 999

    def test_format_identifier(self) -> None:
        """Test that Format has correct identifier"""
        format_handler = PickleFormat[SimpleData]()
        identifier = format_handler.identifier

        assert "formed.workflow.format" in identifier
        assert "PickleFormat" in identifier

    def test_registry_contains_all_formats(self) -> None:
        """Test that all formats are registered"""
        # Access the registry via Registrable
        registry = Format._registry[Format]

        assert "pickle" in registry
        assert "json" in registry
        assert "dataset" in registry
        assert "auto" in registry

        # Verify by_name works
        pickle_cls = Format.by_name("pickle")
        assert pickle_cls == PickleFormat

        json_cls = Format.by_name("json")
        assert json_cls == JsonFormat

        dataset_cls = Format.by_name("dataset")
        assert dataset_cls == DatasetFormat

        auto_cls = Format.by_name("auto")
        assert auto_cls == AutoFormat
