"""Tests for formed.integrations.ml.workflow module"""

import dataclasses
import tempfile
from pathlib import Path

import formed.integrations.ml.types as mlt
from formed.integrations.ml import DataModule, MetadataTransform, ScalarTransform
from formed.integrations.ml.workflow import (
    DataModuleAndInstances,
    DataModuleAndInstancesFormat,
    generate_instances,
    train_datamodule,
    train_datamodule_with_instances,
)


# Test data structures
@dataclasses.dataclass
class SimpleExample:
    id: str
    value: int
    text: str


class SimpleDataModule(
    DataModule[
        mlt.DataModuleModeT,
        SimpleExample,
        "SimpleDataModule[mlt.AsInstance]",
        "SimpleDataModule[mlt.AsBatch]",
    ]
):
    id: MetadataTransform[SimpleExample, str]
    value: ScalarTransform[int]
    text: MetadataTransform[SimpleExample, str]


class TestDataModuleAndInstances:
    """Tests for DataModuleAndInstances dataclass"""

    def test_datamodule_and_instances_creation(self) -> None:
        """Test creating DataModuleAndInstances"""
        datamodule = SimpleDataModule(
            id=MetadataTransform(),
            value=ScalarTransform(),
            text=MetadataTransform(),
        )

        examples = [
            SimpleExample(id="ex1", value=10, text="hello"),
            SimpleExample(id="ex2", value=20, text="world"),
        ]

        instances = []
        for ex in examples:
            instance = datamodule(ex)
            assert instance is not None
            instances.append(instance)

        result = DataModuleAndInstances(datamodule=datamodule, instances=instances)

        assert result.datamodule is datamodule
        assert list(result.instances) == instances

    def test_datamodule_and_instances_frozen(self) -> None:
        """Test that DataModuleAndInstances is frozen"""
        datamodule = SimpleDataModule(
            id=MetadataTransform(),
            value=ScalarTransform(),
            text=MetadataTransform(),
        )

        result = DataModuleAndInstances(datamodule=datamodule, instances=[])

        # Should not be able to modify frozen dataclass
        import pytest

        with pytest.raises(AttributeError):
            result.datamodule = None  # type: ignore[misc]


class TestDataModuleAndInstancesFormat:
    """Tests for DataModuleAndInstancesFormat"""

    def test_write_and_read(self) -> None:
        """Test writing and reading DataModuleAndInstances"""
        datamodule = SimpleDataModule(
            id=MetadataTransform(),
            value=ScalarTransform(),
            text=MetadataTransform(),
        )

        examples = [
            SimpleExample(id="ex1", value=10, text="hello"),
            SimpleExample(id="ex2", value=20, text="world"),
        ]

        # Train datamodule
        with datamodule.train():
            for ex in examples:
                datamodule(ex)

        instances = []
        for ex in examples:
            instance = datamodule(ex)
            assert instance is not None
            instances.append(instance)

        artifact = DataModuleAndInstances(datamodule=datamodule, instances=instances)

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "artifact"
            directory.mkdir(parents=True, exist_ok=True)

            # Write
            format_handler = DataModuleAndInstancesFormat()
            format_handler.write(artifact, directory)

            # Check that files were created
            assert (directory / "instances").exists()
            assert (directory / "datamodule").exists()

            # Read
            loaded = format_handler.read(directory)

            # Verify datamodule structure (we can't compare objects directly)
            assert isinstance(loaded.datamodule, SimpleDataModule)

            # Verify instances
            loaded_instances = list(loaded.instances)
            assert len(loaded_instances) == 2


class TestTrainDataModule:
    """Tests for train_datamodule step"""

    def test_train_datamodule_basic(self) -> None:
        """Test training a datamodule"""
        datamodule = SimpleDataModule(
            id=MetadataTransform(),
            value=ScalarTransform(),
            text=MetadataTransform(),
        )

        examples = [
            SimpleExample(id="ex1", value=10, text="hello"),
            SimpleExample(id="ex2", value=20, text="world"),
            SimpleExample(id="ex3", value=30, text="test"),
        ]

        # Train datamodule (note: train_datamodule is a step, so we call it directly)
        result = train_datamodule(datamodule, examples)

        # Result should be the same datamodule
        assert result is datamodule

        # Verify datamodule processed all examples
        # After training, datamodule should be able to process new instances
        instance = result(examples[0])
        assert instance is not None
        assert instance.id == "ex1"
        assert instance.value == 10
        assert instance.text == "hello"

    def test_train_datamodule_empty_dataset(self) -> None:
        """Test training datamodule with empty dataset"""
        datamodule = SimpleDataModule(
            id=MetadataTransform(),
            value=ScalarTransform(),
            text=MetadataTransform(),
        )

        result = train_datamodule(datamodule, [])

        assert result is datamodule


class TestTrainDataModuleWithInstances:
    """Tests for train_datamodule_with_instances step"""

    def test_train_datamodule_with_instances_basic(self) -> None:
        """Test training datamodule and generating instances"""
        datamodule = SimpleDataModule(
            id=MetadataTransform(),
            value=ScalarTransform(),
            text=MetadataTransform(),
        )

        examples = [
            SimpleExample(id="ex1", value=10, text="hello"),
            SimpleExample(id="ex2", value=20, text="world"),
        ]

        result = train_datamodule_with_instances(datamodule, examples)

        # Check result structure
        assert isinstance(result, DataModuleAndInstances)
        assert result.datamodule is datamodule

        # Check instances
        instances = list(result.instances)
        assert len(instances) == 2
        assert instances[0].id == "ex1"
        assert instances[0].value == 10
        assert instances[1].id == "ex2"
        assert instances[1].value == 20

    def test_train_datamodule_with_instances_lazy_generation(self) -> None:
        """Test that instances are generated lazily"""
        datamodule = SimpleDataModule(
            id=MetadataTransform(),
            value=ScalarTransform(),
            text=MetadataTransform(),
        )

        examples = [SimpleExample(id=f"ex{i}", value=i * 10, text=f"text{i}") for i in range(100)]

        result = train_datamodule_with_instances(datamodule, examples)

        # Instances should be a generator (iterable)
        # We can consume it partially
        instances_iter = iter(result.instances)
        first_instance = next(instances_iter)
        assert first_instance.id == "ex0"
        assert first_instance.value == 0


class TestGenerateInstances:
    """Tests for generate_instances step"""

    def test_generate_instances_basic(self) -> None:
        """Test generating instances as Dataset"""
        datamodule = SimpleDataModule(
            id=MetadataTransform(),
            value=ScalarTransform(),
            text=MetadataTransform(),
        )

        # Pre-train datamodule
        examples = [
            SimpleExample(id="ex1", value=10, text="hello"),
            SimpleExample(id="ex2", value=20, text="world"),
            SimpleExample(id="ex3", value=30, text="test"),
        ]

        with datamodule.train():
            for ex in examples:
                datamodule(ex)

        # Generate instances
        from formed.common.dataset import Dataset

        result = generate_instances(datamodule, examples)

        # Result should be a Dataset
        assert isinstance(result, Dataset)
        assert len(result) == 3

        # Check instances
        assert result[0].id == "ex1"
        assert result[0].value == 10
        assert result[1].id == "ex2"
        assert result[1].value == 20
        assert result[2].id == "ex3"
        assert result[2].value == 30

    def test_generate_instances_empty_dataset(self) -> None:
        """Test generating instances from empty dataset"""
        datamodule = SimpleDataModule(
            id=MetadataTransform(),
            value=ScalarTransform(),
            text=MetadataTransform(),
        )

        from formed.common.dataset import Dataset

        result = generate_instances(datamodule, [])

        assert isinstance(result, Dataset)
        assert len(result) == 0

    def test_generate_instances_large_dataset(self) -> None:
        """Test generating instances from large dataset"""
        datamodule = SimpleDataModule(
            id=MetadataTransform(),
            value=ScalarTransform(),
            text=MetadataTransform(),
        )

        # Large dataset
        examples = [SimpleExample(id=f"ex{i}", value=i * 10, text=f"text{i}") for i in range(1000)]

        with datamodule.train():
            for ex in examples[:10]:  # Train on subset
                datamodule(ex)

        from formed.common.dataset import Dataset

        result = generate_instances(datamodule, examples)

        assert isinstance(result, Dataset)
        assert len(result) == 1000

        # Spot check some instances
        assert result[0].id == "ex0"
        assert result[500].id == "ex500"
        assert result[999].id == "ex999"
