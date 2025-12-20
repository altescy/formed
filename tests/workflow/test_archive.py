"""Tests for WorkflowGraph archive functionality."""

import pytest

from formed.types import IJsonCompatible
from formed.workflow.archive import WorkflowGraphArchive, WorkflowStepArchive
from formed.workflow.graph import WorkflowGraph
from formed.workflow.step import WorkflowStepInfo, step


class TestWorkflowStepArchive:
    @staticmethod
    def test_workflow_step_archive_json_compatible():
        """Test that WorkflowStepArchive is IJsonCompatible."""
        assert issubclass(WorkflowStepArchive, IJsonCompatible)

    @staticmethod
    def test_live_step_info_is_live():
        """Test that a live step info correctly reports is_live()."""

        @step("test_step_archive::test_step_live")
        def test_step_live(value: int) -> int:
            return value

        # Use WorkflowGraph.from_config to create a proper live step
        from formed.workflow.graph import WorkflowGraph

        graph = WorkflowGraph.from_config(
            {"steps": {"test": {"type": "test_step_archive::test_step_live", "value": 42}}}
        )
        step_info = graph["test"]

        assert step_info.is_live()
        assert not step_info.is_archived()

    @staticmethod
    def test_archived_step_info_is_archived():
        """Test that an archived step info correctly reports is_archived()."""
        archive = WorkflowStepArchive(
            name="test",
            step_type="test_step",
            fingerprint="abc123",
            format_identifier="pickle",
            version="1.0",
            source_hash="xyz789",
            config={"value": 42},
            deterministic=True,
            cacheable=None,
            should_be_cached=True,
            dependency_fingerprints={},
        )

        step_info = WorkflowStepInfo(
            name="test",
            step=archive,
            dependencies=frozenset(),
        )

        assert step_info.is_archived()
        assert not step_info.is_live()

    @staticmethod
    def test_archived_step_cannot_access_step_class():
        """Test that archived steps cannot access step_class."""
        archive = WorkflowStepArchive(
            name="test",
            step_type="test_step",
            fingerprint="abc123",
            format_identifier="pickle",
            version="1.0",
            source_hash="xyz789",
            config={"value": 42},
            deterministic=True,
            cacheable=None,
            should_be_cached=True,
            dependency_fingerprints={},
        )

        step_info = WorkflowStepInfo(
            name="test",
            step=archive,
            dependencies=frozenset(),
        )

        with pytest.raises(TypeError, match="is archived"):
            _ = step_info.step_class

    @staticmethod
    def test_live_step_cannot_access_archive_property():
        """Test that live steps cannot access archive property."""

        @step("test_step_archive::test_archive_prop")
        def test_archive_prop(value: int) -> int:
            return value

        # Use WorkflowGraph.from_config to create a proper live step
        from formed.workflow.graph import WorkflowGraph

        graph = WorkflowGraph.from_config(
            {"steps": {"test": {"type": "test_step_archive::test_archive_prop", "value": 42}}}
        )
        step_info = graph["test"]

        with pytest.raises(TypeError, match="is live"):
            _ = step_info.archive

    @staticmethod
    def test_archived_step_provides_metadata():
        """Test that archived steps provide metadata from archive."""
        archive = WorkflowStepArchive(
            name="test",
            step_type="test_step",
            fingerprint="abc123",
            format_identifier="pickle",
            version="1.0",
            source_hash="xyz789",
            config={"value": 42},
            deterministic=True,
            cacheable=None,
            should_be_cached=True,
            dependency_fingerprints={},
        )

        step_info = WorkflowStepInfo(
            name="test",
            step=archive,
            dependencies=frozenset(),
        )

        assert step_info.fingerprint == "abc123"
        assert step_info.version == "1.0"
        assert step_info.deterministic is True
        assert step_info.cacheable is None
        assert step_info.should_be_cached is True

    @staticmethod
    def test_from_archive_without_dependencies():
        """Test creating WorkflowStepInfo from archive without dependencies."""
        archive = WorkflowStepArchive(
            name="test",
            step_type="test_step",
            fingerprint="abc123",
            format_identifier="pickle",
            version="1.0",
            source_hash="xyz789",
            config={"value": 42},
            deterministic=True,
            cacheable=None,
            should_be_cached=True,
            dependency_fingerprints={},
        )

        step_info = WorkflowStepInfo.from_archive(archive, {})

        assert step_info.name == "test"
        assert step_info.fingerprint == "abc123"
        assert len(step_info.dependencies) == 0

    @staticmethod
    def test_from_archive_with_dependencies():
        """Test creating WorkflowStepInfo from archive with dependencies."""
        # Create dependency archive
        dep_archive = WorkflowStepArchive(
            name="dep",
            step_type="dep_step",
            fingerprint="dep123",
            format_identifier="pickle",
            version="1.0",
            source_hash="xyz789",
            config={},
            deterministic=True,
            cacheable=None,
            should_be_cached=True,
            dependency_fingerprints={},
        )

        dep_step_info = WorkflowStepInfo.from_archive(dep_archive, {})

        # Create main archive with dependency
        archive = WorkflowStepArchive(
            name="test",
            step_type="test_step",
            fingerprint="abc123",
            format_identifier="pickle",
            version="1.0",
            source_hash="xyz789",
            config={"dep": {"type": "ref", "ref": "dep"}},
            deterministic=True,
            cacheable=None,
            should_be_cached=True,
            dependency_fingerprints={"dep": {"fingerprint": "dep123", "fieldref": None}},
        )

        fingerprint_map = {"dep123": dep_step_info}
        step_info = WorkflowStepInfo.from_archive(archive, fingerprint_map)

        assert step_info.name == "test"
        assert step_info.fingerprint == "abc123"
        assert len(step_info.dependencies) == 1

        # Check dependency
        dep_path, dep_info = next(iter(step_info.dependencies))
        assert dep_path == ("dep",)
        assert dep_info.fingerprint == "dep123"
        assert dep_info.name == "dep"

    @staticmethod
    def test_from_archive_with_fieldref():
        """Test that fieldref is preserved in archive."""
        archive = WorkflowStepArchive(
            name="test",
            step_type="test_step",
            fingerprint="abc123",
            format_identifier="pickle",
            version="1.0",
            source_hash="xyz789",
            config={},
            deterministic=True,
            cacheable=None,
            should_be_cached=True,
            dependency_fingerprints={},
            fieldref="encoder",
        )

        step_info = WorkflowStepInfo.from_archive(archive, {})

        assert step_info.fieldref == "encoder"


class TestWorkflowGraphArchive:
    @staticmethod
    def test_to_archive_simple_graph():
        """Test converting a simple graph to archive."""

        @step("test_graph_archive::simple_step")
        def simple_step(value: int) -> int:
            return value

        # Create a simple graph with two steps
        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "step1": {"type": "test_graph_archive::simple_step", "value": 42},
                    "step2": {"type": "test_graph_archive::simple_step", "value": 100},
                }
            }
        )

        archive = graph.to_archive()

        assert "step1" in archive.steps
        assert "step2" in archive.steps
        assert archive.steps["step1"].name == "step1"
        assert archive.steps["step2"].name == "step2"
        assert archive.execution_order == ["step1", "step2"]

    @staticmethod
    def test_from_archive_simple_graph():
        """Test reconstructing a simple graph from archive."""

        @step("test_graph_archive::simple_step2")
        def simple_step2(value: int) -> int:
            return value

        # Create and archive a graph
        original_graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "step1": {"type": "test_graph_archive::simple_step2", "value": 42},
                    "step2": {"type": "test_graph_archive::simple_step2", "value": 100},
                }
            }
        )
        archive = original_graph.to_archive()

        # Reconstruct from archive
        restored_graph = WorkflowGraph.from_archive(archive)

        assert len(list(restored_graph)) == 2
        step1_info = restored_graph["step1"]
        step2_info = restored_graph["step2"]

        assert step1_info.name == "step1"
        assert step1_info.is_archived()
        assert step2_info.name == "step2"
        assert step2_info.is_archived()

    @staticmethod
    def test_from_archive_with_dependencies():
        """Test reconstructing a graph with dependencies from archive."""

        @step("test_graph_archive::with_deps")
        def with_deps(value: int) -> int:
            return value

        # Create a graph with dependencies
        original_graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "step1": {"type": "test_graph_archive::with_deps", "value": 42},
                    "step2": {"type": "test_graph_archive::with_deps", "value": {"type": "ref", "ref": "step1"}},
                }
            }
        )
        archive = original_graph.to_archive()

        # Reconstruct from archive
        restored_graph = WorkflowGraph.from_archive(archive)

        step1_info = restored_graph["step1"]
        step2_info = restored_graph["step2"]

        # step2 should have step1 as a dependency
        assert len(step2_info.dependencies) == 1
        dep_path, dep_info = next(iter(step2_info.dependencies))
        assert dep_info.name == "step1"
        assert dep_info.fingerprint == step1_info.fingerprint

    @staticmethod
    def test_to_archive_preserves_fingerprints():
        """Test that to_archive preserves fingerprints correctly."""

        @step("test_graph_archive::preserves_fp")
        def preserves_fp(value: int) -> int:
            return value

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "step1": {"type": "test_graph_archive::preserves_fp", "value": 42},
                }
            }
        )

        # Get fingerprint from live graph
        live_fingerprint = graph["step1"].fingerprint

        # Archive and check fingerprint is preserved
        archive = graph.to_archive()
        assert archive.steps["step1"].fingerprint == live_fingerprint

        # Restore and check fingerprint matches
        restored_graph = WorkflowGraph.from_archive(archive)
        assert restored_graph["step1"].fingerprint == live_fingerprint

    @staticmethod
    def test_to_archive_raises_on_archived_graph():
        """Test that to_archive raises TypeError on archived graph."""
        # Create an already-archived graph
        archive = WorkflowGraphArchive(
            steps={
                "step1": WorkflowStepArchive(
                    name="step1",
                    step_type="test_step",
                    fingerprint="abc123",
                    format_identifier="pickle",
                    version="1.0",
                    source_hash="xyz789",
                    config={"value": 42},
                    deterministic=True,
                    cacheable=None,
                    should_be_cached=True,
                    dependency_fingerprints={},
                )
            },
            execution_order=["step1"],
        )

        graph = WorkflowGraph.from_archive(archive)

        with pytest.raises(TypeError, match="Only live graphs"):
            graph.to_archive()

    @staticmethod
    def test_roundtrip_archive():
        """Test that archive -> restore -> archive produces identical result."""

        @step("test_graph_archive::roundtrip")
        def roundtrip(value: int) -> int:
            return value

        original_graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "step1": {"type": "test_graph_archive::roundtrip", "value": 42},
                    "step2": {"type": "test_graph_archive::roundtrip", "value": {"type": "ref", "ref": "step1"}},
                }
            }
        )

        # First archive
        archive1 = original_graph.to_archive()

        # Restore
        restored_graph = WorkflowGraph.from_archive(archive1)

        # Can't archive again because restored graph is archived
        with pytest.raises(TypeError, match="Only live graphs"):
            restored_graph.to_archive()

    @staticmethod
    def test_from_archive_with_fieldref():
        """Test that fieldref is preserved through archive."""

        @step("test_graph_archive::fieldref_step")
        def fieldref_step(value: int) -> dict:
            return {"encoder": value}

        original_graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "step1": {"type": "test_graph_archive::fieldref_step", "value": 42},
                    "step2": {
                        "type": "test_graph_archive::fieldref_step",
                        "value": {"type": "ref", "ref": "step1.encoder"},
                    },
                }
            }
        )

        archive = original_graph.to_archive()

        # Reconstruct
        restored_graph = WorkflowGraph.from_archive(archive)

        step2_info = restored_graph["step2"]
        # The fieldref should be stored on the dependency WorkflowStepInfo
        # In the graph construction, when we reference "step1.encoder",
        # a new WorkflowStepInfo is created with fieldref="encoder"
        assert len(step2_info.dependencies) == 1
        dep_path, dep_info = next(iter(step2_info.dependencies))
        assert dep_info.fieldref == "encoder"
        assert dep_info.name == "step1"
