"""Archive data structures for workflow execution persistence.

This module defines NamedTuple structures for serializing workflow executions
to JSON format. These archives capture all metadata needed to restore past
executions, including step fingerprints, source code hashes, and dependency
information.
"""

from typing import Literal, NamedTuple, cast

from typing_extensions import Self

from formed.types import JsonValue


class WorkflowStepArchive(NamedTuple):
    """Archived snapshot of a WorkflowStep's execution-time metadata.

    This structure captures all information needed to:

    1. Look up cached results (`fingerprint`, `format_identifier`)
    2. Understand what ran (`version`, `source_hash`, `config`)
    3. Reconstruct dependency references (`dependency_fingerprints`)

    All steps are stored flat in `WorkflowGraphArchive.steps`, and dependencies
    are referenced by fingerprint rather than nested recursively.
    """

    # Identity
    name: str
    step_type: str  # The registered type name (e.g., "load_data", "torch::train")

    # Cache lookup keys
    fingerprint: str
    format_identifier: str

    # Execution metadata
    version: str  # Explicit version or hash of normalized source
    source_hash: str  # Hash of the normalized source code
    config: dict[str, JsonValue]  # The actual config used (with refs NOT resolved - keep original)

    # Properties
    deterministic: bool
    cacheable: bool | None
    should_be_cached: bool  # Pre-computed for convenience

    # Dependencies: Store fingerprints, param paths, and optional fieldrefs
    # The actual WorkflowStepInfo objects are stored separately in WorkflowGraphArchive
    # Format: param_path -> {"fingerprint": str, "fieldref": str | None}
    dependency_fingerprints: dict[str, dict[str, JsonValue]]  # param_path -> {fingerprint, fieldref}

    # Field reference (for WorkflowStepRef behavior)
    fieldref: str | None = None  # Optional field reference like "model.encoder"

    def json(self) -> dict[str, JsonValue]:
        """Convert to JSON-serializable dict."""
        return self._asdict()

    @classmethod
    def from_json(cls, data: JsonValue) -> Self:
        """Create from JSON-deserialized dict."""
        assert isinstance(data, dict)
        return cls(**cast(dict, data))


class WorkflowGraphArchive(NamedTuple):
    """Archived snapshot of a WorkflowGraph's execution-time state.

    All steps are stored flat here (not nested). Dependencies between steps
    are represented by fingerprints in `WorkflowStepArchive.dependency_fingerprints`.
    """

    # Map from step name to its archive
    # All steps are stored flat here, not nested
    steps: dict[str, WorkflowStepArchive]

    # Topologically sorted step names (for correct dependency resolution)
    execution_order: list[str]

    def json(self) -> dict[str, JsonValue]:
        """Convert to JSON-serializable dict."""
        return {
            "steps": {name: step.json() for name, step in self.steps.items()},
            "execution_order": cast(list, self.execution_order),
        }

    @classmethod
    def from_json(cls, data: dict[str, JsonValue]) -> Self:
        """Create from JSON-deserialized dict."""
        steps = {
            name: WorkflowStepArchive.from_json(step_data) for name, step_data in cast(dict, data["steps"]).items()
        }
        execution_order = data["execution_order"]
        return cls(steps=steps, execution_order=cast(list, execution_order))


class WorkflowExecutionArchive(NamedTuple):
    """Complete execution snapshot saved to execution.json.

    This is the top-level structure that organizers save and restore.
    State is NOT included here - it's saved separately as it's mutable.
    """

    # Format versioning
    format_version: Literal["2.0"]

    # Execution identity
    id: str

    # The archived graph
    graph: WorkflowGraphArchive

    # Metadata (git info, packages, environment, etc.)
    metadata: dict[str, JsonValue]

    def json(self) -> dict[str, JsonValue]:
        """Convert to JSON-serializable dict."""
        return {
            "format_version": self.format_version,
            "id": self.id,
            "graph": self.graph.json(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, data: dict[str, JsonValue]) -> Self:
        """Create from JSON-deserialized dict."""
        graph = WorkflowGraphArchive.from_json(cast(dict, data["graph"]))
        return cls(
            format_version=cast(Literal["2.0"], data["format_version"]),
            id=cast(str, data["id"]),
            graph=graph,
            metadata=cast(dict, data["metadata"]),
        )
