"""Workflow graph construction and dependency resolution.

This module provides the WorkflowGraph class which parses workflow configurations
and builds directed acyclic graphs (DAGs) of workflow steps with dependency tracking.

Key Features:
    - Parse Jsonnet workflow configurations
    - Automatic dependency detection via references
    - Topological sorting for execution order
    - Cycle detection in dependencies
    - DAG-based workflow representation

Example:
    >>> from formed.workflow import WorkflowGraph
    >>>
    >>> # Load workflow from Jsonnet config
    >>> graph = WorkflowGraph.from_jsonnet("workflow.jsonnet")
    >>>
    >>> # Access steps in topological order
    >>> for step_info in graph:
    ...     print(f"Step: {step_info.name}")
    ...     print(f"Dependencies: {step_info.dependencies}")
    >>>
    >>> # Get specific step
    >>> preprocess_step = graph["preprocess"]
    >>> print(preprocess_step.fingerprint)

"""

import sys
from collections.abc import Iterator, Mapping
from typing import Any, Optional, TextIO, TypedDict

from colt import ConfigurationError, Lazy

from formed.common.dag import DAG
from formed.common.jsonnet import FromJsonnet
from formed.types import JsonValue

from .archive import WorkflowGraphArchive, WorkflowStepArchive
from .colt import COLT_BUILDER, WorkflowRef
from .constants import WORKFLOW_REFKEY
from .step import WorkflowStep, WorkflowStepInfo
from .types import StrictParamPath


class WorkflowGraphConfig(TypedDict):
    steps: dict[str, JsonValue]


class WorkflowGraph(FromJsonnet):
    __COLT_BUILDER__ = COLT_BUILDER

    @classmethod
    def _build_step_info(
        cls,
        steps: Mapping[str, Lazy[WorkflowStep]],
    ) -> Mapping[str, WorkflowStepInfo]:
        if not steps:
            return {}

        builder = next(iter(steps.values()))._builder

        def find_dependencies(obj: Any, path: tuple[str, ...]) -> frozenset[tuple[StrictParamPath, str, Optional[str]]]:
            refs: set[tuple[StrictParamPath, str, Optional[str]]] = set()
            if WorkflowRef.is_ref(builder, obj):
                step_name, field_name = WorkflowRef._parse_ref(str(obj[WORKFLOW_REFKEY]))
                refs |= {(path, step_name, field_name)}
            if isinstance(obj, WorkflowRef):
                refs |= {(path, obj.step_name, obj.field_name)}
            if isinstance(obj, Mapping):
                for key, value in obj.items():
                    refs |= find_dependencies(value, path + (key,))
            if isinstance(obj, (list, tuple)):
                for i, value in enumerate(obj):
                    refs |= find_dependencies(value, path + (str(i),))
            return frozenset(refs)

        dependencies = {name: find_dependencies(lazy_step.config, ()) for name, lazy_step in steps.items()}

        stack: set[str] = set()
        visited: set[str] = set()
        sorted_step_names: list[str] = []

        def topological_sort(name: str) -> None:
            if name in stack:
                raise ConfigurationError(f"Cycle detected in workflow dependencies: {name} -> {stack}")
            if name in visited:
                return
            stack.add(name)
            visited.add(name)
            for _, dep_name, _ in dependencies[name]:
                topological_sort(dep_name)
            stack.remove(name)
            sorted_step_names.append(name)

        def make_dependency_step(
            path: StrictParamPath,
            step_info: WorkflowStepInfo,
            field_name: Optional[str],
        ) -> tuple[StrictParamPath, WorkflowStepInfo]:
            if field_name:
                # Create a new WorkflowStepInfo with fieldref set
                return (
                    path,
                    WorkflowStepInfo(
                        name=step_info.name,
                        step=step_info.step,
                        dependencies=step_info.dependencies,
                        fieldref=field_name,
                    ),
                )
            return (path, step_info)

        for name in steps.keys():
            topological_sort(name)

        step_name_to_info: dict[str, WorkflowStepInfo] = {}
        for name in sorted_step_names:
            step = steps[name]
            step_dependencies = frozenset(
                make_dependency_step(path, step_name_to_info[dep_name], field_name)
                for path, dep_name, field_name in dependencies[name]
            )
            step_name_to_info[name] = WorkflowStepInfo(name, step, step_dependencies)

        return step_name_to_info

    def __init__(
        self,
        steps: Mapping[str, Lazy[WorkflowStep]],
    ) -> None:
        self._step_info = self._build_step_info(steps)

    def __iter__(self) -> Iterator[WorkflowStepInfo]:
        return iter(self._step_info.values())

    def __getitem__(self, step_name: str) -> WorkflowStepInfo:
        return self._step_info[step_name]

    def get_subgraph(self, step_name: str) -> "WorkflowGraph":
        """Get a subgraph containing a step and all its dependencies.

        Only works with live (non-archived) graphs. For archived graphs,
        dependencies are already resolved in the archive.
        """
        if step_name not in self._step_info:
            raise ValueError(f"Step {step_name} not found in the graph")
        step_info = self._step_info[step_name]

        # Type narrowing: ensure all steps are live
        if not isinstance(step_info.step, Lazy):
            raise TypeError(
                f"Cannot create subgraph from archived step '{step_name}'. "
                f"Subgraph extraction only works with live workflows."
            )

        subgraph_steps: dict[str, Lazy[WorkflowStep]] = {step_name: step_info.step}
        for _, dependant_step_info in step_info.dependencies:
            if not isinstance(dependant_step_info.step, Lazy):
                raise TypeError(f"Cannot create subgraph: dependency '{dependant_step_info.name}' is archived.")
            for sub_step_info in self.get_subgraph(dependant_step_info.name):
                if not isinstance(sub_step_info.step, Lazy):
                    raise TypeError(f"Cannot create subgraph: nested dependency '{sub_step_info.name}' is archived.")
                subgraph_steps[sub_step_info.name] = sub_step_info.step
        return WorkflowGraph(subgraph_steps)

    def visualize(
        self,
        *,
        output: TextIO = sys.stdout,
        additional_info: Mapping[str, str] = {},
    ) -> None:
        def get_node(name: str) -> str:
            if name in additional_info:
                return f"{name}: {additional_info[name]}"
            return name

        dag = DAG(
            {
                get_node(name): {get_node(dep.name) for _, dep in info.dependencies}
                for name, info in self._step_info.items()
            }
        )

        dag.visualize(output=output)

    @classmethod
    def from_config(cls, config: WorkflowGraphConfig) -> "WorkflowGraph":
        return cls.__COLT_BUILDER__(config, WorkflowGraph)

    def to_archive(self) -> WorkflowGraphArchive:
        """Convert graph to archive format for serialization.

        This captures the execution-time state of all steps in a flat structure.
        Only works with live graphs.
        """
        # Ensure all steps are live before archiving
        for step_info in self:
            if not isinstance(step_info.step, Lazy):
                raise TypeError(
                    f"Cannot archive graph containing archived step '{step_info.name}'. "
                    f"Only live graphs can be converted to archives."
                )

        # Convert all steps to archives
        steps: dict[str, WorkflowStepArchive] = {}
        for step_info in self:
            steps[step_info.name] = step_info.to_archive()

        # Compute execution order (topological sort)
        execution_order = [step_info.name for step_info in self]

        return WorkflowGraphArchive(
            steps=steps,
            execution_order=execution_order,
        )

    @classmethod
    def from_archive(cls, archive: WorkflowGraphArchive) -> "WorkflowGraph":
        """Reconstruct graph from archive.

        Handles all dependency resolution internally using a two-pass approach.
        Organizers don't need to know about the complexity.
        """
        # First pass: Create all step infos without dependencies
        # This builds a fingerprint -> WorkflowStepInfo map
        fingerprint_to_info: dict[str, WorkflowStepInfo] = {}

        for step_name in archive.execution_order:
            step_archive = archive.steps[step_name]
            # Create WorkflowStepInfo with archive but no dependencies yet
            step_info = WorkflowStepInfo(
                name=step_archive.name,
                step=step_archive,
                dependencies=frozenset(),
                fieldref=step_archive.fieldref,
            )
            fingerprint_to_info[step_archive.fingerprint] = step_info

        # Second pass: Resolve dependencies using from_archive
        step_name_to_info: dict[str, WorkflowStepInfo] = {}
        for step_name in archive.execution_order:
            step_archive = archive.steps[step_name]
            step_info = WorkflowStepInfo.from_archive(step_archive, fingerprint_to_info)
            step_name_to_info[step_name] = step_info

        # Create graph directly by setting _step_info
        graph = cls.__new__(cls)
        graph._step_info = step_name_to_info
        return graph
