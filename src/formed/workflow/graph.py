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

from .colt import COLT_BUILDER, WorkflowRef
from .constants import WORKFLOW_REFKEY
from .step import WorkflowStep, WorkflowStepInfo, WorkflowStepRef
from .types import StrictParamPath


class WorkflowGraphConfig(TypedDict):
    """Type definition for workflow configuration.

    Attributes:
        steps: Dictionary mapping step names to their configurations.

    """

    steps: dict[str, JsonValue]


class WorkflowGraph(FromJsonnet):
    """Directed acyclic graph of workflow steps with dependency tracking.

    WorkflowGraph parses workflow configurations (typically from Jsonnet),
    resolves step dependencies, and provides topological ordering for execution.
    It detects cycles, validates configurations, and builds a DAG representation.

    Attributes:
        _steps: Mapping of step names to WorkflowStepInfo metadata.
        _dag: Directed acyclic graph representation of step dependencies.

    Example:
        >>> # Define workflow configuration as a dictionary
        >>> config = {
        ...     "steps": {
        ...         "load_data": {
        ...             "type": "load_csv",
        ...             "path": "data.csv"
        ...         },
        ...         "preprocess": {
        ...             "type": "preprocess",
        ...             "data": {"type": "ref", "ref": "load_data"}
        ...         },
        ...         "train": {
        ...             "type": "train_model",
        ...             "data": {"type": "ref", "ref": "preprocess"}
        ...         }
        ...     }
        ... }
        >>>
        >>> # Build graph from config dict
        >>> graph = WorkflowGraph.from_config(config)
        >>>
        >>> # Or load from Jsonnet file
        >>> graph = WorkflowGraph.from_jsonnet("workflow.jsonnet")
        >>>
        >>> # Iterate in topological order (load_data -> preprocess -> train)
        >>> for step_info in graph:
        ...     print(step_info.name)
        >>>
        >>> # Access specific steps
        >>> train_step = graph["train"]
        >>> print(train_step.dependencies)  # Shows dependency on preprocess
        >>>
        >>> # Visualize graph
        >>> graph.visualize(sys.stdout)

    Note:
        - Steps reference each other using {type: "ref", ref: "step_name"}
        - Field-level refs: {type: "ref", ref: "step_name.field"}
        - Dependencies are automatically detected from configuration
        - Cycles in dependencies raise ConfigurationError
        - Topological order ensures dependencies execute before dependents

    """

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
                return (
                    path,
                    WorkflowStepRef(
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
        if step_name not in self._step_info:
            raise ValueError(f"Step {step_name} not found in the graph")
        step_info = self._step_info[step_name]
        subgraph_steps: dict[str, Lazy[WorkflowStep]] = {step_name: step_info.step}
        for _, dependant_step_info in step_info.dependencies:
            for sub_step_info in self.get_subgraph(dependant_step_info.name):
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

    def to_dict(self) -> dict[str, Any]:
        return {"steps": {step_info.name: step_info.step.config for step_info in self}}

    @classmethod
    def from_config(cls, config: WorkflowGraphConfig) -> "WorkflowGraph":
        return cls.__COLT_BUILDER__(config, WorkflowGraph)
