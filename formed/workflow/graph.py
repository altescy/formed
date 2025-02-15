import sys
from collections.abc import Iterator, Mapping
from typing import Any, TextIO, TypedDict

from colt import ConfigurationError, Lazy

from formed.common.dag import DAG
from formed.common.jsonnet import FromJsonnet
from formed.types import JsonValue

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

        def find_dependencies(obj: Any, path: tuple[str, ...]) -> frozenset[tuple[StrictParamPath, str]]:
            refs: set[tuple[StrictParamPath, str]] = set()
            if WorkflowRef.is_ref(builder, obj):
                step_name = str(obj[WORKFLOW_REFKEY])
                refs |= {(path, step_name)}
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
            for _, dep_name in dependencies[name]:
                topological_sort(dep_name)
            stack.remove(name)
            sorted_step_names.append(name)

        for name in steps.keys():
            topological_sort(name)

        step_name_to_info: dict[str, WorkflowStepInfo] = {}
        for name in sorted_step_names:
            step = steps[name]
            step_dependencies = frozenset((path, step_name_to_info[dep_name]) for path, dep_name in dependencies[name])
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
    def from_config(self, config: WorkflowGraphConfig) -> "WorkflowGraph":
        return self.__COLT_BUILDER__(config, WorkflowGraph)
