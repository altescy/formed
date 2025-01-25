import sys
from collections.abc import Callable, Hashable
from typing import Generic, TextIO, TypeVar

T_Node = TypeVar("T_Node", bound=Hashable)


class DAG(Generic[T_Node]):
    def __init__(self, dependencies: dict[T_Node, set[T_Node]]) -> None:
        nodes = set(dependencies.keys() | set(node for deps in dependencies.values() for node in deps))
        empty_dependencies: dict[T_Node, set[T_Node]] = {node: set() for node in nodes}

        self._dependencies = {**empty_dependencies, **dependencies}

    def __len__(self) -> int:
        return len(self._dependencies)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DAG):
            return NotImplemented
        return self._dependencies == other._dependencies

    def __hash__(self) -> int:
        return hash(frozenset((node, frozenset(deps)) for node, deps in self._dependencies.items()))

    @property
    def nodes(self) -> set[T_Node]:
        return set(self._dependencies.keys())

    def subgraph(self, nodes: set[T_Node]) -> "DAG":
        return DAG({node: nodes & deps for node, deps in self._dependencies.items() if node in nodes})

    def in_degree(self, node: T_Node) -> int:
        return len(self._dependencies[node])

    def successors(self, node: T_Node) -> set[T_Node]:
        return {successor for successor, deps in self._dependencies.items() if node in deps}

    def weekly_connected_components(self) -> set["DAG"]:
        groups: list[set[T_Node]] = []

        for node, deps in self._dependencies.items():
            current_group = {node} | deps
            for group in groups:
                if group & current_group:
                    group.update(current_group)
                    break
            else:
                groups.append(current_group)

        for i, group in enumerate(groups):
            for other_group in groups[i + 1 :]:
                if group & other_group:
                    group.update(other_group)
                    other_group.clear()

        groups = [group for group in groups if group]

        return {DAG({node: self._dependencies[node] for node in group}) for group in groups}

    def visualize(
        self,
        *,
        indent: int = 2,
        output: TextIO = sys.stdout,
        rename: Callable[[T_Node], str] = str,
    ) -> None:
        locations: dict[T_Node, tuple[int, int]] = {}

        def _process_dag(dag: DAG, level: int) -> None:
            for subdag in sorted(dag.weekly_connected_components(), key=lambda g: sorted(g.nodes)):
                _process_component(subdag, level)

        def _process_component(dag: DAG, level: int) -> None:
            sources = set(node for node in dag.nodes if dag.in_degree(node) == 0)
            for i, node in enumerate(sorted(sources)):
                if node in locations:
                    raise ValueError(f"Node {node} already placed")
                locations[node] = (len(locations), level + i)

            subdag = dag.subgraph(dag.nodes - sources)
            _process_dag(subdag, level + len(sources))

        _process_dag(self, 0)

        a = [[" "] * level * indent for _, level in sorted(locations.values())]
        for node, (position, level) in sorted(locations.items(), key=lambda x: x[1][0]):
            successors = sorted(self.successors(node), key=lambda v: locations[v][0])
            if not successors:
                continue

            last_successor_position, last_successor_level = locations[successors[-1]]
            for successor_position in range(position + 1, last_successor_position):
                a[successor_position][level * indent] = "│"
            for successor in successors[:-1]:
                successor_position, successor_level = locations[successor]
                a[successor_position][level * indent] = (
                    "┼" if level > 0 and a[successor_position][level * indent - 1] == "─" else "├"
                )
                for offset in range(level * indent + 1, successor_level * indent):
                    a[successor_position][offset] = "─"
            a[last_successor_position][level * indent] = (
                "┴" if level > 0 and a[last_successor_position][level * indent - 1] == "─" else "╰"
            )
            for offset in range(level * indent + 1, last_successor_level * indent):
                a[last_successor_position][offset] = "─"

        output.writelines(
            "".join(a[position]) + "• " + rename(node) + "\n"
            for node, (position, _) in sorted(locations.items(), key=lambda x: x[1][0])
        )
