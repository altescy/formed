"""Directed Acyclic Graph (DAG) utilities for dependency management and visualization.

This module provides the DAG class for representing and manipulating directed acyclic
graphs. It's used primarily for workflow dependency visualization and analysis.

Examples:
    >>> # Create a simple DAG
    >>> dag = DAG({
    ...     "train": {"preprocess", "load_data"},
    ...     "evaluate": {"train"},
    ...     "preprocess": {"load_data"},
    ...     "load_data": set()
    ... })
    >>>
    >>> # Visualize the graph
    >>> dag.visualize()
    >>> # Get successors
    >>> print(dag.successors("load_data"))  # {"preprocess", "train"}

"""

import sys
from collections.abc import Callable, Hashable
from typing import Generic, TextIO, TypeVar

NodeT = TypeVar("NodeT", bound=Hashable)


class DAG(Generic[NodeT]):
    """Directed acyclic graph with dependency tracking and visualization.

    DAG represents a directed acyclic graph where each node can have dependencies
    (edges pointing to other nodes). It provides methods for analyzing graph structure,
    computing weakly connected components, and visualizing the graph structure.

    Type Parameters:
        NodeT: Node type, must be hashable.

    Attributes:
        _dependencies: Mapping from each node to its set of dependency nodes.

    Examples:
        >>> # Define workflow dependencies
        >>> dag = DAG({
        ...     "step3": {"step1", "step2"},
        ...     "step2": {"step1"},
        ...     "step1": set()
        ... })
        >>>
        >>> # Query graph properties
        >>> print(dag.nodes)  # {"step1", "step2", "step3"}
        >>> print(dag.in_degree("step3"))  # 2
        >>> print(dag.successors("step1"))  # {"step2", "step3"}
        >>>
        >>> # Visualize dependencies
        >>> dag.visualize()
        • step1
        ├─• step2
        ╰─• step3

    Note:
        - All nodes referenced in dependencies are automatically included in the graph
        - Nodes can be any hashable type (strings, tuples, etc.)
        - The graph does not validate acyclicity on construction

    """

    def __init__(self, dependencies: dict[NodeT, set[NodeT]]) -> None:
        nodes = set(dependencies.keys() | set(node for deps in dependencies.values() for node in deps))
        empty_dependencies: dict[NodeT, set[NodeT]] = {node: set() for node in nodes}

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
    def nodes(self) -> set[NodeT]:
        return set(self._dependencies.keys())

    def subgraph(self, nodes: set[NodeT]) -> "DAG":
        """Create a subgraph containing only the specified nodes.

        Args:
            nodes: Set of nodes to include in the subgraph.

        Returns:
            A new DAG containing only the specified nodes and their dependencies
            that are also in the node set.

        """
        return DAG({node: nodes & deps for node, deps in self._dependencies.items() if node in nodes})

    def in_degree(self, node: NodeT) -> int:
        """Get the number of dependencies for a node.

        Args:
            node: The node to query.

        Returns:
            The number of nodes that this node depends on.

        """
        return len(self._dependencies[node])

    def successors(self, node: NodeT) -> set[NodeT]:
        """Get all nodes that depend on the specified node.

        Args:
            node: The node to find successors for.

        Returns:
            Set of nodes that have this node as a dependency.

        """
        return {successor for successor, deps in self._dependencies.items() if node in deps}

    def weekly_connected_components(self) -> set["DAG"]:
        """Compute weakly connected components of the graph.

        A weakly connected component is a maximal subgraph where every pair of nodes
        is connected by some path (ignoring edge direction).

        Returns:
            Set of DAG objects, each representing one weakly connected component.

        Examples:
            >>> # Graph with two disconnected components
            >>> dag = DAG({
            ...     "a": {"b"},
            ...     "b": set(),
            ...     "c": {"d"},
            ...     "d": set()
            ... })
            >>> components = dag.weekly_connected_components()
            >>> len(components)  # 2 components
            2

        """
        parent: dict[NodeT, NodeT] = {node: node for node in self._dependencies}

        def find(node: NodeT) -> NodeT:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(x: NodeT, y: NodeT) -> None:
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_x] = root_y

        for node, deps in self._dependencies.items():
            for dep in deps:
                union(node, dep)

        groups: dict[NodeT, set[NodeT]] = {}
        for node in self._dependencies:
            root = find(node)
            groups.setdefault(root, set()).add(node)

        return {self.subgraph(group) for group in groups.values()}

    def visualize(
        self,
        *,
        indent: int = 2,
        output: TextIO = sys.stdout,
        rename: Callable[[NodeT], str] = str,
    ) -> None:
        """Visualize the DAG structure as ASCII art.

        Renders the dependency graph using box-drawing characters, showing the
        relationships between nodes in a tree-like format.

        Args:
            indent: Number of spaces to indent each level. Defaults to 2.
            output: Text stream to write visualization to. Defaults to stdout.
            rename: Function to convert nodes to display strings. Defaults to str().

        Examples:
            >>> dag = DAG({
            ...     "train": {"preprocess"},
            ...     "preprocess": {"load"},
            ...     "load": set()
            ... })
            >>> dag.visualize()
            • load
            ╰─• preprocess
              ╰─• train

        """
        locations: dict[NodeT, tuple[int, int]] = {}

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
