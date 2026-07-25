from io import StringIO
from textwrap import dedent

import pytest

from formed.common.dag import DAG


class TestDagutils:
    @staticmethod
    @pytest.mark.parametrize(
        "dag, node, expected",
        [
            (DAG({"a": {"b"}, "b": {"c"}, "c": set()}), "a", 1),
            (DAG({"a": {"b"}, "b": {"c"}, "c": set()}), "b", 1),
            (DAG({"a": {"b"}, "b": {"c"}, "c": set()}), "c", 0),
        ],
    )
    def test_in_degree(dag: DAG, node: str, expected: int) -> None:
        assert dag.in_degree(node) == expected

    @staticmethod
    @pytest.mark.parametrize(
        "dag, nodes, expected",
        [
            (
                DAG({"a": {"b"}, "b": {"c"}, "c": set()}),
                {"a", "b"},
                DAG({"a": {"b"}, "b": set()}),
            ),
            (
                DAG({"a": {"b"}, "b": {"c"}, "c": set()}),
                {"b", "c"},
                DAG({"b": {"c"}, "c": set()}),
            ),
        ],
    )
    def test_subgraph(dag: DAG, nodes: set[str], expected: DAG) -> None:
        assert dag.subgraph(nodes) == expected

    @staticmethod
    @pytest.mark.parametrize(
        "dag, node, expected",
        [
            (DAG({"a": {"b"}, "b": {"c"}, "c": set()}), "a", set()),
            (DAG({"a": {"b"}, "b": {"c"}, "c": set()}), "b", {"a"}),
            (DAG({"a": {"b"}, "b": {"c"}, "c": set()}), "c", {"b"}),
        ],
    )
    def test_successors(dag: DAG, node: str, expected: set[str]) -> None:
        assert dag.successors(node) == expected

    @staticmethod
    @pytest.mark.parametrize(
        "dag, expected",
        [
            (
                DAG[str]({"a": set(), "b": set()}),
                {DAG[str]({"a": set()}), DAG[str]({"b": set()})},
            ),
            (
                DAG({"a": set(), "b": {"a"}, "c": {"a"}}),
                {DAG({"a": set(), "b": {"a"}, "c": {"a"}})},
            ),
        ],
    )
    def test_weekly_connected_components(dag: DAG, expected: set[DAG]) -> None:
        assert dag.weekly_connected_components() == expected

    @staticmethod
    def test_weekly_connected_components_no_overlap() -> None:
        # Regression test: a graph where the old greedy grouping algorithm
        # produced overlapping components and therefore duplicated nodes.
        dag = DAG(
            {
                "a": set(),
                "b": set(),
                "c": set(),
                "d": {"c"},
                "e": {"b", "d"},
                "f": {"a", "c"},
            }
        )
        components = dag.weekly_connected_components()
        seen: set[str] = set()
        for component in components:
            assert not (seen & component.nodes)
            seen |= component.nodes
        assert seen == dag.nodes
        assert len(components) == 1

    @staticmethod
    def test_visualize_overlapping_components() -> None:
        # Regression test: visualize should not raise "Node ... already placed".
        dag = DAG(
            {
                "a": set(),
                "b": set(),
                "c": set(),
                "d": {"c"},
                "e": {"b", "d"},
                "f": {"a", "c"},
            }
        )
        output = StringIO()
        dag.visualize(output=output)
        assert "c" in output.getvalue()

    @staticmethod
    @pytest.mark.parametrize(
        "dag, expected",
        [
            (
                DAG(
                    {
                        "a": set(),
                        "b": {"a"},
                        "c": {"a", "b"},
                        "d": {"b", "c"},
                        "e": {"a", "c"},
                        "f": {"b"},
                        "x": set(),
                        "y": {"x"},
                    }
                ),
                dedent(
                    """
                    • a
                    ├─• b
                    ├─┼─• c
                    │ ├─┼─• d
                    ╰─│─┴─• e
                      ╰─• f
                    • x
                    ╰─• y
                    """
                ).lstrip(),
            ),
        ],
    )
    def test_visialize(dag: DAG, expected: str) -> None:
        output = StringIO()
        dag.visualize(output=output)
        assert output.getvalue() == expected
