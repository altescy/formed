from typing import Any

from formed.workflow import WorkflowGraph, step


def test_get_subgraph_preserves_filtered_json_config() -> None:
    @step(name="test_graph_value", exist_ok=True)
    def value(value: int) -> int:
        return value

    @step(name="test_graph_add", exist_ok=True)
    def add(left: int, right: int) -> int:
        return left + right

    config: dict[str, Any] = {
        "steps": {
            "dependency": {"type": "test_graph_value", "value": 1},
            "target": {
                "type": "test_graph_add",
                "left": {"type": "ref", "ref": "dependency"},
                "right": 2,
            },
            "unrelated": {"type": "test_graph_value", "value": 3},
        }
    }

    graph = WorkflowGraph.from_json(config)
    subgraph = graph.get_subgraph("target")

    assert list(step_info.name for step_info in subgraph) == ["dependency", "target"]
    assert subgraph.json() == {
        "steps": {
            "dependency": config["steps"]["dependency"],
            "target": config["steps"]["target"],
        }
    }
    assert graph.json() == config


def test_get_subgraph_without_json_config_remains_supported() -> None:
    @step(name="test_graph_programmatic", exist_ok=True)
    def value(value: int) -> int:
        return value

    graph = WorkflowGraph.from_config(
        {"steps": {"target": {"type": "test_graph_programmatic", "value": 1}}}  # type: ignore[typeddict-item]
    )

    subgraph = graph.get_subgraph("target")

    assert [step_info.name for step_info in subgraph] == ["target"]
