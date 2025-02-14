from pathlib import Path

import datasets
import pytest

from formed.integrations.datasets import workflow  # noqa: F401
from formed.workflow import DefaultWorkflowExecutor, MemoryWorkflowOrganizer, WorkflowGraph, step


@pytest.fixture
def dataset() -> datasets.Dataset:
    n = 10
    features = datasets.Features(  # type: ignore[no-untyped-call]
        {
            "tokens": datasets.Sequence(datasets.Value("string")),
            "labels": datasets.Sequence(datasets.ClassLabel(names=["negative", "positive"])),
            "answers": datasets.Sequence(
                {
                    "text": datasets.Value("string"),
                    "answer_start": datasets.Value("int32"),
                }
            ),
            "id": datasets.Value("int64"),
        }
    )
    dataset = datasets.Dataset.from_dict(
        {
            "tokens": [["foo"] * 5] * n,
            "labels": [[1] * 5] * n,
            "answers": [{"answer_start": [97], "text": ["1976"]}] * 10,
            "id": list(range(n)),
        },
        features=features,
    )
    return dataset


@pytest.fixture
def dataset_path(tmp_path: Path, dataset: datasets.Dataset) -> Path:
    dataset_path = tmp_path / "dataset"
    dataset.save_to_disk(str(dataset_path))
    return dataset_path


def test_datasets_workflow(
    dataset_path: Path,
) -> None:
    graph = WorkflowGraph.from_config(
        {
            "steps": {
                "dataset": {
                    "type": "datasets::load_dataset",
                    "path": str(dataset_path),
                },
                "splitted_dataset": {
                    "type": "datasets::train_test_split",
                    "dataset": {"type": "ref", "ref": "dataset"},
                    "test_key": "validation",
                },
            },
        },
    )

    executor = DefaultWorkflowExecutor()
    organizer = MemoryWorkflowOrganizer()
    context = organizer.run(executor, graph)
    splitted_dataset = context.cache[graph["splitted_dataset"]]
    assert isinstance(splitted_dataset, dict)
    assert set(splitted_dataset.keys()) == {"train", "validation"}
    assert all(isinstance(dataset, datasets.Dataset) for dataset in splitted_dataset.values())
