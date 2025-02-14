import sys
from pathlib import Path

import datasets
import pytest
from sentence_transformers import SentenceTransformer

from formed.integrations.datasets import workflow as datasets_workflow  # noqa: F401
from formed.integrations.sentence_transformers import workflow as transformers_workflow  # noqa: F401
from formed.workflow import DefaultWorkflowExecutor, MemoryWorkflowOrganizer, WorkflowGraph


@pytest.fixture
def dataset() -> datasets.Dataset:
    n = 10
    features = datasets.Features(  # type: ignore[no-untyped-call]
        {
            "anchor": datasets.Value("string"),
            "positive": datasets.Value("string"),
        }
    )
    dataset = datasets.Dataset.from_dict(
        {
            "anchor": [" ".join(["foo"] * 5)] * n,
            "positive": [" ".join(["foo"] * 5)] * n,
        },
        features=features,
    )
    return dataset


@pytest.fixture
def dataset_path(tmp_path: Path, dataset: datasets.Dataset) -> Path:
    dataset_path = tmp_path / "dataset"
    dataset.save_to_disk(str(dataset_path))
    return dataset_path


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_sentence_transformers_workflow(
    dataset_path: Path,
) -> None:
    graph = WorkflowGraph.from_config(
        {
            "steps": {
                "dataset": {
                    "type": "datasets::load_dataset",
                    "path": str(dataset_path),
                },
                "model": {
                    "type": "sentence_transformers::train",
                    "model": {
                        "modules": [
                            {
                                "type": "sentence_transformers.models:StaticEmbedding",
                                "tokenizer": {
                                    "type": "tokenizers:Tokenizer.from_pretrained",
                                    "identifier": "hf-internal-testing/tiny-random-BertModel",
                                },
                                "embedding_dim": 16,
                            },
                        ],
                    },
                    "dataset": {"type": "ref", "ref": "dataset"},
                    "loss": {"type": "sentence_transformers.losses.MultipleNegativesRankingLoss"},
                    "loss_modifier": {
                        "type": "sentence_transformers.losses.MatryoshkaLoss",
                        "matryoshka_dims": [16, 8],
                    },
                    "args": {
                        "per_device_train_batch_size": 2,
                        "num_train_epochs": 1,
                        "learning_rate": 1e-4,
                        "warmup_ratio": 0.1,
                        "report_to": "none",
                        "do_train": True,
                    },
                },
            },
        },
    )

    executor = DefaultWorkflowExecutor()
    organizer = MemoryWorkflowOrganizer()
    context = organizer.run(executor, graph)
    model = context.cache[graph["model"]]
    assert isinstance(model, SentenceTransformer)
