from pathlib import Path

import datasets
import pytest
import transformers

from formed.integrations.datasets import workflow as datasets_workflow  # noqa: F401
from formed.integrations.transformers import (
    workflow as transformers_workflow,  # noqa: F401
)
from formed.workflow import (
    DefaultWorkflowExecutor,
    MemoryWorkflowOrganizer,
    WorkflowGraph,
)


@pytest.fixture
def dataset() -> datasets.Dataset:
    n = 10
    features = datasets.Features(  # type: ignore[no-untyped-call]
        {
            "text": datasets.Value("string"),
            "id": datasets.Value("int64"),
        }
    )
    dataset = datasets.Dataset.from_dict(
        {
            "text": [" ".join(["foo"] * 5)] * n,
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


def test_transformers_workflow(
    dataset_path: Path,
) -> None:
    graph = WorkflowGraph.from_config(
        {
            "steps": {
                "dataset": {
                    "type": "datasets::load",
                    "path": str(dataset_path),
                },
                "tokenizer": {
                    "type": "transformers::load_tokenizer",
                    "pretrained_model_name_or_path": "hf-internal-testing/tiny-random-BertModel",
                },
                "tokenized_dataset": {
                    "type": "transformers::tokenize",
                    "dataset": {"type": "ref", "ref": "dataset"},
                    "tokenizer": {"type": "ref", "ref": "tokenizer"},
                    "max_length": 10,
                    "truncation": True,
                },
                "backbone_model": {
                    "type": "transformers::load_model",
                    "auto_class": "AutoModelForMaskedLM",
                    "model_name_or_path": "hf-internal-testing/tiny-random-BertModel",
                },
                "finetuned_model": {
                    "type": "transformers::train_model",
                    "model": {"type": "ref", "ref": "backbone_model"},
                    "dataset": {"type": "ref", "ref": "tokenized_dataset"},
                    "args": {
                        "per_device_train_batch_size": 2,
                        "per_device_eval_batch_size": 2,
                        "num_train_epochs": 1,
                        "learning_rate": 1e-4,
                        "warmup_ratio": 0.1,
                        "report_to": "none",
                        "do_train": True,
                    },
                    "data_collator": {
                        "type": "transformers:DataCollatorForLanguageModeling",
                        "tokenizer": {"type": "ref", "ref": "tokenizer"},
                        "mlm_probability": 0.15,
                        "pad_to_multiple_of": None,
                    },
                    "processing_class": {"type": "ref", "ref": "tokenizer"},
                    "callbacks": [
                        {"type": "formed.integrations.transformers.training:MlflowTrainerCallback"},
                    ],
                },
            },
        },
    )

    executor = DefaultWorkflowExecutor()
    organizer = MemoryWorkflowOrganizer()
    context = organizer.run(executor, graph)
    finetuned_model = context.cache[graph["finetuned_model"]]
    assert finetuned_model is not None
    assert isinstance(finetuned_model, transformers.PreTrainedModel)
