from collections.abc import Mapping, Sequence

from formed.integrations.ml import DataModule, TextFieldTransform
from formed.integrations.ml.workflow import build_datamodule, split_dataset


def test_build_datamodule() -> None:
    dataset = [{"text": "hello world"}, {"text": "goodbye world"}]
    datamodule = build_datamodule(
        dataset,
        DataModule(fields={"text": TextFieldTransform()}),
    )
    assert datamodule.field("text").stats["index_size"] == 3


def test_split_dataset() -> None:
    dataset = [{"text": "foo bar"}] * 10
    splitted_datasets = split_dataset(dataset, {"train": 0.7, "val": 0.1, "test": 0.2})
    assert isinstance(splitted_datasets, Mapping)
    assert set(splitted_datasets.keys()) == {"train", "val", "test"}
    assert len(splitted_datasets["train"]) == 7
    assert len(splitted_datasets["val"]) == 1
    assert len(splitted_datasets["test"]) == 2
