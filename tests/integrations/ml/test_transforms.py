import dataclasses
import string
from collections.abc import Sequence
from typing import Any, Generic, Optional, Union

import numpy
import pytest
from typing_extensions import TypeVar

from formed.common.iterutils import batched
from formed.integrations.ml import (
    BaseTransform,
    DataModule,
    Extra,
    LabelIndexer,
    MetadataTransform,
    Tokenizer,
    TokenSequenceIndexer,
)
from formed.integrations.ml.types import (  # noqa: F401
    AnalyzedText,
    AsBatch,
    AsInstance,
    Batch,
    DataModuleModeT,
    SupportsReconstruct,
)

InputT = TypeVar("InputT", default=Any)
OutputT = TypeVar("OutputT")
TextBatchT = TypeVar("TextBatchT", bound=Batch, default=Any)
DataModuleT = TypeVar("DataModuleT", bound="DataModule", default=Any)
TextTransformT = TypeVar(
    "TextTransformT",
    bound=BaseTransform[Union[str, Sequence[str], AnalyzedText]],
    default=Any,
    covariant=True,
)


class TestTextClassificationDataModule:
    @dataclasses.dataclass
    class ClassificationExample:
        id: str
        text: Union[str, Sequence[str]]
        label: Optional[Union[int, str]] = None

    class TextClassificationDataModule(
        DataModule[
            DataModuleModeT,
            InputT,
            "TextClassificationDataModule[AsInstance, InputT, TextTransformT]",
            "TextClassificationDataModule[AsBatch, InputT, TextTransformT]",
        ],
        Generic[DataModuleModeT, InputT, TextTransformT],
    ):
        id: MetadataTransform[Any, str]
        text: TextTransformT
        label: Extra[LabelIndexer] = Extra.default()

    @pytest.fixture
    @staticmethod
    def dummy_text_classification_dataset() -> list[ClassificationExample]:
        characters = string.ascii_lowercase + " "
        dataset = []
        for i in range(50):
            num_words = 5 + (i % 5)
            words = [
                "".join(characters[(i + j) % len(characters)] for _ in range(3 + ((i + j) % 5)))
                for j in range(num_words)
            ]
            text = " ".join(words)
            label = "positive" if i % 2 == 0 else "negative"
            dataset.append(
                TestTextClassificationDataModule.ClassificationExample(
                    id=f"example_{i}",
                    text=text,
                    label=label,
                )
            )
        return dataset

    def test_text_classification_datamodule(
        self,
        dummy_text_classification_dataset: list[ClassificationExample],
    ) -> None:
        train_dataset = dummy_text_classification_dataset[:40]
        test_dataset = dummy_text_classification_dataset[40:]

        datamodule = self.TextClassificationDataModule(
            id=MetadataTransform(),
            text=Tokenizer(surfaces=TokenSequenceIndexer()),
            label=LabelIndexer(),
        )
        assert isinstance(datamodule.label, SupportsReconstruct)

        with datamodule.train():
            train_instances = [datamodule.instance(example) for example in train_dataset]
        assert len(train_instances) == len(train_dataset)

        test_instances = [datamodule.instance(example) for example in test_dataset]
        assert len(test_instances) == len(test_dataset)

        train_batches = [datamodule.batch(batch) for batch in batched(train_instances, batch_size=8)]
        assert len(train_batches) == 5

        first_batch = train_batches[0]
        assert first_batch.id == [f"example_{i}" for i in range(8)]
        assert isinstance(first_batch.text.surfaces.ids, numpy.ndarray)
        assert first_batch.text.surfaces.ids.shape == (8, 9)
        assert isinstance(first_batch.text.surfaces.mask, numpy.ndarray)
        assert first_batch.text.surfaces.mask.shape == (8, 9)
        assert first_batch.text.surfaces.mask.sum(axis=1).tolist() == [5, 6, 7, 8, 9, 5, 6, 7]
