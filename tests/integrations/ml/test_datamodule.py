from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Union

from formed.integrations.ml import (
    BasicBatchSampler,
    DataLoader,
    DataModule,
    Dataset,
    LabelFieldTransform,
    LabelIndexer,
    TextFieldTransform,
    TokenIndexer,
    debatched,
)


def test_datamodule() -> None:
    @dataclass
    class Text2TextExample:
        source: Union[str, Sequence[str]]
        target: Union[str, Sequence[str]]
        language: str = "en"

    text2text_dataset = [
        Text2TextExample(source="how are you?", target="I am fine."),
        Text2TextExample(source="what is your name?", target="My name is John."),
        Text2TextExample(source="where are you?", target="I am in New-York."),
        Text2TextExample(source="what is the time?", target="It is 10:00 AM."),
        Text2TextExample(source="comment ça va?", target="Je vais bien.", language="fr"),
    ]

    shared_token_indexer = TokenIndexer(default="<unk>", reserved=["<pad>", "<unk>"])
    language_indexer = LabelIndexer[str]()

    text2text_datamodule = DataModule[Text2TextExample](
        fields={
            "source": TextFieldTransform(indexer=shared_token_indexer, pad_token="<pad>"),
            "target": TextFieldTransform(indexer=shared_token_indexer, pad_token="<pad>"),
            "language": LabelFieldTransform(indexer=language_indexer),
        }
    )

    text2text_datamodule.build(text2text_dataset)

    dataloader = DataLoader(BasicBatchSampler(batch_size=2))

    text2text_instances = Dataset.from_iterable(text2text_datamodule(text2text_dataset))
    assert len(text2text_instances) == 5

    batches = list(dataloader(text2text_instances))
    assert len(batches) == 3

    reconstructions = [
        Text2TextExample(
            source=text2text_datamodule.field("source").reconstruct(item["source"]),
            target=text2text_datamodule.field("target").reconstruct(item["target"]),
            language=text2text_datamodule.field("language").reconstruct(item["language"]),
        )
        for batch in batches
        for item in debatched(batch)
        if isinstance(item, Mapping)
    ]
    assert len(reconstructions) == 5

    assert [item.language for item in reconstructions] == ["en", "en", "en", "en", "fr"]
    assert [item.source for item in reconstructions] == [
        ["how", "are", "you", "?"],
        ["what", "is", "your", "name", "?"],
        ["where", "are", "you", "?"],
        ["what", "is", "the", "time", "?"],
        ["comment", "ça", "va", "?"],
    ]
    assert [item.target for item in reconstructions] == [
        ["I", "am", "fine", "."],
        ["My", "name", "is", "John", "."],
        ["I", "am", "in", "New-York", "."],
        ["It", "is", "10", ":", "00", "AM", "."],
        ["Je", "vais", "bien", "."],
    ]
