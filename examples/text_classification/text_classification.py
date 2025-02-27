import dataclasses
import logging
import sys
from collections.abc import Mapping
from typing import Annotated, Optional, Union

import datasets
import flax
import jax
import optax
from flax import nnx, struct
from rich.logging import RichHandler

from formed.integrations.flax import FlaxModel, FlaxTrainer
from formed.integrations.ml import (
    DataModule,
    Dataset,
    FieldConfig,
    LabelFieldTransform,
    TextFieldTransform,
    use_datamodule,
)

logging.basicConfig(
    level="INFO",
    datefmt="[%X]",
    handlers=[RichHandler()],
)


@FlaxModel.register("text_classifier")
class TextClassifier(
    FlaxModel[
        "TextClassifier.Input",
        "TextClassifier.Output",
        "TextClassifier.Params",
    ]
):
    @struct.dataclass
    class Input:
        text: Annotated[Mapping[str, jax.Array], FieldConfig("text", TextFieldTransform)]
        label: Annotated[Optional[jax.Array], FieldConfig("label", LabelFieldTransform)] = None

    @struct.dataclass
    class Output:
        probs: jax.Array
        loss: Optional[jax.Array] = None
        metrics: Optional[Mapping[str, jax.Array]] = None

    Params = type[None]

    def __init__(
        self,
        rngs: Union[int, nnx.Rngs] = 0,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)

        datamodule = use_datamodule()
        vocab_size = datamodule.field("text").stats["index_size"]
        num_labels = datamodule.field("label").stats["index_size"]

        self.embedder = nnx.Embed(vocab_size, hidden_dim, rngs=rngs)
        self.classifier = nnx.Linear(hidden_dim, num_labels, rngs=rngs)

    def __call__(
        self,
        inputs: Input,
        params: Optional[Params] = None,
        *,
        train: bool = False,
    ) -> Output:
        mask = inputs.text["mask"]
        token_embeddings = self.embedder(inputs.text["token_ids"])

        token_embeddings = token_embeddings * mask[:, :, None]
        text_embedding = token_embeddings.sum(axis=1) / mask.sum(axis=1)[:, None]

        logits = self.classifier(text_embedding)
        probs = jax.nn.softmax(logits)

        loss: Optional[jax.Array] = None
        metrics: Optional[Mapping[str, jax.Array]] = None
        if inputs.label is not None:
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, inputs.label).mean()
            accuracy = (inputs.label == probs.argmax(axis=-1)).mean()
            metrics = {"accuracy": accuracy}

        return self.Output(probs=probs, loss=loss, metrics=metrics)


if __name__ == "__main__":
    train_dataset = datasets.load_dataset("stanfordnlp/imdb", split="train")

    datamodule = TextClassifier.default_data_module()
    assert datamodule is not None

    datamodule.build(train_dataset)
    datamodule.activate()

    train_instances = Dataset.from_iterable(datamodule(train_dataset))

    model = TextClassifier()

    trainer = FlaxTrainer(max_epochs=5, logging_strategy="step", logging_interval=100)  # type: ignore
    trainer.train(nnx.Rngs(0), model, train_instances)
