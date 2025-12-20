import dataclasses

import torch

import formed.integrations.ml.types as mlt
import formed.integrations.torch as ft
import formed.integrations.torch.modules as ftm

from .datamodules import TextClassificationDataModule


@dataclasses.dataclass
class TorchClassifierOutput:
    probs: torch.Tensor
    label: torch.Tensor
    loss: torch.Tensor | None = None


@ft.BaseTorchModel.register("textclf::torch_text_classifier")
class TextClassifier(ft.BaseTorchModel[TextClassificationDataModule[mlt.AsBatch], TorchClassifierOutput]):
    def __init__(
        self,
        num_classes: int,
        embedder: ftm.BaseEmbedder,
        vectorizer: ftm.BaseSequenceVectorizer,
        encoder: ftm.BaseSequenceEncoder | None = None,
        feedforward: ftm.FeedForward | None = None,
        sampler: ftm.BaseLabelSampler | None = None,
        loss: ftm.BaseClassificationLoss | None = None,
        dropout: float = 0.1,
    ) -> None:
        sampler = sampler or ftm.ArgmaxLabelSampler()
        loss = loss or ftm.CrossEntropyLoss()

        feature_dim = ft.determine_ndim(
            embedder.get_output_dim(),
            encoder.get_output_dim() if encoder is not None else None,
            vectorizer.get_output_dim(),
            feedforward.get_output_dim() if feedforward is not None else None,
        )

        super().__init__()

        self._embedder = embedder
        self._vectorizer = vectorizer
        self._encoder = encoder
        self._feedforward = feedforward
        self._dropout = torch.nn.Dropout(dropout)
        self._classifier = torch.nn.Linear(feature_dim, num_classes)
        self._sampler = sampler
        self._loss = loss

    def forward(
        self,
        inputs: TextClassificationDataModule[mlt.AsBatch],
        params: None = None,
    ) -> TorchClassifierOutput:
        del params

        embeddings, mask = self._embedder(inputs.text)

        if self._encoder is not None:
            embeddings = self._encoder(embeddings, mask=mask)

        vector = self._vectorizer(embeddings, mask=mask)

        if self._feedforward is not None:
            vector = self._feedforward(vector)

        vector = self._dropout(vector)
        logits = self._classifier(vector)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        label = self._sampler(logits)

        loss: torch.Tensor | None = None
        if inputs.label is not None:
            loss = self._loss(logits, inputs.label)

        return TorchClassifierOutput(
            probs=probs,
            label=label,
            loss=loss,
        )
