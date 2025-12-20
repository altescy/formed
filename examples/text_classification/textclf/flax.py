import jax
from flax import nnx, struct

import formed.integrations.flax as fl
import formed.integrations.flax.modules as flm
import formed.integrations.ml.types as mlt

from .datamodules import TextClassificationDataModule


@struct.dataclass
class FlaxClassifierOutput:
    probs: jax.Array
    label: jax.Array
    loss: jax.Array | None = None


@fl.BaseFlaxModel.register("textclf::flax_text_classifier")
class FlaxTextClassifier(fl.BaseFlaxModel[TextClassificationDataModule[mlt.AsBatch], FlaxClassifierOutput]):
    def __init__(
        self,
        num_classes: int,
        embedder: flm.BaseEmbedder[mlt.BatchT],
        vectorizer: flm.BaseSequenceVectorizer,
        encoder: flm.BaseSequenceEncoder | None = None,
        feedforward: flm.FeedForward | None = None,
        sampler: flm.BaseLabelSampler | None = None,
        loss: flm.BaseClassificationLoss | None = None,
        dropout: float = 0.1,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = rngs or fl.require_rngs()

        feature_dim = fl.determine_ndim(
            embedder.get_output_dim(),
            encoder.get_output_dim() if encoder is not None else None,
            vectorizer.get_output_dim(),
            feedforward.get_output_dim() if feedforward is not None else None,
        )

        self._embedder = embedder
        self._vectorizer = vectorizer
        self._encoder = encoder
        self._feedforward = feedforward
        self._dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0.0 else None
        self._classifier = nnx.Linear(feature_dim, num_classes, rngs=rngs)
        self._sampler = sampler or flm.ArgmaxLabelSampler()
        self._loss = loss or flm.CrossEntropyLoss()

    def __call__(
        self,
        inputs: TextClassificationDataModule[mlt.AsBatch],
        params: None = None,
    ) -> FlaxClassifierOutput:
        embeddings, mask = self._embedder(inputs.text)

        if self._encoder is not None:
            embeddings = self._encoder(embeddings, mask=mask)

        features = self._vectorizer(embeddings, mask=mask)

        if self._feedforward is not None:
            features = self._feedforward(features)

        if self._dropout is not None:
            features = self._dropout(features)

        logits = self._classifier(features)
        probs = jax.nn.softmax(logits, axis=-1)
        label = self._sampler(logits)

        loss: jax.Array | None = None
        if inputs.label is not None:
            loss = self._loss(logits, inputs.label)

        return FlaxClassifierOutput(probs=probs, label=label, loss=loss)
