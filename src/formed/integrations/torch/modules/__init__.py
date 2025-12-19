"""PyTorch neural network modules for formed.

This package provides reusable PyTorch modules for building NLP models,
including embedders, encoders, and other building blocks.

"""

from .embedders import AnalyzedTextEmbedder, BaseEmbedder, EmbedderOutput, TokenEmbedder
from .encoders import (
    BasePositionalEncoder,
    BaseSequenceEncoder,
    FeedForwardSequenceEncoder,
    GRUSequenceEncoder,
    LearnablePositionalEncoder,
    LSTMSequenceEncoder,
    ResidualSequenceEncoder,
    RotaryPositionalEncoder,
    SinusoidalPositionalEncoder,
    StackedSequenceEncoder,
    TransformerEncoder,
)
from .feedforward import FeedForward
from .losses import (
    BaseClassificationLoss,
    BaseRegressionLoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    MeanSquaredErrorLoss,
)
from .masks import BaseAttentionMask, CausalMask, CombinedMask, SlidingWindowAttentionMask
from .samplers import (
    ArgmaxLabelSampler,
    BaseLabelSampler,
    BaseMultilabelSampler,
    BernoulliMultilabelSampler,
    MultinomialLabelSampler,
    ThresholdMultilabelSampler,
    TopKMultilabelSampler,
)
from .vectorizers import BagOfEmbeddingsSequenceVectorizer, BaseSequenceVectorizer
from .weighters import BalancedByDistributionLabelWeighter, BaseLabelWeighter, StaticLabelWeighter

__all__ = [
    # embedders
    "AnalyzedTextEmbedder",
    "BaseEmbedder",
    "EmbedderOutput",
    "TokenEmbedder",
    # encoders
    "BasePositionalEncoder",
    "BaseSequenceEncoder",
    "FeedForwardSequenceEncoder",
    "GRUSequenceEncoder",
    "LearnablePositionalEncoder",
    "LSTMSequenceEncoder",
    "ResidualSequenceEncoder",
    "RotaryPositionalEncoder",
    "SinusoidalPositionalEncoder",
    "StackedSequenceEncoder",
    "TransformerEncoder",
    # feedforward
    "FeedForward",
    # losses
    "BCEWithLogitsLoss",
    "BaseClassificationLoss",
    "BaseRegressionLoss",
    "CrossEntropyLoss",
    "MeanSquaredErrorLoss",
    # masks
    "BaseAttentionMask",
    "CausalMask",
    "CombinedMask",
    "SlidingWindowAttentionMask",
    # samplers
    "ArgmaxLabelSampler",
    "BaseLabelSampler",
    "BaseMultilabelSampler",
    "BernoulliMultilabelSampler",
    "MultinomialLabelSampler",
    "ThresholdMultilabelSampler",
    "TopKMultilabelSampler",
    # vectorizers
    "BagOfEmbeddingsSequenceVectorizer",
    "BaseSequenceVectorizer",
    # weighters
    "BalancedByDistributionLabelWeighter",
    "BaseLabelWeighter",
    "StaticLabelWeighter",
]
