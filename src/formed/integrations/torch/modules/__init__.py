"""PyTorch neural network modules for formed.

This package provides reusable PyTorch modules for building NLP models,
including embedders, encoders, and other building blocks.

"""

from .embedders import AnalyzedTextEmbedder, BaseEmbedder, EmbedderOutput, TokenEmbedder
from .encoders import BaseSequenceEncoder, GRUSequenceEncoder, LSTMSequenceEncoder
from .feedforward import FeedForward
from .losses import BaseClassificationLoss, BaseRegressionLoss, CrossEntropyLoss, MeanSquaredErrorLoss
from .samplers import ArgmaxLabelSampler, BaseLabelSampler, MultinomialLabelSampler
from .vectorizers import BagOfEmbeddingsSequenceVectorizer, BaseSequenceVectorizer
from .weighters import BalancedByDistributionLabelWeighter, BaseLabelWeighter, StaticLabelWeighter

__all__ = [
    # embedders
    "AnalyzedTextEmbedder",
    "BaseEmbedder",
    "EmbedderOutput",
    "TokenEmbedder",
    # encoders
    "BaseSequenceEncoder",
    "GRUSequenceEncoder",
    "LSTMSequenceEncoder",
    # feedforward
    "FeedForward",
    # losses
    "BaseClassificationLoss",
    "BaseRegressionLoss",
    "CrossEntropyLoss",
    "MeanSquaredErrorLoss",
    # samplers
    "ArgmaxLabelSampler",
    "BaseLabelSampler",
    "MultinomialLabelSampler",
    # vectorizers
    "BagOfEmbeddingsSequenceVectorizer",
    "BaseSequenceVectorizer",
    # weighters
    "BalancedByDistributionLabelWeighter",
    "BaseLabelWeighter",
    "StaticLabelWeighter",
]
