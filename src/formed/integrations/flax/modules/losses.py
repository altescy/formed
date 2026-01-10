import abc
from typing import Generic, Literal

import jax
from colt import Registrable
from flax import nnx
from typing_extensions import TypeVar

from ..types import ArrayCompatible
from ..utils import ensure_jax_array
from .weighters import BaseLabelWeighter

_ParamsT = TypeVar("_ParamsT", default=None)


class BaseClassificationLoss(nnx.Module, Registrable, Generic[_ParamsT], abc.ABC):
    """Abstract base class for classification loss functions.

    A ClassificationLoss defines a strategy for computing loss based on model logits and true labels.

    Type Parameters:
        _ParamsT: Type of additional parameters used during loss computation.

    """

    @abc.abstractmethod
    def __call__(self, logits: jax.Array, labels: ArrayCompatible, params: _ParamsT | None = None) -> jax.Array:
        """Compute the classification loss.

        Args:
            logits: Model output logits of shape (..., num_classes).
            labels: True target labels of shape (...).
            params: Optional additional parameters for loss computation.

        Returns:
            Computed loss as a scalar jax.Array.
        """
        raise NotImplementedError


@BaseClassificationLoss.register("cross_entropy")
class CrossEntropyLoss(BaseClassificationLoss[_ParamsT]):
    """Cross-entropy loss for classification tasks.

    Args:
        weighter: An optional label weighter to assign weights to each class.
        reduce: Reduction method for loss computation ("mean" or "sum").
    """

    def __init__(
        self,
        weighter: BaseLabelWeighter[_ParamsT] | None = None,
        reduce: Literal["mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        self._weighter = weighter
        self._reduce = reduce

    def __call__(
        self,
        logits: jax.Array,
        labels: ArrayCompatible,
        params: _ParamsT | None = None,
    ) -> jax.Array:
        labels = ensure_jax_array(labels)
        one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        if self._weighter is not None:
            weights = self._weighter(logits, labels, params)
            loss = -(one_hot_labels * log_probs * weights).sum(axis=-1)
        else:
            loss = -(one_hot_labels * log_probs).sum(axis=-1)

        if self._reduce == "mean":
            return loss.mean()
        elif self._reduce == "sum":
            return loss.sum()

        raise ValueError(f"Unknown reduce operation: {self._reduce}")
