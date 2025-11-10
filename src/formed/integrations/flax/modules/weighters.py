import abc
from typing import Generic

import jax
from colt import Registrable
from flax import nnx
from typing_extensions import TypeVar

from ..types import ArrayCompatible
from ..utils import ensure_jax_array

_ParamsT = TypeVar("_ParamsT", default=None)


class BaseLabelWeighter(nnx.Module, Registrable, Generic[_ParamsT], abc.ABC):
    """Abstract base class for label weighters.

    A LabelWeighter defines a strategy for assigning weights to each label
    based on model logits and true targets.

    Type Parameters:
        _ParamsT: Type of additional parameters used during weighting.
    """

    @abc.abstractmethod
    def __call__(
        self,
        logits: jax.Array,
        targets: jax.Array,
        params: _ParamsT | None = None,
    ) -> jax.Array:
        """Compute weights for each target label.

        Args:
            logits: Model output logits of shape (..., num_classes).
            targets: True target labels of shape (...).
            params: Optional additional parameters for weighting.

        Returns:
            Weights for each logit of shape (..., num_classes) or broadcastable to that shape.
        """
        raise NotImplementedError


@BaseLabelWeighter.register("static")
class StaticLabelWeighter(BaseLabelWeighter):
    """Label weighter that assigns static weights to each class.

    Args:
        weights: An array of shape (num_classes,) containing the weight for each class.
    """

    def __init__(self, weights: ArrayCompatible) -> None:
        super().__init__()
        self._weights = nnx.Param(ensure_jax_array(weights), mutable=False)

    def __call__(
        self,
        logits: jax.Array,
        targets: ArrayCompatible,
        params: None = None,
    ) -> jax.Array:
        del logits, targets, params
        return self._weights[None, ...]


@BaseLabelWeighter.register("balanced_by_distribution")
class BalancedByDistributionLabelWeighter(BaseLabelWeighter):
    """Label weighter that balances classes based on their distribution.

    Args:
        distribution: An array of shape (num_classes,) representing the class distribution.
        eps: A small epsilon value to avoid division by zero.
    """

    def __init__(self, distribution: ArrayCompatible, eps: float = 1e-8) -> None:
        self._distribution = nnx.Param(ensure_jax_array(distribution), mutable=False)
        self._eps = eps

    def __call__(
        self,
        logits: jax.Array,
        targets: ArrayCompatible,
        params: None = None,
    ) -> jax.Array:
        del logits, targets, params
        weights = 1.0 / (self._distribution * len(self._distribution) + self._eps)
        return weights[None, ...]
