"""Label weighters for classification tasks.

This module provides weighters that assign weights to class labels,
useful for handling imbalanced datasets.

Key Components:
    - BaseLabelWeighter: Abstract base class for label weighters
    - StaticLabelWeighter: Uses fixed weights per class
    - BalancedByDistributionLabelWeighter: Balances based on class distribution

Example:
    >>> from formed.integrations.torch.modules import StaticLabelWeighter
    >>> import torch
    >>>
    >>> # Static weights for 3 classes
    >>> weights = torch.tensor([1.0, 2.0, 3.0])  # Weight rare classes more
    >>> weighter = StaticLabelWeighter(weights=weights)
    >>>
    >>> logits = torch.randn(4, 3)
    >>> labels = torch.tensor([0, 1, 2, 0])
    >>> class_weights = weighter(logits, labels)  # Shape: (1, 3)

"""

import abc
from typing import Generic, TypeVar

import torch
import torch.nn as nn
from colt import Registrable

_ParamsT = TypeVar("_ParamsT", bound=object | None)


class BaseLabelWeighter(nn.Module, Registrable, Generic[_ParamsT], abc.ABC):
    """Abstract base class for label weighters.

    A LabelWeighter defines a strategy for assigning weights to each label
    based on model logits and true targets.

    Type Parameters:
        _ParamsT: Type of additional parameters used during weighting.

    """

    @abc.abstractmethod
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        params: _ParamsT | None = None,
    ) -> torch.Tensor:
        """Compute weights for each target label.

        Args:
            logits: Model output logits of shape (..., num_classes).
            targets: True target labels of shape (...).
            params: Optional additional parameters for weighting.

        Returns:
            Weights for each logit of shape (1, num_classes) or broadcastable to logits shape.

        """
        raise NotImplementedError


@BaseLabelWeighter.register("static")
class StaticLabelWeighter(BaseLabelWeighter[None]):
    """Label weighter that assigns static weights to each class.

    Args:
        weights: A tensor of shape (num_classes,) containing the weight for each class.

    Example:
        >>> # Weight class 1 twice as much as class 0
        >>> weights = torch.tensor([1.0, 2.0, 1.0])
        >>> weighter = StaticLabelWeighter(weights=weights)
        >>> logits = torch.randn(4, 3)
        >>> labels = torch.tensor([0, 1, 2, 0])
        >>> class_weights = weighter(logits, labels)

    """

    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_weights", weights.float())
        self._weights: torch.Tensor

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        params: None = None,
    ) -> torch.Tensor:
        """Return static weights.

        Args:
            logits: Ignored.
            targets: Ignored.
            params: Ignored.

        Returns:
            Weights of shape (1, num_classes).

        """
        return self._weights.unsqueeze(0)


@BaseLabelWeighter.register("balanced_by_distribution")
class BalancedByDistributionLabelWeighter(BaseLabelWeighter[None]):
    """Label weighter that balances classes based on their distribution.

    The weight for each class is computed as: 1 / (distribution * num_classes + eps)

    Args:
        distribution: A tensor of shape (num_classes,) representing the class distribution
            (should sum to 1.0).
        eps: A small epsilon value to avoid division by zero.

    Example:
        >>> # Class distribution: 50%, 30%, 20%
        >>> distribution = torch.tensor([0.5, 0.3, 0.2])
        >>> weighter = BalancedByDistributionLabelWeighter(distribution=distribution)
        >>> logits = torch.randn(4, 3)
        >>> labels = torch.tensor([0, 1, 2, 0])
        >>> class_weights = weighter(logits, labels)
        >>> # Rare classes get higher weights

    """

    def __init__(self, distribution: torch.Tensor, eps: float = 1e-8) -> None:
        super().__init__()
        self.register_buffer("_distribution", distribution.float())
        self._distribution: torch.Tensor
        self._eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        params: None = None,
    ) -> torch.Tensor:
        """Compute balanced weights.

        Args:
            logits: Ignored.
            targets: Ignored.
            params: Ignored.

        Returns:
            Weights of shape (1, num_classes).

        """
        num_classes = len(self._distribution)
        weights = 1.0 / (self._distribution * num_classes + self._eps)
        return weights.unsqueeze(0)
