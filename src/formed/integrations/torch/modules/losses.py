"""Loss functions for classification tasks.

This module provides loss functions for classification with support for
label weighting and different reduction strategies.

Key Components:
    - BaseClassificationLoss: Abstract base class for classification losses
    - CrossEntropyLoss: Standard cross-entropy loss with optional weighting

Example:
    >>> from formed.integrations.torch.modules import CrossEntropyLoss
    >>> import torch
    >>>
    >>> # Simple cross-entropy
    >>> loss_fn = CrossEntropyLoss()
    >>> logits = torch.randn(4, 10)  # (batch_size, num_classes)
    >>> labels = torch.randint(0, 10, (4,))  # (batch_size,)
    >>> loss = loss_fn(logits, labels)
    >>>
    >>> # With label weighting
    >>> from formed.integrations.torch.modules import StaticLabelWeighter
    >>> weighter = StaticLabelWeighter(weights=torch.ones(10))
    >>> loss_fn = CrossEntropyLoss(weighter=weighter)

"""

import abc
from typing import Generic, Literal, Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from colt import Registrable

from ..types import TensorCompatible
from ..utils import ensure_torch_tensor
from .weighters import BaseLabelWeighter

_ParamsT = TypeVar("_ParamsT", bound=Optional[object])


class BaseClassificationLoss(nn.Module, Registrable, Generic[_ParamsT], abc.ABC):
    """Abstract base class for classification loss functions.

    A ClassificationLoss defines a strategy for computing loss based on model logits and true labels.

    Type Parameters:
        _ParamsT: Type of additional parameters used during loss computation.

    """

    @abc.abstractmethod
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, params: Optional[_ParamsT] = None) -> torch.Tensor:
        """Compute the classification loss.

        Args:
            logits: Model output logits of shape (..., num_classes).
            labels: True target labels of shape (...).
            params: Optional additional parameters for loss computation.

        Returns:
            Computed loss as a scalar tensor.

        """
        raise NotImplementedError

    def __call__(
        self, logits: torch.Tensor, labels: TensorCompatible, params: Optional[_ParamsT] = None
    ) -> torch.Tensor:
        return super().__call__(logits, labels, params)


@BaseClassificationLoss.register("cross_entropy")
class CrossEntropyLoss(BaseClassificationLoss[_ParamsT]):
    """Cross-entropy loss for classification tasks.

    Args:
        weighter: An optional label weighter to assign weights to each class.
        reduce: Reduction method - "mean" or "sum".

    Example:
        >>> loss_fn = CrossEntropyLoss()
        >>> logits = torch.randn(4, 10)
        >>> labels = torch.randint(0, 10, (4,))
        >>> loss = loss_fn(logits, labels)

    """

    def __init__(
        self,
        weighter: Optional[BaseLabelWeighter[_ParamsT]] = None,
        reduce: Literal["mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        self._weighter = weighter
        self._reduce = reduce

    def forward(
        self,
        logits: torch.Tensor,
        labels: TensorCompatible,
        params: Optional[_ParamsT] = None,
    ) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: Logits of shape (..., num_classes).
            labels: Labels of shape (...).
            params: Optional parameters for the weighter.

        Returns:
            Loss scalar.

        """
        labels = ensure_torch_tensor(labels)

        num_classes = logits.shape[-1]
        one_hot_labels = F.one_hot(labels.long(), num_classes=num_classes).float()
        log_probs = F.log_softmax(logits, dim=-1)

        if self._weighter is not None:
            weights = self._weighter(logits, labels, params)
            loss = -(one_hot_labels * log_probs * weights).sum(dim=-1)
        else:
            loss = -(one_hot_labels * log_probs).sum(dim=-1)

        if self._reduce == "mean":
            return loss.mean()
        elif self._reduce == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduce operation: {self._reduce}")


@BaseClassificationLoss.register("bce_with_logits")
class BCEWithLogitsLoss(BaseClassificationLoss[_ParamsT]):
    """Binary cross-entropy loss with logits for multilabel classification tasks.

    This loss combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid followed by BCELoss.

    Args:
        weighter: An optional label weighter to assign weights to each class.
        reduce: Reduction method - "mean" or "sum".
        pos_weight: Optional weight for positive examples per class.

    Example:
        >>> loss_fn = BCEWithLogitsLoss()
        >>> logits = torch.randn(4, 10)  # (batch_size, num_classes)
        >>> labels = torch.randint(0, 2, (4, 10)).float()  # (batch_size, num_classes)
        >>> loss = loss_fn(logits, labels)

    """

    def __init__(
        self,
        weighter: Optional[BaseLabelWeighter[_ParamsT]] = None,
        reduce: Literal["mean", "sum"] = "mean",
        pos_weight: Optional[TensorCompatible] = None,
    ) -> None:
        super().__init__()
        self._weighter = weighter
        self._reduce = reduce
        self._pos_weight = ensure_torch_tensor(pos_weight) if pos_weight is not None else None

    def forward(
        self,
        logits: torch.Tensor,
        labels: TensorCompatible,
        params: Optional[_ParamsT] = None,
    ) -> torch.Tensor:
        """Compute BCE with logits loss.

        Args:
            logits: Logits of shape (..., num_classes).
            labels: Binary labels of shape (..., num_classes).
            params: Optional parameters for the weighter.

        Returns:
            Loss scalar.

        """
        labels = ensure_torch_tensor(labels).float()

        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self._pos_weight, reduction="none")

        if self._weighter is not None:
            weights = self._weighter(logits, labels, params)
            loss = loss * weights

        if self._reduce == "mean":
            return loss.mean()
        elif self._reduce == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduce operation: {self._reduce}")


class BaseRegressionLoss(nn.Module, Registrable, Generic[_ParamsT], abc.ABC):
    """Abstract base class for regression loss functions.

    A RegressionLoss defines a strategy for computing loss based on model predictions and true labels.

    Type Parameters:
        _ParamsT: Type of additional parameters used during loss computation.

    """

    @abc.abstractmethod
    def forward(
        self, predictions: torch.Tensor, labels: torch.Tensor, params: Optional[_ParamsT] = None
    ) -> torch.Tensor:
        """Compute the regression loss.

        Args:
            predictions: Model output predictions of shape (...).
            labels: True target labels of shape (...).
            params: Optional additional parameters for loss computation.

        Returns:
            Computed loss as a scalar tensor.

        """
        raise NotImplementedError

    def __call__(
        self, predictions: torch.Tensor, targets: TensorCompatible, params: Optional[_ParamsT] = None
    ) -> torch.Tensor:
        return super().__call__(predictions, targets, params)


@BaseRegressionLoss.register("mse")
class MeanSquaredErrorLoss(BaseRegressionLoss[_ParamsT]):
    """Mean Squared Error (MSE) loss for regression tasks.

    Args:
        reduce: Reduction method - "mean" or "sum".

    Example:
        >>> loss_fn = MeanSquaredErrorLoss()
        >>> predictions = torch.randn(4)
        >>> labels = torch.randn(4)
        >>> loss = loss_fn(predictions, labels)

    """

    def __init__(
        self,
        reduce: Literal["mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        self._reduce = reduce

    def forward(
        self,
        predictions: torch.Tensor,
        labels: TensorCompatible,
        params: Optional[_ParamsT] = None,
    ) -> torch.Tensor:
        """Compute MSE loss.

        Args:
            predictions: Predictions of shape (...).
            labels: Labels of shape (...).
            params: Ignored.

        Returns:
            Loss scalar.

        """
        labels = ensure_torch_tensor(labels)

        loss = (predictions - labels).pow(2)

        if self._reduce == "mean":
            return loss.mean()
        elif self._reduce == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduce operation: {self._reduce}")
