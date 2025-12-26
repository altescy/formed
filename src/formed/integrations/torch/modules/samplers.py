"""Label samplers for classification tasks.

This module provides samplers that convert model logits into discrete labels.

Key Components:
    - `BaseLabelSampler`: Abstract base class for label samplers
    - `ArgmaxLabelSampler`: Selects the label with highest logit
    - `MultinomialLabelSampler`: Samples from categorical distribution
    - `BaseMultilabelSampler`: Abstract base class for multilabel samplers
    - `ThresholdMultilabelSampler`: Selects labels above a threshold
    - `TopKMultilabelSampler`: Selects top-k labels
    - `BernoulliMultilabelSampler`: Samples labels from independent Bernoulli distributions


Examples:
    >>> from formed.integrations.torch.modules import ArgmaxLabelSampler, MultinomialLabelSampler
    >>> import torch
    >>>
    >>> logits = torch.randn(4, 10)  # (batch_size, num_classes)
    >>>
    >>> # Argmax sampling (deterministic)
    >>> argmax_sampler = ArgmaxLabelSampler()
    >>> labels = argmax_sampler(logits)
    >>>
    >>> # Multinomial sampling (stochastic)
    >>> multi_sampler = MultinomialLabelSampler()
    >>> labels = multi_sampler(logits, temperature=0.8)

"""

import abc
from typing import Generic, Optional, TypedDict, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from colt import Registrable

_ParamsT = TypeVar("_ParamsT", bound=Optional[object])


class BaseLabelSampler(nn.Module, Registrable, Generic[_ParamsT], abc.ABC):
    """Abstract base class for label samplers.

    A LabelSampler defines a strategy for sampling labels based on model logits.

    Type Parameters:
        _ParamsT: Type of additional parameters used during sampling.

    """

    @abc.abstractmethod
    def forward(self, logits: torch.Tensor, params: Optional[_ParamsT] = None) -> torch.Tensor:
        """Sample labels from logits.

        Args:
            logits: Model output logits of shape `(..., num_classes)`.
            **kwargs: Additional parameters for sampling.

        Returns:
            Sampled labels of shape `(...)`.

        """
        raise NotImplementedError

    def __call__(self, logits: torch.Tensor, params: Optional[_ParamsT] = None) -> torch.Tensor:
        return super().__call__(logits, params=params)


@BaseLabelSampler.register("argmax")
class ArgmaxLabelSampler(BaseLabelSampler[None]):
    """Label sampler that selects the label with the highest logit.

    Examples:
        >>> sampler = ArgmaxLabelSampler()
        >>> logits = torch.randn(4, 10)
        >>> labels = sampler(logits)  # Shape: (4,)

    """

    def forward(self, logits: torch.Tensor, params: None = None) -> torch.Tensor:
        """Select the argmax label.

        Args:
            logits: Logits of shape `(..., num_classes)`.
            **kwargs: Ignored.

        Returns:
            Labels of shape `(...)`.

        """
        return logits.argmax(dim=-1)


class MultinomialLabelSamplerParams(TypedDict, total=False):
    """Parameters for MultinomialLabelSampler.

    Attributes:
        temperature: Sampling temperature to control randomness.
            Higher temperature = more random, lower = more deterministic.

    """

    temperature: float


@BaseLabelSampler.register("multinomial")
class MultinomialLabelSampler(BaseLabelSampler[MultinomialLabelSamplerParams]):
    """Label sampler that samples labels from a multinomial distribution.

    Examples:
        >>> sampler = MultinomialLabelSampler()
        >>> logits = torch.randn(4, 10)
        >>>
        >>> # Sample with default temperature
        >>> labels = sampler(logits)
        >>>
        >>> # Sample with temperature scaling
        >>> labels = sampler(logits, temperature=0.5)

    """

    def forward(self, logits: torch.Tensor, params: Optional[MultinomialLabelSamplerParams] = None) -> torch.Tensor:
        """Sample labels from categorical distribution.

        Args:
            logits: Logits of shape `(..., num_classes)`.
            temperature: Sampling temperature to control randomness.
                Higher temperature = more random, lower = more deterministic.
            **kwargs: Ignored.

        Returns:
            Sampled labels of shape `(...)`.

        """
        temperature = params.get("temperature", 1.0) if params is not None else 1.0
        if temperature != 1.0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(probs.shape[:-1])


class BaseMultilabelSampler(nn.Module, Registrable, Generic[_ParamsT], abc.ABC):
    """Abstract base class for multilabel samplers.

    A MultilabelSampler defines a strategy for sampling multiple labels
    based on model logits.

    Type Parameters:
        _ParamsT: Type of additional parameters used during sampling.

    """

    @abc.abstractmethod
    def forward(self, logits: torch.Tensor, params: Optional[_ParamsT] = None) -> torch.Tensor:
        """Sample multiple labels from logits.

        Args:
            logits: Model output logits of shape `(..., num_classes)`.
            **kwargs: Additional parameters for sampling.

        Returns:
            Sampled labels of shape `(..., num_labels)`.

        """
        raise NotImplementedError

    def __call__(self, logits: torch.Tensor, params: Optional[_ParamsT] = None) -> torch.Tensor:
        return super().__call__(logits, params=params)


class ThresholdMultilabelSamplerParams(TypedDict, total=False):
    """Parameters for ThresholdMultilabelSampler.

    Attributes:
        threshold: Probability threshold for selecting labels.

    """

    threshold: float


@BaseMultilabelSampler.register("threshold")
class ThresholdMultilabelSampler(BaseMultilabelSampler[ThresholdMultilabelSamplerParams]):
    """Multilabel sampler that selects labels above a certain threshold.

    Examples:
        >>> sampler = ThresholdMultilabelSampler(threshold=0.5)
        >>> logits = torch.randn(4, 10)
        >>> labels = sampler(logits)  # Shape: (4, num_labels)

    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        logits: torch.Tensor,
        params: Optional[ThresholdMultilabelSamplerParams] = None,
    ) -> torch.Tensor:
        """Select labels above the threshold.

        Args:
            logits: Logits of shape `(..., num_classes)`.
            params: Optional parameters containing threshold.

        Returns:
            Labels of shape `(..., num_labels)`.

        """
        threshold = (params or {}).get("threshold", self.threshold)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()


class TopKMultilabelSamplerParams(TypedDict, total=False):
    """Parameters for TopKMultilabelSampler.

    Attributes:
        k: Number of top labels to select.

    """

    k: int


@BaseMultilabelSampler.register("topk")
class TopKMultilabelSampler(BaseMultilabelSampler[TopKMultilabelSamplerParams]):
    """Multilabel sampler that selects the top-k labels.

    Examples:
        >>> sampler = TopKMultilabelSampler(k=3)
        >>> logits = torch.randn(4, 10)
        >>> labels = sampler(logits)  # Shape: (4, num_labels)

    """

    def __init__(self, k: int = 1) -> None:
        super().__init__()
        self.k = k

    def forward(
        self,
        logits: torch.Tensor,
        params: Optional[TopKMultilabelSamplerParams] = None,
    ) -> torch.Tensor:
        """Select the top-k labels.

        Args:
            logits: Logits of shape `(..., num_classes)`.
            **kwargs: Ignored.

        Returns:
            Labels of shape `(..., num_labels)`.

        """
        k = (params or {}).get("k", self.k)
        topk_indices = logits.topk(k, dim=-1).indices
        labels = torch.zeros_like(logits).scatter_(-1, topk_indices, 1.0)
        return labels


@BaseMultilabelSampler.register("bernoulli")
class BernoulliMultilabelSampler(BaseMultilabelSampler[None]):
    """Multilabel sampler that samples labels from independent Bernoulli distributions.

    Examples:
        >>> sampler = BernoulliMultilabelSampler()
        >>> logits = torch.randn(4, 10)
        >>> labels = sampler(logits)  # Shape: (4, num_labels)

    """

    def forward(self, logits: torch.Tensor, params: None = None) -> torch.Tensor:
        """Sample labels from Bernoulli distributions.

        Args:
            logits: Logits of shape `(..., num_classes)`.
            **kwargs: Ignored.

        Returns:
            Sampled labels of shape `(..., num_labels)`.

        """
        probs = torch.sigmoid(logits)
        return torch.bernoulli(probs)
