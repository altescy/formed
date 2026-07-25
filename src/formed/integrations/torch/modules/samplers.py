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
    - `BaseSequenceSampler`: Abstract base class for autoregressive sequence samplers
    - `GreedySequenceSampler`: Generates sequences using greedy selection


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
import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypedDict, TypeVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from colt import Registrable

from ..model import BaseTorchModel

_ParamsT = TypeVar("_ParamsT", bound=Optional[object])
_ModelInputT = TypeVar("_ModelInputT")
_ModelOutputT = TypeVar("_ModelOutputT")
_ModelParamsT = TypeVar("_ModelParamsT")
_StateT = TypeVar("_StateT")
_SequenceSamplerParamsT = TypeVar("_SequenceSamplerParamsT", bound=Optional[object])


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
            params: Additional parameters for sampling.

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
            params: Ignored.

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
            params: Optional parameters containing temperature for sampling.

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
            params: Additional parameters for sampling.

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
            params: Optional parameters containing k for top-k selection.

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
            params: Ignored.

        Returns:
            Sampled labels of shape `(..., num_labels)`.

        """
        probs = torch.sigmoid(logits)
        return torch.bernoulli(probs)


@dataclass
class SequenceSamplingStepInput(Generic[_ModelInputT, _StateT]):
    """Input passed to a model-specific sequence sampling adapter.

    ``sequences`` is ``None`` on the first step. On subsequent steps it has
    shape ``(effective_batch_size, generated_length)``.

    """

    inputs: _ModelInputT
    sequences: torch.Tensor | None = None
    state: _StateT | None = None
    batch_indices: torch.Tensor | None = None
    step: int = 0


@dataclass
class SequenceSamplingStepOutput(Generic[_StateT]):
    """Logits and decoder state returned for one sampling step."""

    logits: torch.Tensor
    state: _StateT | None = None


@dataclass
class SequenceSamplingModelParams(Generic[_StateT]):
    """Standard model parameters used by the default sampling adapter."""

    sequences: torch.Tensor | None = None
    state: _StateT | None = None
    batch_indices: torch.Tensor | None = None
    step: int = 0


@dataclass
class SequenceSamplingModelOutput(Generic[_StateT]):
    """Standard model output consumed by the default sampling adapter."""

    logits: torch.Tensor
    state: _StateT | None = None


class BaseSequenceSamplingAdapter(
    Registrable,
    Generic[_ModelInputT, _ModelOutputT, _ModelParamsT, _StateT],
    abc.ABC,
):
    """Adapt a model's public input and output types to one sampling step."""

    @abc.abstractmethod
    def step(
        self,
        model: BaseTorchModel[_ModelInputT, _ModelOutputT, _ModelParamsT],
        request: SequenceSamplingStepInput[_ModelInputT, _StateT],
    ) -> SequenceSamplingStepOutput[_StateT]:
        """Run one model step and extract candidate logits and decoder state."""
        raise NotImplementedError


@BaseSequenceSamplingAdapter.register("default")
class DefaultSequenceSamplingAdapter(
    BaseSequenceSamplingAdapter[
        _ModelInputT,
        SequenceSamplingModelOutput[_StateT],
        SequenceSamplingModelParams[_StateT],
        _StateT,
    ],
):
    """Adapt models that use formed's standard sequence sampling types."""

    def step(
        self,
        model: BaseTorchModel[
            _ModelInputT,
            SequenceSamplingModelOutput[_StateT],
            SequenceSamplingModelParams[_StateT],
        ],
        request: SequenceSamplingStepInput[_ModelInputT, _StateT],
    ) -> SequenceSamplingStepOutput[_StateT]:
        output = model(
            request.inputs,
            params=SequenceSamplingModelParams(
                sequences=request.sequences,
                state=request.state,
                batch_indices=request.batch_indices,
                step=request.step,
            ),
        )
        return SequenceSamplingStepOutput(logits=output.logits, state=output.state)


@dataclass(frozen=True)
class SequenceSamplingContext:
    """Batched state shared with candidate rules and stopping criteria."""

    sequences: torch.Tensor
    lengths: torch.Tensor
    finished: torch.Tensor
    batch_indices: torch.Tensor
    step: int


@dataclass
class SequenceCandidateRuleUpdate(Generic[_StateT]):
    """State transition and optional completion signal from a candidate rule."""

    state: _StateT | None = None
    finished: torch.Tensor | None = None


class BaseSequenceCandidateRule(nn.Module, Registrable, Generic[_StateT], abc.ABC):
    """Modify candidate scores before selection and optionally maintain state."""

    @abc.abstractmethod
    def forward(
        self,
        scores: torch.Tensor,
        context: SequenceSamplingContext,
        state: _StateT | None = None,
    ) -> torch.Tensor:
        """Return candidate scores with this rule applied."""
        raise NotImplementedError

    def update(
        self,
        samples: torch.Tensor,
        context: SequenceSamplingContext,
        state: _StateT | None = None,
    ) -> SequenceCandidateRuleUpdate[_StateT]:
        """Update rule state after selection; stateless rules need not override this."""
        del samples, context
        return SequenceCandidateRuleUpdate(state=state)


class BaseSequenceConstraint(BaseSequenceCandidateRule[_StateT], abc.ABC):
    """Hard candidate rule that disallows invalid candidates."""


class BaseSequenceScoreModifier(BaseSequenceCandidateRule[_StateT], abc.ABC):
    """Soft candidate rule that adjusts candidate scores."""


class SequenceTerminationReason(enum.IntEnum):
    """Reason why generation ended for an individual sequence."""

    UNFINISHED = 0
    MAX_STEPS = 1
    STOPPING_CRITERION = 2
    END_OF_SEQUENCE = 3
    CONSTRAINT_SATISFIED = 4
    CONSTRAINT_DEAD_END = 5


class BaseSequenceStoppingCriterion(nn.Module, Registrable, abc.ABC):
    """Determine which effective batch elements should stop after selection."""

    reason = SequenceTerminationReason.STOPPING_CRITERION

    @abc.abstractmethod
    def forward(self, context: SequenceSamplingContext) -> torch.Tensor:
        """Return a boolean tensor of shape ``(effective_batch_size,)``."""
        raise NotImplementedError


@BaseSequenceStoppingCriterion.register("end_of_sequence")
class EndOfSequenceStoppingCriterion(BaseSequenceStoppingCriterion):
    """Stop when the latest candidate is the configured end-of-sequence index."""

    reason = SequenceTerminationReason.END_OF_SEQUENCE

    def __init__(self, end_index: int) -> None:
        super().__init__()
        self.end_index = end_index

    def forward(self, context: SequenceSamplingContext) -> torch.Tensor:
        return context.sequences[:, -1].eq(self.end_index)


class SequenceSamplerParams(TypedDict, total=False):
    """Optional runtime overrides for sequence sampling."""

    max_steps: int


@dataclass(frozen=True)
class SampledSequenceBatch:
    """A batch of sampled token IDs structurally compatible with sequence indexers."""

    ids: torch.Tensor
    mask: torch.Tensor

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class SequenceSamplerOutput:
    """Sequences, validity masks, scores, and termination metadata."""

    sequences: torch.Tensor
    lengths: torch.Tensor
    mask: torch.Tensor
    sequence_scores: torch.Tensor | None = None
    termination_reasons: torch.Tensor | None = None

    def get_sequence_batch(self, rank: int = 0) -> SampledSequenceBatch:
        """Return one n-best rank for the whole batch."""
        num_return_sequences = self.sequences.size(1)
        if not 0 <= rank < num_return_sequences:
            raise IndexError(f"rank must be in [0, {num_return_sequences}), got {rank}")
        return SampledSequenceBatch(
            ids=self.sequences[:, rank],
            mask=self.mask[:, rank],
        )

    @property
    def best_sequences(self) -> SampledSequenceBatch:
        """Return the highest-ranked sampled sequence for every batch item."""
        return self.get_sequence_batch(0)


class BaseSequenceSampler(
    nn.Module,
    Registrable,
    Generic[_ModelInputT, _ModelOutputT, _ModelParamsT, _StateT, _SequenceSamplerParamsT],
    abc.ABC,
):
    """Abstract base class for batched autoregressive sequence samplers."""

    def __init__(
        self,
        adapter: BaseSequenceSamplingAdapter[_ModelInputT, _ModelOutputT, _ModelParamsT, _StateT],
        candidate_rules: Sequence[BaseSequenceCandidateRule[Any]] = (),
        stopping_criteria: Sequence[BaseSequenceStoppingCriterion] = (),
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.candidate_rules = cast(Sequence[BaseSequenceCandidateRule[Any]], nn.ModuleList(candidate_rules))
        self.stopping_criteria = cast(Sequence[BaseSequenceStoppingCriterion], nn.ModuleList(stopping_criteria))

    @abc.abstractmethod
    def forward(
        self,
        model: BaseTorchModel[_ModelInputT, _ModelOutputT, _ModelParamsT],
        inputs: _ModelInputT,
        initial_state: _StateT | None = None,
        params: _SequenceSamplerParamsT | None = None,
    ) -> SequenceSamplerOutput:
        """Generate sequences from a model in batch-first form."""
        raise NotImplementedError


@BaseSequenceSampler.register("greedy")
class GreedySequenceSampler(
    BaseSequenceSampler[
        _ModelInputT,
        _ModelOutputT,
        _ModelParamsT,
        _StateT,
        SequenceSamplerParams,
    ],
):
    """Generate one sequence per batch item using greedy candidate selection."""

    def __init__(
        self,
        adapter: BaseSequenceSamplingAdapter[_ModelInputT, _ModelOutputT, _ModelParamsT, _StateT],
        max_steps: int,
        candidate_rules: Sequence[BaseSequenceCandidateRule[Any]] = (),
        stopping_criteria: Sequence[BaseSequenceStoppingCriterion] = (),
    ) -> None:
        super().__init__(adapter, candidate_rules, stopping_criteria)
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        self.max_steps = max_steps

    def forward(
        self,
        model: BaseTorchModel[_ModelInputT, _ModelOutputT, _ModelParamsT],
        inputs: _ModelInputT,
        initial_state: _StateT | None = None,
        params: SequenceSamplerParams | None = None,
    ) -> SequenceSamplerOutput:
        """Generate batched sequences subject to injected rules and criteria."""
        max_steps = params.get("max_steps", self.max_steps) if params is not None else self.max_steps
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")

        sequences: torch.Tensor | None = None
        lengths: torch.Tensor | None = None
        sequence_scores: torch.Tensor | None = None
        finished: torch.Tensor | None = None
        termination_reasons: torch.Tensor | None = None
        batch_indices: torch.Tensor | None = None
        rule_states: list[Any | None] = [None] * len(self.candidate_rules)
        state = initial_state

        for step in range(max_steps):
            step_output = self.adapter.step(
                model,
                SequenceSamplingStepInput(
                    inputs=inputs,
                    sequences=sequences,
                    state=state,
                    batch_indices=batch_indices,
                    step=step,
                ),
            )
            scores = step_output.logits
            if scores.ndim != 2:
                raise ValueError(
                    "Sequence sampling scores must have shape "
                    f"(effective_batch_size, num_candidates), got {tuple(scores.shape)}"
                )
            if scores.size(-1) == 0:
                raise ValueError("Sequence sampling requires at least one candidate")

            batch_size = scores.size(0)
            if sequences is None:
                sequences = torch.empty((batch_size, 0), dtype=torch.long, device=scores.device)
                batch_indices = torch.arange(batch_size, device=scores.device)
                lengths = torch.zeros(batch_size, dtype=torch.long, device=scores.device)
                sequence_scores = torch.zeros(batch_size, dtype=scores.dtype, device=scores.device)
                finished = torch.zeros(batch_size, dtype=torch.bool, device=scores.device)
                termination_reasons = torch.full(
                    (batch_size,),
                    SequenceTerminationReason.UNFINISHED,
                    dtype=torch.long,
                    device=scores.device,
                )
            elif batch_size != sequences.size(0):
                raise ValueError("The effective batch size returned by the adapter changed during sampling")

            assert batch_indices is not None
            assert lengths is not None
            assert sequence_scores is not None
            assert finished is not None
            assert termination_reasons is not None

            context = SequenceSamplingContext(
                sequences=sequences,
                lengths=lengths,
                finished=finished,
                batch_indices=batch_indices,
                step=step,
            )
            candidate_scores = scores
            for rule, rule_state in zip(self.candidate_rules, rule_states):
                candidate_scores = rule(candidate_scores, context, rule_state)
                if candidate_scores.shape != scores.shape:
                    raise ValueError("Candidate rules must preserve the shape of candidate scores")

            active = ~finished
            has_candidate = torch.isfinite(candidate_scores).any(dim=-1)
            dead_end = active & ~has_candidate
            safe_scores = candidate_scores.clone()
            safe_scores[~active | dead_end] = 0

            log_probs = torch.log_softmax(safe_scores, dim=-1)
            samples = safe_scores.argmax(dim=-1)
            sample_scores = log_probs.gather(-1, samples.unsqueeze(-1)).squeeze(-1)
            selected = active & ~dead_end
            sample_scores = torch.where(selected, sample_scores, torch.zeros_like(sample_scores))
            lengths = lengths + selected.long()
            sequences = torch.cat((sequences, samples.unsqueeze(-1)), dim=-1)
            sequence_scores = sequence_scores + sample_scores

            termination_reasons = torch.where(
                dead_end,
                torch.full_like(termination_reasons, SequenceTerminationReason.CONSTRAINT_DEAD_END),
                termination_reasons,
            )
            finished = finished | dead_end
            context = SequenceSamplingContext(
                sequences=sequences,
                lengths=lengths,
                finished=finished,
                batch_indices=batch_indices,
                step=step,
            )

            constraint_finished = torch.zeros_like(finished)
            for index, (rule, rule_state) in enumerate(zip(self.candidate_rules, rule_states)):
                update = rule.update(samples, context, rule_state)
                rule_states[index] = update.state
                if update.finished is not None:
                    if update.finished.shape != finished.shape:
                        raise ValueError("Candidate rule completion masks must match the effective batch shape")
                    constraint_finished |= update.finished & ~finished

            termination_reasons = torch.where(
                constraint_finished,
                torch.full_like(termination_reasons, SequenceTerminationReason.CONSTRAINT_SATISFIED),
                termination_reasons,
            )
            finished = finished | constraint_finished
            assert finished is not None
            context = SequenceSamplingContext(
                sequences=sequences,
                lengths=lengths,
                finished=finished,
                batch_indices=batch_indices,
                step=step,
            )

            current_finished = finished
            for criterion in self.stopping_criteria:
                criterion_mask = criterion(context)
                if criterion_mask.shape != current_finished.shape:
                    raise ValueError("Stopping criteria must return the effective batch shape")
                criterion_finished = criterion_mask & ~current_finished
                termination_reasons = torch.where(
                    criterion_finished,
                    torch.full_like(termination_reasons, criterion.reason),
                    termination_reasons,
                )
                current_finished = current_finished | criterion_finished
                context = SequenceSamplingContext(
                    sequences=sequences,
                    lengths=lengths,
                    finished=current_finished,
                    batch_indices=batch_indices,
                    step=step,
                )

            finished = current_finished
            state = step_output.state
            assert finished is not None
            if finished.all():
                break

        assert sequences is not None
        assert lengths is not None
        assert sequence_scores is not None
        assert finished is not None
        assert termination_reasons is not None

        termination_reasons = torch.where(
            ~finished,
            torch.full_like(termination_reasons, SequenceTerminationReason.MAX_STEPS),
            termination_reasons,
        )
        generated_length = int(lengths.max().item()) if lengths.numel() else 0
        sequences = sequences[:, :generated_length]
        mask = torch.arange(generated_length, device=sequences.device).unsqueeze(0) < lengths.unsqueeze(1)
        return SequenceSamplerOutput(
            sequences=sequences.unsqueeze(1),
            lengths=lengths.unsqueeze(1),
            mask=mask.unsqueeze(1),
            sequence_scores=sequence_scores.unsqueeze(1),
            termination_reasons=termination_reasons.unsqueeze(1),
        )
