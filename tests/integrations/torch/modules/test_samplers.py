"""Tests for samplers module."""

from dataclasses import dataclass
from typing import Self

import pytest
import torch

from formed.integrations.torch.model import BaseTorchModel
from formed.integrations.torch.modules.samplers import (
    AllBeamsFinishedTerminationPolicy,
    ArgmaxLabelSampler,
    BaseSequenceCandidateRule,
    BaseSequenceConstraint,
    BaseSequenceHypothesisScorer,
    BaseSequenceSampler,
    BaseSequenceSamplingAdapter,
    BeamSearchSequenceSampler,
    BeamSearchSequenceSamplerParams,
    BeamSearchTerminationContext,
    BernoulliMultilabelSampler,
    CumulativeLogProbabilityScorer,
    DefaultSequenceSamplingAdapter,
    EndOfSequenceStoppingCriterion,
    EnoughFinishedHypothesesTerminationPolicy,
    GreedySequenceSampler,
    LengthPenaltySequenceHypothesisScorer,
    MultinomialLabelSampler,
    ScoreBoundTerminationPolicy,
    SequenceCandidateRuleUpdate,
    SequenceHypothesisScoringContext,
    SequenceSamplerParams,
    SequenceSamplingContext,
    SequenceSamplingModelOutput,
    SequenceSamplingModelParams,
    SequenceSamplingStepInput,
    SequenceSamplingStepOutput,
    SequenceTerminationReason,
    ThresholdMultilabelSampler,
    TopKMultilabelSampler,
)
from formed.workflow.colt import COLT_BUILDER


@dataclass
class ToyDecoderState:
    values: torch.Tensor

    def reorder(self, indices: torch.Tensor) -> Self:
        return type(self)(self.values.index_select(0, indices))


@dataclass
class CounterRuleState:
    values: torch.Tensor

    def reorder(self, indices: torch.Tensor) -> Self:
        return type(self)(self.values.index_select(0, indices))


class ScheduledLogitsModel(BaseTorchModel[torch.Tensor, torch.Tensor, None]):
    """Return a precomputed schedule of logits for sampler tests."""

    def forward(self, inputs: torch.Tensor, params: None = None) -> torch.Tensor:
        return inputs


class StandardSamplingModel(
    BaseTorchModel[
        torch.Tensor,
        SequenceSamplingModelOutput[None],
        SequenceSamplingModelParams[None],
    ]
):
    """Model using the standard sampling parameters and output."""

    def forward(
        self,
        inputs: torch.Tensor,
        params: SequenceSamplingModelParams[None] | None = None,
    ) -> SequenceSamplingModelOutput[None]:
        assert params is not None
        return SequenceSamplingModelOutput(logits=inputs[:, params.step])


@BaseSequenceSamplingAdapter.register("scheduled_logits_test")
class ScheduledLogitsAdapter(BaseSequenceSamplingAdapter[torch.Tensor, torch.Tensor, None, ToyDecoderState]):
    def step(
        self,
        model: BaseTorchModel[torch.Tensor, torch.Tensor, None],
        request: SequenceSamplingStepInput[torch.Tensor, ToyDecoderState],
    ) -> SequenceSamplingStepOutput[ToyDecoderState]:
        scheduled_logits = model(request.inputs)
        if request.step == 0:
            assert request.batch_indices is None
            logits = scheduled_logits[:, request.step]
        else:
            assert request.batch_indices is not None
            logits = scheduled_logits[request.batch_indices, request.step]
        state = request.state or ToyDecoderState(torch.arange(logits.size(0), device=logits.device))
        if request.batch_indices is not None:
            assert torch.equal(state.values, request.batch_indices)
        return SequenceSamplingStepOutput(logits=logits, state=state)


@BaseSequenceSamplingAdapter.register("branching_logits_test")
class BranchingLogitsAdapter(BaseSequenceSamplingAdapter[torch.Tensor, torch.Tensor, None, ToyDecoderState]):
    """Return logits whose second step depends on the selected first token."""

    def step(
        self,
        model: BaseTorchModel[torch.Tensor, torch.Tensor, None],
        request: SequenceSamplingStepInput[torch.Tensor, ToyDecoderState],
    ) -> SequenceSamplingStepOutput[ToyDecoderState]:
        del model
        if request.sequences is None:
            batch_size = request.inputs.size(0)
            logits = torch.tensor([[0.0, -0.1]], device=request.inputs.device).expand(batch_size, -1)
            state = ToyDecoderState(torch.arange(batch_size, device=request.inputs.device))
        else:
            assert request.batch_indices is not None
            assert request.state is not None
            assert torch.equal(request.state.values, request.batch_indices)
            first_tokens = request.sequences[:, 0]
            preferred = torch.tensor([0.0, 0.0], device=request.inputs.device)
            alternative = torch.tensor([10.0, 0.0], device=request.inputs.device)
            logits = torch.where(first_tokens.unsqueeze(1).eq(0), preferred, alternative)
            state = request.state
        return SequenceSamplingStepOutput(logits=logits, state=state)


@BaseSequenceCandidateRule.register("finish_after_two_test")
class FinishAfterTwoCandidatesConstraint(BaseSequenceConstraint[CounterRuleState]):
    def forward(
        self,
        scores: torch.Tensor,
        context: SequenceSamplingContext,
        state: CounterRuleState | None = None,
    ) -> torch.Tensor:
        return scores

    def update(
        self,
        samples: torch.Tensor,
        context: SequenceSamplingContext,
        state: CounterRuleState | None = None,
    ) -> SequenceCandidateRuleUpdate[CounterRuleState]:
        del samples
        if state is None:
            state = CounterRuleState(torch.zeros_like(context.lengths))
        state = CounterRuleState(state.values + (~context.finished).long())
        return SequenceCandidateRuleUpdate(state=state, finished=state.values >= 2)


class RejectFirstBatchConstraint(BaseSequenceConstraint[None]):
    def forward(
        self,
        scores: torch.Tensor,
        context: SequenceSamplingContext,
        state: None = None,
    ) -> torch.Tensor:
        scores = scores.clone()
        scores[context.batch_indices == 0] = -torch.inf
        return scores


class FirstBatchEndOfSequenceCriterion(EndOfSequenceStoppingCriterion):
    def forward(self, context: SequenceSamplingContext) -> torch.Tensor:
        return super().forward(context) & context.batch_indices.eq(0)


class ForwardOnlyHypothesisScorer(BaseSequenceHypothesisScorer):
    def forward(self, context: SequenceHypothesisScoringContext) -> torch.Tensor:
        return context.sequence_scores


class TestGreedySequenceSampler:
    def test_constructs_from_config(self) -> None:
        sampler = COLT_BUILDER(
            {
                "type": "greedy",
                "adapter": {"type": "scheduled_logits_test"},
                "max_steps": 4,
            },
            BaseSequenceSampler,
        )

        assert isinstance(sampler, GreedySequenceSampler)
        assert isinstance(sampler.adapter, ScheduledLogitsAdapter)

    def test_default_adapter_constructs_from_config(self) -> None:
        sampler = COLT_BUILDER(
            {
                "type": "greedy",
                "adapter": {"type": "default"},
                "max_steps": 4,
            },
            BaseSequenceSampler,
        )

        assert isinstance(sampler, GreedySequenceSampler)
        assert isinstance(sampler.adapter, DefaultSequenceSamplingAdapter)

    def test_rules_and_criteria_construct_from_config(self) -> None:
        sampler = COLT_BUILDER(
            {
                "type": "greedy",
                "adapter": {"type": "scheduled_logits_test"},
                "candidate_rules": [{"type": "finish_after_two_test"}],
                "stopping_criteria": [
                    {
                        "type": "end_of_sequence",
                        "end_index": 2,
                    }
                ],
                "max_steps": 4,
            },
            BaseSequenceSampler,
        )

        assert isinstance(sampler.candidate_rules[0], FinishAfterTwoCandidatesConstraint)
        assert isinstance(sampler.stopping_criteria[0], EndOfSequenceStoppingCriterion)

    def test_batched_sampling_stops_after_every_sequence_reaches_eos(self) -> None:
        model = ScheduledLogitsModel()
        sampler = GreedySequenceSampler(
            adapter=ScheduledLogitsAdapter(),
            max_steps=4,
            stopping_criteria=[EndOfSequenceStoppingCriterion(end_index=2)],
        )
        logits = torch.tensor(
            [
                [[4.0, 1.0, 0.0], [0.0, 1.0, 4.0], [0.0, 4.0, 1.0], [4.0, 0.0, 1.0]],
                [[0.0, 4.0, 1.0], [4.0, 1.0, 0.0], [0.0, 1.0, 4.0], [4.0, 0.0, 1.0]],
            ]
        )

        output = sampler(model, logits)

        assert output.sequences.shape == (2, 1, 3)
        assert torch.equal(output.sequences[:, 0], torch.tensor([[0, 2, 0], [1, 0, 2]]))
        assert torch.equal(output.lengths, torch.tensor([[2], [3]]))
        assert torch.equal(output.mask[:, 0], torch.tensor([[True, True, False], [True, True, True]]))
        assert torch.equal(
            output.termination_reasons,
            torch.tensor([[SequenceTerminationReason.END_OF_SEQUENCE], [SequenceTerminationReason.END_OF_SEQUENCE]]),
        )
        assert output.sequence_scores is not None
        assert output.sequence_scores.shape == (2, 1)
        assert torch.equal(output.best_sequences.ids, output.sequences[:, 0])
        assert torch.equal(output.best_sequences.mask, output.mask[:, 0])
        assert len(output.best_sequences) == 2

    def test_sequence_batch_validates_nbest_rank(self) -> None:
        model = ScheduledLogitsModel()
        sampler = GreedySequenceSampler(adapter=ScheduledLogitsAdapter(), max_steps=1)
        output = sampler(model, torch.tensor([[[3.0, 1.0]]]))

        assert torch.equal(output.get_sequence_batch(0).ids, torch.tensor([[0]]))
        with pytest.raises(IndexError, match="rank"):
            output.get_sequence_batch(1)

    def test_runtime_params_partially_override_defaults(self) -> None:
        model = ScheduledLogitsModel()
        sampler = GreedySequenceSampler(adapter=ScheduledLogitsAdapter(), max_steps=4)
        logits = torch.tensor([[[3.0, 1.0], [1.0, 3.0], [3.0, 1.0], [1.0, 3.0]]])

        output = sampler(model, logits, params=SequenceSamplerParams(max_steps=3))

        assert torch.equal(output.sequences, torch.tensor([[[0, 1, 0]]]))
        assert torch.equal(output.lengths, torch.tensor([[3]]))

    def test_constructor_requires_positive_max_steps(self) -> None:
        with pytest.raises(ValueError, match="max_steps must be greater than zero"):
            GreedySequenceSampler(adapter=ScheduledLogitsAdapter(), max_steps=0)

    def test_runtime_max_steps_must_be_positive(self) -> None:
        sampler = GreedySequenceSampler(adapter=ScheduledLogitsAdapter(), max_steps=4)

        with pytest.raises(ValueError, match="max_steps must be greater than zero"):
            sampler(
                ScheduledLogitsModel(),
                torch.randn(1, 1, 2),
                params=SequenceSamplerParams(max_steps=0),
            )

    def test_default_adapter_supports_standard_model_contract(self) -> None:
        model = StandardSamplingModel()
        sampler = GreedySequenceSampler(
            adapter=DefaultSequenceSamplingAdapter(),
            max_steps=3,
        )
        logits = torch.tensor([[[3.0, 1.0], [1.0, 3.0], [3.0, 1.0]]])

        output = sampler(model, logits)

        assert torch.equal(output.sequences, torch.tensor([[[0, 1, 0]]]))
        assert torch.equal(output.lengths, torch.tensor([[3]]))

    def test_stateful_constraint_can_finish_generation(self) -> None:
        sampler = GreedySequenceSampler(
            adapter=ScheduledLogitsAdapter(),
            max_steps=4,
            candidate_rules=[FinishAfterTwoCandidatesConstraint()],
        )
        logits = torch.tensor([[[3.0, 1.0], [1.0, 3.0], [3.0, 1.0], [1.0, 3.0]]])

        output = sampler(ScheduledLogitsModel(), logits)

        assert torch.equal(output.sequences, torch.tensor([[[0, 1]]]))
        assert torch.equal(output.lengths, torch.tensor([[2]]))
        assert torch.equal(
            output.termination_reasons,
            torch.tensor([[SequenceTerminationReason.CONSTRAINT_SATISFIED]]),
        )

    def test_constraint_dead_end_is_distinguished_from_completion(self) -> None:
        sampler = GreedySequenceSampler(
            adapter=ScheduledLogitsAdapter(),
            max_steps=2,
            candidate_rules=[RejectFirstBatchConstraint()],
        )
        logits = torch.tensor(
            [
                [[3.0, 1.0], [1.0, 3.0]],
                [[1.0, 3.0], [3.0, 1.0]],
            ]
        )

        output = sampler(ScheduledLogitsModel(), logits)

        assert torch.equal(output.lengths, torch.tensor([[0], [2]]))
        assert torch.equal(output.mask[:, 0], torch.tensor([[False, False], [True, True]]))
        assert torch.equal(
            output.termination_reasons,
            torch.tensor([[SequenceTerminationReason.CONSTRAINT_DEAD_END], [SequenceTerminationReason.MAX_STEPS]]),
        )


class TestBeamSearchSequenceSampler:
    @pytest.mark.parametrize(
        "hypothesis_scorer",
        [CumulativeLogProbabilityScorer(), LengthPenaltySequenceHypothesisScorer(alpha=0.6)],
        ids=["cumulative", "length_penalty"],
    )
    def test_matches_exhaustive_search_and_safe_termination(
        self,
        hypothesis_scorer: BaseSequenceHypothesisScorer,
    ) -> None:
        # Keep frontier scores distinct because torch.topk does not guarantee
        # a stable ordering for ties across platforms.
        logits = torch.tensor([[[2.0, 1.0, 0.0], [1.5, 0.8, 0.6], [1.0, 0.4, 0.0]]])
        log_probs = torch.log_softmax(logits[0], dim=-1)
        exhaustive: list[tuple[float, float, tuple[int, ...]]] = []

        def enumerate_sequences(prefix: tuple[int, ...], raw_score: float) -> None:
            step = len(prefix)
            for token in range(log_probs.size(-1)):
                sequence = (*prefix, token)
                score = raw_score + float(log_probs[step, token])
                finished = token == 2 or len(sequence) == log_probs.size(0)
                if finished:
                    context = SequenceHypothesisScoringContext(
                        sequence_scores=torch.tensor([score]),
                        lengths=torch.tensor([len(sequence)]),
                        finished=torch.tensor([token == 2]),
                        max_steps=log_probs.size(0),
                    )
                    ranking_score = float(hypothesis_scorer(context)[0])
                    exhaustive.append((ranking_score, score, sequence))
                else:
                    enumerate_sequences(sequence, score)

        enumerate_sequences((), 0.0)
        expected = sorted(exhaustive, reverse=True)[:3]

        def sample(policy: AllBeamsFinishedTerminationPolicy | ScoreBoundTerminationPolicy):
            sampler = BeamSearchSequenceSampler(
                adapter=ScheduledLogitsAdapter(),
                max_steps=3,
                beam_size=3,
                num_return_sequences=3,
                hypothesis_scorer=hypothesis_scorer,
                termination_policy=policy,
                stopping_criteria=[EndOfSequenceStoppingCriterion(end_index=2)],
            )
            return sampler(ScheduledLogitsModel(), logits)

        full_output = sample(AllBeamsFinishedTerminationPolicy())
        bounded_output = sample(ScoreBoundTerminationPolicy())

        for output in (full_output, bounded_output):
            assert output.sequence_scores is not None
            assert output.ranking_scores is not None
            for rank, (expected_ranking, expected_raw, expected_sequence) in enumerate(expected):
                length = int(output.lengths[0, rank])
                assert tuple(output.sequences[0, rank, :length].tolist()) == expected_sequence
                assert output.sequence_scores[0, rank].item() == pytest.approx(expected_raw)
                assert output.ranking_scores[0, rank].item() == pytest.approx(expected_ranking)

    def test_constructs_from_config(self) -> None:
        sampler = COLT_BUILDER(
            {
                "type": "beam_search",
                "adapter": {"type": "branching_logits_test"},
                "max_steps": 2,
                "beam_size": 2,
                "num_return_sequences": 2,
                "hypothesis_scorer": {"type": "length_penalty", "alpha": 0.6},
                "termination_policy": {"type": "score_bound"},
            },
            BaseSequenceSampler,
        )

        assert isinstance(sampler, BeamSearchSequenceSampler)
        assert isinstance(sampler.adapter, BranchingLogitsAdapter)
        assert isinstance(sampler.hypothesis_scorer, LengthPenaltySequenceHypothesisScorer)
        assert isinstance(sampler.termination_policy, ScoreBoundTerminationPolicy)

    def test_returns_ranked_nbest_and_reorders_decoder_state(self) -> None:
        sampler = BeamSearchSequenceSampler(
            adapter=BranchingLogitsAdapter(),
            max_steps=2,
            beam_size=2,
            num_return_sequences=2,
        )

        output = sampler(ScheduledLogitsModel(), torch.zeros(2, 1))

        assert output.sequences.shape == (2, 2, 2)
        assert torch.equal(output.sequences[:, 0], torch.tensor([[1, 0], [1, 0]]))
        assert torch.equal(output.sequences[:, 1, 0], torch.zeros(2, dtype=torch.long))
        assert output.sequence_scores is not None
        assert output.ranking_scores is not None
        assert torch.all(output.sequence_scores[:, 0] >= output.sequence_scores[:, 1])
        assert torch.equal(output.lengths, torch.full((2, 2), 2))
        assert torch.all(output.mask)

    def test_beam_size_one_matches_greedy(self) -> None:
        inputs = torch.tensor([[[3.0, 1.0], [1.0, 3.0], [3.0, 1.0]]])
        greedy = GreedySequenceSampler(adapter=ScheduledLogitsAdapter(), max_steps=3)
        beam = BeamSearchSequenceSampler(
            adapter=ScheduledLogitsAdapter(),
            max_steps=3,
            beam_size=1,
        )

        greedy_output = greedy(ScheduledLogitsModel(), inputs)
        beam_output = beam(ScheduledLogitsModel(), inputs)

        assert torch.equal(beam_output.sequences, greedy_output.sequences)
        assert torch.equal(beam_output.lengths, greedy_output.lengths)
        assert torch.equal(beam_output.sequence_scores, greedy_output.sequence_scores)

    def test_finished_hypotheses_are_retained_without_expansion(self) -> None:
        sampler = BeamSearchSequenceSampler(
            adapter=BranchingLogitsAdapter(),
            max_steps=3,
            beam_size=2,
            num_return_sequences=2,
            stopping_criteria=[EndOfSequenceStoppingCriterion(end_index=1)],
        )

        output = sampler(ScheduledLogitsModel(), torch.zeros(1, 1))

        assert torch.equal(output.lengths, torch.tensor([[1, 2]]))
        assert torch.equal(output.mask[0, 0], torch.tensor([True, False]))
        assert output.termination_reasons is not None
        assert output.termination_reasons[0, 0] == SequenceTerminationReason.END_OF_SEQUENCE
        assert output.termination_reasons[0, 1] == SequenceTerminationReason.END_OF_SEQUENCE

    def test_reorders_stateful_candidate_rule_state(self) -> None:
        sampler = BeamSearchSequenceSampler(
            adapter=BranchingLogitsAdapter(),
            max_steps=3,
            beam_size=2,
            num_return_sequences=2,
            candidate_rules=[FinishAfterTwoCandidatesConstraint()],
        )

        output = sampler(ScheduledLogitsModel(), torch.zeros(1, 1))

        assert torch.equal(output.lengths, torch.full((1, 2), 2))
        assert output.termination_reasons is not None
        assert torch.all(output.termination_reasons.eq(SequenceTerminationReason.CONSTRAINT_SATISFIED))

    def test_termination_policy_prunes_unfinished_hypotheses(self) -> None:
        sampler = BeamSearchSequenceSampler(
            adapter=BranchingLogitsAdapter(),
            max_steps=3,
            beam_size=2,
            stopping_criteria=[EndOfSequenceStoppingCriterion(end_index=1)],
            termination_policy=EnoughFinishedHypothesesTerminationPolicy(),
        )

        output = sampler(ScheduledLogitsModel(), torch.zeros(1, 1))

        assert torch.equal(output.sequences, torch.tensor([[[1]]]))
        assert torch.equal(output.lengths, torch.tensor([[1]]))
        assert output.termination_reasons is not None
        assert output.termination_reasons[0, 0] == SequenceTerminationReason.END_OF_SEQUENCE

    def test_one_input_can_terminate_while_the_rest_of_the_batch_continues(self) -> None:
        sampler = BeamSearchSequenceSampler(
            adapter=BranchingLogitsAdapter(),
            max_steps=3,
            beam_size=2,
            stopping_criteria=[FirstBatchEndOfSequenceCriterion(end_index=1)],
            termination_policy=EnoughFinishedHypothesesTerminationPolicy(),
        )

        output = sampler(ScheduledLogitsModel(), torch.zeros(2, 1))

        assert torch.equal(output.lengths, torch.tensor([[1], [3]]))
        assert output.termination_reasons is not None
        assert output.termination_reasons[0, 0] == SequenceTerminationReason.END_OF_SEQUENCE
        assert output.termination_reasons[1, 0] == SequenceTerminationReason.MAX_STEPS

    def test_runtime_params_override_beam_defaults(self) -> None:
        sampler = BeamSearchSequenceSampler(
            adapter=BranchingLogitsAdapter(),
            max_steps=3,
            beam_size=2,
        )

        output = sampler(
            ScheduledLogitsModel(),
            torch.zeros(1, 1),
            params=BeamSearchSequenceSamplerParams(
                max_steps=2,
                beam_size=2,
                num_return_sequences=2,
            ),
        )

        assert output.sequences.shape == (1, 2, 2)

    def test_validates_parameters(self) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            BeamSearchSequenceSampler(adapter=BranchingLogitsAdapter(), max_steps=0, beam_size=2)
        with pytest.raises(ValueError, match="beam_size"):
            BeamSearchSequenceSampler(adapter=BranchingLogitsAdapter(), max_steps=2, beam_size=0)
        with pytest.raises(ValueError, match="num_return_sequences"):
            BeamSearchSequenceSampler(
                adapter=BranchingLogitsAdapter(),
                max_steps=2,
                beam_size=2,
                num_return_sequences=3,
            )


class TestSequenceHypothesisScorer:
    def test_custom_scorer_gets_a_safe_default_upper_bound(self) -> None:
        scorer = ForwardOnlyHypothesisScorer()
        context = SequenceHypothesisScoringContext(
            sequence_scores=torch.tensor([-1.0, -2.0]),
            lengths=torch.tensor([1, 2]),
            finished=torch.tensor([False, False]),
            max_steps=3,
        )

        result = scorer.upper_bound(context)

        assert torch.isposinf(result).all()

    def test_cumulative_log_probability_is_identity(self) -> None:
        scores = torch.tensor([-1.0, -2.0])
        scorer = CumulativeLogProbabilityScorer()

        result = scorer(
            SequenceHypothesisScoringContext(
                sequence_scores=scores,
                lengths=torch.tensor([1, 2]),
                finished=torch.tensor([True, False]),
                max_steps=3,
            )
        )

        assert torch.equal(result, scores)

    def test_length_penalty_normalizes_by_length(self) -> None:
        scorer = LengthPenaltySequenceHypothesisScorer(alpha=1.0)

        result = scorer(
            SequenceHypothesisScoringContext(
                sequence_scores=torch.tensor([-1.0, -1.0]),
                lengths=torch.tensor([1, 7]),
                finished=torch.tensor([True, True]),
                max_steps=7,
            )
        )

        assert torch.allclose(result, torch.tensor([-1.0, -0.5]))

    def test_length_penalty_requires_non_negative_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            LengthPenaltySequenceHypothesisScorer(alpha=-0.1)

    def test_length_penalty_upper_bound_uses_max_length_for_unfinished_hypotheses(self) -> None:
        scorer = LengthPenaltySequenceHypothesisScorer(alpha=1.0)
        context = SequenceHypothesisScoringContext(
            sequence_scores=torch.tensor([-1.0, -1.0]),
            lengths=torch.tensor([1, 1]),
            finished=torch.tensor([True, False]),
            max_steps=7,
        )

        result = scorer.upper_bound(context)

        assert torch.allclose(result, torch.tensor([-1.0, -0.5]))


class TestBeamSearchTerminationPolicy:
    @staticmethod
    def context(
        ranking_scores: torch.Tensor,
        upper_bound_scores: torch.Tensor,
        finished: torch.Tensor,
        num_return_sequences: int = 1,
    ) -> BeamSearchTerminationContext:
        return BeamSearchTerminationContext(
            sequence_scores=ranking_scores,
            ranking_scores=ranking_scores,
            upper_bound_scores=upper_bound_scores,
            lengths=torch.ones_like(ranking_scores, dtype=torch.long),
            finished=finished,
            step=1,
            max_steps=4,
            num_return_sequences=num_return_sequences,
        )

    def test_all_finished_requires_every_beam(self) -> None:
        context = self.context(
            ranking_scores=torch.tensor([[-1.0, -2.0], [-1.0, -2.0]]),
            upper_bound_scores=torch.tensor([[-1.0, -2.0], [-1.0, -2.0]]),
            finished=torch.tensor([[True, False], [True, True]]),
        )

        result = AllBeamsFinishedTerminationPolicy()(context)

        assert torch.equal(result, torch.tensor([False, True]))

    def test_enough_finished_is_explicitly_heuristic(self) -> None:
        context = self.context(
            ranking_scores=torch.tensor([[-2.0, -1.0]]),
            upper_bound_scores=torch.tensor([[-2.0, -1.0]]),
            finished=torch.tensor([[True, False]]),
        )

        result = EnoughFinishedHypothesesTerminationPolicy()(context)

        assert torch.equal(result, torch.tensor([True]))

    def test_score_bound_requires_unfinished_upper_bound_to_be_worse(self) -> None:
        context = self.context(
            ranking_scores=torch.tensor([[-1.0, -2.0], [-2.0, -1.0]]),
            upper_bound_scores=torch.tensor([[-1.0, -2.0], [-2.0, -1.0]]),
            finished=torch.tensor([[True, False], [True, False]]),
        )

        result = ScoreBoundTerminationPolicy()(context)

        assert torch.equal(result, torch.tensor([True, False]))


class TestArgmaxLabelSampler:
    def test_basic_sampling(self):
        """Test argmax label sampling with 2D logits."""
        sampler = ArgmaxLabelSampler()
        logits = torch.tensor([[1.0, 3.0, 2.0], [0.5, 0.2, 0.8], [2.0, 1.5, 3.5]])

        labels = sampler(logits)

        assert labels.shape == (3,)
        assert torch.equal(labels, torch.tensor([1, 2, 2]))

    def test_deterministic(self):
        """Test that argmax sampling is deterministic."""
        sampler = ArgmaxLabelSampler()
        logits = torch.randn(10, 5)

        labels1 = sampler(logits)
        labels2 = sampler(logits)

        assert torch.equal(labels1, labels2)

    def test_3d_logits(self):
        """Test argmax sampling with 3D logits (e.g., sequence labeling)."""
        sampler = ArgmaxLabelSampler()
        batch_size, seq_len, num_classes = 4, 8, 10
        logits = torch.randn(batch_size, seq_len, num_classes)

        labels = sampler(logits)

        assert labels.shape == (batch_size, seq_len)
        # Verify each label is in valid range
        assert torch.all(labels >= 0)
        assert torch.all(labels < num_classes)

    def test_single_sample(self):
        """Test argmax sampling with single sample."""
        sampler = ArgmaxLabelSampler()
        logits = torch.tensor([[1.0, 5.0, 2.0]])

        labels = sampler(logits)

        assert labels.shape == (1,)
        assert labels.item() == 1

    def test_ties(self):
        """Test argmax behavior with tied logits."""
        sampler = ArgmaxLabelSampler()
        # PyTorch argmax returns the first occurrence
        logits = torch.tensor([[2.0, 2.0, 1.0], [1.0, 3.0, 3.0]])

        labels = sampler(logits)

        assert labels.shape == (2,)
        assert labels[0].item() == 0  # First occurrence of max
        assert labels[1].item() == 1  # First occurrence of max


class TestMultinomialLabelSampler:
    def test_basic_sampling(self):
        """Test multinomial label sampling."""
        sampler = MultinomialLabelSampler()
        batch_size, num_classes = 20, 5
        logits = torch.randn(batch_size, num_classes)

        labels = sampler(logits)

        assert labels.shape == (batch_size,)
        assert torch.all(labels >= 0)
        assert torch.all(labels < num_classes)

    def test_stochastic(self):
        """Test that multinomial sampling is stochastic."""
        sampler = MultinomialLabelSampler()
        logits = torch.randn(100, 10)

        labels1 = sampler(logits)
        labels2 = sampler(logits)

        # With high probability, at least some labels should differ
        assert not torch.equal(labels1, labels2)

    def test_temperature_low(self):
        """Test multinomial sampling with low temperature (more deterministic)."""
        sampler = MultinomialLabelSampler()
        # Logits with clear winner
        logits = torch.tensor([[10.0, 1.0, 1.0]] * 50)

        labels = sampler(logits, params={"temperature": 0.1})

        # With low temperature, should mostly pick the highest logit
        assert (labels == 0).float().mean() > 0.8

    def test_temperature_high(self):
        """Test multinomial sampling with high temperature (more random)."""
        sampler = MultinomialLabelSampler()
        # Logits with clear winner
        logits = torch.tensor([[10.0, 1.0, 1.0]] * 100)

        labels = sampler(logits, params={"temperature": 10.0})

        # With high temperature, distribution should be more uniform
        # Not all samples should be 0
        unique_labels = torch.unique(labels)
        assert len(unique_labels) > 1

    def test_temperature_default(self):
        """Test multinomial sampling with default temperature."""
        sampler = MultinomialLabelSampler()
        logits = torch.randn(10, 5)

        labels = sampler(logits)

        assert labels.shape == (10,)
        assert torch.all(labels >= 0)
        assert torch.all(labels < 5)

    def test_3d_logits(self):
        """Test multinomial sampling with 3D logits."""
        sampler = MultinomialLabelSampler()
        batch_size, seq_len, num_classes = 4, 8, 10
        logits = torch.randn(batch_size, seq_len, num_classes)

        labels = sampler(logits)

        assert labels.shape == (batch_size, seq_len)
        assert torch.all(labels >= 0)
        assert torch.all(labels < num_classes)


class TestThresholdMultilabelSampler:
    def test_default_threshold(self):
        """Test threshold multilabel sampling with default threshold."""
        sampler = ThresholdMultilabelSampler()
        # Create logits that will produce probs around 0.5
        logits = torch.tensor([[0.0, 5.0, -5.0], [2.0, -2.0, 0.5]])

        labels = sampler(logits)

        assert labels.shape == (2, 3)
        # Check that outputs are binary
        assert torch.all((labels == 0.0) | (labels == 1.0))

    def test_custom_threshold(self):
        """Test threshold multilabel sampling with custom threshold."""
        sampler = ThresholdMultilabelSampler(threshold=0.7)
        # Create logits that map to known probabilities
        logits = torch.tensor([[1.0, -1.0, 2.0], [0.0, 3.0, -2.0]])

        labels = sampler(logits)

        assert labels.shape == (2, 3)
        # Verify threshold behavior
        probs = torch.sigmoid(logits)
        expected = (probs >= 0.7).float()
        assert torch.equal(labels, expected)

    def test_all_above_threshold(self):
        """Test when all logits are above threshold."""
        sampler = ThresholdMultilabelSampler(threshold=0.3)
        logits = torch.tensor([[5.0, 5.0, 5.0]])

        labels = sampler(logits)

        assert torch.all(labels == 1.0)

    def test_all_below_threshold(self):
        """Test when all logits are below threshold."""
        sampler = ThresholdMultilabelSampler(threshold=0.8)
        logits = torch.tensor([[-5.0, -5.0, -5.0]])

        labels = sampler(logits)

        assert torch.all(labels == 0.0)

    def test_deterministic(self):
        """Test that threshold sampling is deterministic."""
        sampler = ThresholdMultilabelSampler(threshold=0.5)
        logits = torch.randn(10, 8)

        labels1 = sampler(logits)
        labels2 = sampler(logits)

        assert torch.equal(labels1, labels2)


class TestTopKMultilabelSampler:
    def test_k_equals_1(self):
        """Test top-k sampling with k=1."""
        sampler = TopKMultilabelSampler(k=1)
        logits = torch.tensor([[1.0, 3.0, 2.0], [0.5, 0.2, 0.8]])

        labels = sampler(logits)

        assert labels.shape == (2, 3)
        # Each row should have exactly 1 label
        assert torch.equal(labels.sum(dim=-1), torch.tensor([1.0, 1.0]))
        # Check which labels are selected
        assert torch.equal(labels[0], torch.tensor([0.0, 1.0, 0.0]))
        assert torch.equal(labels[1], torch.tensor([0.0, 0.0, 1.0]))

    def test_k_equals_3(self):
        """Test top-k sampling with k=3."""
        sampler = TopKMultilabelSampler(k=3)
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])

        labels = sampler(logits)

        assert labels.shape == (1, 5)
        # Should have exactly 3 labels
        assert labels.sum().item() == 3.0
        # Top 3 should be indices 1, 4, 2 (logits 5.0, 4.0, 3.0)
        expected = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0]])
        assert torch.equal(labels, expected)

    def test_k_equals_num_classes(self):
        """Test top-k when k equals number of classes."""
        sampler = TopKMultilabelSampler(k=3)
        logits = torch.tensor([[1.0, 2.0, 3.0]])

        labels = sampler(logits)

        # Should select all available classes
        assert labels.sum().item() == 3.0
        assert torch.all(labels == 1.0)

    def test_deterministic(self):
        """Test that top-k sampling is deterministic."""
        sampler = TopKMultilabelSampler(k=2)
        logits = torch.randn(10, 8)

        labels1 = sampler(logits)
        labels2 = sampler(logits)

        assert torch.equal(labels1, labels2)

    def test_ties(self):
        """Test top-k behavior with tied logits."""
        sampler = TopKMultilabelSampler(k=2)
        logits = torch.tensor([[2.0, 2.0, 1.0, 2.0]])

        labels = sampler(logits)

        # Should select exactly k=2 labels
        assert labels.sum().item() == 2.0

    def test_batch_dimension(self):
        """Test top-k with batch dimension."""
        sampler = TopKMultilabelSampler(k=2)
        batch_size = 5
        num_classes = 10
        logits = torch.randn(batch_size, num_classes)

        labels = sampler(logits)

        assert labels.shape == (batch_size, num_classes)
        # Each sample should have exactly k labels
        assert torch.all(labels.sum(dim=-1) == 2.0)


class TestBernoulliMultilabelSampler:
    def test_basic_sampling(self):
        """Test Bernoulli multilabel sampling."""
        sampler = BernoulliMultilabelSampler()
        logits = torch.tensor([[0.0, 5.0, -5.0], [2.0, -2.0, 0.5]])

        labels = sampler(logits)

        assert labels.shape == (2, 3)
        # Check that outputs are binary
        assert torch.all((labels == 0.0) | (labels == 1.0))

    def test_stochastic(self):
        """Test that Bernoulli sampling is stochastic."""
        sampler = BernoulliMultilabelSampler()
        logits = torch.zeros(50, 10)  # Equal probability for all

        labels1 = sampler(logits)
        labels2 = sampler(logits)

        # With high probability, at least some labels should differ
        assert not torch.equal(labels1, labels2)

    def test_extreme_logits(self):
        """Test Bernoulli sampling with extreme logits."""
        sampler = BernoulliMultilabelSampler()

        # Very high logits (prob ≈ 1)
        high_logits = torch.tensor([[10.0, 10.0, 10.0]] * 10)
        labels_high = sampler(high_logits)
        # Most should be 1
        assert labels_high.float().mean() > 0.9

        # Very low logits (prob ≈ 0)
        low_logits = torch.tensor([[-10.0, -10.0, -10.0]] * 10)
        labels_low = sampler(low_logits)
        # Most should be 0
        assert labels_low.float().mean() < 0.1

    def test_independent_sampling(self):
        """Test that Bernoulli sampling is independent per class."""
        sampler = BernoulliMultilabelSampler()
        # Different probabilities for each class
        logits = torch.tensor([[5.0, -5.0, 0.0]] * 100)

        labels = sampler(logits)

        # First class should be mostly 1
        assert labels[:, 0].float().mean() > 0.9
        # Second class should be mostly 0
        assert labels[:, 1].float().mean() < 0.1
        # Third class should be around 0.5
        assert 0.3 < labels[:, 2].float().mean() < 0.7

    def test_batch_dimension(self):
        """Test Bernoulli sampling with batch dimension."""
        sampler = BernoulliMultilabelSampler()
        batch_size = 8
        num_classes = 12
        logits = torch.randn(batch_size, num_classes)

        labels = sampler(logits)

        assert labels.shape == (batch_size, num_classes)
        assert torch.all((labels == 0.0) | (labels == 1.0))

    def test_3d_logits(self):
        """Test Bernoulli sampling with 3D logits."""
        sampler = BernoulliMultilabelSampler()
        batch_size, seq_len, num_classes = 4, 8, 10
        logits = torch.randn(batch_size, seq_len, num_classes)

        labels = sampler(logits)

        assert labels.shape == (batch_size, seq_len, num_classes)
        assert torch.all((labels == 0.0) | (labels == 1.0))
