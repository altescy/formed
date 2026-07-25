from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

import formed.integrations.ml.types as mlt
import formed.integrations.torch as ft
import formed.integrations.torch.modules as ftm

from .datamodules import Seq2SeqDataModule


@dataclass
class Seq2SeqTrainingOutput:
    logits: torch.Tensor
    predictions: torch.Tensor
    loss: torch.Tensor


SamplingParams: TypeAlias = ftm.SequenceSamplingModelParams[ftm.LSTMDecoderState]
ModelOutput: TypeAlias = Seq2SeqTrainingOutput | ftm.SequenceSamplingModelOutput[ftm.LSTMDecoderState]


@ft.BaseTorchModel.register("seq2seq::model")
class LSTMSeq2SeqModel(ft.BaseTorchModel[Seq2SeqDataModule[mlt.AsBatch], ModelOutput, SamplingParams]):
    def __init__(
        self,
        source_embedder: nn.Module,
        target_embedder: nn.Module,
        encoder: ftm.BaseSequenceEncoder,
        decoder_state_initializer: ftm.BaseSequenceDecoderStateInitializer[ftm.LSTMDecoderState],
        decoder: ftm.BaseSequenceDecoder[ftm.LSTMDecoderState, None],
        output_projection: nn.Module,
        target_pad_index: int,
        target_bos_index: int,
    ) -> None:
        super().__init__()
        self._target_pad_index = target_pad_index
        self._target_bos_index = target_bos_index
        self._source_embedder = source_embedder
        self._target_embedder = target_embedder
        self._encoder = encoder
        self._decoder_state_initializer = decoder_state_initializer
        self._decoder = decoder
        self._output_projection = output_projection

    def _encode(self, inputs: Seq2SeqDataModule[mlt.AsBatch]) -> ftm.LSTMDecoderState:
        source_ids = ft.ensure_torch_tensor(inputs.source.ids)
        source_mask = ft.ensure_torch_tensor(inputs.source.mask)
        embeddings = self._source_embedder(source_ids)
        encoder_outputs = self._encoder(embeddings, mask=source_mask)
        return self._decoder_state_initializer(encoder_outputs, mask=source_mask)

    def forward(
        self,
        inputs: Seq2SeqDataModule[mlt.AsBatch],
        params: SamplingParams | None = None,
    ) -> ModelOutput:
        if params is None:
            target_ids = ft.ensure_torch_tensor(inputs.target.ids)
            target_mask = ft.ensure_torch_tensor(inputs.target.mask)
            decoder_ids = target_ids[:, :-1]
            decoder_mask = target_mask[:, :-1]
            targets = target_ids[:, 1:]
            outputs, _ = self._decoder(
                self._target_embedder(decoder_ids),
                mask=decoder_mask,
                initial_state=self._encode(inputs),
            )
            logits = self._output_projection(outputs)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
                ignore_index=self._target_pad_index,
            )
            return Seq2SeqTrainingOutput(logits=logits, predictions=logits.argmax(dim=-1), loss=loss)

        state = self._encode(inputs) if params.state is None else params.state
        if params.sequences is None:
            source_ids = ft.ensure_torch_tensor(inputs.source.ids)
            token_ids = torch.full(
                (source_ids.size(0), 1),
                self._target_bos_index,
                dtype=torch.long,
                device=source_ids.device,
            )
        else:
            token_ids = params.sequences[:, -1:]
        outputs, state = self._decoder(self._target_embedder(token_ids), initial_state=state)
        return ftm.SequenceSamplingModelOutput(
            logits=self._output_projection(outputs[:, -1]),
            state=state,
        )
