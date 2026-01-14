"""Workflow steps for Sentence Transformers integration.

This module provides workflow steps for loading, training, and converting
sentence transformer models.

Available Steps:
    - `sentence_transformers::load`: Load a pre-trained sentence transformer model.
    - `sentence_transformers::train`: Train a sentence transformer model.
    - `sentence_transformers::convert_tokenizer`: Convert a sentence transformer tokenizer to a formed Tokenizer (requires ml integration).

"""

from collections.abc import Callable, Mapping
from contextlib import suppress
from os import PathLike
from pathlib import Path
from typing import Any, Generic, cast

import datasets
import minato
import torch
from colt import Lazy
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import SentenceEvaluator
from transformers import PreTrainedTokenizerBase, TrainerCallback
from transformers.data.data_collator import DataCollator
from transformers.trainer_utils import EvalPrediction

from formed.types import NotSpecified
from formed.workflow import Format, step, use_step_workdir

from .types import SentenceTransformerT
from .utils import load_sentence_transformer


@Format.register("sentence_transformer::model")
class SentenceTransformerFormat(Generic[SentenceTransformerT], Format[SentenceTransformerT]):
    def write(self, artifact: SentenceTransformerT, directory: Path) -> None:
        artifact.save_pretrained(str(directory / "model"))

    def read(self, directory: Path) -> SentenceTransformerT:
        return cast(SentenceTransformerT, SentenceTransformer(str(directory / "model")))


@step("sentence_transformers::load", cacheable=False)
def load_pretrained_model(
    model_name_or_path: str | PathLike,
    **kwargs: Any,
) -> SentenceTransformer:
    """Load a pre-trained sentence transformer model.

    Args:
        model_name_or_path: Model identifier or path to model directory.
        **kwargs: Additional arguments to pass to SentenceTransformer constructor.

    Returns:
        Loaded SentenceTransformer model.
    """
    with suppress(Exception):
        model_name_or_path = minato.cached_path(model_name_or_path)
    return SentenceTransformer(str(model_name_or_path), **kwargs)


@step("sentence_transformers::train", format=SentenceTransformerFormat())
def train_sentence_transformer(
    model: SentenceTransformer,
    loss: Mapping[str, Lazy[torch.nn.Module]] | Lazy[torch.nn.Module],
    args: Lazy[SentenceTransformerTrainingArguments],
    dataset: None
    | (
        datasets.Dataset
        | datasets.DatasetDict
        | Mapping[
            str,
            datasets.Dataset | datasets.DatasetDict,
        ]
    ) = None,
    loss_modifier: None
    | (
        Mapping[str, list[Lazy[torch.nn.Module]] | Lazy[torch.nn.Module]]
        | list[Lazy[torch.nn.Module]]
        | Lazy[torch.nn.Module]
    ) = None,
    data_collator: DataCollator | None = None,  # pyright: ignore[reportInvalidTypeForm]
    tokenizer: PreTrainedTokenizerBase | None = None,
    evaluator: SentenceEvaluator | list[SentenceEvaluator] | None = None,
    callbacks: list[TrainerCallback] | None = None,
    model_init: Callable[[], SentenceTransformer] | None = None,
    compute_metrics: Callable[[EvalPrediction], dict] | None = None,
    optimizers: tuple[
        Lazy[torch.optim.Optimizer] | None,
        Lazy[torch.optim.lr_scheduler.LambdaLR] | None,
    ] = (None, None),
    preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    train_dataset_key: str = "train",
    eval_dataset_key: str = "validation",
) -> SentenceTransformer:
    """Train a sentence transformer model.

    This step trains a SentenceTransformer model using the provided loss function,
    datasets, and training arguments.

    Args:
        model: SentenceTransformer model to train.
        loss: Loss function(s) for training (single or mapping by dataset key).
        args: Training arguments configuration.
        dataset: Training/validation datasets.
        loss_modifier: Optional modifier(s) to apply to the loss function.
        data_collator: Optional data collator for batching.
        tokenizer: Optional tokenizer.
        evaluator: Optional evaluator(s) for validation.
        callbacks: Optional training callbacks.
        model_init: Optional model initialization function.
        compute_metrics: Optional metrics computation function.
        optimizers: Optional optimizer and learning rate scheduler.
        preprocess_logits_for_metrics: Optional logits preprocessing function.
        train_dataset_key: Key for training dataset split.
        eval_dataset_key: Key for evaluation dataset split.

    Returns:
        Trained SentenceTransformer model.
    """
    workdir = use_step_workdir()

    args_ = args.construct(output_dir=str(workdir))

    if isinstance(dataset, datasets.Dataset):
        train_dataset = dataset
        eval_dataset = None
    else:
        train_dataset = dataset.get(train_dataset_key) if dataset and args_.do_train else None
        eval_dataset = dataset.get(eval_dataset_key) if dataset and args_.do_eval else None

    loss_: torch.nn.Module | dict[str, torch.nn.Module]
    if isinstance(loss, Mapping):
        loss_ = {k: ll.construct(model=model) for k, ll in loss.items()}
    else:
        loss_ = loss.construct(model=model)
    if loss_modifier:
        if isinstance(loss_modifier, Mapping):
            assert isinstance(loss_, dict)
            for k, m in loss_modifier.items():
                if not isinstance(m, list):
                    m = [m]
                for n in m:
                    loss_[k] = n.construct(model=model, loss=loss_[k])
        else:
            if not isinstance(loss_modifier, list):
                loss_modifier = [loss_modifier]
            if isinstance(loss_, dict):
                for k, ll in loss_.items():
                    for m in loss_modifier:
                        loss_[k] = m.construct(model=model, loss=ll)
            else:
                for m in loss_modifier:
                    loss_ = m.construct(model=model, loss=loss_)

    lazy_optimizer, lazy_lr_scheduler = optimizers
    optimizer = lazy_optimizer.construct(params=model.parameters()) if lazy_optimizer else None
    lr_scheduler = lazy_lr_scheduler.construct(optimizer=optimizer) if lazy_lr_scheduler else None

    trainer = SentenceTransformerTrainer(
        model=model,
        loss=loss_,
        args=args_,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        evaluator=evaluator,
        callbacks=callbacks,
        model_init=model_init,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler),  # type: ignore[arg-type]
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    return model


with suppress(ImportError):
    from formed.integrations.ml import Param, Tokenizer, TokenSequenceIndexer

    from .analyzers import SentenceTransformerAnalyzer

    @step("sentence_transformers::convert_tokenizer", format="json")
    def convert_tokenizer(
        model_name_or_path: str | PathLike,
        pad_token: str | None | NotSpecified = NotSpecified.VALUE,
        unk_token: str | None | NotSpecified = NotSpecified.VALUE,
        bos_token: str | None | NotSpecified = NotSpecified.VALUE,
        eos_token: str | None | NotSpecified = NotSpecified.VALUE,
        freeze: bool = True,
        accessor: str | Callable | None = None,
    ) -> Tokenizer:
        """Convert a sentence transformer model's tokenizer to a formed Tokenizer.

        This step extracts the tokenizer from a sentence transformer model and
        converts it into a formed Tokenizer with specified special tokens.

        Args:
            model_name_or_path: Model identifier or path to model directory.
            pad_token: Padding token (uses model default if not specified).
            unk_token: Unknown token (uses model default if not specified).
            bos_token: Beginning-of-sequence token (uses model default if not specified).
            eos_token: End-of-sequence token (uses model default if not specified).
            freeze: Whether to freeze the vocabulary.
            accessor: Optional accessor for token extraction.

        Returns:
            Converted formed Tokenizer.

        Raises:
            AssertionError: If pad_token is not specified and not available in the model.
        """
        model = load_sentence_transformer(model_name_or_path)

        def get_token(given: str | None | NotSpecified, default: Any) -> str | None:
            if not isinstance(given, NotSpecified):
                return given
            if isinstance(default, str):
                return default
            return None

        vocab = model.tokenizer.get_vocab().copy()
        pad_token = get_token(pad_token, getattr(model.tokenizer, "pad_token", None))
        unk_token = get_token(unk_token, getattr(model.tokenizer, "unk_token", None))
        bos_token = get_token(bos_token, getattr(model.tokenizer, "bos_token", None))
        eos_token = get_token(eos_token, getattr(model.tokenizer, "eos_token", None))

        assert isinstance(pad_token, str), "pad_token must be specified or available in the tokenizer"

        surface_indexer = TokenSequenceIndexer(
            vocab=vocab,
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            freeze=freeze,
        )
        analyzer = SentenceTransformerAnalyzer(model_name_or_path)
        return Tokenizer(
            surfaces=surface_indexer,
            analyzer=Param.cast(analyzer),
            accessor=accessor,
        )
