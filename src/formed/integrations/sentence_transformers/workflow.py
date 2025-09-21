from collections.abc import Callable, Mapping
from contextlib import suppress
from os import PathLike
from pathlib import Path
from typing import Any, Generic, Optional, Union, cast

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

from formed.workflow import Format, step, use_step_workdir

from .types import SentenceTransformerT


@Format.register("sentence_transformer")
class SentenceTransformerFormat(Generic[SentenceTransformerT], Format[SentenceTransformerT]):
    def write(self, artifact: SentenceTransformerT, directory: Path) -> None:
        artifact.save_pretrained(str(directory / "model"))

    def read(self, directory: Path) -> SentenceTransformerT:
        return cast(SentenceTransformerT, SentenceTransformer(str(directory / "model")))


@step("sentence_transformers::load_pretrained_model", cacheable=False)
def load_pretrained_model(
    model_name_or_path: Union[str, PathLike],
    **kwargs: Any,
) -> SentenceTransformer:
    with suppress(Exception):
        model_name_or_path = minato.cached_path(model_name_or_path)
    return SentenceTransformer(str(model_name_or_path), **kwargs)


@step("sentence_transformers::train", format=SentenceTransformerFormat())
def train_sentence_transformer(
    model: SentenceTransformer,
    loss: Union[Mapping[str, Lazy[torch.nn.Module]], Lazy[torch.nn.Module]],
    args: Lazy[SentenceTransformerTrainingArguments],
    dataset: Optional[
        Union[
            datasets.Dataset,
            datasets.DatasetDict,
            Mapping[
                str,
                Union[datasets.Dataset, datasets.DatasetDict],
            ],
        ]
    ] = None,
    loss_modifier: Optional[
        Union[
            Mapping[str, Union[list[Lazy[torch.nn.Module]], Lazy[torch.nn.Module]]],
            list[Lazy[torch.nn.Module]],
            Lazy[torch.nn.Module],
        ]
    ] = None,
    data_collator: Optional[DataCollator] = None,  # pyright: ignore[reportInvalidTypeForm]
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    evaluator: Optional[Union[SentenceEvaluator, list[SentenceEvaluator]]] = None,
    callbacks: Optional[list[TrainerCallback]] = None,
    model_init: Optional[Callable[[], SentenceTransformer]] = None,
    compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
    optimizers: tuple[
        Optional[Lazy[torch.optim.Optimizer]],
        Optional[Lazy[torch.optim.lr_scheduler.LambdaLR]],
    ] = (None, None),
    preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    train_dataset_key: str = "train",
    eval_dataset_key: str = "validation",
) -> SentenceTransformer:
    workdir = use_step_workdir()

    args_ = args.construct(output_dir=str(workdir))

    if isinstance(dataset, datasets.Dataset):
        train_dataset = dataset
        eval_dataset = None
    else:
        train_dataset = dataset.get(train_dataset_key) if dataset and args_.do_train else None
        eval_dataset = dataset.get(eval_dataset_key) if dataset and args_.do_eval else None

    loss_: Union[torch.nn.Module, dict[str, torch.nn.Module]]
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
