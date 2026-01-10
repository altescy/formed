"""Workflow integration for Flax model training.

This module provides workflow steps for training Flax models, allowing
them to be integrated into the formed workflow system with automatic
caching and dependency tracking.

Available Steps:
    - `flax::train`: Train a Flax model using the provided trainer.
    - `flax::evaluate`: Evaluate a Flax model on a dataset.

Examples:
    >>> from formed.integrations.flax import train_flax_model
    >>>
    >>> # In workflow configuration (jsonnet):
    >>> # {
    >>> #   steps: {
    >>> #     train: {
    >>> #       type: "flax::train",
    >>> #       model: { type: "my_model", ... },
    >>> #       trainer: { type: "flax_trainer", ... },
    >>> #       train_dataset: { type: "ref", ref: "preprocess" },
    >>> #       random_seed: 42
    >>> #     }
    >>> #   }
    >>> # }

"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, cast

import cloudpickle
import orbax.checkpoint
from colt import Lazy
from flax import nnx

from formed.common.ctxutils import closing
from formed.common.rich import progress
from formed.workflow import Format, WorkflowStepResultFlag, step, use_step_logger
from formed.workflow.colt import COLT_BUILDER, COLT_TYPEKEY
from formed.workflow.utils import WorkflowJSONDecoder, WorkflowJSONEncoder

from .model import BaseFlaxModel
from .random import use_rngs
from .training import FlaxTrainer
from .types import IDataLoader, IEvaluator, ItemT, ModelInputT, ModelOutputT, ModelParamsT


@Format.register("flax::model")
class FlaxModelFormat(Format[BaseFlaxModel]):
    def _get_config_path(self, directory: Path) -> Path:
        return directory / "config.json"

    def _get_pickle_path(self, directory: Path) -> Path:
        return directory / "model.pkl"

    def _get_checkpointer(self, directory: Path) -> orbax.checkpoint.CheckpointManager:
        return orbax.checkpoint.CheckpointManager(
            directory / "checkpoint",
            orbax.checkpoint.PyTreeCheckpointer(),
            options=orbax.checkpoint.CheckpointManagerOptions(create=True),
        )

    def write(self, artifact: BaseFlaxModel, directory: Path) -> None:
        if (config := getattr(artifact, "__model_config__", None)) is not None:
            config = dict(artifact.__model_config__)
            config[COLT_TYPEKEY] = f"{artifact.__class__.__module__}:{artifact.__class__.__name__}"
            del artifact.__model_config__
            self._get_config_path(directory).write_text(json.dumps(config, cls=WorkflowJSONEncoder))
            self._get_checkpointer(directory).save(0, artifact)
        else:
            with self._get_pickle_path(directory).open("wb") as f:
                cloudpickle.dump(artifact, f)

    def read(self, directory: Path) -> BaseFlaxModel:
        if (pickle_path := self._get_pickle_path(directory)).exists():
            with pickle_path.open("rb") as f:
                return cloudpickle.load(f)

        with use_rngs(0):
            model = COLT_BUILDER(
                json.loads(
                    self._get_config_path(directory).read_text(),
                    cls=WorkflowJSONDecoder,
                )
            )

        return cast(BaseFlaxModel, self._get_checkpointer(directory).restore(0, items=model))


@step("flax::train", format=FlaxModelFormat())
def train_flax_model(
    model: Lazy[BaseFlaxModel],
    trainer: FlaxTrainer,
    train_dataset: Sequence[ItemT],
    val_dataset: Sequence[ItemT] | None = None,
    random_seed: int = 0,
) -> BaseFlaxModel:
    """Train a Flax model using the provided trainer.

    This workflow step trains a Flax NNX model on the provided datasets,
    returning the trained model. The training process is cached based on
    the model architecture, trainer configuration, and dataset fingerprints.

    Args:
        model: Flax model to train.
        trainer: Trainer configuration with dataloaders and callbacks.
        train_dataset: Training dataset items.
        val_dataset: Optional validation dataset items.
        random_seed: Random seed for reproducibility.

    Returns:
        Trained Flax model with updated parameters.

    Examples:
        >>> # Use in Python code
        >>> trained_model = train_flax_model(
        ...     model=my_model,
        ...     trainer=trainer,
        ...     train_dataset=train_data,
        ...     val_dataset=val_data,
        ...     random_seed=42
        ... )

    """

    with use_rngs(random_seed):
        model_instance = model.construct()
        state = trainer.train(model_instance, train_dataset, val_dataset)

    model_instance = nnx.merge(state.graphdef, state.params, *state.additional_states)
    model_instance.__model_config__ = model.config
    return model_instance


@step("flax::evaluate", format="json")
def evaluate_flax_model(
    model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
    evaluator: IEvaluator[ModelInputT, ModelOutputT],
    dataset: list[ItemT],
    dataloader: IDataLoader[ItemT, ModelInputT],
    params: ModelParamsT | None = None,
    random_seed: int | None = None,
) -> Annotated[dict[str, float], WorkflowStepResultFlag.METRICS]:
    """Evaluate a Flax model on a dataset using the provided evaluator.

    Args:
        model: Flax model to evaluate.
        evaluator: Evaluator to compute metrics.
        dataset: Dataset items for evaluation.
        dataloader: DataLoader to convert items to model inputs.
        params: Optional model parameters to use for evaluation.

    Returns:
        Dictionary of computed evaluation metrics.
    """

    logger = use_step_logger(__name__)

    with use_rngs(random_seed):
        model.eval()
        evaluator.reset()

        with (
            closing(dataloader(dataset)) as loader,
            progress(loader, desc="Evaluating model") as iterator,
        ):
            for inputs in iterator:
                output = model(inputs, params)
                evaluator.update(inputs, output)

        metrics = evaluator.compute()
        logger.info("Evaluation metrics: %s", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    return metrics
