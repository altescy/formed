"""Workflow integration for Flax model training.

This module provides workflow steps for training Flax models, allowing
them to be integrated into the formed workflow system with automatic
caching and dependency tracking.

Example:
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

from collections.abc import Sequence
from typing import Annotated, Optional

from colt import Lazy
from flax import nnx

from formed.common.rich import progress
from formed.workflow import WorkflowStepResultFlag, step, use_step_logger

from .model import BaseFlaxModel
from .random import use_rngs
from .training import FlaxTrainer
from .types import IDataLoader, IEvaluator, ItemT, ModelInputT, ModelOutputT, ModelParamsT


@step("flax::train", format="pickle")
def train_flax_model(
    model: Lazy[BaseFlaxModel],
    trainer: FlaxTrainer,
    train_dataset: Sequence[ItemT],
    val_dataset: Optional[Sequence[ItemT]] = None,
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

    Example:
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
        state = trainer.train(model.construct(), train_dataset, val_dataset)
    return nnx.merge(state.graphdef, state.params, *state.additional_states)


@step("flax::evaluate", format="json")
def evaluate_flax_model(
    model: BaseFlaxModel[ModelInputT, ModelOutputT, ModelParamsT],
    evaluator: IEvaluator[ModelInputT, ModelOutputT],
    dataset: list[ItemT],
    dataloader: IDataLoader[ItemT, ModelInputT],
    params: ModelParamsT | None = None,
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

    evaluator.reset()

    with progress(dataloader(dataset), desc="Evaluating model") as iterator:
        for inputs in iterator:
            output = model(inputs, params)
            evaluator.update(inputs, output)

    metrics = evaluator.compute()
    logger.info("Evaluation metrics: %s", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    return metrics
