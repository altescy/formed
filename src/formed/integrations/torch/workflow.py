"""Workflow integration for PyTorch model training.

This module provides workflow steps for training PyTorch models, allowing
them to be integrated into the formed workflow system with automatic
caching and dependency tracking.

Example:
    >>> from formed.integrations.torch import train_torch_model
    >>>
    >>> # In workflow configuration (jsonnet):
    >>> # {
    >>> #   steps: {
    >>> #     train: {
    >>> #       type: "torch::train",
    >>> #       model: { type: "my_model", ... },
    >>> #       trainer: { type: "torch_trainer", ... },
    >>> #       train_dataset: { type: "ref", ref: "preprocess" },
    >>> #       random_seed: 42
    >>> #     }
    >>> #   }
    >>> # }

"""

from collections.abc import Sequence
from typing import Annotated, cast

import torch

from formed.common.rich import progress
from formed.workflow import WorkflowStepResultFlag, step, use_step_logger

from .model import BaseTorchModel
from .training import TorchTrainer
from .types import IDataLoader, IEvaluator, ItemT, ModelInputT, ModelOutputT, ModelParamsT


@step("torch::train", format="pickle")
def train_torch_model(
    model: BaseTorchModel,
    trainer: TorchTrainer,
    train_dataset: Sequence[ItemT],
    val_dataset: Sequence[ItemT] | None = None,
    random_seed: int = 0,
) -> BaseTorchModel:
    """Train a PyTorch model using the provided trainer.

    This workflow step trains a PyTorch model on the provided datasets,
    returning the trained model. The training process is cached based on
    the model architecture, trainer configuration, and dataset fingerprints.

    Args:
        model: PyTorch model to train.
        trainer: Trainer configuration with dataloaders and callbacks.
        train_dataset: Training dataset items.
        val_dataset: Optional validation dataset items.
        random_seed: Random seed for reproducibility.

    Returns:
        Trained PyTorch model with updated parameters.

    Example:
        >>> # Use in Python code
        >>> trained_model = train_torch_model(
        ...     model=my_model,
        ...     trainer=trainer,
        ...     train_dataset=train_data,
        ...     val_dataset=val_data,
        ...     random_seed=42
        ... )

    """
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Train the model
    state = trainer.train(model, train_dataset, val_dataset)

    # Return the trained model
    return cast(BaseTorchModel, state.model)


@step("torch::evaluate", format="json")
def evaluate_torch_model(
    model: BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT],
    evaluator: IEvaluator[ModelInputT, ModelOutputT],
    dataset: list[ItemT],
    dataloader: IDataLoader[ItemT, ModelInputT],
    params: ModelParamsT | None = None,
    random_seed: int | None = None,
) -> Annotated[dict[str, float], WorkflowStepResultFlag.METRICS]:
    """Evaluate a PyTorch model on a dataset using the provided evaluator.

    This workflow step evaluates a PyTorch model on the provided dataset,
    computing metrics using the evaluator. Evaluation is performed in
    evaluation mode (no gradient computation).

    Args:
        model: PyTorch model to evaluate.
        evaluator: Evaluator to compute metrics.
        dataset: Dataset items for evaluation.
        dataloader: DataLoader to convert items to model inputs.
        params: Optional model parameters to use for evaluation.
        random_seed: Optional random seed for reproducibility.

    Returns:
        Dictionary of computed evaluation metrics.

    Example:
        >>> # Use in Python code
        >>> metrics = evaluate_torch_model(
        ...     model=trained_model,
        ...     evaluator=my_evaluator,
        ...     dataset=test_data,
        ...     dataloader=test_loader
        ... )

    """
    logger = use_step_logger(__name__)

    # Set random seed if provided
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    # Set model to evaluation mode
    model.eval()
    evaluator.reset()

    # Evaluate model
    with torch.no_grad():
        with progress(dataloader(dataset), desc="Evaluating model") as iterator:
            for inputs in iterator:
                output = model(inputs, params)
                evaluator.update(inputs, output)

    metrics = evaluator.compute()
    logger.info("Evaluation metrics: %s", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    return metrics
