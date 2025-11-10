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
from typing import Optional

from flax import nnx

from formed.workflow import step

from .model import BaseFlaxModel
from .random import use_rngs
from .training import FlaxTrainer
from .types import ItemT


@step("flax::train")
def train_flax_model(
    model: BaseFlaxModel,
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
        state = trainer.train(model, train_dataset, val_dataset)
    return nnx.merge(state.graphdef, state.params, *state.additional_states)
