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

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, TypeVar, cast

import cloudpickle
import torch
from colt import Lazy

from formed.common.ctxutils import closing
from formed.common.rich import progress
from formed.workflow import Format, WorkflowStepResultFlag, step, use_step_logger
from formed.workflow.colt import COLT_BUILDER, COLT_TYPEKEY
from formed.workflow.utils import WorkflowJSONDecoder, WorkflowJSONEncoder

from .model import BaseTorchModel
from .training import TorchTrainer
from .types import IDataLoader, IEvaluator, ItemT, ModelInputT, ModelOutputT, ModelParamsT
from .utils import set_random_seed

_ModelT = TypeVar("_ModelT", bound=BaseTorchModel)


@Format.register("torch_model")
class TorchModelFormat(Format[_ModelT]):
    def _get_config_path(self, directory: Path) -> Path:
        return directory / "config.json"

    def _get_state_path(self, directory: Path) -> Path:
        return directory / "state.pth"

    def _get_pickle_path(self, directory: Path) -> Path:
        return directory / "model.pkl"

    def write(self, artifact: _ModelT, directory: Path) -> None:
        if artifact.__model_config__ is not None:
            config = dict(artifact.__model_config__)
            config[COLT_TYPEKEY] = f"{artifact.__class__.__module__}:{artifact.__class__.__name__}"
            self._get_config_path(directory).write_text(
                json.dumps(
                    artifact.__model_config__,
                    indent=2,
                    cls=WorkflowJSONEncoder,
                )
            )
            torch.save(
                artifact.state_dict(),
                self._get_state_path(directory),
            )
        else:
            with self._get_pickle_path(directory).open("wb") as f:
                cloudpickle.dump(artifact, f)

    def read(self, directory: Path) -> _ModelT:
        if (pickle_path := self._get_pickle_path(directory)).exists():
            with pickle_path.open("rb") as f:
                model = cloudpickle.load(f)
            return cast(_ModelT, model)

        config = json.loads(
            self._get_config_path(directory).read_text(),
            cls=WorkflowJSONDecoder,
        )
        state_dict = torch.load(self._get_state_path(directory))
        model = COLT_BUILDER(config)
        model.load_state_dict(state_dict)
        return model


@step("torch::train", format=TorchModelFormat())
def train_torch_model(
    model: Lazy[BaseTorchModel],
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
    set_random_seed(random_seed)

    # Build model from Lazy
    model_instance = model.construct()

    # Set config for selialization
    model_instance.__model_config__ = model.config

    # Train the model
    state = trainer.train(model_instance, train_dataset, val_dataset)

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
