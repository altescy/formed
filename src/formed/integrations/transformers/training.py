from collections.abc import Mapping
from typing import Any, Optional, Union

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from formed.workflow import use_step_logger


class MlflowTrainerCallback(TrainerCallback):  # type: ignore[misc]
    def __init__(self) -> None:
        from formed.integrations.mlflow.workflow import MlflowLogger

        self._mlflow_logger: Optional[MlflowLogger] = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        from formed.integrations.mlflow.workflow import use_mlflow_logger

        logger = use_step_logger(__name__)
        self._mlflow_logger = use_mlflow_logger()
        if self._mlflow_logger is None:
            logger.warning("MLflow logger is not available. Skipping logging.")

    def on_log(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Mapping[str, Any],
        model: Optional[torch.nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        if self._mlflow_logger is None:
            return

        logger = use_step_logger(__name__)

        if not state.is_world_process_zero:
            return

        for key, value in logs.items():
            numerical_value: Union[int, float]
            if isinstance(value, (int, float)):
                numerical_value = value
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                numerical_value = value.item()
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{value}" of type {type(value)} for key "{key}" as a metric. '
                    "MLflow's log_metric() only accepts float and int types so we dropped this attribute."
                )
                continue

            self._mlflow_logger.log_metric(key, numerical_value, step=state.global_step)
