import datetime
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import mlflow
import mlflow.artifacts
from mlflow.entities import Experiment as MlflowExperiment
from mlflow.entities import Run as MlflowRun
from mlflow.tracking import artifact_utils
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.mlflow_tags import (
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_RUN_NAME,
    MLFLOW_RUN_NOTE,
)

from formed.types import JsonValue
from formed.workflow import (
    WorkflowExecutionID,
    WorkflowExecutionInfo,
    WorkflowExecutionState,
    WorkflowExecutionStatus,
    WorkflowStepInfo,
    WorkflowStepState,
    WorkflowStepStatus,
)
from formed.workflow.utils import as_jsonvalue

MlflowParamValue = Union[int, float, str, None]
MlflowParams = dict[str, MlflowParamValue]


class WorkflowRunType(str, Enum):
    STEP = "step"
    EXECUTION = "execution"


class WorkflowCacheStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"


class MlflowRunStatus(str, Enum):
    RUNNING = "RUNNING"
    SCHEDULED = "SCHEDULED"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"

    @classmethod
    def from_step_status(cls, status: WorkflowStepStatus) -> "MlflowRunStatus":
        if status == WorkflowStepStatus.PENDING:
            return cls.SCHEDULED
        if status == WorkflowStepStatus.RUNNING:
            return cls.RUNNING
        if status == WorkflowStepStatus.COMPLETED:
            return cls.FINISHED
        if status == WorkflowStepStatus.FAILURE:
            return cls.FAILED
        if status == WorkflowStepStatus.CANCELED:
            return cls.KILLED
        raise ValueError(f"Invalid step status: {status}")

    @classmethod
    def from_execution_status(cls, status: WorkflowExecutionStatus) -> "MlflowRunStatus":
        if status == WorkflowExecutionStatus.PENDING:
            return cls.SCHEDULED
        if status == WorkflowExecutionStatus.RUNNING:
            return cls.RUNNING
        if status == WorkflowExecutionStatus.COMPLETED:
            return cls.FINISHED
        if status == WorkflowExecutionStatus.FAILURE:
            return cls.FAILED
        if status == WorkflowExecutionStatus.CANCELED:
            return cls.KILLED
        raise ValueError(f"Invalid execution status: {status}")

    def to_step_status(self) -> WorkflowStepStatus:
        if self == self.RUNNING:
            return WorkflowStepStatus.RUNNING
        if self == self.SCHEDULED:
            return WorkflowStepStatus.PENDING
        if self == self.FINISHED:
            return WorkflowStepStatus.COMPLETED
        if self == self.FAILED:
            return WorkflowStepStatus.FAILURE
        if self == self.KILLED:
            return WorkflowStepStatus.CANCELED
        raise ValueError(f"Invalid run status: {self}")

    def to_execution_status(self) -> WorkflowExecutionStatus:
        if self == self.RUNNING:
            return WorkflowExecutionStatus.RUNNING
        if self == self.SCHEDULED:
            return WorkflowExecutionStatus.PENDING
        if self == self.FINISHED:
            return WorkflowExecutionStatus.COMPLETED
        if self == self.FAILED:
            return WorkflowExecutionStatus.FAILURE
        if self == self.KILLED:
            return WorkflowExecutionStatus.CANCELED
        raise ValueError(f"Invalid run status: {self}")


class MlflowTag(str, Enum):
    # MLflow tags
    MLFLOW_PARENT_RUN_ID = MLFLOW_PARENT_RUN_ID
    MLFLOW_RUN_NAME = MLFLOW_RUN_NAME
    MLFLOW_RUN_NOTE = MLFLOW_RUN_NOTE

    # Custom tags
    MLFACTORY_RUN_TYPE = "formed.workflow.run_type"
    MLFACTORY_STEP_FINGERPRINT = "formed.workflow.step.fingerprint"
    MLFACTORY_STEP_CACHE_STATUS = "formed.workflow.step.cache_status"


def flatten_params(d: Mapping[str, JsonValue]) -> MlflowParams:
    result = {}
    for key, value in d.items():
        if isinstance(value, (list, tuple)):
            value = {str(i): v for i, v in enumerate(value)}
        if isinstance(value, dict):
            result.update({f"{key}.{k}": v for k, v in flatten_params(value).items()})
        else:
            result[key] = value
    return result


def build_filter_string(
    run_type: Optional[WorkflowRunType] = None,
    step_info: Optional[Union[str, WorkflowStepInfo]] = None,
    execution_info: Optional[Union[str, WorkflowExecutionInfo]] = None,
    additional_filters: Optional[str] = None,
) -> str:
    conditions: list[str] = []
    if run_type is not None:
        conditions.append(f"tags.{MlflowTag.MLFACTORY_RUN_TYPE.value} = '{run_type.value}'")
    if step_info is not None:
        step_fingerprint = step_info.fingerprint if isinstance(step_info, WorkflowStepInfo) else step_info
        conditions.append(f"tags.{MlflowTag.MLFACTORY_STEP_FINGERPRINT.value} = '{step_fingerprint}'")
    if execution_info is not None:
        execution_id = execution_info.id if isinstance(execution_info, WorkflowExecutionInfo) else execution_info
        conditions.append(f"tags.{MlflowTag.MLFLOW_RUN_NAME.value} = '{execution_id}'")
    if additional_filters is not None:
        conditions.append(additional_filters)
    return " AND ".join(conditions)


def is_mlflow_using_local_artifact_storage(
    mlflow_run: Union[str, MlflowRun],
) -> bool:
    mlflow_run_id = mlflow_run.info.run_id if isinstance(mlflow_run, MlflowRun) else mlflow_run
    mlflow_artifact_uri = urlparse(artifact_utils.get_artifact_uri(run_id=mlflow_run_id))  # type: ignore[no-untyped-call]
    return bool(mlflow_artifact_uri.scheme == "file")


def get_mlflow_local_artifact_storage_path(
    mlflow_run: Union[str, MlflowRun],
) -> Optional[Path]:
    mlflow_run_id = mlflow_run.info.run_id if isinstance(mlflow_run, MlflowRun) else mlflow_run
    mlflow_artifact_uri = urlparse(artifact_utils.get_artifact_uri(run_id=mlflow_run_id))  # type: ignore[no-untyped-call]
    if mlflow_artifact_uri.scheme == "file":
        return Path(mlflow_artifact_uri.path)
    return None


def fetch_child_mlflow_runs(
    client: MlflowClient,
    experiment: Union[str, MlflowExperiment],
    mlflow_run: Union[str, MlflowRun],
) -> Iterator[MlflowRun]:
    experiment = get_mlflow_experiment(experiment)
    mlflow_run_id = mlflow_run.info.run_id if isinstance(mlflow_run, MlflowRun) else mlflow_run
    page_token: Optional[str] = None
    while True:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.{MlflowTag.MLFLOW_PARENT_RUN_ID.value} = '{mlflow_run_id}'",
            page_token=page_token,
        )
        yield from iter(runs)
        if runs.token is None:
            break
        page_token = runs.token


def update_mlflow_tags(
    client: MlflowClient,
    run: Union[str, MlflowRun],
    tags: dict[MlflowTag, str],
) -> None:
    run_id = run.info.run_id if isinstance(run, MlflowRun) else run
    for tag, value in tags.items():
        client.set_tag(run_id=run_id, key=tag.value, value=value)


def fetch_mlflow_runs(
    client: MlflowClient,
    experiment: Union[str, MlflowExperiment],
    *,
    run_type: Optional[WorkflowRunType] = None,
    step_info: Optional[Union[str, WorkflowStepInfo]] = None,
    execution_info: Optional[Union[str, WorkflowExecutionInfo]] = None,
    with_children: bool = False,
) -> Iterator[MlflowRun]:
    experiment = get_mlflow_experiment(experiment)

    page_token: Optional[str] = None
    filter_string = build_filter_string(run_type, step_info, execution_info)
    while True:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            page_token=page_token,
        )
        for run in runs:
            yield run
            if with_children:
                yield from fetch_child_mlflow_runs(client, experiment, run)
        if runs.token is None:
            break
        page_token = runs.token


def fetch_mlflow_run(
    client: MlflowClient,
    experiment: Union[str, MlflowExperiment],
    *,
    run_type: Optional[WorkflowRunType] = None,
    step_info: Optional[Union[str, WorkflowStepInfo]] = None,
    execution_info: Optional[Union[str, WorkflowExecutionInfo]] = None,
) -> Optional[MlflowRun]:
    return next(
        fetch_mlflow_runs(
            client,
            experiment,
            run_type=run_type,
            step_info=step_info,
            execution_info=execution_info,
        ),
        None,
    )


def generate_new_execution_id(client: MlflowClient, experiment: Union[str, MlflowExperiment]) -> WorkflowExecutionID:
    experiment = get_mlflow_experiment(experiment)
    while True:
        execution_id = uuid.uuid4().hex[:8]
        if fetch_mlflow_run(client, experiment, execution_info=execution_id) is None:
            return WorkflowExecutionID(execution_id)


def add_mlflow_run(
    client: MlflowClient,
    experiment: Union[str, MlflowExperiment],
    step_or_execution_info: Union[WorkflowStepInfo, WorkflowExecutionInfo],
    parent_run_id: Optional[str] = None,
) -> MlflowRun:
    experiment = get_mlflow_experiment(experiment)

    run_name: str
    params: MlflowParams
    tags: dict[MlflowTag, str]

    if isinstance(step_or_execution_info, WorkflowStepInfo):
        run_name = step_or_execution_info.name
        params = get_step_params(step_or_execution_info)
        tags = get_step_tags(step_or_execution_info)
    elif isinstance(step_or_execution_info, WorkflowExecutionInfo):
        assert step_or_execution_info.id is not None
        run_name = step_or_execution_info.id
        params = get_execution_params(step_or_execution_info)
        tags = get_execution_tags(step_or_execution_info)
    else:
        raise ValueError(f"Unsupported type: {type(step_or_execution_info)}")

    if parent_run_id is not None:
        tags[MlflowTag.MLFLOW_PARENT_RUN_ID] = parent_run_id

    run = client.create_run(
        experiment_id=experiment.experiment_id,
        run_name=run_name,
        tags=context_registry.resolve_tags({tag.value: value for tag, value in tags.items()}),
    )
    for key, value in params.items():
        client.log_param(run.info.run_id, key, value)

    return run


def download_mlflow_artifacts(
    client: MlflowClient,
    experiment: Union[str, MlflowExperiment],
    step_or_execution_info: Union[WorkflowStepInfo, WorkflowExecutionInfo],
    directory: Union[str, PathLike],
    artifact_path: Optional[str] = None,
) -> None:
    run_type = (
        WorkflowRunType.STEP if isinstance(step_or_execution_info, WorkflowStepInfo) else WorkflowRunType.EXECUTION
    )
    step_info = step_or_execution_info if isinstance(step_or_execution_info, WorkflowStepInfo) else None
    execution_info = step_or_execution_info if isinstance(step_or_execution_info, WorkflowExecutionInfo) else None
    run = fetch_mlflow_run(
        client,
        experiment,
        run_type=run_type,
        step_info=step_info,
        execution_info=execution_info,
    )
    if run is None:
        raise FileNotFoundError("Run not found")
    directory = Path(directory)
    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id,
            artifact_path=artifact_path,
            dst_path=temp_dir,
        )
        download_path = Path(temp_dir)
        if artifact_path is not None:
            download_path = download_path / artifact_path
        directory.mkdir(parents=True, exist_ok=True)
        for path in download_path.glob("*"):
            shutil.move(path, directory / path.name)


def upload_mlflow_artifacts(
    client: MlflowClient,
    experiment: Union[str, MlflowExperiment],
    step_or_execution_info: Union[WorkflowStepInfo, WorkflowExecutionInfo],
    directory: Union[str, PathLike],
    artifact_path: Optional[str] = None,
) -> None:
    run = fetch_mlflow_run(
        client,
        experiment,
        step_info=step_or_execution_info if isinstance(step_or_execution_info, WorkflowStepInfo) else None,
        execution_info=step_or_execution_info if isinstance(step_or_execution_info, WorkflowExecutionInfo) else None,
    )
    if run is None:
        raise ValueError("Run not found")
    client.log_artifacts(
        run_id=run.info.run_id,
        local_dir=str(directory),
        artifact_path=artifact_path,
    )


def terminate_mlflow_run(
    client: MlflowClient,
    experiment: Union[str, MlflowExperiment],
    step_or_execution_state: Union[WorkflowStepState, WorkflowExecutionState],
) -> None:
    step_info: Optional[str] = None
    execution_info: Optional[str] = None
    run_status: MlflowRunStatus
    end_time: Optional[int] = None
    if isinstance(step_or_execution_state, WorkflowStepState):
        if step_or_execution_state.status not in (
            WorkflowStepStatus.COMPLETED,
            WorkflowStepStatus.FAILURE,
            WorkflowStepStatus.CANCELED,
        ):
            raise ValueError(f"Invalid step status: {step_or_execution_state.status}")
        step_info = step_or_execution_state.fingerprint
        run_status = MlflowRunStatus.from_step_status(step_or_execution_state.status)
        end_time = (
            int(step_or_execution_state.finished_at.timestamp() * 1000) if step_or_execution_state.finished_at else None
        )
    elif isinstance(step_or_execution_state, WorkflowExecutionState):
        if step_or_execution_state.status not in (
            WorkflowExecutionStatus.COMPLETED,
            WorkflowExecutionStatus.FAILURE,
            WorkflowExecutionStatus.CANCELED,
        ):
            raise ValueError(f"Invalid execution status: {step_or_execution_state.status}")
        execution_info = step_or_execution_state.execution_id
        run_status = MlflowRunStatus.from_execution_status(step_or_execution_state.status)
        end_time = (
            int(step_or_execution_state.finished_at.timestamp() * 1000) if step_or_execution_state.finished_at else None
        )
    experiment = get_mlflow_experiment(experiment)
    run = fetch_mlflow_run(
        client,
        experiment,
        step_info=step_info,
        execution_info=execution_info,
    )
    if run is None:
        raise ValueError("Run not found")
    client.set_terminated(
        run_id=run.info.run_id,
        status=run_status.value,
        end_time=end_time,
    )


def remove_mlflow_run(
    client: MlflowClient,
    experiment: Union[str, MlflowExperiment],
    step_or_execution_info: Union[WorkflowStepInfo, WorkflowExecutionInfo],
) -> None:
    run = fetch_mlflow_run(
        client,
        experiment,
        step_info=step_or_execution_info if isinstance(step_or_execution_info, WorkflowStepInfo) else None,
        execution_info=step_or_execution_info if isinstance(step_or_execution_info, WorkflowExecutionInfo) else None,
    )
    if run is None:
        raise ValueError("Run not found")
    client.delete_run(run.info.run_id)


def get_mlflow_experiment(experiment: Union[str, MlflowExperiment]) -> MlflowExperiment:
    if isinstance(experiment, str):
        client = MlflowClient()
        experiment_or_none = client.get_experiment_by_name(experiment)
        if experiment_or_none is None:
            mlflow.create_experiment(experiment)
            experiment_or_none = client.get_experiment_by_name(experiment)
            assert experiment_or_none is not None
        experiment = experiment_or_none
    return experiment


def get_step_params(step_info: WorkflowStepInfo) -> MlflowParams:
    config = as_jsonvalue(step_info.step.config)
    assert isinstance(config, dict)
    return flatten_params(config)


def get_execution_params(execution_info: WorkflowExecutionInfo) -> MlflowParams:
    config = execution_info.graph.json()
    assert isinstance(config, dict)
    return flatten_params(config)


def get_step_tags(step_info: WorkflowStepInfo) -> dict[MlflowTag, str]:
    return {
        MlflowTag.MLFACTORY_RUN_TYPE: WorkflowRunType.STEP.value,
        MlflowTag.MLFACTORY_STEP_FINGERPRINT: step_info.fingerprint,
    }


def get_execution_tags(execution_info: WorkflowExecutionInfo) -> dict[MlflowTag, str]:
    assert execution_info.id is not None
    return {
        MlflowTag.MLFLOW_RUN_NAME: execution_info.id,
        MlflowTag.MLFACTORY_RUN_TYPE: WorkflowRunType.EXECUTION.value,
    }


def get_mlflow_tags_from_run(run: MlflowRun) -> dict[MlflowTag, str]:
    return {tag: run.data.tags[tag.value] for tag in MlflowTag if tag.value in run.data.tags}


def get_execution_state_from_run(run: MlflowRun) -> WorkflowExecutionState:
    tags = get_mlflow_tags_from_run(run)
    if MlflowTag.MLFACTORY_RUN_TYPE not in tags:
        raise ValueError("Run type not found")
    if tags[MlflowTag.MLFACTORY_RUN_TYPE] != WorkflowRunType.EXECUTION.value:
        raise ValueError(f"Invalid run type: {tags[MlflowTag.MLFACTORY_RUN_TYPE]}")
    if run.info.run_name is None:
        raise ValueError("Run name not found")
    execution_id = WorkflowExecutionID(run.info.run_name)
    status = MlflowRunStatus(run.info.status).to_execution_status()
    started_at = datetime.datetime.fromtimestamp(run.info.start_time / 1000)
    finished_at = datetime.datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None
    return WorkflowExecutionState(
        execution_id=execution_id,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
    )
