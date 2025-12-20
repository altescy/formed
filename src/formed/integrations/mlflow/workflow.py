import contextvars
import os
import shutil
from collections.abc import Sequence
from contextlib import suppress
from io import StringIO
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, TypeVar, Union, cast, overload

import mlflow
import mlflow.artifacts
from filelock import BaseFileLock, FileLock
from mlflow.entities import Experiment as MlflowExperiment
from mlflow.tracking.client import MlflowClient

from formed.common.logutils import LogCapture
from formed.types import JsonValue
from formed.workflow import (
    WorkflowCache,
    WorkflowCallback,
    WorkflowExecutionContext,
    WorkflowExecutionID,
    WorkflowExecutionInfo,
    WorkflowExecutor,
    WorkflowGraph,
    WorkflowOrganizer,
    WorkflowStep,
    WorkflowStepContext,
    WorkflowStepInfo,
    WorkflowStepResultFlag,
    WorkflowStepStatus,
    get_step_logger_from_info,
    use_step_context,
)

from . import utils as mlflow_utils
from .constants import DEFAULT_MLFLOW_DIRECTORY, DEFAULT_MLFLOW_EXPERIMENT_NAME
from .utils import (
    MlflowRun,
    MlflowRunStatus,
    MlflowTag,
    WorkflowCacheStatus,
    fetch_mlflow_run,
    get_mlflow_experiment,
)

if TYPE_CHECKING:
    with suppress(ImportError):
        from matplotlib.figure import Figure as MatplotlibFigure
    with suppress(ImportError):
        from mlflow import Image as MlflowImage
    with suppress(ImportError):
        from numpy import ndarray as NumpyArray
    with suppress(ImportError):
        from pandas import DataFrame as PandasDataFrame
    with suppress(ImportError):
        from PIL.Image import Image as PILImage
    with suppress(ImportError):
        from plotly.graph_objs import Figure as PlotlyFigure

logger = getLogger(__name__)

T = TypeVar("T")

_MLFLOW_EXPERIMENT = contextvars.ContextVar[Optional[MlflowExperiment]]("_MLFLOW_EXPERIMENT", default=None)


@WorkflowCache.register("mlflow")
class MlflowWorkflowCache(WorkflowCache):
    _CACHE_DIRNAME: ClassVar[str] = "cache"
    _DEFAULT_DIRECTORY: ClassVar[Path] = DEFAULT_MLFLOW_DIRECTORY / "cache"

    def __init__(
        self,
        experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT_NAME,
        directory: Optional[Union[str, PathLike]] = None,
        mlflow_client: Optional[MlflowClient] = None,
    ) -> None:
        self._client = mlflow_client or MlflowClient()
        self._experiment_name = experiment_name
        self._directory = Path(directory or self._DEFAULT_DIRECTORY)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _get_step_cache_dir(self, step_info: WorkflowStepInfo) -> Path:
        run = mlflow_utils.fetch_mlflow_run(
            self._client,
            self._experiment_name,
            step_info=step_info,
        )
        mlflow_artifact_dir = mlflow_utils.get_mlflow_local_artifact_storage_path(run) if run is not None else None
        if mlflow_artifact_dir is not None:
            return mlflow_artifact_dir / self._CACHE_DIRNAME
        return self._directory / step_info.fingerprint

    def _get_step_cache_lock(self, step_info: WorkflowStepInfo) -> BaseFileLock:
        return FileLock(str(self._get_step_cache_dir(step_info).with_suffix(".lock")))

    def __getitem__(self, step_info: WorkflowStepInfo[WorkflowStep[T]]) -> T:
        if step_info not in self:
            raise KeyError(step_info)
        step_cache_dir = self._get_step_cache_dir(step_info)
        with self._get_step_cache_lock(step_info):
            if not step_cache_dir.exists():
                try:
                    mlflow_utils.download_mlflow_artifacts(
                        self._client,
                        self._experiment_name,
                        step_info,
                        step_cache_dir,
                        artifact_path=self._CACHE_DIRNAME,
                    )
                except FileNotFoundError:
                    raise KeyError(step_info)
                except Exception:
                    shutil.rmtree(step_cache_dir, ignore_errors=True)
                    raise
            return cast(T, step_info.format.read(step_cache_dir))

    def __setitem__(self, step_info: WorkflowStepInfo[WorkflowStep[T]], value: T) -> None:
        run = mlflow_utils.fetch_mlflow_run(
            self._client,
            self._experiment_name,
            step_info=step_info,
        )
        if run is None:
            raise RuntimeError(f"Run for step {step_info} not found")
        elif run.info.status != MlflowRunStatus.RUNNING.value:
            raise ValueError(f"Run {run.info.run_id} is not running")
        step_cache_dir = self._get_step_cache_dir(step_info)
        with self._get_step_cache_lock(step_info):
            mlflow_utils.update_mlflow_tags(
                self._client,
                run,
                {MlflowTag.MLFACTORY_STEP_CACHE_STATUS: WorkflowCacheStatus.PENDING.value},
            )
            try:
                step_cache_dir.mkdir(parents=True, exist_ok=True)
                step_info.format.write(value, step_cache_dir)
                if not mlflow_utils.is_mlflow_using_local_artifact_storage(run):
                    mlflow_utils.upload_mlflow_artifacts(
                        self._client,
                        self._experiment_name,
                        step_info,
                        step_cache_dir,
                        artifact_path=self._CACHE_DIRNAME,
                    )
            except Exception as e:
                mlflow_utils.update_mlflow_tags(
                    self._client,
                    run,
                    {MlflowTag.MLFACTORY_STEP_CACHE_STATUS: WorkflowCacheStatus.INACTIVE.value},
                )
                raise e
            else:
                mlflow_utils.update_mlflow_tags(
                    self._client,
                    run,
                    {MlflowTag.MLFACTORY_STEP_CACHE_STATUS: WorkflowCacheStatus.ACTIVE.value},
                )

    def __delitem__(self, step_info: WorkflowStepInfo) -> None:
        run = mlflow_utils.fetch_mlflow_run(
            self._client,
            self._experiment_name,
            step_info=step_info,
        )
        if run is None:
            raise KeyError(step_info)
        mlflow_utils.update_mlflow_tags(
            self._client,
            run,
            {MlflowTag.MLFACTORY_STEP_CACHE_STATUS: WorkflowCacheStatus.INACTIVE.value},
        )
        if not mlflow_utils.is_mlflow_using_local_artifact_storage(run):
            step_cache_dir = self._get_step_cache_dir(step_info)
            with self._get_step_cache_lock(step_info):
                shutil.rmtree(step_cache_dir, ignore_errors=True)

    def __contains__(self, step_info: WorkflowStepInfo) -> bool:
        run = mlflow_utils.fetch_mlflow_run(
            self._client,
            self._experiment_name,
            step_info=step_info,
        )
        return (
            run is not None
            and run.data.tags.get(MlflowTag.MLFACTORY_STEP_CACHE_STATUS) == WorkflowCacheStatus.ACTIVE.value
        )


@WorkflowCallback.register("mlflow")
class MlflowWorkflowCallback(WorkflowCallback):
    _LOG_FILENAME: ClassVar[str] = "out.log"
    _STEP_METADATA_ARTIFACT_FILENAME: ClassVar[str] = "step.json"
    _EXECUTION_METADATA_ARTIFACT_FILENAME: ClassVar[str] = "execution.json"

    def __init__(
        self,
        experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT_NAME,
        mlflow_client: Optional[MlflowClient] = None,
        log_execution_metrics: bool = False,
    ) -> None:
        self._client = mlflow_client or MlflowClient()
        self._experiment_name = experiment_name
        self._execution_run: Optional[MlflowRun] = None
        self._execution_log: Optional[LogCapture[StringIO]] = None
        self._step_log: dict[WorkflowStepInfo, LogCapture[StringIO]] = {}
        self._log_execution_metrics = log_execution_metrics
        self._step_run_ids: dict[str, str] = {}
        self._dependents_map: dict[str, set[str]] = {}

    def on_execution_start(
        self,
        execution_context: WorkflowExecutionContext,
    ) -> None:
        assert self._execution_run is None
        execution_info = execution_context.info
        if execution_info.id is None:
            execution_info.id = mlflow_utils.generate_new_execution_id(self._client, self._experiment_name)
        self._execution_log = LogCapture(StringIO())
        self._execution_log.start()
        self._execution_run = mlflow_utils.add_mlflow_run(
            self._client,
            self._experiment_name,
            execution_info,
        )
        # Use WorkflowExecutionInfo.to_json_dict() for proper serialization
        self._client.log_dict(
            run_id=self._execution_run.info.run_id,
            dictionary=execution_info.json(),
            artifact_file=self._EXECUTION_METADATA_ARTIFACT_FILENAME,
        )

        # Initialize tracking for notes
        self._dependents_map = self._build_dependents_map(execution_info.graph)
        self._step_run_ids = {}

        # Set initial execution note
        initial_note = self._generate_execution_note_markdown(
            execution_info,
            self._step_run_ids,
            self._dependents_map,
        )
        self._update_run_note(self._execution_run.info.run_id, initial_note)

    def on_execution_end(
        self,
        execution_context: "WorkflowExecutionContext",
    ) -> None:
        assert self._execution_run is not None
        mlflow_utils.terminate_mlflow_run(
            self._client,
            self._experiment_name,
            execution_context.state,
        )
        if self._execution_log is not None:
            self._execution_log.stop()
            self._client.log_text(
                run_id=self._execution_run.info.run_id,
                text=self._execution_log.stream.getvalue(),
                artifact_file=self._LOG_FILENAME,
            )
            self._execution_log.stream.close()
        self._execution_run = None

    def on_step_start(
        self,
        step_context: WorkflowStepContext,
        execution_context: WorkflowExecutionContext,
    ) -> None:
        assert self._execution_run is not None
        step_info = step_context.info
        run = mlflow_utils.add_mlflow_run(
            self._client,
            self._experiment_name,
            step_info,
            parent_run_id=self._execution_run.info.run_id,
        )
        self._client.log_dict(
            run_id=run.info.run_id,
            dictionary=step_info.json(),
            artifact_file=self._STEP_METADATA_ARTIFACT_FILENAME,
        )
        self._step_log[step_info] = LogCapture(StringIO(), logger=get_step_logger_from_info(step_info))
        self._step_log[step_info].start()

        # Store step run ID
        self._step_run_ids[step_info.name] = run.info.run_id

        # Set step note
        step_note = self._generate_step_note_markdown(
            step_info,
            execution_context.info,
            self._step_run_ids,
            self._dependents_map,
        )
        self._update_run_note(run.info.run_id, step_note)

        # Update execution note with new run ID
        execution_note = self._generate_execution_note_markdown(
            execution_context.info,
            self._step_run_ids,
            self._dependents_map,
        )
        self._update_run_note(self._execution_run.info.run_id, execution_note)

    def _build_dependents_map(self, graph: WorkflowGraph) -> dict[str, set[str]]:
        """Build reverse dependency map: step_name -> set of steps that depend on it."""
        dependents: dict[str, set[str]] = {name: set() for name in graph._step_info.keys()}
        for step_name, step_info in graph._step_info.items():
            for _, dep_info in step_info.dependencies:
                dependents[dep_info.name].add(step_name)
        return dependents

    def _get_mlflow_run_url(self, run_id: str) -> str:
        """Generate MLflow UI URL for a run."""
        experiment = get_mlflow_experiment(self._experiment_name)
        experiment_id = experiment.experiment_id
        return f"/#/experiments/{experiment_id}/runs/{run_id}"

    def _update_run_note(self, run_id: str, markdown: str) -> None:
        """Update MLFLOW_RUN_NOTE tag for a run."""
        self._client.set_tag(run_id, MlflowTag.MLFLOW_RUN_NOTE.value, markdown)

    def _generate_execution_note_markdown(
        self,
        execution_info: WorkflowExecutionInfo,
        step_run_ids: dict[str, str],
        dependents_map: dict[str, set[str]],
    ) -> str:
        """Generate markdown note for execution run.

        Args:
            execution_info: Execution information
            step_run_ids: Map of step names to their MLflow run IDs (may be incomplete)
            dependents_map: Reverse dependency map

        Returns:
            Markdown string with execution summary
        """
        lines = []
        lines.append(f"# Workflow Execution: {execution_info.id}")
        lines.append("")
        lines.append("## Steps")
        lines.append("")
        lines.append("| Step | Dependencies | Dependents |")
        lines.append("|------|--------------|------------|")

        for step_name, step_info in execution_info.graph._step_info.items():
            # Step name with link if available
            if step_name in step_run_ids:
                step_link = f"[{step_name}]({self._get_mlflow_run_url(step_run_ids[step_name])})"
            else:
                step_link = step_name

            # Dependencies (use set to avoid duplicates)
            dep_names = {dep_info.name for _, dep_info in step_info.dependencies}
            if dep_names:
                dep_links = []
                for dep_name in sorted(dep_names):
                    if dep_name in step_run_ids:
                        dep_links.append(f"[{dep_name}]({self._get_mlflow_run_url(step_run_ids[dep_name])})")
                    else:
                        dep_links.append(dep_name)
                dependencies_str = ", ".join(dep_links)
            else:
                dependencies_str = "-"

            # Dependents
            dependent_names = dependents_map.get(step_name, set())
            if dependent_names:
                dependent_links = []
                for dependent_name in sorted(dependent_names):
                    if dependent_name in step_run_ids:
                        dependent_links.append(
                            f"[{dependent_name}]({self._get_mlflow_run_url(step_run_ids[dependent_name])})"
                        )
                    else:
                        dependent_links.append(dependent_name)
                dependents_str = ", ".join(dependent_links)
            else:
                dependents_str = "-"

            lines.append(f"| {step_link} | {dependencies_str} | {dependents_str} |")

        lines.append("")
        lines.append("## Dependency Graph")
        lines.append("")
        lines.append("```")

        # Use DAG.visualize for text-based graph
        from io import StringIO

        from formed.common.dag import DAG

        dag = DAG(
            {
                step_name: {dep.name for _, dep in info.dependencies}
                for step_name, info in execution_info.graph._step_info.items()
            }
        )
        graph_output = StringIO()
        dag.visualize(output=graph_output)
        lines.append(graph_output.getvalue().rstrip())

        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    def _generate_step_note_markdown(
        self,
        step_info: WorkflowStepInfo,
        execution_info: WorkflowExecutionInfo,
        step_run_ids: dict[str, str],
        dependents_map: dict[str, set[str]],
    ) -> str:
        """Generate markdown note for step run.

        Args:
            step_info: Step information
            execution_info: Execution information
            step_run_ids: Map of step names to their MLflow run IDs (may be incomplete)
            dependents_map: Reverse dependency map

        Returns:
            Markdown string with step details
        """
        lines = []
        lines.append(f"# Step: {step_info.name}")
        lines.append("")

        # Link to parent execution
        if self._execution_run is not None:
            execution_url = self._get_mlflow_run_url(self._execution_run.info.run_id)
            lines.append(f"**Execution:** [{execution_info.id}]({execution_url})")
            lines.append("")

        # Dependencies (upstream) - use set to avoid duplicates
        dep_names = {dep_info.name for _, dep_info in step_info.dependencies}
        if dep_names:
            lines.append("## Dependencies (Upstream)")
            lines.append("")
            for dep_name in sorted(dep_names):
                if dep_name in step_run_ids:
                    dep_url = self._get_mlflow_run_url(step_run_ids[dep_name])
                    lines.append(f"- [{dep_name}]({dep_url})")
                else:
                    lines.append(f"- {dep_name}")
            lines.append("")

        # Dependents (downstream)
        dependent_names = dependents_map.get(step_info.name, set())
        if dependent_names:
            lines.append("## Dependents (Downstream)")
            lines.append("")
            for dependent_name in sorted(dependent_names):
                if dependent_name in step_run_ids:
                    dependent_url = self._get_mlflow_run_url(step_run_ids[dependent_name])
                    lines.append(f"- [{dependent_name}]({dependent_url})")
                else:
                    lines.append(f"- {dependent_name}")
            lines.append("")

        lines.append("")

        return "\n".join(lines)

    def on_step_end(
        self,
        step_context: WorkflowStepContext,
        execution_context: WorkflowExecutionContext,
    ) -> None:
        step_info = step_context.info
        mlflow_utils.terminate_mlflow_run(
            self._client,
            self._experiment_name,
            step_context.state,
        )
        if (step_log := self._step_log.pop(step_info, None)) is not None:
            run = mlflow_utils.fetch_mlflow_run(
                self._client,
                self._experiment_name,
                step_info=step_info,
            )
            if run is None:
                raise RuntimeError(f"Run for step {step_info} not found")
            step_log.stop()
            self._client.log_text(
                run_id=run.info.run_id,
                text=step_log.stream.getvalue(),
                artifact_file=self._LOG_FILENAME,
            )
            if (
                WorkflowStepResultFlag.METRICS in WorkflowStepResultFlag.get_flags(step_info)
                and step_context.state.status == WorkflowStepStatus.COMPLETED
            ):
                metrics = execution_context.cache[step_info]
                assert isinstance(metrics, dict), f"Expected dict, got {type(metrics)}"
                for key, value in metrics.items():
                    self._client.log_metric(run.info.run_id, key, value)
                if self._log_execution_metrics:
                    assert self._execution_run is not None
                    for key, value in metrics.items():
                        key = f"{step_info.name}/{key}"
                        self._client.log_metric(self._execution_run.info.run_id, key, value)
            step_log.stream.close()


@WorkflowOrganizer.register("mlflow")
class MlflowWorkflowOrganizer(WorkflowOrganizer):
    def __init__(
        self,
        experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT_NAME,
        cache: Optional[WorkflowCache] = None,
        callbacks: Optional[Union[WorkflowCallback, Sequence[WorkflowCallback]]] = None,
        log_execution_metrics: Optional[bool] = None,
    ) -> None:
        self._client = MlflowClient()
        self._experiment_name = experiment_name

        cache = cache or MlflowWorkflowCache(
            experiment_name=experiment_name,
            mlflow_client=self._client,
        )
        if callbacks is None:
            callbacks = []
        elif isinstance(callbacks, WorkflowCallback):
            callbacks = [callbacks]
        if any(isinstance(callback, MlflowWorkflowCallback) for callback in callbacks):
            if log_execution_metrics is not None:
                logger.warning(
                    "Ignoring `log_execution_metrics` parameter because `MlflowWorkflowCallback` is already present"
                )
        else:
            mlflow_callback = MlflowWorkflowCallback(
                experiment_name,
                mlflow_client=self._client,
                log_execution_metrics=log_execution_metrics or False,
            )
            callbacks = [mlflow_callback] + list(callbacks)

        super().__init__(cache, callbacks)

    def run(
        self,
        executor: WorkflowExecutor,
        execution: Union[WorkflowGraph, WorkflowExecutionInfo],
    ) -> WorkflowExecutionContext:
        cxt = contextvars.copy_context()

        super_run = super().run

        def _run() -> WorkflowExecutionContext:
            experiment = get_mlflow_experiment(self._experiment_name)
            _MLFLOW_EXPERIMENT.set(experiment)
            return super_run(executor, execution)

        return cxt.run(_run)

    def get(self, execution_id: WorkflowExecutionID) -> Optional[WorkflowExecutionContext]:
        run = mlflow_utils.fetch_mlflow_run(
            self._client,
            self._experiment_name,
            execution_info=execution_id,
        )
        if run is None:
            return None
        artifact_uri = run.info.artifact_uri
        if not artifact_uri:
            raise RuntimeError(f"Run {run.info.run_id} has no artifact URI")

        # Load execution data using proper deserialization
        # Download artifact to temporary file
        execution_data = mlflow.artifacts.load_dict(
            artifact_uri + "/" + MlflowWorkflowCallback._EXECUTION_METADATA_ARTIFACT_FILENAME
        )
        execution_info = WorkflowExecutionInfo.from_json(execution_data)

        execution_state = mlflow_utils.get_execution_state_from_run(run)
        return WorkflowExecutionContext(execution_info, execution_state, self.cache, self.callback)

    def exists(self, execution_id: WorkflowExecutionID) -> bool:
        run = mlflow_utils.fetch_mlflow_run(
            self._client,
            self._experiment_name,
            execution_info=execution_id,
        )
        return run is not None

    def remove(self, execution_id: WorkflowExecutionID) -> None:
        for run in mlflow_utils.fetch_mlflow_runs(
            self._client,
            self._experiment_name,
            execution_info=execution_id,
            with_children=True,
        ):
            logger.info(f"Removing run {run.info.run_id}")
            self._client.delete_run(run.info.run_id)


class MlflowLogger:
    _ARTIFACT_PATH: ClassVar[str] = "artifacts"

    def __init__(self, run: MlflowRun):
        self.run = run

    @overload
    def _get_artifact_path(self, artifact_path: None) -> None: ...

    @overload
    def _get_artifact_path(self, artifact_path: str) -> str: ...

    def _get_artifact_path(self, artifact_path: Optional[str]) -> Optional[str]:
        if artifact_path is None:
            return None
        return os.path.join(self._ARTIFACT_PATH, artifact_path)

    @property
    def mlflow_client(self) -> MlflowClient:
        return MlflowClient()

    def log_metric(
        self,
        key: str,
        value: float,
        timestamp: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        self.mlflow_client.log_metric(
            run_id=self.run.info.run_id,
            key=key,
            value=value,
            timestamp=timestamp,
            step=step,
        )

    def log_metrics(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.log_metric(key, value)

    def log_table(
        self,
        data: Union[dict[str, Sequence[Union[str, bool, int, float]]], "PandasDataFrame"],
        artifact_path: str,
    ) -> None:
        self.mlflow_client.log_table(
            run_id=self.run.info.run_id,
            data=data,
            artifact_file=self._get_artifact_path(artifact_path),
        )

    def log_text(
        self,
        text: str,
        artifact_path: str,
    ) -> None:
        self.mlflow_client.log_text(
            run_id=self.run.info.run_id,
            text=text,
            artifact_file=self._get_artifact_path(artifact_path),
        )

    def log_dict(
        self,
        dictionary: dict[str, JsonValue],
        artifact_path: str,
    ) -> None:
        self.mlflow_client.log_dict(
            run_id=self.run.info.run_id,
            dictionary=dictionary,
            artifact_file=self._get_artifact_path(artifact_path),
        )

    def log_figure(
        self,
        figure: Union["MatplotlibFigure", "PlotlyFigure"],
        artifact_path: str,
    ) -> None:
        self.mlflow_client.log_figure(
            run_id=self.run.info.run_id,
            figure=figure,
            artifact_file=self._get_artifact_path(artifact_path),
        )

    def log_image(
        self,
        image: Union["NumpyArray", "PILImage", "MlflowImage"],
        artifact_path: Optional[str] = None,
    ) -> None:
        self.mlflow_client.log_image(
            run_id=self.run.info.run_id,
            image=image,
            artifact_file=self._get_artifact_path(artifact_path),
        )

    def log_artifact(
        self,
        local_path: Union[str, PathLike],
        artifact_path: Optional[str] = None,
    ) -> None:
        self.mlflow_client.log_artifact(
            run_id=self.run.info.run_id,
            local_path=local_path,
            artifact_path=self._get_artifact_path(artifact_path),
        )

    def log_artifacts(
        self,
        local_dir: Union[str, PathLike],
        artifact_path: Optional[str] = None,
    ) -> None:
        self.mlflow_client.log_artifacts(
            run_id=self.run.info.run_id,
            local_dir=str(local_dir),
            artifact_path=self._get_artifact_path(artifact_path),
        )


def use_mlflow_experiment() -> Optional[MlflowExperiment]:
    return _MLFLOW_EXPERIMENT.get()


def use_mlflow_logger() -> Optional[MlflowLogger]:
    if (experiment := use_mlflow_experiment()) is None:
        return None

    if (context := use_step_context()) is None:
        return None

    client = MlflowClient()
    if (run := fetch_mlflow_run(client, experiment, step_info=context.info)) is None:
        return None

    return MlflowLogger(run)
