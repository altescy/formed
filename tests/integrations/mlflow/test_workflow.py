from pathlib import Path
from typing import Annotated

import pytest

mlflow = pytest.importorskip("mlflow")

from formed.integrations.mlflow.workflow import MlflowWorkflowCallback  # noqa: E402
from formed.workflow import DefaultWorkflowExecutor, WorkflowGraph, WorkflowStepResultFlag, step  # noqa: E402


def test_logs_metrics_from_non_cached_step(tmp_path: Path) -> None:
    previous_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tmp_path.as_uri())
    try:
        @step("test_mlflow_workflow::non_cached_metrics", cacheable=False)
        def _() -> Annotated[dict[str, float], WorkflowStepResultFlag.METRICS]:
            return {"loss": 0.5}

        graph = WorkflowGraph.from_json(
            {"steps": {"metrics": {"type": "test_mlflow_workflow::non_cached_metrics"}}}
        )
        experiment_name = "non-cached-metrics"

        DefaultWorkflowExecutor()(graph, callback=MlflowWorkflowCallback(experiment_name))

        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        assert experiment is not None
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        assert any(run.data.metrics.get("loss") == 0.5 for run in runs)
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
