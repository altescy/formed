import cmath
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Union

import pytest

from formed.commands import create_subcommand
from formed.workflow import step, use_step_logger

ComplexOrTuple = Union[int, float, complex, tuple[Union[int, float], Union[int, float]]]


class TestWorkflowCommand:
    @staticmethod
    @pytest.fixture
    def euler_workflow() -> None:
        def make_complex(x: ComplexOrTuple) -> complex:
            if isinstance(x, complex):
                return x
            if isinstance(x, (int, float)):
                return complex(x)
            return complex(*x)

        @step(exist_ok=True)
        def cadd(a: ComplexOrTuple, b: ComplexOrTuple) -> ComplexOrTuple:
            return make_complex(a) + make_complex(b)

        @step(exist_ok=True)
        def csub(a: ComplexOrTuple, b: ComplexOrTuple) -> ComplexOrTuple:
            return make_complex(a) - make_complex(b)

        @step(exist_ok=True)
        def cexp(x: ComplexOrTuple) -> ComplexOrTuple:
            return cmath.exp(make_complex(x))

        @step(exist_ok=True)
        def cmul(a: ComplexOrTuple, b: ComplexOrTuple) -> ComplexOrTuple:
            return make_complex(a) * make_complex(b)

        @step(exist_ok=True)
        def csin(x: ComplexOrTuple) -> ComplexOrTuple:
            return cmath.sin(make_complex(x))

        @step(exist_ok=True)
        def ccos(x: ComplexOrTuple) -> ComplexOrTuple:
            return cmath.cos(make_complex(x))

        @step(name="print", exist_ok=True)
        def print_(input: Any) -> None:
            logger = use_step_logger()
            assert logger is not None
            logger.info(f"{input=}")

    @staticmethod
    @pytest.fixture
    def euler_config(tmp_path: Path) -> Iterator[Path]:
        config = """
        local i = [0.0, 1.0];
        local pi = [3.1415926535, 0.0];

        {
            "steps": {
                "i_times_pi": {
                    "type": "cmul",
                    "a": i,
                    "b": pi
                },
                "pow_e": {
                    "type": "cexp",
                    "x": { "type": "ref", "ref": "i_times_pi" }
                },
                "plus_one": {
                    "type": "cadd",
                    "a": { "type": "ref", "ref": "pow_e" },
                    "b": [1, 0]
                },
                "print": {
                    "type": "print",
                    "input": { "type": "ref", "ref": "plus_one" }
                }
            }
        }
        """

        config_path = tmp_path / "euler_config.jsonnet"
        with config_path.open("w") as f:
            f.write(config)

        yield config_path

    @staticmethod
    @pytest.mark.usefixtures("euler_workflow")
    def test_workflow_run_command(euler_config: Path) -> None:
        app = create_subcommand("workflow_test")
        args = app.parser.parse_args(["workflow", "run", str(euler_config)])

        app(args)

    @staticmethod
    @pytest.mark.usefixtures("euler_workflow")
    def test_workflow_run_command_with_tags(euler_config: Path, tmp_path: Path) -> None:
        formed_dir = tmp_path / "formed"
        settings_path = tmp_path / "formed.yml"
        settings_path.write_text(
            f"""workflow:
  organizer:
    type: filesystem
    directory: {formed_dir}
  tags:
    - project:demo
"""
        )

        app = create_subcommand("workflow_test")
        args = app.parser.parse_args(
            [
                "workflow",
                "--settings",
                str(settings_path),
                "run",
                str(euler_config),
                "--tags",
                "experiment:euler",
                "team:math",
            ]
        )

        app(args)

        # Verify execution tags were persisted
        executions_dir = formed_dir / "executions"
        execution_dirs = list(executions_dir.iterdir())
        assert len(execution_dirs) == 1
        execution_dir = execution_dirs[0]
        tags_path = execution_dir / "tags.json"
        assert tags_path.exists()
        persisted_tags = json.loads(tags_path.read_text())
        assert set(persisted_tags) == {"experiment:euler", "project:demo", "team:math"}


class TestWorkflowSettingsTags:
    @staticmethod
    def test_workflow_settings_builds_execution_metadata() -> None:
        from formed.workflow.settings import WorkflowSettings

        settings = WorkflowSettings(tags=["foo", "bar"])
        metadata = settings.build_execution_metadata()

        assert metadata.tags == ("foo", "bar")
