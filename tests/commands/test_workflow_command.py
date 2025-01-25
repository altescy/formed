import cmath
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

        @step
        def cadd(a: ComplexOrTuple, b: ComplexOrTuple) -> ComplexOrTuple:
            return make_complex(a) + make_complex(b)

        @step
        def csub(a: ComplexOrTuple, b: ComplexOrTuple) -> ComplexOrTuple:
            return make_complex(a) - make_complex(b)

        @step
        def cexp(x: ComplexOrTuple) -> ComplexOrTuple:
            return cmath.exp(make_complex(x))

        @step
        def cmul(a: ComplexOrTuple, b: ComplexOrTuple) -> ComplexOrTuple:
            return make_complex(a) * make_complex(b)

        @step
        def csin(x: ComplexOrTuple) -> ComplexOrTuple:
            return cmath.sin(make_complex(x))

        @step
        def ccos(x: ComplexOrTuple) -> ComplexOrTuple:
            return cmath.cos(make_complex(x))

        @step(name="print")
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
