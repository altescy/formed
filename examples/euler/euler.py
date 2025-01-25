import cmath
from typing import Any, Union

from formed import workflow

ComplexOrTuple = Union[complex, tuple]


def make_complex(x: Union[int, float, ComplexOrTuple]) -> complex:
    if isinstance(x, complex):
        return x
    elif isinstance(x, (int, float)):
        return complex(x)
    else:
        return complex(*x)


@workflow.step
def cadd(a: ComplexOrTuple, b: ComplexOrTuple) -> complex:
    return make_complex(a) + make_complex(b)


@workflow.step
def csub(a: ComplexOrTuple, b: ComplexOrTuple) -> complex:
    return make_complex(a) - make_complex(b)


@workflow.step
def cexp(x: ComplexOrTuple, base: ComplexOrTuple = cmath.e) -> complex:
    return make_complex(base) ** make_complex(x)


@workflow.step
def cmul(a: ComplexOrTuple, b: ComplexOrTuple) -> complex:
    return make_complex(a) * make_complex(b)


@workflow.step
def csin(x: ComplexOrTuple) -> complex:
    return cmath.sin(make_complex(x))


@workflow.step
def ccos(x: ComplexOrTuple) -> complex:
    return cmath.cos(make_complex(x))


@workflow.step("print")
def print_(input: Any) -> None:
    logger = workflow.use_step_logger(__name__)
    logger.info(f"input: {input}")
