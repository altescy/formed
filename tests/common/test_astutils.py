import pytest

from formed.common.astutils import normalize_source


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("a = 1", "a = 1", True),
        ("a = 1", "b = 1", True),
        ("a = 1", "a = 2", False),
        (
            """
            def foo(a, b):
                return a + b
            """.strip(),
            """
            def foo(x, y):
                # comment
                return x + y
            """.strip(),
            True,
        ),
        (
            """
            def foo(a, b):
                return a + b
            """.strip(),
            """
            def foo(x, y):
                # comment
                return x * y
            """.strip(),
            False,
        ),
    ],
)
def test_normalize_source(a: str, b: str, expected: bool) -> None:
    assert (normalize_source(a) == normalize_source(b)) == expected
