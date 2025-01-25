import pytest

from formed.common.hashutils import hash_object, murmurhash3


def test_hash_object_returns_same_value_for_same_object() -> None:
    assert hash_object("hello") == hash_object("hello")
    assert hash_object("world") == hash_object("world")
    assert hash_object({"hello": "world"}) == hash_object({"hello": "world"})


def test_hash_object_returns_different_value_for_different_object() -> None:
    assert hash_object("hello") != hash_object("world")
    assert hash_object({"hello": "world"}) != hash_object({"world": "hello"})


@pytest.mark.parametrize(
    "s,expected",
    [
        ("test", 0xBA6BD213),
        ("Hello, world!", 0xC0363E43),
        ("The quick brown fox jumps over the lazy dog", 0x2E4FF723),
    ],
)
def test_murmurhash3_returns_correct_value(s: str, expected: int) -> None:
    assert murmurhash3(s) == expected
