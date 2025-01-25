import pytest

from formed.common.base58 import b58decode, b58encode


@pytest.mark.parametrize(
    "s,expected",
    [
        (b"hello", b"Cn8eVZg"),
        (b"world", b"EUYUqQf"),
        (b"hello world", b"StV1DL6CwTryKyV"),
    ],
)
def test_b58encode(s: bytes, expected: bytes) -> None:
    assert b58encode(s) == expected


@pytest.mark.parametrize(
    "s,expected",
    [
        (b"Cn8eVZg", b"hello"),
        (b"EUYUqQf", b"world"),
        (b"StV1DL6CwTryKyV", b"hello world"),
    ],
)
def test_b58decode(s: bytes, expected: bytes) -> None:
    assert b58decode(s) == expected
