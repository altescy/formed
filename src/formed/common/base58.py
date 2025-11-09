"""Base58 encoding utilities for compact representation.

Base58 is a binary-to-text encoding scheme that uses 58 alphanumeric characters,
excluding visually similar characters (0, O, I, l) to reduce transcription errors.
It's commonly used for encoding hashes and fingerprints in a human-readable format.

The alphabet used: 123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz

Example:
    >>> from formed.common.base58 import b58encode, b58decode
    >>>
    >>> # Encode binary data or strings
    >>> encoded = b58encode(b"hello")
    >>> print(encoded)  # b'Cn8eVZg'
    >>>
    >>> # Decode back to original
    >>> decoded = b58decode(encoded)
    >>> print(decoded)  # b'hello'

"""

from typing import Final, Union

_ALPHABET: Final[bytes] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58encode(s: Union[str, bytes]) -> bytes:
    """Encode bytes or string to Base58.

    Args:
        s: Input bytes or ASCII string to encode.

    Returns:
        Base58-encoded bytes.

    Example:
        >>> b58encode(b"test")
        b'3yZe7d'
        >>> b58encode("hello")
        b'Cn8eVZg'

    """
    if isinstance(s, str):
        s = s.encode("ascii")
    original_length = len(s)
    s = s.lstrip(b"\0")
    stripped_length = len(s)
    n = int.from_bytes(s, "big")
    res = bytearray()
    while n:
        n, r = divmod(n, 58)
        res.append(_ALPHABET[r])
    return _ALPHABET[0:1] * (original_length - stripped_length) + bytes(reversed(res))


def b58decode(v: Union[str, bytes]) -> bytes:
    """Decode Base58-encoded data back to bytes.

    Args:
        v: Base58-encoded bytes or ASCII string.

    Returns:
        The original bytes.

    Example:
        >>> encoded = b58encode(b"test")
        >>> b58decode(encoded)
        b'test'
        >>> b58decode("Cn8eVZg")
        b'hello'

    """
    v = v.rstrip()
    if isinstance(v, str):
        v = v.encode("ascii")
    original_length = len(v)
    v = v.lstrip(_ALPHABET[0:1])
    stripped_length = len(v)
    n = 0
    for c in v:
        n = n * 58 + _ALPHABET.index(c)
    return n.to_bytes(original_length - stripped_length + (n.bit_length() + 7) // 8, "big")
