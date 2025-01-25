from typing import Final, Union

_ALPHABET: Final[bytes] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58encode(s: Union[str, bytes]) -> bytes:
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
