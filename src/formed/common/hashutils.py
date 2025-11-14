"""Object hashing utilities for content-based fingerprinting.

This module provides utilities for hashing arbitrary Python objects using
serialization and cryptographic hash functions. It's used for cache keys
and fingerprinting in workflow systems.

Key Features:
    - Hash arbitrary Python objects using dill serialization
    - BLAKE2b hashing for cryptographic strength
    - MurmurHash3 for string hashing (non-cryptographic)

Example:
    >>> from formed.common.hashutils import hash_object, murmurhash3
    >>>
    >>> # Hash complex objects
    >>> config = {"model": "bert", "epochs": 10, "layers": [1, 2, 3]}
    >>> fingerprint = hash_object(config)
    >>>
    >>> # Fast string hashing
    >>> hash_val = murmurhash3("my_key", seed=42)

"""

import hashlib
import io
from typing import Any

import cloudpickle


def hash_object_bytes(o: Any) -> bytes:
    """Hash a Python object to bytes using BLAKE2b.

    Serializes the object using dill and computes a BLAKE2b hash digest.
    This provides a stable, cryptographically strong fingerprint for
    arbitrary Python objects.

    Args:
        o: The object to hash (any dill-serializable object).

    Returns:
        The BLAKE2b hash digest as bytes.

    Example:
        >>> obj = {"key": "value", "nested": [1, 2, 3]}
        >>> hash_bytes = hash_object_bytes(obj)
        >>> len(hash_bytes)  # BLAKE2b produces 64 bytes
        64

    """
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        cloudpickle.dump(o, buffer)
        m.update(buffer.getbuffer())
    return m.digest()


def hash_object(o: Any) -> int:
    """Hash a Python object to an integer using BLAKE2b.

    Convenience wrapper around hash_object_bytes() that converts the
    digest to an integer representation.

    Args:
        o: The object to hash (any dill-serializable object).

    Returns:
        The hash as a large integer.

    Example:
        >>> config1 = {"model": "bert"}
        >>> config2 = {"model": "gpt"}
        >>> hash_object(config1) != hash_object(config2)
        True

    """
    return int.from_bytes(hash_object_bytes(o), "big")


def murmurhash3(key: str, seed: int = 0) -> int:
    """Compute MurmurHash3 hash of a string.

    MurmurHash3 is a fast, non-cryptographic hash function suitable for
    hash tables and other applications where speed is more important than
    cryptographic strength.

    Args:
        key: The string to hash.
        seed: Hash seed for generating different hash values. Defaults to 0.

    Returns:
        A 32-bit unsigned integer hash value.

    Example:
        >>> # Hash strings quickly
        >>> murmurhash3("token")
        123456789  # Example value
        >>>
        >>> # Use different seeds for different hash spaces
        >>> murmurhash3("token", seed=0) != murmurhash3("token", seed=1)
        True

    Note:
        - This is NOT cryptographically secure
        - Used for vocabulary hashing, feature hashing, etc.
        - Produces 32-bit output (0 to 4294967295)

    """
    length = len(key)

    remainder = length & 3
    n = length - remainder
    h1 = seed
    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    i = 0

    while i < n:
        k1 = (
            (ord(key[i]) & 0xFF)
            | ((ord(key[i + 1]) & 0xFF) << 8)
            | (ord(key[i + 2]) & 0xFF) << 16
            | (ord(key[i + 3]) & 0xFF) << 24
        )
        i += 4

        k1 = (((k1 & 0xFFFF) * c1) + ((((k1 >> 16) * c1) & 0xFFFF) << 16)) & 0xFFFFFFFF
        k1 = (k1 << 15) | (k1 >> 17)
        k1 = (((k1 & 0xFFFF) * c2) + ((((k1 >> 16) * c2) & 0xFFFF) << 16)) & 0xFFFFFFFF

        h1 ^= k1
        h1 = (h1 << 13) | (h1 >> 19)
        h1b = (((h1 & 0xFFFF) * 5) + ((((h1 >> 16) * 5) & 0xFFFF) << 16)) & 0xFFFFFFFF
        h1 = ((h1b & 0xFFFF) + 0x6B64) + ((((h1b >> 16) + 0xE654) & 0xFFFF) << 16)

    k1 = 0
    if remainder == 3:
        k1 ^= (ord(key[i + 2]) & 0xFF) << 16
        k1 ^= (ord(key[i + 1]) & 0xFF) << 8
        k1 ^= ord(key[i]) & 0xFF
    elif remainder == 2:
        k1 ^= (ord(key[i + 1]) & 0xFF) << 8
        k1 ^= ord(key[i]) & 0xFF
    elif remainder == 1:
        k1 ^= ord(key[i]) & 0xFF
    k1 = (((k1 & 0xFFFF) * c1) + ((((k1 >> 16) * c1) & 0xFFFF) << 16)) & 0xFFFFFFFF
    k1 = (k1 << 15) | (k1 >> 17)
    k1 = (((k1 & 0xFFFF) * c2) + ((((k1 >> 16) * c2) & 0xFFFF) << 16)) & 0xFFFFFFFF
    h1 ^= k1

    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (((h1 & 0xFFFF) * 0x85EBCA6B) + ((((h1 >> 16) * 0x85EBCA6B) & 0xFFFF) << 16)) & 0xFFFFFFFF
    h1 ^= h1 >> 13
    h1 = (((h1 & 0xFFFF) * 0xC2B2AE35) + ((((h1 >> 16) * 0xC2B2AE35) & 0xFFFF) << 16)) & 0xFFFFFFFF
    h1 ^= h1 >> 16

    return h1 >> 0
