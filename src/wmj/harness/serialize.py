"""wmj.harness.serialize — the one shared canonical JSON serializer.

Every machine-readable artefact this project writes goes through
canonical_serialize, so NF-1's byte-identity comparison has one code
path to trust (cross-cutting spec ADR-002 rule 4).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from wmj.errors import WmjError


class NonFiniteValueError(WmjError, ValueError):
    """Raised when a NaN or infinite value would be serialized."""


def canonical_serialize(obj: Any) -> bytes:
    """Serialize obj to the project's one canonical JSON byte form.

    In plain words: turns Python/NumPy data into JSON bytes that come
    out exactly the same every time — same key order, same float
    formatting, same encoding — so two runs can be compared byte for
    byte (NF-1).
    """
    native = _to_native_types(obj)
    try:
        text = json.dumps(native, sort_keys=True, indent=2, allow_nan=False)
    except ValueError as exc:
        raise NonFiniteValueError(
            f"canonical_serialize refuses a non-finite value (NaN/Infinity): {exc} "
            f"(cross-cutting ADR-002 rule 4)"
        ) from exc
    return text.encode("utf-8")


def _to_native_types(obj: Any) -> Any:
    """Recursively convert NumPy arrays/scalars to native Python types.

    .tolist() on an ndarray and .item() on a NumPy scalar both return
    native Python floats/ints — never triggering NumPy's own __repr__,
    which changed under NEP 51 (NumPy >=2.0) in a way that breaks a
    hand-rolled repr()-based serializer. json.dumps never calls
    __repr__ on a float either, so once everything here is native,
    json.dumps is the only thing that ever renders a float to text.
    """
    if isinstance(obj, np.ndarray):
        return _to_native_types(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: _to_native_types(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native_types(value) for value in obj]
    return obj
