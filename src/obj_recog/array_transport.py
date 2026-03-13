from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any
import uuid

import numpy as np


@dataclass(frozen=True, slots=True)
class SharedMemoryArrayRef:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SharedMemoryArrayRef:
        return cls(
            name=str(payload["name"]),
            shape=tuple(int(value) for value in payload["shape"]),
            dtype=str(payload["dtype"]),
        )


class SharedMemoryArrayWriter:
    def __init__(self, *, prefix: str = "obj-recog") -> None:
        self._prefix = str(prefix)
        self._buffers: dict[str, tuple[shared_memory.SharedMemory, tuple[int, ...], str]] = {}

    def write(self, key: str, array: np.ndarray) -> SharedMemoryArrayRef:
        contiguous = np.ascontiguousarray(np.asarray(array))
        shape = tuple(int(value) for value in contiguous.shape)
        dtype = contiguous.dtype.str
        required_bytes = int(contiguous.nbytes)
        existing = self._buffers.get(str(key))
        if (
            existing is None
            or existing[1] != shape
            or existing[2] != dtype
            or existing[0].size != required_bytes
        ):
            self._dispose_buffer(str(key))
            name = f"{self._prefix}-{uuid.uuid4().hex}"
            shm = shared_memory.SharedMemory(name=name, create=True, size=required_bytes)
            self._buffers[str(key)] = (shm, shape, dtype)
        shm, _, _ = self._buffers[str(key)]
        target = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
        target[...] = contiguous
        return SharedMemoryArrayRef(name=shm.name, shape=shape, dtype=dtype)

    def close(self) -> None:
        for key in list(self._buffers):
            self._dispose_buffer(key)

    def _dispose_buffer(self, key: str) -> None:
        existing = self._buffers.pop(str(key), None)
        if existing is None:
            return
        shm = existing[0]
        try:
            shm.close()
        finally:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


def load_shared_memory_array(reference: SharedMemoryArrayRef, *, copy: bool = True) -> np.ndarray:
    shm = shared_memory.SharedMemory(name=reference.name, create=False)
    try:
        array = np.ndarray(reference.shape, dtype=np.dtype(reference.dtype), buffer=shm.buf)
        return array.copy() if copy else array
    finally:
        shm.close()
