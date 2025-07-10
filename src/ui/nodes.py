from __future__ import annotations
from abc import ABC, abstractmethod
from functools import total_ordering
from pathlib import Path
from typing import Any, Generic, Optional, Protocol, TypeVar, List, Literal
from operator import (
    add,
    sub,
    mul,
    truediv,
    floordiv,
    mod,
    pow,
    and_,
    or_,
    xor,
    lshift,
    rshift,
)

T = TypeVar("T")  # The concrete Python type each node represents


class _Comparable(Protocol):
    def __lt__(self, other: Any) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...


@total_ordering
class UINode(Generic[T], ABC):
    """
    A transparent wrapper around a real Python value that tracks node-graph
    state (dirty/connected) while *behaving* like the underlying value.
    """

    def __init__(
        self, id: str, name: str, description: str = "", default_value: T = None
    ) -> None:
        self.id: str = id
        self.name: str = name
        self.description: str = description
        self._value: Optional[T] = default_value

        self.type: str = self.__class__.__name__.replace("UINode", "").lower()
        self.is_dirty: bool = False
        self.is_connected: bool = False

    # -----------------------------------------------------------
    # Value lifecycle helpers
    # -----------------------------------------------------------
    @property
    def value(self) -> T:
        """Return the current value or the subclass' default."""
        return self._value if self._value is not None else self.default_value()

    @value.setter
    def value(self, new_value: T) -> None:
        self._value = new_value
        self.is_dirty = True

    def as_param(self) -> dict[str, Any]:
        return {self.id: self.value}

    @abstractmethod
    def default_value(self) -> T: ...

    # -----------------------------------------------------------
    # Transparent delegation
    # -----------------------------------------------------------
    def __getattr__(self, item: str):  # fall back to the wrapped value
        try:
            return getattr(self.value, item)
        except AttributeError as exc:
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {item!r}"
            ) from exc

    # -----------------------------------------------------------
    # Core dunder overrides so Python treats us as the wrapped type
    # -----------------------------------------------------------
    def __str__(self) -> str:  # str(node)
        return str(self.value)

    def __repr__(self) -> str:  # interactive debugging
        return f"<{self.__class__.__name__} {self.name}={self.value!r}>"

    def __bool__(self) -> bool:  # truth-value testing
        return bool(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __float__(self) -> float:
        return float(self.value)

    # arithmetic can be forwarded implicitly via __getattr__,
    # but equality / ordering need explicit hooks so Python knows what to use
    def __eq__(self, other: Any) -> bool:
        other_val = other.value if isinstance(other, UINode) else other
        return self.value == other_val

    def __lt__(self, other: Any) -> bool:
        if isinstance(self.value, _Comparable):
            other_val = other.value if isinstance(other, UINode) else other
            return self.value < other_val
        return NotImplemented

    def _binary(self, other: Any, op):
        """Forward a binary operator to the underlying value."""
        other_val = other.value if isinstance(other, UINode) else other
        return op(self.value, other_val)

    def _rbinary(self, other: Any, op):
        """Forward the *reversed* binary operator (other ∘ self)."""
        other_val = other.value if isinstance(other, UINode) else other
        return op(other_val, self.value)

    # ── arithmetic ────────────────────────────────────────────
    __add__ = lambda self, other: self._binary(other, add)
    __radd__ = lambda self, other: self._rbinary(other, add)
    __sub__ = lambda self, other: self._binary(other, sub)
    __rsub__ = lambda self, other: self._rbinary(other, sub)
    __mul__ = lambda self, other: self._binary(other, mul)
    __rmul__ = lambda self, other: self._rbinary(other, mul)
    __truediv__ = lambda self, other: self._binary(other, truediv)
    __rtruediv__ = lambda self, other: self._rbinary(other, truediv)
    __floordiv__ = lambda self, other: self._binary(other, floordiv)
    __rfloordiv__ = lambda self, other: self._rbinary(other, floordiv)
    __mod__ = lambda self, other: self._binary(other, mod)
    __rmod__ = lambda self, other: self._rbinary(other, mod)
    __pow__ = lambda self, other: self._binary(other, pow)
    __rpow__ = lambda self, other: self._rbinary(other, pow)

    # ── bitwise (for ints/bools) ──────────────────────────────
    __and__ = lambda self, other: self._binary(other, and_)
    __rand__ = lambda self, other: self._rbinary(other, and_)
    __or__ = lambda self, other: self._binary(other, or_)
    __ror__ = lambda self, other: self._rbinary(other, or_)
    __xor__ = lambda self, other: self._binary(other, xor)
    __rxor__ = lambda self, other: self._rbinary(other, xor)
    __lshift__ = lambda self, other: self._binary(other, lshift)
    __rlshift__ = lambda self, other: self._rbinary(other, lshift)
    __rshift__ = lambda self, other: self._binary(other, rshift)
    __rrshift__ = lambda self, other: self._rbinary(other, rshift)


class TextNode(UINode[str]):
    def default_value(self) -> str:
        return ""


class NumberNode(UINode[int]):
    def default_value(self) -> int:
        return 0


class FloatNode(UINode[float]):
    def default_value(self) -> float:
        return 0.0


class BoolNode(UINode[bool]):
    def default_value(self) -> bool:
        return False


class ListNode(UINode[List[Any]]):
    def default_value(self) -> List[Any]:
        return []


class FileNode(UINode[Path]):
    """
    Represents any file-like resource (local path, S3 URI, etc.).
    Extra helpers can be added here (exists, size, mime-type, …).
    """

    def default_value(self) -> Path:
        return Path("")  # empty path
