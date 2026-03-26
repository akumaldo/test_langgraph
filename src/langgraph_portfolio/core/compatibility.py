from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib.util
from typing import Any, ClassVar, get_origin, get_type_hints


try:  # pragma: no cover - exercised only when pydantic is installed
    from pydantic import BaseModel as _PydanticBaseModel
    from pydantic import Field as _PydanticField
except Exception:  # pragma: no cover - optional dependency
    _PydanticBaseModel = None
    _PydanticField = None


@dataclass(frozen=True)
class FieldSpec:
    default: Any = ...
    default_factory: Any | None = None

    def resolve(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        if self.default is ...:
            return None
        return deepcopy(self.default)


def Field(*, default: Any = ..., default_factory: Any | None = None) -> Any:
    if _PydanticField is not None:  # pragma: no cover - optional dependency
        kwargs: dict[str, Any] = {}
        if default is not ...:
            kwargs["default"] = default
        if default_factory is not None:
            kwargs["default_factory"] = default_factory
        return _PydanticField(**kwargs)
    return FieldSpec(default=default, default_factory=default_factory)


if _PydanticBaseModel is not None:  # pragma: no cover - optional dependency
    BaseModel = _PydanticBaseModel
else:

    class BaseModel:
        """Small Pydantic-compatible stand-in used when the dependency is absent."""

        model_config: ClassVar[dict[str, Any]] = {}

        def __init__(self, **data: Any) -> None:
            field_names = self._field_names()
            for name in field_names:
                if name.startswith("_"):
                    continue
                if name in data:
                    value = data[name]
                else:
                    value = self._default_for(name)
                setattr(self, name, value)

            for key, value in data.items():
                if key not in field_names:
                    setattr(self, key, value)

        @classmethod
        def model_validate(cls, data: dict[str, Any]) -> "BaseModel":
            return cls(**data)

        def model_dump(self) -> dict[str, Any]:
            return {name: self._dump_value(getattr(self, name, None)) for name in self._field_names()}

        def model_copy(self, *, update: dict[str, Any] | None = None) -> "BaseModel":
            payload = self.model_dump()
            if update:
                payload.update(update)
            return self.__class__(**payload)

        def _default_for(self, name: str) -> Any:
            value = getattr(self.__class__, name, None)
            if isinstance(value, FieldSpec):
                return value.resolve()
            if value is not None:
                return deepcopy(value)
            return None

        def _field_names(self) -> list[str]:
            hints = get_type_hints(self.__class__, include_extras=True)
            return [
                name
                for name, hint in hints.items()
                if not name.startswith("_") and get_origin(hint) is not ClassVar
            ]

        def _dump_value(self, value: Any) -> Any:
            if hasattr(value, "model_dump"):
                return value.model_dump()
            if isinstance(value, dict):
                return {key: self._dump_value(item) for key, item in value.items()}
            if isinstance(value, list):
                return [self._dump_value(item) for item in value]
            if isinstance(value, tuple):
                return [self._dump_value(item) for item in value]
            if hasattr(value, "__dict__") and not isinstance(value, type):
                return {
                    key: self._dump_value(item)
                    for key, item in vars(value).items()
                    if not key.startswith("_")
                }
            return value

        def __repr__(self) -> str:  # pragma: no cover - convenience only
            return f"{self.__class__.__name__}({self.model_dump()!r})"


def framework_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None
