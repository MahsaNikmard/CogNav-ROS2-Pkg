from dataclasses import dataclass, field
from typing import Any


@dataclass
class Blackboard:
    """Key-value store shared by the leaves of one mission."""

    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value
