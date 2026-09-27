"""Shared context for discovery sources."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from src.config import Config
from src.http import HttpClient

log = logging.getLogger("research_librarian.discovery")


@dataclass
class SourceContext:
    cfg: Config
    http: HttpClient
    errors: List[str] = field(default_factory=list)

    def warn(self, source: str, msg: str) -> None:
        line = f"{source}: {msg}"
        log.warning(line)
        self.errors.append(line)

    @property
    def max_results(self) -> int:
        return int(self.cfg.get("scan.max_results_per_query", 200))
