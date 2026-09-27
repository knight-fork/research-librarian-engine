"""Configuration loading: config.yaml, authors.yaml and environment secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = ROOT / "reports"
STATE_PATH = DATA_DIR / "state.json"
AUDIT_PATH = DATA_DIR / "audit.jsonl"
PROPOSALS_PATH = DATA_DIR / "proposals.json"


@dataclass
class Secrets:
    zotero_library_id: Optional[str] = None
    zotero_library_type: str = "user"
    zotero_api_key: Optional[str] = None
    semantic_scholar_api_key: Optional[str] = None
    openalex_api_key: Optional[str] = None
    ncbi_api_key: Optional[str] = None
    openreview_username: Optional[str] = None
    openreview_password: Optional[str] = None
    contact_email: Optional[str] = None
    gemini_api_key: Optional[str] = None

    @property
    def zotero_configured(self) -> bool:
        return bool(self.zotero_library_id and self.zotero_api_key)


@dataclass
class Config:
    raw: Dict[str, Any]
    authors: List[Dict[str, Any]] = field(default_factory=list)
    secrets: Secrets = field(default_factory=Secrets)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def auto_add_threshold(self) -> float:
        return float(self.get("thresholds.auto_add", 0.82))

    @property
    def review_threshold(self) -> float:
        return float(self.get("thresholds.review", 0.68))

    @property
    def venue_weights(self) -> Dict[str, float]:
        return {k: float(v) for k, v in (self.get("venue_weights", {}) or {}).items()}

    @property
    def high_priority_venues(self) -> List[str]:
        return list(self.get("high_priority_venues", []) or [])

    def source_enabled(self, name: str) -> bool:
        return bool(self.get(f"sources.{name}", False))

    @property
    def profile(self) -> str:
        """general: relevance from the query's own terms (any field). radiology: built-in CXR/CT/mammography ontology."""
        return str(self.get("profile", "general")).strip().lower()

    @property
    def is_radiology(self) -> bool:
        return self.profile == "radiology"

    @property
    def writes_enabled(self) -> bool:
        return bool(self.get("safety.writes_enabled", False))


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name, "").strip()
    return value or None


def load_config(config_path: Optional[Path] = None, authors_path: Optional[Path] = None) -> Config:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
    config_path = config_path or ROOT / "config.yaml"
    authors_path = authors_path or ROOT / "authors.yaml"
    raw = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    authors: List[Dict[str, Any]] = []
    if authors_path.exists():
        authors = (yaml.safe_load(authors_path.read_text()) or {}).get("authors") or []
    secrets = Secrets(
        zotero_library_id=_env("ZOTERO_LIBRARY_ID"),
        zotero_library_type=(_env("ZOTERO_LIBRARY_TYPE") or "user").split()[0].lower(),
        zotero_api_key=_env("ZOTERO_API_KEY"),
        semantic_scholar_api_key=_env("SEMANTIC_SCHOLAR_API_KEY"),
        openalex_api_key=_env("OPENALEX_API_KEY"),
        ncbi_api_key=_env("NCBI_API_KEY"),
        openreview_username=_env("OPENREVIEW_USERNAME"),
        openreview_password=_env("OPENREVIEW_PASSWORD"),
        contact_email=_env("CONTACT_EMAIL"),
        gemini_api_key=_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
    )
    for d in (DATA_DIR, CACHE_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return Config(raw=raw or {}, authors=authors, secrets=secrets)
