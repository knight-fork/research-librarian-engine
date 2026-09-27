"""Canonical paper schema and search-request types."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional


@dataclass
class Paper:
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    publication_date: Optional[str] = None  # ISO yyyy-mm-dd (or yyyy-mm / yyyy)
    venue: str = ""
    venue_type: str = "preprint"  # conference | journal | preprint | workshop
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    openalex_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    url: str = ""
    pdf_url: Optional[str] = None
    abstract: str = ""
    citation_count: Optional[int] = None
    topics: List[str] = field(default_factory=list)
    modalities: List[str] = field(default_factory=list)
    source: str = ""
    relevance_score: float = 0.0
    quality_score: float = 0.0
    final_score: float = 0.0
    discovery_reason: str = ""

    # --- extensions beyond the canonical schema ---
    venue_key: str = "unknown"          # canonical venue id, e.g. "miccai"
    sources: List[str] = field(default_factory=list)   # every source that returned this paper
    queries: List[str] = field(default_factory=list)   # discovery queries that surfaced it
    publication_type: str = ""          # raw type from source (editorial, review, ...)
    pmcid: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    issn: Optional[str] = None
    pdf_candidates: Dict[str, str] = field(default_factory=dict)  # kind -> url
    subtopics: List[str] = field(default_factory=list)  # taxonomy tags
    classification: Dict[str, Any] = field(default_factory=dict)
    score_breakdown: Dict[str, Any] = field(default_factory=dict)
    decision: str = ""                  # accept | review | reject
    reject_reason: str = ""
    author_watch: Optional[str] = None
    existing_zotero_key: Optional[str] = None
    duplicate_of: Optional[str] = None
    related_preprint_key: Optional[str] = None  # Zotero preprint this published paper upgrades
    relationship: Optional[str] = None  # citation-expansion relationship to a seed paper
    references: List[str] = field(default_factory=list)  # external ids of references, when fetched
    author_ids: List[str] = field(default_factory=list)  # "openalex:A..." / "s2:..." for exact author-watch matching
    graph_evidence: Dict[str, Any] = field(default_factory=dict)  # citation-graph evidence (e.g. baseline score / contexts)
    extra_tags: List[str] = field(default_factory=list)  # per-paper Zotero tags (e.g. "relation:baseline")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Paper":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def is_peer_reviewed(self) -> bool:
        return self.venue_type in ("conference", "journal")

    def age_years(self, today: Optional[date] = None) -> float:
        today = today or date.today()
        d = parse_date(self.publication_date) or (date(self.year, 7, 1) if self.year else None)
        if d is None:
            return 3.0
        return max(0.0, (today - d).days / 365.25)

    def identifiers(self) -> Dict[str, str]:
        ids = {}
        for k in ("doi", "pmid", "arxiv_id", "openalex_id", "semantic_scholar_id"):
            v = getattr(self, k)
            if v:
                ids[k] = v
        return ids


def parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    parts = str(value)[:10].replace("/", "-").split("-")
    try:
        y = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 and parts[1] else 1
        d = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        return date(y, max(1, min(m, 12)), max(1, min(d, 28 if m == 2 else 30 if m in (4, 6, 9, 11) else 31)))
    except (ValueError, IndexError):
        return None


@dataclass
class SearchRequest:
    """Deterministic search parameters (what NL commands compile into)."""
    intent: str = "topic_search"  # scan | topic_search | author_search | venue_search | similar | citations | related | missing_lit | add_doi
    topics: List[str] = field(default_factory=list)
    modalities: List[str] = field(default_factory=list)
    free_text: Optional[str] = None
    author: Optional[str] = None
    author_openalex_id: Optional[str] = None
    author_s2_id: Optional[str] = None
    venues: List[str] = field(default_factory=list)
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    top_venues_only: bool = False
    seed: Optional[str] = None           # DOI / title / Zotero key for expansion intents
    collection: Optional[str] = None     # for missing-literature analysis
    limit: Optional[int] = None
    action: str = "add_by_threshold"     # add_by_threshold | report_only
    source_tag: str = "manual-prompt"    # auto-discovery | author-watch | manual-prompt
    strict_request_match: bool = True    # require requested topic/modality match
    extra_tags: List[str] = field(default_factory=list)  # added to every Zotero item written (e.g. "alert:<id>")
    label: Optional[str] = None          # human-readable run name for reports (e.g. the alert name)
    include_seed: bool = False           # baselines: also add the seed paper itself
    target_collection: Optional[str] = None  # file accepted papers into this collection path (from the library root)
    parse_notes: List[str] = field(default_factory=list)  # how a prompt was interpreted (typo fixes, ambiguities)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
