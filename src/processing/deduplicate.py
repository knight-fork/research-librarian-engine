"""Deduplication among candidates and against the Zotero library.

Match order: DOI -> PMID -> arXiv id -> OpenAlex / Semantic Scholar id -> normalized title -> fuzzy title."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from src.models import Paper
from src.processing.normalize import is_preprint_doi, merge, normalize_title

ID_FIELDS = ("doi", "pmid", "arxiv_id", "openalex_id", "semantic_scholar_id")


def title_similarity(a: str, b: str) -> float:
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    # Cheap length gate before the O(n*m) ratio.
    if min(len(na), len(nb)) / max(len(na), len(nb)) < 0.85:
        return 0.0
    sm = SequenceMatcher(None, na, nb)
    if sm.real_quick_ratio() < 0.9 or sm.quick_ratio() < 0.9:
        return 0.0
    return sm.ratio()


@dataclass
class Record:
    """A minimal identity record (a Zotero item or a candidate)."""
    key: str
    title: str
    ids: Dict[str, str] = field(default_factory=dict)
    item_type: str = ""
    year: Optional[int] = None


_DISTINGUISHING = re.compile(r"\b(?:part|phase|stage|type|grade|volume|vol)\s+([ivx]+|\d+)\b|\b(\d+[a-z]?)\b")
_LEADING_STOP = re.compile(r"^(?:(?:a|an|the|towards?|on)\s+)+")


def distinguishing_tokens(normalized_title: str) -> set:
    """Tokens that make near-identical titles different works: 'part ii', '3d', '2024', 'chexpert 14'."""
    return {a or b for a, b in _DISTINGUISHING.findall(normalized_title)}


def same_work_title(a: str, b: str, threshold: float = 0.96) -> bool:
    """Fuzzy title equality that never equates 'Part I'/'Part II', '2D'/'3D' or different years."""
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb or distinguishing_tokens(na) != distinguishing_tokens(nb):
        return False
    return title_similarity(na, nb) >= threshold


class DedupIndex:
    def __init__(self, fuzzy_threshold: float = 0.96):
        self.fuzzy_threshold = fuzzy_threshold
        self.by_id: Dict[Tuple[str, str], str] = {}
        self.by_title: Dict[str, List[str]] = {}
        self.records: Dict[str, Record] = {}
        self.titles: Dict[str, List[str]] = {}          # key -> every normalized title seen for that work
        self.key_ids: Dict[str, Dict[str, set]] = {}    # key -> field -> identifier values
        self._buckets: Dict[str, List[str]] = {}

    def add(self, rec: Record) -> None:
        """Add a record; adding the same key again unions its identifiers and titles."""
        self.records.setdefault(rec.key, rec)
        ids = self.key_ids.setdefault(rec.key, {})
        for k, v in rec.ids.items():
            if v:
                self.by_id.setdefault((k, str(v).lower()), rec.key)
                ids.setdefault(k, set()).add(str(v).lower())
        nt = normalize_title(rec.title)
        if nt and nt not in self.titles.setdefault(rec.key, []):
            self.titles[rec.key].append(nt)
            self.by_title.setdefault(nt, []).append(rec.key)
            self._buckets.setdefault(self._bucket(nt), []).append(rec.key)

    @staticmethod
    def _bucket(nt: str) -> str:
        return _LEADING_STOP.sub("", nt).replace(" ", "")[:6]

    def _conflicts(self, ids: Dict[str, str], key: str) -> bool:
        """Different publisher DOIs or different arXiv ids prove two records are different works."""
        have = self.key_ids.get(key, {})
        doi = (ids.get("doi") or "").lower()
        if doi and not is_preprint_doi(doi):
            theirs = {d for d in have.get("doi", set()) if not is_preprint_doi(d)}
            if theirs and doi not in theirs:
                return True
        arx = (ids.get("arxiv_id") or "").lower()
        if arx and have.get("arxiv_id") and arx not in have["arxiv_id"]:
            return True
        return False

    def find(self, ids: Dict[str, str], title: str) -> Tuple[Optional[str], str]:
        """Return (record key, match method) or (None, '')."""
        for k in ID_FIELDS:
            v = ids.get(k)
            if v and (k, str(v).lower()) in self.by_id:
                return self.by_id[(k, str(v).lower())], k
        nt = normalize_title(title)
        if not nt:
            return None, ""
        for key in self.by_title.get(nt, []):
            if not self._conflicts(ids, key):
                return key, "title"
        tokens = distinguishing_tokens(nt)
        if len(self.records) < 20000:
            candidates = list(self.records.keys())
        else:
            candidates = self._buckets.get(self._bucket(nt), [])
        best, best_key = 0.0, None
        for key in candidates:
            for other in self.titles.get(key, []):
                if distinguishing_tokens(other) != tokens:
                    continue
                sim = title_similarity(nt, other)
                if sim > best and not self._conflicts(ids, key):
                    best, best_key = sim, key
        if best_key and best >= self.fuzzy_threshold:
            return best_key, f"fuzzy_title({best:.3f})"
        return None, ""

    def find_paper(self, p: Paper) -> Tuple[Optional[str], str]:
        return self.find(p.identifiers(), p.title)


def dedupe_candidates(papers: List[Paper], fuzzy_threshold: float = 0.96) -> List[Paper]:
    """Collapse records of the same paper returned by different sources / queries.

    Every title and identifier a cluster has carried stays indexed, so the result does not depend on the
    order in which sources returned their records."""
    index = DedupIndex(fuzzy_threshold)
    out: Dict[str, Paper] = {}
    for p in papers:
        key, _ = index.find_paper(p)
        if key is not None:
            out[key] = merge(out[key], p)
            index.add(Record(key=key, title=p.title, ids=p.identifiers()))
            index.add(Record(key=key, title=out[key].title, ids=out[key].identifiers()))
        else:
            key = f"c{len(out)}"
            out[key] = p
            index.add(Record(key=key, title=p.title, ids=p.identifiers()))
    return list(out.values())
