"""Paper <-> Zotero item mapping and library indexing for dedup."""
from __future__ import annotations

import html
import re
from datetime import date
from typing import Any, Dict, List, Optional

from src.models import Paper
from src.processing.deduplicate import DedupIndex, Record
from src.processing.normalize import normalize_arxiv_id, normalize_doi, normalize_pmid, venue_tag

MODALITY_TAG = {"cxr": "modality:cxr", "ct": "modality:ct", "mammo": "modality:mammo", "radiology-general": "modality:radiology-general"}
TOPIC_TAG = {
    "foundation-model": "topic:foundation-model", "vlm": "topic:vlm", "uncertainty": "topic:uncertainty",
    "calibration": "topic:calibration", "conformal-prediction": "topic:conformal-prediction", "ood": "topic:ood",
    "distribution-shift": "topic:distribution-shift", "selective-prediction": "topic:selective-prediction",
    "robustness": "topic:robustness", "report-generation": "topic:report-generation",
    "image-text-pretraining": "topic:image-text-pretraining", "vlm-reliability": "topic:vlm-reliability",
}


def split_name(name: str) -> Dict[str, str]:
    name = " ".join(name.split())
    if "," in name:
        last, first = [x.strip() for x in name.split(",", 1)]
        return {"creatorType": "author", "firstName": first, "lastName": last}
    parts = name.split(" ")
    if len(parts) == 1:
        return {"creatorType": "author", "name": name}
    # Keep lowercase particles with the last name (van, de, von, ...).
    i = len(parts) - 1
    while i > 1 and parts[i - 1].lower() in ("van", "von", "de", "der", "den", "da", "di", "del", "la", "le", "dos", "du"):
        i -= 1
    return {"creatorType": "author", "firstName": " ".join(parts[:i]), "lastName": " ".join(parts[i:])}


def build_tags(p: Paper, decision: str, source_tag: str, high_priority_score: float = 0.90) -> List[str]:
    tags = [MODALITY_TAG[m] for m in p.modalities if m in MODALITY_TAG]
    tags += [TOPIC_TAG[t] for t in p.topics if t in TOPIC_TAG]
    tags += list(p.subtopics)
    tags.append(f"source:{source_tag}")
    if p.author_watch and source_tag != "author-watch":
        tags.append("source:author-watch")
    tags.append("status:unread")
    if decision == "review":
        tags.append("status:review-required")
    elif p.final_score >= high_priority_score:
        tags.append("status:high-priority")
    vt = venue_tag(p.venue_key)
    if vt:
        tags.append(vt)
    if p.venue_type == "preprint":
        tags.append("type:preprint")
    return list(dict.fromkeys(tags))


def build_extra(p: Paper) -> str:
    lines = []
    if p.arxiv_id:
        lines.append(f"arXiv: {p.arxiv_id}")
    if p.pmid:
        lines.append(f"PMID: {p.pmid}")
    if p.pmcid:
        lines.append(f"PMCID: {p.pmcid}")
    if p.openalex_id:
        lines.append(f"OpenAlex: {p.openalex_id}")
    if p.semantic_scholar_id:
        lines.append(f"Semantic Scholar: {p.semantic_scholar_id}")
    return "\n".join(lines)


def provenance_note(p: Paper, today: Optional[date] = None) -> str:
    today = today or date.today()
    b = p.score_breakdown or {}
    rows = [
        "Auto-discovered by research-librarian.",
        f"Reason: {p.discovery_reason or '-'}",
        f"Source: {', '.join(p.sources) or p.source}",
        f"Discovery query: {'; '.join(p.queries[:5]) or '-'}",
        f"Discovery date: {today.isoformat()}",
        f"Relevance score: {p.relevance_score:.2f} | Quality score: {p.quality_score:.2f} | Final score: {p.final_score:.2f}",
        f"Decision: {p.decision}" + (f" ({p.reject_reason})" if p.reject_reason else ""),
    ]
    if b:
        rows.append("Breakdown: " + ", ".join(f"{k}={v:.2f}" for k, v in b.items() if isinstance(v, (int, float))))
    return "".join(f"<p>{html.escape(r)}</p>" for r in rows)


def to_zotero_item(p: Paper, collection_keys: List[str], tags: List[str]) -> Dict[str, Any]:
    creators = [split_name(a) for a in p.authors]
    common: Dict[str, Any] = {
        "title": p.title,
        "creators": creators,
        "abstractNote": p.abstract,
        "date": p.publication_date or (str(p.year) if p.year else ""),
        "url": p.url or "",
        "extra": build_extra(p),
        "tags": [{"tag": t} for t in tags],
        "collections": collection_keys,
        "relations": {},
    }
    doi = p.doi or ""
    if p.venue_type == "journal":
        item = {"itemType": "journalArticle", **common, "publicationTitle": p.venue, "volume": p.volume or "",
                "issue": p.issue or "", "pages": p.pages or "", "DOI": doi, "ISSN": p.issn or ""}
    elif p.venue_type in ("conference", "workshop"):
        item = {"itemType": "conferencePaper", **common, "proceedingsTitle": p.venue, "conferenceName": _conference_name(p),
                "pages": p.pages or "", "DOI": doi, "publisher": p.publisher or "", "volume": p.volume or ""}
    else:
        repo = "arXiv" if p.arxiv_id else (p.venue or "")
        item = {"itemType": "preprint", **common, "repository": repo,
                "archiveID": f"arXiv:{p.arxiv_id}" if p.arxiv_id else "", "DOI": doi or (f"10.48550/arXiv.{p.arxiv_id}" if p.arxiv_id else "")}
    return item


def _conference_name(p: Paper) -> str:
    from src.processing.normalize import VENUE_DISPLAY
    name = VENUE_DISPLAY.get(p.venue_key)
    if name and p.year:
        return f"{name} {p.year}"
    return name or ""


def note_item(parent_key: Optional[str], html_note: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    item: Dict[str, Any] = {"itemType": "note", "note": html_note, "tags": [{"tag": t} for t in (tags or [])], "collections": [], "relations": {}}
    if parent_key:
        item["parentItem"] = parent_key
    return item


# ---------------------------------------------------------------- library index

_EXTRA_PATTERNS = {
    "arxiv_id": re.compile(r"arxiv:\s*([^\s]+)", re.I),
    "pmid": re.compile(r"pmid:\s*(\d+)", re.I),
    "openalex_id": re.compile(r"openalex:\s*(W\d+)", re.I),
    "semantic_scholar_id": re.compile(r"semantic scholar:\s*([0-9a-f]{40})", re.I),
    "doi": re.compile(r"doi:\s*(10\.\S+)", re.I),
}


def record_from_item(data: Dict[str, Any]) -> Optional[Record]:
    if data.get("itemType") in ("note", "attachment", "annotation") or data.get("deleted"):
        return None
    ids: Dict[str, str] = {}
    doi = normalize_doi(data.get("DOI"))
    extra = data.get("extra") or ""
    for k, rx in _EXTRA_PATTERNS.items():
        m = rx.search(extra)
        if m and k not in ids:
            ids[k] = m.group(1)
    if doi:
        ids["doi"] = doi
    elif ids.get("doi"):
        ids["doi"] = normalize_doi(ids["doi"]) or ""
    url = data.get("url") or ""
    arx = normalize_arxiv_id(data.get("archiveID") or "") or (normalize_arxiv_id(url) if "arxiv.org" in url else None) \
        or normalize_arxiv_id(ids.get("arxiv_id")) or (normalize_arxiv_id(doi) if doi and doi.startswith("10.48550/") else None)
    if arx:
        ids["arxiv_id"] = arx
    else:
        ids.pop("arxiv_id", None)
    if ids.get("pmid"):
        ids["pmid"] = normalize_pmid(ids["pmid"]) or ""
    if "pubmed.ncbi.nlm.nih.gov" in url and "pmid" not in ids:
        pm = normalize_pmid(url)
        if pm:
            ids["pmid"] = pm
    year = None
    m = re.search(r"(19|20)\d{2}", data.get("date") or "")
    if m:
        year = int(m.group(0))
    return Record(key=data["key"], title=data.get("title") or "", ids={k: v for k, v in ids.items() if v},
                  item_type=data.get("itemType", ""), year=year)


def build_library_index(items: List[Dict[str, Any]], fuzzy_threshold: float = 0.96) -> DedupIndex:
    index = DedupIndex(fuzzy_threshold)
    for data in items:
        rec = record_from_item(data)
        if rec:
            index.add(rec)
    return index
