"""Crossref: DOI validation and canonical metadata."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.discovery.base import SourceContext
from src.http import HttpError
from src.models import Paper
from src.processing.normalize import canonical_venue, finalize, strip_markup

BASE = "https://api.crossref.org"

TYPE_HINT = {"journal-article": "journal", "proceedings-article": "conference", "posted-content": "preprint",
             "book-chapter": "conference"}


def _params(ctx: SourceContext, extra: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(extra)
    if ctx.cfg.secrets.contact_email:
        p["mailto"] = ctx.cfg.secrets.contact_email
    return p


def _date(m: Dict[str, Any]) -> Optional[str]:
    for k in ("issued", "published-online", "published-print", "created"):
        parts = ((m.get(k) or {}).get("date-parts") or [[None]])[0]
        if parts and parts[0]:
            return "-".join(f"{x:02d}" if i else str(x) for i, x in enumerate(parts))
    return None


def to_paper(m: Dict[str, Any]) -> Paper:
    ctype = m.get("type", "")
    containers = [c for c in (m.get("container-title") or []) if c]
    # Prefer the volume title over the series ("Lecture Notes in Computer Science").
    specific = [c for c in containers if not c.lower().startswith(("lecture notes", "communications in computer", "proceedings of spie"))]
    container = (specific or containers or [""])[0]
    event = (m.get("event") or {}).get("name") if isinstance(m.get("event"), dict) else None
    for a in m.get("assertion") or []:
        if a.get("name") == "conference_name" and not event:
            event = a.get("value")
    venue = container or event or ""
    # MICCAI & co. are published as LNCS book chapters whose container title names the conference.
    vkey, vtype = canonical_venue(f"{venue} {event or ''}".strip(), TYPE_HINT.get(ctype, ""))
    if ctype == "posted-content":
        vtype = "preprint"
        if vkey not in ("arxiv", "medrxiv"):
            vkey = "preprint"
    authors = []
    for a in m.get("author") or []:
        name = " ".join(x for x in (a.get("given"), a.get("family")) if x) or a.get("name", "")
        authors.append(name)
    date = _date(m)
    p = Paper(
        title=strip_markup((m.get("title") or [""])[0]),
        authors=authors, publication_date=date,
        year=int(date[:4]) if date else None,
        venue=venue, venue_key=vkey, venue_type=vtype,
        doi=m.get("DOI"), url=m.get("URL") or "",
        abstract=strip_markup(m.get("abstract") or ""),
        citation_count=m.get("is-referenced-by-count"),
        source="crossref", publication_type=ctype,
        volume=m.get("volume"), issue=m.get("issue"), pages=m.get("page"),
        publisher=m.get("publisher"), issn=(m.get("ISSN") or [None])[0],
    )
    if ctype in ("journal-article",) and (m.get("subtype") == "editorial"):
        p.publication_type = "editorial"
    return finalize(p)


def get_doi(ctx: SourceContext, doi: str) -> Optional[Paper]:
    try:
        data = ctx.http.get(f"{BASE}/works/{doi}", params=_params(ctx, {}), use_cache=True, cache_ttl=7 * 86400)
    except HttpError as exc:
        if exc.status != 404:
            ctx.warn("crossref", f"DOI {doi}: {exc}")
        return None
    return to_paper(data.get("message") or {})


def find_by_title(ctx: SourceContext, title: str, rows: int = 5) -> List[Paper]:
    try:
        data = ctx.http.get(f"{BASE}/works", params=_params(ctx, {"query.bibliographic": title, "rows": rows}), use_cache=True)
    except HttpError as exc:
        ctx.warn("crossref", f"title search failed: {exc}")
        return []
    return [to_paper(m) for m in (data.get("message") or {}).get("items") or []]
