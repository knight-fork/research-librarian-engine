"""OpenReview (API v2): ICLR / NeurIPS and other OpenReview-hosted venues.

Anonymous API access is blocked by a bot challenge, so this source needs
OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD. Without them it is skipped with a warning."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.discovery.base import SourceContext
from src.http import HttpError
from src.models import Paper
from src.processing.classify import MODALITY_TERMS, _hits
from src.processing.normalize import finalize

API = "https://api2.openreview.net"
_TOKEN: Dict[str, Optional[str]] = {}


def _token(ctx: SourceContext) -> Optional[str]:
    s = ctx.cfg.secrets
    if not (s.openreview_username and s.openreview_password):
        return None
    if s.openreview_username not in _TOKEN:
        try:
            data = ctx.http.request("POST", f"{API}/login", json_body={"id": s.openreview_username, "password": s.openreview_password})
            _TOKEN[s.openreview_username] = data.get("token")
        except HttpError as exc:
            ctx.warn("openreview", f"login failed: {exc}")
            _TOKEN[s.openreview_username] = None
    return _TOKEN[s.openreview_username]


def _v(content: Dict[str, Any], key: str, default=None):
    val = content.get(key)
    return val.get("value", default) if isinstance(val, dict) else (val if val is not None else default)


def to_paper(note: Dict[str, Any], venue_key: str, year: int) -> Paper:
    c = note.get("content") or {}
    venue_str = _v(c, "venue", "") or ""
    vtype = "workshop" if "workshop" in venue_str.lower() else "conference"
    pdf = _v(c, "pdf")
    p = Paper(
        title=_v(c, "title", ""), authors=list(_v(c, "authors", []) or []), year=year,
        venue=venue_str or f"{venue_key.upper()} {year}", venue_key=venue_key, venue_type=vtype,
        abstract=_v(c, "abstract", ""), url=f"https://openreview.net/forum?id={note.get('forum') or note.get('id')}",
        source="openreview", publication_type="conference-paper",
    )
    if pdf:
        p.pdf_candidates["proceedings"] = "https://openreview.net" + pdf if pdf.startswith("/") else pdf
    return finalize(p)


def venue_papers(ctx: SourceContext, venue_id: str, venue_key: str, year: int, prefilter: bool = True) -> List[Paper]:
    """All accepted papers for a venue id, optionally pre-filtered to radiology-looking titles/abstracts."""
    tok = _token(ctx)
    if not tok:
        ctx.warn("openreview", "skipped (set OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD to enable)")
        return []
    out: List[Paper] = []
    offset = 0
    while True:
        try:
            data = ctx.http.get(f"{API}/notes", params={"content.venueid": venue_id, "limit": 1000, "offset": offset},
                                headers={"Authorization": f"Bearer {tok}"}, use_cache=True, cache_ttl=7 * 86400)
        except HttpError as exc:
            ctx.warn("openreview", f"{venue_id}: {exc}")
            break
        notes = data.get("notes") or []
        for n in notes:
            p = to_paper(n, venue_key, year)
            if prefilter and not _looks_radiology(p):
                continue
            p.queries.append(f"openreview:{venue_id}")
            out.append(p)
        if len(notes) < 1000:
            break
        offset += 1000
    return out


def _looks_radiology(p: Paper) -> bool:
    text = f"{p.title} {p.abstract}"
    return any(_hits(terms, text)[0] > 0 for terms in MODALITY_TERMS.values())
