"""PMLR: ICML / MIDL and other PMLR-published proceedings.

PMLR is a proceedings platform, not a venue: papers inherit the configured conference."""
from __future__ import annotations

import html
import re
from typing import List

from src.discovery.base import SourceContext
from src.http import HttpError
from src.models import Paper
from src.processing.classify import MODALITY_TERMS, _hits
from src.processing.normalize import finalize, strip_markup

BASE = "https://proceedings.mlr.press"
_PAPER_RE = re.compile(r'<div class="paper">(.*?)</div>', re.S)
VENUE_NAMES = {"icml": "International Conference on Machine Learning", "midl": "Medical Imaging with Deep Learning"}


def _parse_volume(page: str) -> List[dict]:
    rows = []
    for block in _PAPER_RE.findall(page):
        title = re.search(r'<p class="title">(.*?)</p>', block, re.S)
        authors = re.search(r'<span class="authors">(.*?)</span>', block, re.S)
        abs_url = re.search(r'href="(https?://proceedings\.mlr\.press/v\d+/[^"]+\.html)"', block)
        pdf_url = re.search(r'href="([^"]+\.pdf)"', block)
        pages = re.search(r"PMLR \d+:([\w\-]+)", block)
        if not title:
            continue
        rows.append({
            "title": strip_markup(title.group(1)),
            "authors": [a.strip() for a in html.unescape(re.sub(r"<[^>]+>", "", authors.group(1))).replace("\xa0", " ").split(",")] if authors else [],
            "abs_url": abs_url.group(1) if abs_url else None,
            "pdf_url": pdf_url.group(1) if pdf_url else None,
            "pages": pages.group(1) if pages else None,
        })
    return rows


def _abstract(ctx: SourceContext, url: str) -> tuple:
    try:
        page = ctx.http.get(url, expect="text", use_cache=True, cache_ttl=30 * 86400)
    except HttpError:
        return "", None
    m = re.search(r'<div id="abstract" class="abstract">(.*?)</div>', page, re.S)
    d = re.search(r'name="citation_publication_date" content="([\d/]+)"', page)
    return (strip_markup(m.group(1)) if m else ""), (d.group(1).replace("/", "-") if d else None)


def volume_papers(ctx: SourceContext, volume: int, venue_key: str, year: int) -> List[Paper]:
    """Papers in a PMLR volume. For large volumes only radiology-looking titles are kept."""
    try:
        page = ctx.http.get(f"{BASE}/v{volume}/", expect="text", use_cache=True, cache_ttl=7 * 86400)
    except HttpError as exc:
        ctx.warn("pmlr", f"v{volume}: {exc}")
        return []
    out = []
    rows = _parse_volume(page)
    # Large general-ML volumes (ICML): only fetch abstracts for radiology-looking titles.
    prefilter = len(rows) > 300
    for row in rows:
        title = row["title"]
        if title.lower() in ("preface", "front matter"):
            continue
        if prefilter and not any(_hits(terms, title)[0] > 0 for terms in MODALITY_TERMS.values()):
            continue
        abstract, pub_date = _abstract(ctx, row["abs_url"]) if row["abs_url"] else ("", None)
        p = Paper(
            title=title, authors=row["authors"], year=year, publication_date=pub_date,
            venue=f"Proceedings of {VENUE_NAMES.get(venue_key, venue_key.upper())} (PMLR v{volume})",
            venue_key=venue_key, venue_type="conference", url=row["abs_url"] or "", abstract=abstract,
            pages=row["pages"], volume=str(volume), publisher="PMLR", source="pmlr", publication_type="proceedings-article",
        )
        if row["pdf_url"]:
            p.pdf_candidates["proceedings"] = row["pdf_url"]
        p.queries.append(f"pmlr:v{volume}")
        out.append(finalize(p))
    return out
