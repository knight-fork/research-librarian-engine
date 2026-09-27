"""arXiv API: early versions / preprints."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Optional

from src.discovery.base import SourceContext
from src.discovery.query import BoolQuery
from src.models import Paper
from src.processing.normalize import canonical_venue, finalize, normalize_arxiv_id

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
CATEGORIES = ["cs.CV", "eess.IV", "cs.LG", "cs.AI", "cs.CL", "physics.med-ph", "stat.ML"]


def parse_feed(xml_text: str) -> List[Paper]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", NS):
        aid = normalize_arxiv_id(e.findtext("a:id", default="", namespaces=NS))
        if not aid:
            continue
        journal_ref = (e.findtext("arxiv:journal_ref", default="", namespaces=NS) or "").strip()
        doi = e.findtext("arxiv:doi", default=None, namespaces=NS)
        p = Paper(
            title=" ".join((e.findtext("a:title", default="", namespaces=NS) or "").split()),
            authors=[a.findtext("a:name", default="", namespaces=NS) for a in e.findall("a:author", NS)],
            publication_date=(e.findtext("a:published", default="", namespaces=NS) or "")[:10] or None,
            venue="arXiv", venue_key="arxiv", venue_type="preprint",
            arxiv_id=aid, doi=doi, abstract=" ".join((e.findtext("a:summary", default="", namespaces=NS) or "").split()),
            url=f"https://arxiv.org/abs/{aid}", source="arxiv",
        )
        if journal_ref:
            # Keep the preprint as a preprint but remember the claimed venue (resolved via DOI/merge later).
            key, vtype = canonical_venue(journal_ref)
            p.classification["journal_ref"] = journal_ref
            if doi and key not in ("unknown",):
                p.venue, p.venue_key, p.venue_type = journal_ref, key, vtype
        out.append(finalize(p))
    return out


def search(ctx: SourceContext, q: BoolQuery, date_from: Optional[str] = None, date_to: Optional[str] = None,
           limit: Optional[int] = None) -> List[Paper]:
    query = f"({q.arxiv()}) AND (" + " OR ".join(f"cat:{c}" for c in CATEGORIES) + ")"
    if date_from or date_to:
        a = (date_from or "1991-01-01").replace("-", "") + "0000"
        b = (date_to or "2999-12-31").replace("-", "") + "2359"
        query += f" AND submittedDate:[{a} TO {b}]"
    limit = limit or ctx.max_results
    out: List[Paper] = []
    start = 0
    while start < limit:
        n = min(100, limit - start)
        xml_text = ctx.http.get(API, params={"search_query": query, "start": start, "max_results": n,
                                             "sortBy": "submittedDate", "sortOrder": "descending"}, expect="text", use_cache=True)
        batch = parse_feed(xml_text)
        for p in batch:
            p.queries.append(f"arxiv:{q.label}")
        out.extend(batch)
        if len(batch) < n:
            break
        start += n
    return out


def get(ctx: SourceContext, arxiv_id: str) -> Optional[Paper]:
    xml_text = ctx.http.get(API, params={"id_list": arxiv_id}, expect="text", use_cache=True)
    papers = parse_feed(xml_text)
    return papers[0] if papers else None
