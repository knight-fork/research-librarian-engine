"""Semantic Scholar: topic search, author-centric discovery, references / citations, recommendations."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.discovery.base import SourceContext
from src.discovery.query import BoolQuery
from src.http import HttpError
from src.models import Paper
from src.processing.normalize import canonical_venue, finalize

BASE = "https://api.semanticscholar.org/graph/v1"
REC = "https://api.semanticscholar.org/recommendations/v1"
FIELDS = ("title,authors,year,publicationDate,venue,publicationVenue,externalIds,url,abstract,citationCount,"
          "openAccessPdf,publicationTypes,journal")


def _headers(ctx: SourceContext) -> Dict[str, str]:
    key = ctx.cfg.secrets.semantic_scholar_api_key
    return {"x-api-key": key} if key else {}


def to_paper(d: Dict[str, Any]) -> Paper:
    ext = d.get("externalIds") if isinstance(d.get("externalIds"), dict) else {}
    pv = d.get("publicationVenue") if isinstance(d.get("publicationVenue"), dict) else {}
    journal = d.get("journal") if isinstance(d.get("journal"), dict) else {}
    venue = pv.get("name") or d.get("venue") or journal.get("name") or ""
    hint = (pv.get("type") or "").lower()
    ptypes = [t.lower() for t in (d.get("publicationTypes") or [])]
    if not hint:
        hint = "conference" if "conference" in ptypes else "journal" if "journalarticle" in ptypes else ""
    vkey, vtype = canonical_venue(venue, hint)
    # S2 venue strings for arXiv-only papers are often "arXiv.org" or empty.
    if vkey in ("unknown",) and ext.get("ArXiv") and not ext.get("DOI"):
        vkey, vtype = "arxiv", "preprint"
    jname = (journal.get("name") or "").strip().lower()
    jvol = str(journal.get("volume") or "").strip()
    arxiv_pseudo_journal = jname in ("arxiv", "arxiv.org") or jvol.lower().startswith("abs/")
    oa = d.get("openAccessPdf") if isinstance(d.get("openAccessPdf"), dict) else {}
    pdfs = {}
    if oa.get("url"):
        pdfs["arxiv" if "arxiv.org" in oa["url"] else "official_oa"] = oa["url"]
    p = Paper(
        title=d.get("title") or "",
        authors=[a.get("name", "") for a in d.get("authors") or []],
        year=d.get("year"), publication_date=d.get("publicationDate"),
        venue=venue, venue_key=vkey, venue_type=vtype,
        doi=ext.get("DOI"), arxiv_id=ext.get("ArXiv"), pmid=ext.get("PubMed"),
        pmcid=("PMC" + str(ext["PubMedCentral"])) if ext.get("PubMedCentral") else None,
        semantic_scholar_id=d.get("paperId"), url=d.get("url") or "",
        abstract=d.get("abstract") or "", citation_count=d.get("citationCount"),
        source="semantic_scholar", publication_type=",".join(ptypes),
        volume=None if arxiv_pseudo_journal or not jvol else jvol,
        pages=None if arxiv_pseudo_journal else (str(journal.get("pages") or "").strip() or None),
        pdf_candidates=pdfs,
        author_ids=[f"s2:{a['authorId']}" for a in d.get("authors") or [] if isinstance(a, dict) and a.get("authorId")],
    )
    if "review" in ptypes and "journalarticle" not in ptypes:
        p.publication_type = "review"
    if "editorial" in ptypes or "lettersandcomments" in ptypes:
        p.publication_type = "editorial"
    p.pdf_url = pdfs.get("official_oa")
    p.url = ""  # prefer DOI/arXiv canonical URL over the S2 page
    finalize(p)
    if not p.url:
        p.url = d.get("url") or ""
    return p


def _date_range(date_from: Optional[str], date_to: Optional[str]) -> Optional[str]:
    if not date_from and not date_to:
        return None
    return f"{date_from or ''}:{date_to or ''}"


def search(ctx: SourceContext, q: BoolQuery, date_from: Optional[str] = None, date_to: Optional[str] = None,
           limit: Optional[int] = None, venue: Optional[str] = None) -> List[Paper]:
    limit = limit or ctx.max_results
    params: Dict[str, Any] = {"query": q.semantic_scholar(), "fields": FIELDS, "sort": "publicationDate:desc"}
    dr = _date_range(date_from, date_to)
    if dr:
        params["publicationDateOrYear"] = dr
    if venue:
        params["venue"] = venue
    out: List[Paper] = []
    token = None
    while len(out) < limit:
        if token:
            params["token"] = token
        data = ctx.http.get(f"{BASE}/paper/search/bulk", params=params, headers=_headers(ctx), use_cache=True)
        for d in data.get("data") or []:
            p = to_paper(d)
            p.queries.append(f"semantic_scholar:{q.label}")
            out.append(p)
            if len(out) >= limit:
                break
        token = data.get("token")
        if not token:
            break
    return out


def get_paper(ctx: SourceContext, pid: str) -> Optional[Paper]:
    """pid: S2 id, 'DOI:..', 'ARXIV:..', 'PMID:..', or a URL."""
    try:
        d = ctx.http.get(f"{BASE}/paper/{pid}", params={"fields": FIELDS}, headers=_headers(ctx), use_cache=True)
    except HttpError as exc:
        if exc.status != 404:
            ctx.warn("semantic_scholar", f"lookup {pid} failed: {exc}")
        return None
    return to_paper(d) if d else None


def batch(ctx: SourceContext, ids: List[str]) -> List[Optional[Paper]]:
    """Batch lookup (ids like 'DOI:10...' / 'ARXIV:...'), up to 500 per call. Order is preserved."""
    out: List[Optional[Paper]] = []
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        try:
            data = ctx.http.request("POST", f"{BASE}/paper/batch", params={"fields": FIELDS}, json_body={"ids": chunk},
                                    headers=_headers(ctx))
        except HttpError as exc:
            if "No valid paper ids" not in exc.body:
                ctx.warn("semantic_scholar", f"batch lookup failed: {exc}")
            out.extend([None] * len(chunk))
            continue
        out.extend(to_paper(d) if d else None for d in data)
    return out


def search_title(ctx: SourceContext, title: str) -> Optional[Paper]:
    try:
        data = ctx.http.get(f"{BASE}/paper/search/match", params={"query": title, "fields": FIELDS}, headers=_headers(ctx), use_cache=True)
    except HttpError:
        return None
    rows = data.get("data") or []
    return to_paper(rows[0]) if rows else None


def resolve_author(ctx: SourceContext, name: str) -> List[Dict[str, Any]]:
    data = ctx.http.get(f"{BASE}/author/search", params={"query": name, "fields": "name,affiliations,paperCount,citationCount,hIndex,papers.title", "limit": 10},
                        headers=_headers(ctx), use_cache=True)
    cands = []
    for a in data.get("data") or []:
        titles = " ".join((p.get("title") or "") for p in (a.get("papers") or [])[:100]).lower()
        affinity = sum(titles.count(k) for k in ("radiolog", "x-ray", "mammogra", "chest", " ct ", "tomograph", "medical imag", "foundation", "vision-language"))
        cands.append({"source": "semantic_scholar", "id": a.get("authorId"), "name": a.get("name"),
                      "works_count": a.get("paperCount", 0), "cited_by_count": a.get("citationCount", 0),
                      "institutions": a.get("affiliations") or [], "affinity": affinity,
                      "score": min(affinity, 30) + min(a.get("paperCount", 0), 300) / 30})
    cands.sort(key=lambda c: -c["score"])
    return cands


def author_papers(ctx: SourceContext, author_id: str, date_from: Optional[str] = None, limit: int = 1000,
                  date_to: Optional[str] = None) -> List[Paper]:
    out: List[Paper] = []
    offset = 0
    while offset < limit:
        data = ctx.http.get(f"{BASE}/author/{author_id}/papers", params={"fields": FIELDS, "limit": 500, "offset": offset},
                            headers=_headers(ctx), use_cache=True)
        rows = data.get("data") or []
        for d in rows:
            p = to_paper(d)
            if date_from and p.publication_date and p.publication_date < date_from:
                continue
            if date_from and not p.publication_date and p.year and p.year < int(date_from[:4]):
                continue
            if date_to and p.publication_date and p.publication_date[:10] > date_to:
                continue
            if date_to and not p.publication_date and p.year and p.year > int(date_to[:4]):
                continue
            p.queries.append(f"semantic_scholar:author:{author_id}")
            out.append(p)
        if data.get("next") is None or not rows:
            break
        offset = data["next"]
    return out


def _edges(ctx: SourceContext, pid: str, kind: str, limit: int) -> List[Paper]:
    key = "citedPaper" if kind == "references" else "citingPaper"
    out: List[Paper] = []
    offset = 0
    while offset < limit:
        try:
            data = ctx.http.get(f"{BASE}/paper/{pid}/{kind}", params={"fields": FIELDS, "limit": min(1000, limit), "offset": offset},
                                headers=_headers(ctx), use_cache=True)
        except HttpError as exc:
            ctx.warn("semantic_scholar", f"{kind} for {pid} failed: {exc}")
            break
        rows = data.get("data") or []
        for r in rows:
            d = r.get(key) or {}
            if d.get("title"):
                out.append(to_paper(d))
        if data.get("next") is None or not rows:
            break
        offset = data["next"]
    return out


def reference_edges(ctx: SourceContext, pid: str, limit: int = 1000) -> List[Tuple[Paper, Dict[str, Any]]]:
    """References with citation evidence: the sentences citing each one, S2 intents and 'influential' flag."""
    out: List[Tuple[Paper, Dict[str, Any]]] = []
    offset = 0
    while offset < limit:
        try:
            data = ctx.http.get(f"{BASE}/paper/{pid}/references", params={"fields": FIELDS + ",contexts,intents,isInfluential",
                                "limit": min(1000, limit), "offset": offset}, headers=_headers(ctx), use_cache=True)
        except HttpError as exc:
            ctx.warn("semantic_scholar", f"references for {pid} failed: {exc}")
            break
        rows = data.get("data") or []
        for r in rows:
            d = r.get("citedPaper") or {}
            if d.get("title"):
                out.append((to_paper(d), {"contexts": r.get("contexts") or [], "intents": r.get("intents") or [],
                                          "influential": bool(r.get("isInfluential"))}))
        if data.get("next") is None or not rows:
            break
        offset = data["next"]
    return out


def references(ctx: SourceContext, pid: str, limit: int = 1000) -> List[Paper]:
    return _edges(ctx, pid, "references", limit)


def citations(ctx: SourceContext, pid: str, limit: int = 1000) -> List[Paper]:
    return _edges(ctx, pid, "citations", limit)


def recommendations(ctx: SourceContext, pid: str, limit: int = 100) -> List[Paper]:
    try:
        data = ctx.http.get(f"{REC}/papers/forpaper/{pid}", params={"fields": FIELDS, "limit": limit, "from": "all-cs"},
                            headers=_headers(ctx), use_cache=True)
    except HttpError as exc:
        ctx.warn("semantic_scholar", f"recommendations failed: {exc}")
        return []
    return [to_paper(d) for d in data.get("recommendedPapers") or []]
