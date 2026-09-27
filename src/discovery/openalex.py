"""OpenAlex: broad discovery, author lookup, venue filtering, citation metadata."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.discovery.base import SourceContext
from src.discovery.query import BoolQuery
from src.models import Paper
from src.processing.normalize import canonical_venue, finalize, normalize_arxiv_id

BASE = "https://api.openalex.org"
SELECT = ("id,doi,title,display_name,publication_date,publication_year,authorships,primary_location,locations,"
          "best_oa_location,ids,cited_by_count,type,abstract_inverted_index,biblio,referenced_works")


def _params(ctx: SourceContext, extra: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(extra)
    if ctx.cfg.secrets.contact_email:
        p["mailto"] = ctx.cfg.secrets.contact_email
    if ctx.cfg.secrets.openalex_api_key:
        p["api_key"] = ctx.cfg.secrets.openalex_api_key
    return p


def _abstract(inv: Optional[Dict[str, List[int]]]) -> str:
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def to_paper(w: Dict[str, Any]) -> Paper:
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    venue = src.get("display_name") or ""
    vkey, vtype = canonical_venue(venue, src.get("type"))
    wtype = (w.get("type") or "").lower()
    if wtype == "preprint" and vtype not in ("journal", "conference"):
        vtype = "preprint"
    arxiv_id = None
    pdfs: Dict[str, str] = {}
    for l in w.get("locations") or []:
        url = (l.get("landing_page_url") or "") + " " + (l.get("pdf_url") or "")
        if "arxiv.org" in url and not arxiv_id:
            arxiv_id = normalize_arxiv_id(url)
        if l.get("pdf_url") and l.get("is_oa"):
            s = (l.get("source") or {}).get("display_name", "").lower()
            if "pubmed central" in s or "europe pmc" in s:
                pdfs.setdefault("pmc", l["pdf_url"])
            elif "arxiv" in s:
                pdfs.setdefault("arxiv", l["pdf_url"])
            else:
                pdfs.setdefault("official_oa", l["pdf_url"])
    best = w.get("best_oa_location") or {}
    if best.get("pdf_url") and "arxiv" not in best["pdf_url"]:
        pdfs.setdefault("official_oa", best["pdf_url"])
    ids = w.get("ids") or {}
    biblio = w.get("biblio") or {}
    pages = None
    if biblio.get("first_page"):
        pages = biblio["first_page"] + (f"-{biblio['last_page']}" if biblio.get("last_page") and biblio["last_page"] != biblio["first_page"] else "")
    p = Paper(
        title=w.get("title") or w.get("display_name") or "",
        authors=[(a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or []],
        year=w.get("publication_year"),
        publication_date=w.get("publication_date"),
        venue=venue, venue_key=vkey, venue_type=vtype,
        doi=w.get("doi"), arxiv_id=arxiv_id, pmid=ids.get("pmid"),
        openalex_id=(w.get("id") or "").rsplit("/", 1)[-1] or None,
        url=loc.get("landing_page_url") or "",
        abstract=_abstract(w.get("abstract_inverted_index")),
        citation_count=w.get("cited_by_count"),
        source="openalex", publication_type=wtype,
        volume=biblio.get("volume"), issue=biblio.get("issue"), pages=pages,
        issn=src.get("issn_l"), pdf_candidates=pdfs,
        references=[r.rsplit("/", 1)[-1] for r in (w.get("referenced_works") or [])],
        author_ids=[f"openalex:{(a.get('author') or {}).get('id', '').rsplit('/', 1)[-1]}"
                    for a in w.get("authorships") or [] if (a.get("author") or {}).get("id")],
    )
    p.pdf_url = pdfs.get("official_oa")
    return finalize(p)


def _paged(ctx: SourceContext, params: Dict[str, Any], limit: int) -> Iterable[Dict[str, Any]]:
    cursor, n = "*", 0
    while cursor and n < limit:
        data = ctx.http.get(f"{BASE}/works", params=_params(ctx, {**params, "cursor": cursor, "per-page": min(200, limit), "select": SELECT}),
                            use_cache=True)
        results = data.get("results") or []
        for r in results:
            yield r
            n += 1
            if n >= limit:
                return
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not results:
            return


def _date_filter(date_from: Optional[str], date_to: Optional[str]) -> List[str]:
    f = []
    if date_from:
        f.append(f"from_publication_date:{date_from}")
    if date_to:
        f.append(f"to_publication_date:{date_to}")
    return f


def search(ctx: SourceContext, q: BoolQuery, date_from: Optional[str] = None, date_to: Optional[str] = None,
           limit: Optional[int] = None) -> List[Paper]:
    filters = [f"title_and_abstract.search:{q.openalex()}"] + _date_filter(date_from, date_to)
    out = []
    for w in _paged(ctx, {"filter": ",".join(filters), "sort": "publication_date:desc"}, limit or ctx.max_results):
        p = to_paper(w)
        p.queries.append(f"openalex:{q.label}")
        out.append(p)
    return out


def resolve_author(ctx: SourceContext, name: str) -> List[Dict[str, Any]]:
    """Return author candidates, best first, scored for radiology / medical-imaging affinity."""
    data = ctx.http.get(f"{BASE}/authors", params=_params(ctx, {"search": name, "per-page": 10}), use_cache=True)
    cands = []
    for a in data.get("results") or []:
        topics = " ".join((t.get("display_name") or "") for t in (a.get("topics") or [])[:15]).lower()
        affinity = sum(k in topics for k in ("radiolog", "medical imag", "image", "deep learning", "breast", "chest", "tomograph", "mammogra", "neural"))
        insts = [i.get("display_name") for i in (a.get("last_known_institutions") or []) if i.get("display_name")]
        cands.append({
            "source": "openalex", "id": a["id"].rsplit("/", 1)[-1], "name": a.get("display_name"),
            "works_count": a.get("works_count", 0), "cited_by_count": a.get("cited_by_count", 0),
            "institutions": insts, "affinity": affinity,
            "score": affinity * 10 + min(a.get("works_count", 0), 300) / 30,
        })
    cands.sort(key=lambda c: -c["score"])
    return cands


def author_works(ctx: SourceContext, author_id: str, date_from: Optional[str] = None, limit: int = 500,
                 date_to: Optional[str] = None) -> List[Paper]:
    filters = [f"author.id:{author_id}"] + _date_filter(date_from, date_to)
    out = []
    for w in _paged(ctx, {"filter": ",".join(filters), "sort": "publication_date:desc"}, limit):
        p = to_paper(w)
        p.queries.append(f"openalex:author:{author_id}")
        out.append(p)
    return out


def get_work(ctx: SourceContext, doi: Optional[str] = None, openalex_id: Optional[str] = None) -> Optional[Paper]:
    key = f"doi:{doi}" if doi else openalex_id
    try:
        w = ctx.http.get(f"{BASE}/works/{key}", params=_params(ctx, {"select": SELECT}), use_cache=True)
    except Exception as exc:  # noqa: BLE001
        ctx.warn("openalex", f"lookup {key} failed: {exc}")
        return None
    return to_paper(w) if w else None


def works_by_ids(ctx: SourceContext, ids: List[str]) -> List[Paper]:
    out = []
    for i in range(0, len(ids), 50):
        chunk = "|".join(ids[i:i + 50])
        for w in _paged(ctx, {"filter": f"openalex_id:{chunk}"}, 50):
            out.append(to_paper(w))
    return out


def citing_works(ctx: SourceContext, openalex_id: str, limit: int = 200) -> List[Paper]:
    return [to_paper(w) for w in _paged(ctx, {"filter": f"cites:{openalex_id}", "sort": "cited_by_count:desc"}, limit)]


def related_works(ctx: SourceContext, openalex_id: str) -> List[Paper]:
    try:
        w = ctx.http.get(f"{BASE}/works/{openalex_id}", params=_params(ctx, {"select": "related_works"}), use_cache=True)
    except Exception as exc:  # noqa: BLE001
        ctx.warn("openalex", f"related works failed: {exc}")
        return []
    ids = [r.rsplit("/", 1)[-1] for r in (w.get("related_works") or [])]
    return works_by_ids(ctx, ids) if ids else []


def source_ids_for_venue(ctx: SourceContext, venue_name: str) -> List[str]:
    data = ctx.http.get(f"{BASE}/sources", params=_params(ctx, {"search": venue_name, "per-page": 5}), use_cache=True)
    return [s["id"].rsplit("/", 1)[-1] for s in data.get("results") or []][:3]


def search_in_sources(ctx: SourceContext, q: BoolQuery, source_ids: List[str], date_from: Optional[str] = None,
                      date_to: Optional[str] = None, limit: Optional[int] = None) -> List[Paper]:
    filters = [f"title_and_abstract.search:{q.openalex()}", "primary_location.source.id:" + "|".join(source_ids)] + _date_filter(date_from, date_to)
    return [to_paper(w) for w in _paged(ctx, {"filter": ",".join(filters)}, limit or ctx.max_results)]
