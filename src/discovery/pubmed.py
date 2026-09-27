"""PubMed (NCBI E-utilities): radiology / biomedical journal discovery and verification."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from src.discovery.base import SourceContext
from src.discovery.query import BoolQuery
from src.models import Paper
from src.processing.normalize import canonical_venue, finalize

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
EXCLUDED_TYPES = {"Editorial", "Comment", "Letter", "Published Erratum", "News", "Retraction of Publication", "Retracted Publication"}


def _params(ctx: SourceContext, extra: Dict) -> Dict:
    p = {"tool": "research-librarian", **extra}
    if ctx.cfg.secrets.ncbi_api_key:
        p["api_key"] = ctx.cfg.secrets.ncbi_api_key
    if ctx.cfg.secrets.contact_email:
        p["email"] = ctx.cfg.secrets.contact_email
    return p


def _text(el: Optional[ET.Element]) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def _pubdate(art: ET.Element) -> Optional[str]:
    for path in ("./PubmedData/History/PubMedPubDate[@PubStatus='pubmed']", "./MedlineCitation/Article/ArticleDate",
                 "./MedlineCitation/Article/Journal/JournalIssue/PubDate"):
        d = art.find(path)
        if d is None:
            continue
        y = _text(d.find("Year"))
        if not y:
            md = _text(d.find("MedlineDate"))
            y = md[:4] if md[:4].isdigit() else ""
        if not y:
            continue
        m = _text(d.find("Month"))
        m = MONTHS.get(m[:3].lower(), int(m) if m.isdigit() else 1) if m else 1
        day = _text(d.find("Day"))
        return f"{y}-{int(m):02d}-{int(day) if day.isdigit() else 1:02d}"
    return None


def parse_articles(xml_text: str) -> List[Paper]:
    root = ET.fromstring(xml_text)
    out = []
    for art in root.findall("./PubmedArticle"):
        mc = art.find("./MedlineCitation")
        a = mc.find("./Article")
        journal = _text(a.find("./Journal/Title"))
        ptypes = [_text(t) for t in a.findall("./PublicationTypeList/PublicationType")]
        ids = {i.get("IdType"): _text(i) for i in art.findall("./PubmedData/ArticleIdList/ArticleId")}
        authors = []
        for au in a.findall("./AuthorList/Author"):
            name = " ".join(x for x in (_text(au.find("ForeName")), _text(au.find("LastName"))) if x) or _text(au.find("CollectiveName"))
            if name:
                authors.append(name)
        abstract = " ".join(_text(t) for t in a.findall("./Abstract/AbstractText"))
        issue = a.find("./Journal/JournalIssue")
        vkey, vtype = canonical_venue(journal, "journal")
        is_preprint = "Preprint" in ptypes or vtype == "preprint"
        ptype = next((t for t in ptypes if t in EXCLUDED_TYPES), "Preprint" if is_preprint else "Review" if "Review" in ptypes else "Journal Article")
        pmcid = ids.get("pmc")
        p = Paper(
            title=_text(a.find("./ArticleTitle")), authors=authors, publication_date=_pubdate(art),
            venue=journal, venue_key=vkey, venue_type="preprint" if is_preprint else "journal", doi=ids.get("doi"), pmid=_text(mc.find("./PMID")),
            pmcid=pmcid, abstract=abstract, source="pubmed", publication_type=ptype.lower(),
            volume=_text(issue.find("Volume")) if issue is not None else None,
            issue=_text(issue.find("Issue")) if issue is not None else None,
            pages=_text(a.find("./Pagination/MedlinePgn")) or None,
            issn=_text(a.find("./Journal/ISSN")) or None,
        )
        if pmcid:
            p.pdf_candidates["pmc"] = f"https://europepmc.org/articles/{pmcid}?pdf=render"
        p.url = f"https://pubmed.ncbi.nlm.nih.gov/{p.pmid}/" if not p.doi else ""
        out.append(finalize(p))
    return out


def search(ctx: SourceContext, q: BoolQuery, date_from: Optional[str] = None, date_to: Optional[str] = None,
           limit: Optional[int] = None, extra_term: Optional[str] = None) -> List[Paper]:
    term = q.pubmed()
    if extra_term:
        term = f"({term}) AND {extra_term}"
    if date_from or date_to:
        term += f' AND ("{(date_from or "1900-01-01").replace("-", "/")}"[dp] : "{(date_to or "3000/12/31").replace("-", "/")}"[dp])'
    res = ctx.http.get(f"{BASE}/esearch.fcgi", params=_params(ctx, {"db": "pubmed", "term": term, "retmax": limit or ctx.max_results,
                                                                     "retmode": "json", "sort": "pub_date"}), use_cache=True)
    pmids = (res.get("esearchresult") or {}).get("idlist") or []
    papers = fetch(ctx, pmids)
    for p in papers:
        p.queries.append(f"pubmed:{q.label}")
    return papers


def fetch(ctx: SourceContext, pmids: List[str]) -> List[Paper]:
    out: List[Paper] = []
    for i in range(0, len(pmids), 200):
        xml_text = ctx.http.request("POST", f"{BASE}/efetch.fcgi", data=_params(ctx, {"db": "pubmed", "id": ",".join(pmids[i:i + 200]), "retmode": "xml"}),
                                    expect="text")
        out.extend(parse_articles(xml_text))
    return out


def author_search(ctx: SourceContext, author: str, date_from: Optional[str] = None, limit: int = 300) -> List[Paper]:
    q = BoolQuery([[author]], f"author:{author}")
    term = f'"{author}"[au]'
    if date_from:
        term += f' AND ("{date_from.replace("-", "/")}"[dp] : "3000/12/31"[dp])'
    res = ctx.http.get(f"{BASE}/esearch.fcgi", params=_params(ctx, {"db": "pubmed", "term": term, "retmax": limit, "retmode": "json"}), use_cache=True)
    papers = fetch(ctx, (res.get("esearchresult") or {}).get("idlist") or [])
    for p in papers:
        p.queries.append(f"pubmed:{q.label}")
    return papers
