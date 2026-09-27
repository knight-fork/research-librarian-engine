"""Metadata normalization: identifiers, titles, venues, and merging records of the same paper."""
from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable, List, Optional, Tuple

from src.models import Paper

# ---------------------------------------------------------------- identifiers

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.I)
_ARXIV_NEW = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(v\d+)?(?![\d])")
_ARXIV_OLD = re.compile(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.I)


def normalize_doi(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    m = _DOI_RE.search(str(value))
    if not m:
        return None
    doi = m.group(0).rstrip(".,;)]}").lower()
    return doi


def normalize_arxiv_id(value: Optional[str]) -> Optional[str]:
    """Return a version-less arXiv id from an id, URL, or arXiv DOI."""
    if not value:
        return None
    s = str(value)
    m = re.search(r"10\.48550/arxiv\.(.+)$", s, re.I)
    if m:
        s = m.group(1)
    m = _ARXIV_NEW.search(s)
    if m:
        return m.group(1)
    m = _ARXIV_OLD.search(s)
    if m and ("arxiv" in s.lower() or "/" in m.group(1)):
        return m.group(1).lower()
    return None


def normalize_pmid(value) -> Optional[str]:
    if value is None:
        return None
    m = re.search(r"(\d{5,9})", str(value))
    return m.group(1) if m else None


# ---------------------------------------------------------------- text

def strip_markup(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


_VERSION_MARKERS = re.compile(r"\b(preprint|extended abstract|extended version|full version|v\d+)\b")


def normalize_title(title: Optional[str]) -> str:
    """lowercase, Unicode-normalize, strip punctuation/markup/version markers, collapse whitespace."""
    if not title:
        return ""
    t = unicodedata.normalize("NFKD", strip_markup(title))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = _VERSION_MARKERS.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_author_name(name: str) -> str:
    n = unicodedata.normalize("NFKD", name or "")
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    n = re.sub(r"[^a-z\s\-]", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def author_matches(query: str, candidate: str, strict: bool = False) -> bool:
    """Name match on surname plus given name.

    Loose (default, used as a byline sanity check after ID-based author resolution): a first initial on
    either side is enough, and a surname-only byline matches. Strict (author-watch scoring): both sides
    must carry the same full given name ('Yu-Yin' == 'Yuyin'); initials or surname-only never match."""
    q, c = normalize_author_name(query).split(), normalize_author_name(candidate).split()
    if not q or not c or q[-1] != c[-1]:
        return False
    if len(q) == 1 or len(c) == 1:
        return not strict
    qf, cf = q[0].replace("-", ""), c[0].replace("-", "")
    if len(qf) > 1 and len(cf) > 1:
        return qf == cf
    return (not strict) and qf[:1] == cf[:1]


# ---------------------------------------------------------------- venues

# (venue_key, venue_type, regex over lowercased venue string). Order matters: specific first.
VENUE_PATTERNS: List[Tuple[str, str, str]] = [
    ("radiology_ai", "journal", r"radiology[\s.:]*artificial intelligence|radiol\.?\s*artif\.?\s*intell"),
    ("tmi", "journal", r"transactions on medical imaging|ieee trans\.? med\.? imaging|^tmi$"),
    ("tpami", "journal", r"pattern analysis and machine intelligence|^tpami$"),
    ("tnnls", "journal", r"neural networks and learning systems|^tnnls$"),
    ("jmlr", "journal", r"^journal of machine learning research$|^jmlr$|j\.? mach\.? learn\.? res"),
    ("medical_image_analysis", "journal", r"^medical image analysis$|^med\.? image anal\.?$"),
    ("european_radiology", "journal", r"^european radiology$|^eur\.? radiol\.?$"),
    ("jdi", "journal", r"journal of digital imaging|imaging informatics in medicine|^j\.? digit\.? imaging"),
    ("aim", "journal", r"^artificial intelligence in medicine$|^artif\.? intell\.? med\.?$"),
    ("radiology", "journal", r"^radiology$"),
    ("miccai", "conference", r"medical image computing and computer[\s\-]assisted intervention|\bmiccai\b"),
    ("midl", "conference", r"medical imaging with deep learning|\bmidl\b"),
    ("isbi", "conference", r"international symposium on biomedical imaging|\bisbi\b"),
    ("rsna", "conference", r"radiological society of north america|\brsna\b"),
    ("neurips", "conference", r"neural information processing systems|\bneurips\b|\bnips\b"),
    ("iclr", "conference", r"international conference on learning representations|\biclr\b"),
    ("icml", "conference", r"international conference on machine learning(?!\s*(and\b|,|&|for\b|in\b|theory|\(icmla|\(icmlc))|\bicml\b"),
    ("cvpr", "conference", r"computer vision and pattern recognition(?!.*workshop)|\bcvpr\b"),
    ("iccv", "conference", r"international conference on computer vision(?!\s*(and\b|,|&|theory|for\b|in\b|\(visapp|\(cvidl))|\biccv\b"),
    ("eccv", "conference", r"european conference on computer vision|\beccv\b"),
    ("aaai", "conference", r"\baaai\b(?!\s*/\s*acm)"),
    ("ijcai", "conference", r"international joint conference on artificial intelligence|\bijcai\b"),
    ("arxiv", "preprint", r"\barxiv\b"),
    ("medrxiv", "preprint", r"\bmedrxiv\b|\bbiorxiv\b|research square|ssrn|preprints\.org|techrxiv"),
]

VENUE_TAG = {
    "medical_image_analysis": "media",
    "radiology_ai": "radiology-ai",
    "european_radiology": "eur-radiol",
}

VENUE_DISPLAY = {
    "neurips": "NeurIPS", "icml": "ICML", "iclr": "ICLR", "cvpr": "CVPR", "iccv": "ICCV", "eccv": "ECCV",
    "miccai": "MICCAI", "midl": "MIDL", "tpami": "TPAMI", "tmi": "IEEE TMI", "medical_image_analysis": "MedIA",
    "radiology_ai": "Radiology: AI", "tnnls": "TNNLS", "jmlr": "JMLR", "radiology": "Radiology", "aaai": "AAAI",
    "ijcai": "IJCAI", "isbi": "ISBI", "european_radiology": "European Radiology", "jdi": "J Digit Imaging",
    "aim": "AI in Medicine", "rsna": "RSNA", "arxiv": "arXiv",
}


def canonical_venue(venue: Optional[str], hint_type: Optional[str] = None) -> Tuple[str, str]:
    """Map a raw venue string to (venue_key, venue_type).

    hint_type is the source's own classification (journal/conference/repository/preprint)."""
    v = strip_markup(venue).lower().strip()
    # Strip leading "proceedings of the (nth)" noise.
    v_clean = re.sub(r"^(\d{4}\s+)?(ieee/cvf\s+)?(proceedings of (the )?)?(\d+(st|nd|rd|th)\s+)?", "", v)
    is_workshop = bool(re.search(r"\bworkshops?\b", v))
    for key, vtype, pat in VENUE_PATTERNS:
        if re.search(pat, v) or re.search(pat, v_clean):
            if is_workshop and vtype == "conference":
                return key, "workshop"
            return key, vtype
    hint = (hint_type or "").lower()
    if "lecture notes in" in v or hint in ("book series", "book-series"):
        return "other_conference", "conference"
    if is_workshop:
        return "workshop", "workshop"
    if hint in ("repository", "preprint", "posted-content", "posted_content"):
        return "preprint", "preprint"
    if hint == "journal" or hint in ("journal-article", "journalarticle"):
        return "other_journal", "journal"
    if hint in ("conference", "proceedings-article", "proceedings", "conference_paper"):
        return "other_conference", "conference"
    if not v:
        return "unknown", "preprint"
    if re.search(r"journal|transactions|letters|review|annals", v):
        return "other_journal", "journal"
    if re.search(r"conference|symposium|proceedings|workshop", v):
        return "other_conference", "conference"
    return "unknown", hint if hint in ("journal", "conference") else "preprint"


def venue_tag(key: str) -> Optional[str]:
    if key in ("unknown", "other_journal", "other_conference", "workshop", "preprint", "medrxiv"):
        return None
    return "venue:" + VENUE_TAG.get(key, key.replace("_", "-"))


def venue_display(p: Paper) -> str:
    base = VENUE_DISPLAY.get(p.venue_key) or p.venue or "Unknown venue"
    return f"{base} {p.year}" if p.year and str(p.year) not in base else base


# ---------------------------------------------------------------- finishing

def finalize(p: Paper) -> Paper:
    """Normalize the fields of a freshly mapped record in place."""
    p.title = strip_markup(p.title).rstrip(".")
    p.abstract = strip_markup(p.abstract)
    p.abstract = re.sub(r"^(abstract|summary)[\s:.-]+", "", p.abstract, flags=re.I)
    p.doi = normalize_doi(p.doi)
    if p.doi and p.doi.startswith("10.48550/arxiv."):
        p.arxiv_id = p.arxiv_id or normalize_arxiv_id(p.doi)
    p.arxiv_id = normalize_arxiv_id(p.arxiv_id) if p.arxiv_id else None
    p.pmid = normalize_pmid(p.pmid)
    p.authors = [re.sub(r"\s+", " ", a).strip() for a in p.authors if a and a.strip()]
    if p.publication_date and not p.year:
        try:
            p.year = int(p.publication_date[:4])
        except ValueError:
            pass
    if p.venue_key in ("", "unknown") or not p.venue_key:
        p.venue_key, vtype = canonical_venue(p.venue, p.venue_type)
        p.venue_type = vtype
    if p.venue_key in ("arxiv", "medrxiv") and p.venue_type != "preprint":
        p.venue_type = "preprint"
    if p.source and p.source not in p.sources:
        p.sources.append(p.source)
    if p.arxiv_id and "arxiv" not in p.pdf_candidates:
        p.pdf_candidates["arxiv"] = f"https://arxiv.org/pdf/{p.arxiv_id}"
    if not p.url:
        if p.doi and not p.doi.startswith("10.48550/"):
            p.url = f"https://doi.org/{p.doi}"
        elif p.arxiv_id:
            p.url = f"https://arxiv.org/abs/{p.arxiv_id}"
    return p


# Which record's venue "wins" when merging: peer-reviewed beats preprint.
_TYPE_RANK = {"journal": 3, "conference": 3, "workshop": 2, "preprint": 1}
_GENERIC_VENUES = ("unknown", "other_journal", "other_conference")
# DOI prefixes minted by preprint servers (arXiv, bioRxiv/medRxiv, Research Square, SSRN, Preprints.org, TechRxiv, OSF).
PREPRINT_DOI_PREFIXES = ("10.48550/", "10.1101/", "10.21203/", "10.2139/", "10.20944/", "10.36227/", "10.31219/", "10.31234/")


def is_preprint_doi(doi: Optional[str]) -> bool:
    return bool(doi) and doi.lower().startswith(PREPRINT_DOI_PREFIXES)


def merge(primary: Paper, other: Paper) -> Paper:
    """Merge `other` into `primary` (same paper, possibly different versions). Returns primary.

    Venue, locators (volume/issue/pages) and dates come from the peer-reviewed version; identifiers and
    descriptive fields are unioned."""
    p_rank, o_rank = _TYPE_RANK.get(primary.venue_type, 0), _TYPE_RANK.get(other.venue_type, 0)
    other_better_venue = (o_rank, other.venue_key not in _GENERIC_VENUES) > (p_rank, primary.venue_key not in _GENERIC_VENUES)
    if other_better_venue:
        for f in ("venue", "venue_key", "venue_type"):
            setattr(primary, f, getattr(other, f))
        for f in ("volume", "issue", "pages", "publisher", "issn"):
            if getattr(other, f):
                setattr(primary, f, getattr(other, f))
            elif o_rank > p_rank:
                setattr(primary, f, None)  # a preprint's locators (e.g. S2 'abs/2401.x') don't describe the published version
        if other.year:
            if other.publication_date or other.year != primary.year:
                primary.publication_date = other.publication_date
            primary.year = other.year
        elif other.publication_date:
            primary.publication_date = other.publication_date
        if other.doi and not is_preprint_doi(other.doi):
            primary.doi = other.doi
            primary.url = other.url or primary.url
    series_only = re.match(r"(lecture notes|communications in computer|proceedings of spie)", (primary.venue or "").lower())
    if series_only and other.venue and not re.match(r"(lecture notes|communications in computer|proceedings of spie)", other.venue.lower()) \
            and o_rank >= _TYPE_RANK.get(primary.venue_type, 0):
        primary.venue, primary.venue_key, primary.venue_type = other.venue, other.venue_key, other.venue_type
    lower_version = _TYPE_RANK.get(other.venue_type, 0) < _TYPE_RANK.get(primary.venue_type, 0)
    for f in ("doi", "arxiv_id", "pmid", "pmcid", "openalex_id", "semantic_scholar_id", "url", "publication_date",
              "year", "volume", "issue", "pages", "publisher", "issn", "pdf_url", "publication_type"):
        if getattr(primary, f) or not getattr(other, f):
            continue
        if lower_version and f in ("volume", "issue", "pages", "publisher", "issn"):
            continue
        if lower_version and f == "publication_date" and primary.year and str(other.publication_date)[:4] != str(primary.year):
            continue
        setattr(primary, f, getattr(other, f))
    # A preprint-server DOI should not displace a publisher DOI.
    if is_preprint_doi(primary.doi) and other.doi and not is_preprint_doi(other.doi):
        primary.doi = other.doi
    if len(other.abstract) > len(primary.abstract):
        primary.abstract = other.abstract
    if len(other.authors) > len(primary.authors):
        primary.authors = other.authors
    if other.citation_count is not None:
        primary.citation_count = max(primary.citation_count or 0, other.citation_count)
    for kind, url in other.pdf_candidates.items():
        primary.pdf_candidates.setdefault(kind, url)
    primary.sources = _uniq(primary.sources + other.sources)
    primary.queries = _uniq(primary.queries + other.queries)
    primary.references = _uniq(primary.references + other.references)
    primary.author_ids = _uniq(primary.author_ids + other.author_ids)
    primary.extra_tags = _uniq(primary.extra_tags + other.extra_tags)
    if other.graph_evidence and (other.graph_evidence.get("score", 0) > primary.graph_evidence.get("score", -1)):
        primary.graph_evidence = other.graph_evidence
    if other.author_watch and not primary.author_watch:
        primary.author_watch = other.author_watch
    if other.relationship and not primary.relationship:
        primary.relationship = other.relationship
    return primary


def _uniq(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out
