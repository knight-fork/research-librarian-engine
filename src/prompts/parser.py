"""Natural-language command parser: compiles prompts into deterministic SearchRequests.

Rule-based first; an optional LLM fallback (llm.enabled) handles phrasing the rules cannot parse.
The LLM only fills search parameters - it never supplies bibliographic metadata."""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import List, Optional, Tuple

from src.models import SearchRequest

TOPIC_SYNONYMS: List[Tuple[str, str]] = [
    (r"foundation[\s-]models?|foundational models?|\bfms?\b|self[\s-]supervised|pre[\s-]?train(ing|ed)?|generalist", "foundation-model"),
    (r"vision[\s-]language|\bvlms?\b|multimodal llms?|\bmllms?\b|image[\s-]text|report generation|\bvqa\b|medical clip|radiology clip", "vlm"),
    (r"uncertaint(y|ies)|uncertainty[\s-]aware|epistemic|aleatoric|confidence estimation|reliabilit(y|ies)|trustworth", "uncertainty"),
    (r"calibrat(ion|ed)", "calibration"),
    (r"conformal", "conformal-prediction"),
    (r"selective prediction|abstention|abstain", "selective-prediction"),
    (r"out[\s-]of[\s-]distribution|\bood\b|failure detection", "ood"),
    (r"distribution shift|domain shift|dataset shift", "distribution-shift"),
    (r"robustness|fairness|shortcut|external validation", "robustness"),
    (r"hallucinat", "vlm-reliability"),
]
MODALITY_SYNONYMS: List[Tuple[str, str]] = [
    (r"chest\s?x[\s-]?rays?|\bcxrs?\b|chest radiograph", "cxr"),
    (r"\bct\b|computed tomography", "ct"),
    (r"mammo(graph(y|ic)|grams?)?\b|breast imaging|tomosynthesis|\bdbt\b", "mammo"),
    (r"\bradiology\b(?! vlm)|medical imaging", "radiology-general"),
]
VENUE_SYNONYMS: List[Tuple[str, str]] = [
    (r"\bmiccai\b", "miccai"), (r"\bmidl\b", "midl"), (r"\bneurips\b|\bnips\b", "neurips"), (r"\bicml\b", "icml"),
    (r"\biclr\b", "iclr"), (r"\bcvpr\b", "cvpr"), (r"\biccv\b", "iccv"), (r"\beccv\b", "eccv"), (r"\btpami\b", "tpami"),
    (r"\btmi\b|transactions on medical imaging", "tmi"), (r"\bmedia\b|medical image analysis", "medical_image_analysis"),
    (r"radiology:? ?(ai|artificial intelligence)", "radiology_ai"), (r"\baaai\b", "aaai"), (r"\bijcai\b", "ijcai"), (r"\bisbi\b", "isbi"),
    (r"\bjmlr\b", "jmlr"), (r"\btnnls\b", "tnnls"), (r"european radiology|\beur\.? radiol", "european_radiology"),
    (r"journal of digital imaging|\bj\.? digit\.? imaging|imaging informatics in medicine", "jdi"),
    (r"artificial intelligence in medicine", "aim"), (r"\brsna\b", "rsna"),
]
# A period only ends the name when it closes a word of 2+ letters (so "J. Smith" / "Curtis P. Langlotz" survive).
AUTHOR_STOP = (r"(?:\s+(?:on|about|related|regarding|in|for|since|from|after|during|between|published|that|with|working|"
               r"covering|et\s+al\b)|(?<![\s.][A-Za-z])(?<!\bDr)(?<!\bProf)\.(?=\s|$)|[,;]|$)")
SCAN_RE = re.compile(r"^\s*(?:please\s+)?(?:(?:run|do|start|perform)\s+(?:a\s+|the\s+)?)?(?:new\s+|scheduled\s+|full\s+)?scan\b"
                     r"|\b(?:run|do|start|perform)\s+(?:a\s+|the\s+)?(?:new\s+|scheduled\s+|full\s+)?scan\b", re.I)


def _date_from(text: str, today: date) -> Tuple[Optional[str], Optional[str]]:
    t = text.lower()
    m = re.search(r"\b(?:since|from|after)\s+((?:19|20)\d{2})\b", t)
    if m:
        return f"{m.group(1)}-01-01", None
    m = re.search(r"\bbetween\s+((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})", t)
    if m:
        return f"{m.group(1)}-01-01", f"{m.group(2)}-12-31"
    m = re.search(r"\b(?:in|of)\s+((?:19|20)\d{2})\b", t) or re.search(r"\b(?:miccai|midl|neurips|icml|iclr|cvpr|iccv|eccv|isbi)\s+((?:19|20)\d{2})\b", t)
    if m:
        return f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"
    m = re.search(r"\b(?:last|past)\s+(\d+|one|two|three|four|five|six|ten)\s+(day|week|month|year)s?", t)
    if m:
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10}
        n = int(m.group(1)) if m.group(1).isdigit() else words[m.group(1)]
        days = {"day": 1, "week": 7, "month": 30, "year": 365}[m.group(2)] * n
        return (today - timedelta(days=days)).isoformat(), None
    if re.search(r"\b(?:last|past) year\b", t):
        return (today - timedelta(days=365)).isoformat(), None
    if re.search(r"\b(recent|latest|new)\b", t):
        return (today - timedelta(days=730)).isoformat(), None
    return None, None


def _collect(text: str, table: List[Tuple[str, str]]) -> List[str]:
    out: List[str] = []
    for pat, key in table:
        if re.search(pat, text, re.I) and key not in out:
            out.append(key)
    return out


# ---------------------------------------------------------------- typo tolerance

# Words that trigger a search field. Misspellings of these (edit distance 1, or 2 for long words) are
# corrected; everything else in the prompt is left alone.
TRIGGER_VOCAB = [
    "mammography", "mammogram", "mammograms", "mammographic", "tomosynthesis", "radiograph", "radiographs",
    "radiography", "radiology", "radiological", "tomography", "computed", "chest", "x-ray", "x-rays", "imaging",
    "foundation", "vision", "language", "vision-language", "multimodal", "pretraining", "pre-training",
    "self-supervised", "generalist", "uncertainty", "uncertainties", "calibration", "calibrated", "conformal",
    "hallucination", "hallucinations", "reliability", "robustness", "fairness", "out-of-distribution",
    "distribution", "selective", "abstention", "generation", "baseline", "baselines", "references", "similar",
    "miccai", "neurips", "icml", "iclr", "cvpr", "iccv", "eccv", "midl", "isbi", "tpami",
]
_VOCAB_SET = set(TRIGGER_VOCAB)
# Stems of domain words: a token starting with one of these that can't be matched is reported, not ignored.
_DOMAIN_STEMS = ("mammo", "radiog", "radiol", "tomog", "tomos", "uncert", "calib", "confor", "halluc", "founda", "multimod")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")


def _edit_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein (optimal string alignment) distance."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def _protected_spans(text: str) -> List[Tuple[int, int]]:
    """Quoted titles, identifiers/URLs and author names are never 'corrected'."""
    spans = []
    for rx in (r"[\"“'][^\"”']{3,}[\"”']", r"\S*10\.\d{4,9}/\S+", r"https?://\S+", r"\b\d{4}\.\d{4,5}\b",
               r"\b(?:by|from|of)\s+((?:[A-Z][\w.'-]*\s*){1,4})"):
        spans += [m.span() for m in re.finditer(rx, text)]
    return spans


def correct_typos(text: str) -> Tuple[str, List[str], List[str]]:
    """Return (corrected text, notes about corrections, unrecognised domain-looking words)."""
    spans = _protected_spans(text)
    pieces, notes, unknown, last = [], [], [], 0
    all_patterns = [p for p, _ in TOPIC_SYNONYMS + MODALITY_SYNONYMS + VENUE_SYNONYMS]
    for m in _TOKEN_RE.finditer(text):
        token, w = m.group(0), m.group(0).lower()
        if len(w) < 5 or w in _VOCAB_SET or any(a <= m.start() < b for a, b in spans):
            continue
        if any(re.search(p, w) for p in all_patterns):
            continue  # already recognised as written
        limit = 1 if len(w) <= 7 else 2
        scored = sorted((_edit_distance(w, v), v) for v in TRIGGER_VOCAB
                        if v[0] == w[0] and abs(len(v) - len(w)) <= limit)
        best = [v for dist, v in scored if dist <= limit and dist == scored[0][0]] if scored else []
        if len(best) == 1 or (best and len({b.rstrip("s") for b in best}) == 1):
            fix = best[0]
            pieces.append(text[last:m.start()] + fix)
            last = m.end()
            notes.append(f"interpreted '{token}' as '{fix}'")
        elif w.startswith(_DOMAIN_STEMS):
            unknown.append(token)
    pieces.append(text[last:])
    return "".join(pieces), notes, unknown


# ---------------------------------------------------------------- baselines / seed extraction

BASELINE_RE = re.compile(r"\bbaselines?\b|\bcompared (?:against|with|to)\b|\bcomparison (?:methods?|models?|papers?)\b|"
                         r"\bcompeting (?:methods?|approaches|models?)\b|\bwhat (?:does|did) (?:it|this paper) compare", re.I)
_PRONOUN_SEED = re.compile(r"^(?:this|that|the|my|its|it|their)(?:\s+(?:paper|article|work|one|study))?$", re.I)


def extract_seed(text: str) -> Optional[str]:
    """DOI, arXiv id, Zotero item key, quoted title, or '... of <Name>' from a prompt."""
    m = re.search(r"10\.\d{4,9}/[^\s\"'<>]+", text)
    if m:
        return m.group(0).rstrip(".,;)")
    m = re.search(r"arxiv(?:\.org/(?:abs|pdf)/|:)?\s*(\d{4}\.\d{4,5})", text, re.I) or re.search(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b", text)
    if m:
        return m.group(1)
    m = re.search(r"[\"“']([^\"”']{4,})[\"”']", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{8}\b", text)
    if m:
        return m.group(0)
    m = re.search(r"\b(?:fetch|add|get|save|import|download)\s+(?:me\s+)?(.+?)\s+(?:and|plus|with)\s+(?:also\s+)?(?:get\s+|fetch\s+|add\s+)?"
                  r"(?:its|their|the)\s+baselines?\b", text, re.I)
    if m and not _PRONOUN_SEED.match(m.group(1).strip()):
        return m.group(1).strip()
    m = re.search(r"\bpaper\s+(?:called|titled|named)\s+(.+?)(?:\s+and\b|[,;.]|$)", text, re.I) or \
        re.search(r"\b(?:baselines?|compared (?:against|with|to)|comparison methods?)\s+(?:of|for|in|used (?:by|in))\s+(.+?)"
                  r"(?:\s+(?:and|that|which|too|also|paper)\b|[,;.?]|$)", text, re.I)
    if m:
        seed = m.group(1).strip()
        return None if _PRONOUN_SEED.match(seed) else seed
    return None


_LEAD_RE = re.compile(
    r"^\s*(?:(?:please|can\s+you|find|fetch|get|search(?:\s+for)?|show|list|add|give|look\s+for|track|monitor|collect|gather|"
    r"retrieve|download|what\s+are|me|all|the|some|any|latest|recent|new|newest|important|relevant|strongest|best|good|key|"
    r"top\s+\d+|papers?|articles?|publications?|work|research|preprints?|studies|literature|on|about|regarding|related\s+to|"
    r"covering|concerning|discussing|that\s+(?:discuss|study|use))\b\s*)+", re.I)
_TRAIL_RE = re.compile(
    r"\s+(?:since|from\s+(?:19|20)\d{2}|after|before|between|in\s+the\s+(?:last|past)|over\s+the\s+(?:last|past)|published|"
    r"from\s+top|at\s+top|in\s+top|that\s+(?:are|i|we)\b|which\b|and\s+(?:add|get|save|put)|to\s+my\s+zotero|missing\s+from|"
    r"not\s+(?:already\s+)?in|every\s+|each\s+|daily|weekly|monthly)\b.*$", re.I)


def extract_subject(text: str, author: Optional[str] = None) -> Optional[str]:
    """What the prompt is *about*: 'fetch me recent papers on graph neural networks since 2023' -> 'graph neural networks'."""
    t = text
    if author:
        t = re.sub(r"\b(?:by|from|of)\s+" + re.escape(author), " ", t, flags=re.I)
    for pat, _ in VENUE_SYNONYMS:
        t = re.sub(pat, " ", t, flags=re.I)
    t = re.sub(r"\b(?:19|20)\d{2}\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _TRAIL_RE.sub("", t)
    prev = None
    while prev != t:  # strip leading verbs / fillers / connectors repeatedly
        prev = t
        t = _LEAD_RE.sub("", t + " ").strip()
    t = re.sub(r"^(?:papers?|articles?|work)\s+(?:on|about|related\s+to)\s+", "", t, flags=re.I)
    t = t.strip(" ,.;:?!-")
    return t or None


def parse(text: str, today: Optional[date] = None, profile: str = "general") -> Tuple[SearchRequest, float]:
    """Return (request, confidence 0-1). Misspelled domain terms are corrected and reported in req.parse_notes.

    In the general profile the request's free_text carries the subject of the prompt (what to search for);
    in the radiology profile it is only set when no built-in topic / modality was recognised."""
    today = today or date.today()
    raw, notes, unknown = correct_typos(text.strip())
    t = raw.lower()
    req = SearchRequest()
    req.parse_notes = notes + [f"couldn't interpret '{u}' - check the spelling" for u in unknown]
    conf = 0.5

    req.topics = _collect(t, TOPIC_SYNONYMS)
    req.modalities = _collect(t, MODALITY_SYNONYMS)
    # "radiology VLMs" / "radiology foundation models" is a general-radiology qualifier, not a modality restriction
    if "radiology-general" in req.modalities and len(req.modalities) == 1 and re.search(r"radiology (vlms?|vision|foundation)", t):
        req.modalities = []
    if len([m for m in req.modalities if m != "radiology-general"]) and "radiology-general" in req.modalities:
        req.modalities.remove("radiology-general")
    req.venues = _collect(t, VENUE_SYNONYMS)
    req.date_from, req.date_to = _date_from(t, today)
    req.top_venues_only = bool(re.search(r"\btop(-tier)? (venues?|conferences?|journals?)\b", t))
    m = re.search(r"\b(?:top|best|strongest)\s+(\d+)\b", t)
    if m:
        req.limit = int(m.group(1))
    if re.search(r"\b(show|list|preview|what are|don't add|do not add|without adding)\b", t):
        req.action = "report_only"

    if SCAN_RE.search(t):
        req.intent, req.source_tag, req.strict_request_match = "scan", "auto-discovery", False
        conf = 0.9
    elif BASELINE_RE.search(t):
        req.intent, req.strict_request_match = "baselines", False
        req.seed = extract_seed(raw)
        # "Fetch this paper / <id> / "<title>" and (get) its baselines too" -> also add the paper itself.
        verb = r"\b(?:fetch|add|get|save|import|download)\s+(?:me\s+)?"
        seed_ref = r"(?:(?:this|the|that)\s+paper|it\b|10\.\d{4}|arxiv|\d{4}\.\d{4}|[\"“'])"
        req.include_seed = bool(re.search(verb + seed_ref, t) or
                                (req.seed and re.search(verb + re.escape(req.seed.lower()[:12]), t)))
        if req.seed:
            conf = 0.85
        else:
            conf = 0.35
            req.parse_notes.append("which paper? give its DOI, arXiv id, Zotero item key, or its title in quotes")
    elif re.search(r"\b(missing|gaps?)\b.*\bcollection\b|\banaly[sz]e my (zotero )?collection", t):
        req.intent = "missing_lit"
        m = re.search(r"collection\s+(?:on|about|called|named)?\s*[\"']?([^\"'.]+?)[\"']?(?:\s+and\b|[.]|$)", raw, re.I)
        req.collection = m.group(1).strip() if m else None
        conf = 0.8 if req.collection else 0.4
    elif re.search(r"\bsimilar to\b", t):
        req.intent = "similar"
        req.seed = _quoted_or_after(raw, r"similar to")
        conf = 0.85 if req.seed else 0.3
    elif re.search(r"\breferences?\b.*\b(cited by|in|of)\b|\bcited by\b", t):
        req.intent = "citations"
        req.seed = _quoted_or_after(raw, r"(?:cited by|references (?:cited )?(?:by|in|of))")
        conf = 0.8 if req.seed else 0.3
    elif re.search(r"related to (this|the) zotero item|related to item|related to zotero", t) or re.search(r"\b[A-Z0-9]{8}\b", raw) and "related" in t:
        req.intent = "related"
        m = re.search(r"\b([A-Z0-9]{8})\b", raw)
        req.seed = m.group(1) if m else _quoted_or_after(raw, r"related to")
        conf = 0.8 if req.seed else 0.3
    elif re.search(r"\b(?:papers|work|publications|articles)\s+(?:by|from|of)\s+(?!the\b|last\b|top\b|\d)", t):
        req.intent, req.source_tag = "author_search", "author-watch"
        m = re.search(r"\b(?:papers|work|publications|articles)\s+(?:by|from|of)\s+(.+?)" + AUTHOR_STOP, raw, re.I)
        name = m.group(1).strip().strip("\"'") if m else None
        if name:
            name = re.sub(r"^(?:dr|prof|professor)\.?\s+", "", name, flags=re.I).strip()
        # Ignore venue names mis-read as authors ("papers from MICCAI")
        if name and not _collect(name.lower(), VENUE_SYNONYMS):
            req.author = name
            conf = 0.85
        else:
            req.intent, req.author = ("venue_search" if req.venues else "topic_search"), None
            conf = 0.7
    elif req.venues:
        req.intent = "venue_search"
        conf = 0.8
    else:
        req.intent = "topic_search"
        conf = 0.75 if (req.topics or req.modalities) else 0.3

    if req.intent in ("topic_search", "venue_search", "author_search") and (profile != "radiology" or not (req.topics or req.modalities)):
        req.free_text = extract_subject(raw, req.author)
        if req.intent == "topic_search" and req.free_text and conf < 0.6:
            conf = 0.7  # a clear subject is enough for a general search
    if unknown:
        conf = min(conf, 0.45)  # a domain word we couldn't read: ask rather than silently widen the search
    if req.intent == "missing_lit" or req.intent in ("similar", "citations", "related"):
        req.strict_request_match = False
    if req.intent == "venue_search" and not req.date_from:
        req.date_from = (today - timedelta(days=365 * 3)).isoformat()
    if req.intent == "author_search":
        req.strict_request_match = bool(req.topics or req.modalities)
    return req, conf


def _quoted_or_after(raw: str, lead: str) -> Optional[str]:
    m = re.search(r"[\"“']([^\"”']{8,})[\"”']", raw)
    if m:
        return m.group(1).strip()
    m = re.search(lead + r"\s+(.+?)(?:\s+that\s+(?:i|are)\b.*|\s+not already.*|[.?]?$)", raw, re.I)
    if m:
        seed = re.sub(r"^(the paper|paper|this paper)\s+", "", m.group(1).strip(), flags=re.I)
        return seed or None
    return None
