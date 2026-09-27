"""Query generation: modality x topic boolean queries, rendered per source."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

MODALITY_QUERY_TERMS = {
    "cxr": ["chest x-ray", "chest radiograph", "chest radiography", "CXR"],
    "ct": ["computed tomography", "CT scans", "chest CT", "CT images", "CT volumes"],
    "mammo": ["mammography", "mammogram", "breast imaging", "breast tomosynthesis"],
    "radiology-general": ["radiology", "medical imaging", "radiograph"],
}

TOPIC_QUERY_TERMS = {
    "foundation-model": ["foundation model", "self-supervised", "pretraining", "pre-training", "zero-shot", "generalist"],
    "vlm": ["vision-language", "vision language model", "report generation", "image-text", "multimodal large language model",
            "visual question answering", "CLIP"],
    "uncertainty": ["uncertainty", "calibration", "conformal", "out-of-distribution", "selective prediction", "failure detection"],
    "calibration": ["calibration", "miscalibration", "confidence estimation"],
    "conformal-prediction": ["conformal prediction", "conformal"],
    "selective-prediction": ["selective prediction", "abstention", "selective classification"],
    "ood": ["out-of-distribution", "OOD detection", "failure detection"],
    "distribution-shift": ["distribution shift", "domain shift", "external validation"],
    "robustness": ["robustness", "distribution shift", "domain shift", "fairness", "shortcut learning"],
    "vlm-reliability": ["hallucination", "calibration", "uncertainty", "factual consistency"],
    "report-generation": ["report generation"],
    "image-text-pretraining": ["image-text pretraining", "vision-language pretraining", "contrastive language-image"],
    "hallucination": ["hallucination", "factual consistency"],
}


@dataclass
class BoolQuery:
    """AND of OR-groups."""
    groups: List[List[str]] = field(default_factory=list)
    label: str = ""

    def terms(self) -> List[str]:
        return [t for g in self.groups for t in g]

    # --- renderers ---
    @staticmethod
    def _q(t: str) -> str:
        return f'"{t}"' if (" " in t or "-" in t) else t

    def openalex(self) -> str:
        # OpenAlex boolean search: AND / OR / quotes. Commas are filter separators, so none allowed.
        return " AND ".join("(" + " OR ".join(self._q(t) for t in g) + ")" for g in self.groups).replace(",", " ")

    def semantic_scholar(self) -> str:
        # /paper/search/bulk syntax: + AND, | OR, quotes for phrases.
        return " + ".join("(" + " | ".join(f'"{t}"' if " " in t or "-" in t else t for t in g) + ")" for g in self.groups)

    def pubmed(self) -> str:
        return " AND ".join("(" + " OR ".join(f'"{t}"[tiab]' for t in g) + ")" for g in self.groups)

    def arxiv(self) -> str:
        def term(t: str) -> str:
            t = t.replace("-", " ")
            return f'abs:"{t}"' if " " in t else f"abs:{t}"
        return " AND ".join("(" + " OR ".join(term(t) for t in g) + ")" for g in self.groups)

    def free_text(self) -> str:
        return " ".join(g[0] for g in self.groups)


def build_queries(modalities: Sequence[str], topics: Sequence[str], free_text: Optional[str] = None) -> List[BoolQuery]:
    """Generate modality x topic-group queries. Fine-grained topics fold into their own group."""
    modalities = list(modalities) or ["cxr", "ct", "mammo", "radiology-general"]
    topics = list(topics) or ["foundation-model", "vlm", "uncertainty"]
    queries: List[BoolQuery] = []
    for m in modalities:
        mterms = MODALITY_QUERY_TERMS.get(m, [m])
        if len(topics) > 1 and set(topics) >= {"vlm", "uncertainty"} and len(topics) == 2:
            # Conjunctive request, e.g. "uncertainty in radiology VLMs"
            queries.append(BoolQuery([mterms, TOPIC_QUERY_TERMS["vlm"][:4], TOPIC_QUERY_TERMS["vlm-reliability"]], f"{m}+vlm+uncertainty"))
            continue
        for t in topics:
            tterms = TOPIC_QUERY_TERMS.get(t, [t])
            groups = [mterms, tterms]
            if free_text:
                groups.append([free_text])
            queries.append(BoolQuery(groups, f"{m}+{t}"))
    return queries


def scan_queries() -> List[BoolQuery]:
    """The default scheduled-scan query set."""
    qs = build_queries(["cxr", "ct", "mammo"], ["foundation-model", "vlm", "uncertainty"])
    qs.append(BoolQuery([MODALITY_QUERY_TERMS["radiology-general"], TOPIC_QUERY_TERMS["uncertainty"]], "radiology+uncertainty"))
    qs.append(BoolQuery([MODALITY_QUERY_TERMS["radiology-general"], ["foundation model", "vision-language"]], "radiology+fm/vlm"))
    qs.append(BoolQuery([["radiology", "medical"], ["vision-language", "VLM", "multimodal large language model"],
                         TOPIC_QUERY_TERMS["vlm-reliability"]], "radiology+vlm+reliability"))
    return qs


# ---------------------------------------------------------------- free-text queries (general profile)

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "in", "on", "to", "with", "without", "using", "use", "based", "via",
    "from", "about", "into", "toward", "towards", "by", "at", "as", "is", "are", "its", "their", "that", "which",
    "paper", "papers", "article", "articles", "study", "studies", "work", "research", "method", "methods",
    "approach", "approaches", "new", "novel", "recent", "latest", "relevant", "important", "me", "my",
}


def _synonym_tables():
    # The built-in vocabulary (see src/prompts/parser.py) doubles as query expansion: a recognised concept
    # becomes an OR-group of its synonyms instead of a literal word.
    from src.prompts.parser import MODALITY_SYNONYMS, TOPIC_SYNONYMS
    out = []
    for pattern, key in TOPIC_SYNONYMS:
        if key in TOPIC_QUERY_TERMS:
            out.append((pattern, TOPIC_QUERY_TERMS[key]))
    for pattern, key in MODALITY_SYNONYMS:
        if key in MODALITY_QUERY_TERMS:
            out.append((pattern, MODALITY_QUERY_TERMS[key]))
    return out


def groups_from_text(text: Optional[str], expand: bool = True) -> List[List[str]]:
    """Free text -> AND of OR-groups.

    Explicit syntax:  (GNN OR "graph neural network") AND "drug discovery"   (also ';' for AND, '|' for OR)
    Plain text:       every significant word / "quoted phrase" must appear; known concepts expand to synonyms."""
    import re
    text = (text or "").strip()
    if not text:
        return []
    if re.search(r"\bAND\b|;|\bOR\b|\|", text):
        groups = []
        for part in re.split(r"\s+AND\s+|;", text):
            part = part.strip().strip("()").strip()
            alts = [a.strip().strip("()").strip().strip('"') for a in re.split(r"\s+OR\s+|\|", part)]
            alts = [a for a in alts if a]
            if alts:
                groups.append(alts)
        return groups
    groups: List[List[str]] = []
    phrases = re.findall(r'"([^"]+)"', text)
    groups += [[p] for p in phrases]
    rest = re.sub(r'"[^"]+"', " ", text)
    if expand:
        for pattern, synonyms in _synonym_tables():
            m = re.search(pattern, rest, re.I)
            if m:
                groups.append(list(dict.fromkeys([m.group(0)] + synonyms)))
                rest = rest[:m.start()] + " " + rest[m.end():]
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-+.]*[A-Za-z0-9+]|[A-Za-z0-9]", rest):
        if w.lower() not in STOPWORDS and len(w) > 1:
            groups.append([w])
    # de-duplicate groups (case-insensitive)
    seen, out = set(), []
    for g in groups:
        key = tuple(x.lower() for x in g)
        if key not in seen:
            seen.add(key)
            out.append(g)
    return out


def text_query(text: str, label: Optional[str] = None) -> BoolQuery:
    return BoolQuery(groups_from_text(text), label or "query")
