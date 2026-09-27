"""Relevance / quality / final scoring and threshold decisions."""
from __future__ import annotations

import math
import re
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

from src.config import Config
from src.models import Paper, SearchRequest
from src.processing import classify as C
from src.processing.normalize import author_matches

TOPIC_KEYS = {
    "foundation-model": "foundation_model_relevance",
    "vlm": "vlm_relevance",
    "uncertainty": "uncertainty_relevance",
    "robustness": "robustness_relevance",
}
# Fine-grained requested topics map onto topic_scores entries.
FINE_TOPICS = ("calibration", "conformal-prediction", "selective-prediction", "ood", "distribution-shift",
               "report-generation", "image-text-pretraining", "hallucination")


def requested_topic_score(cls: Dict, topics: Sequence[str]) -> float:
    if not topics:
        return C.main_topic_score(cls)
    scores = []
    for t in topics:
        if t in TOPIC_KEYS:
            scores.append(float(cls[TOPIC_KEYS[t]]))
        elif t in FINE_TOPICS:
            scores.append(float(cls["topic_scores"].get(t, 0.0)))
        elif t == "vlm-reliability":
            scores.append(min(float(cls["vlm_relevance"]), float(cls["topic_scores"].get("hallucination", 0.0)) + 0.2))
    if not scores:
        return C.main_topic_score(cls)
    # All requested topics should be present: geometric-ish blend of min and mean.
    return round(0.6 * min(scores) + 0.4 * (sum(scores) / len(scores)), 4)


def keyword_match(p: Paper, request: Optional[SearchRequest]) -> float:
    """title_abstract_keyword_match: explicit free-text terms if given, else breadth of title-level signal."""
    text = f"{p.title} {p.abstract}".lower()
    if request and request.free_text:
        words = [w for w in request.free_text.lower().replace(",", " ").split() if len(w) > 2]
        if words:
            return round(sum(1 for w in words if w in text) / len(words), 4)
    cls = p.classification or {}
    title_dims = len(cls.get("title_dimensions", []))
    abstract_dims = sum(1 for k, v in cls.get("matched_terms", {}).items() if v and k != "other")
    return round(min(1.0, 0.3 * title_dims + 0.1 * abstract_dims), 4)


def recency(p: Paper, today: Optional[date] = None) -> float:
    age = p.age_years(today)
    return round(max(0.0, 1.0 - max(0.0, age - 1.0) / 4.0), 4)  # 1.0 within a year, 0 at 5 years


def author_watch_match(p: Paper, cfg: Config) -> Optional[Dict]:
    """Watched-author match: pinned OpenAlex / S2 author ids first, else a strict full-name match.

    Initial-only or surname-only matches never count (namesakes must not lower the auto-add bar)."""
    for a in cfg.authors:
        name = a.get("name")
        pinned = {f"openalex:{a['openalex_id']}" if a.get("openalex_id") else None,
                  f"s2:{a['semantic_scholar_id']}" if a.get("semantic_scholar_id") else None} - {None}
        if pinned and pinned & set(p.author_ids):
            return a
        # If the paper carries ids from a source where this author is pinned and none match, it is someone else.
        pinned_sources = {x.split(":", 1)[0] for x in pinned}
        if pinned_sources & {x.split(":", 1)[0] for x in p.author_ids}:
            continue
        if name and any(author_matches(name, x, strict=True) for x in p.authors):
            return a
    return None


def citation_signal(p: Paper, normalize_age: bool = True, today: Optional[date] = None) -> float:
    cites = p.citation_count or 0
    age = max(p.age_years(today), 0.5)
    per_year = cites / age if normalize_age else cites / 3.0
    signal = min(1.0, math.log1p(per_year) / math.log1p(50))
    if normalize_age:
        w = min(1.0, age / 2.0)  # very new papers: lean on a neutral prior
        signal = w * signal + (1 - w) * 0.5
    return round(signal, 4)


def metadata_confidence(p: Paper) -> float:
    checks = [bool(p.doi or p.arxiv_id or p.pmid), bool(p.authors), bool(p.abstract), p.venue_key not in ("unknown",),
              bool(p.publication_date or p.year)]
    base = sum(checks) / len(checks)
    return round(min(1.0, base * 0.85 + 0.15 * min(1.0, (len(p.sources) - 1) / 2)), 4)


def methodological_relevance(p: Paper) -> float:
    method_tags = [s for s in p.subtopics if s.startswith(("uncertainty:", "fm:"))]
    score = 0.6 + 0.2 * min(2, len(method_tags))
    if "type:survey" in p.subtopics:
        score = min(score, 0.7)
    return round(min(1.0, score), 4)


def _quality(p: Paper, cfg: Config, today: Optional[date]) -> Tuple[float, Dict[str, float]]:
    weights = cfg.venue_weights
    vw = weights.get(p.venue_key, weights.get(p.venue_type if p.venue_type == "preprint" else "unknown", 0.4))
    peer = {"journal": 1.0, "conference": 1.0, "workshop": 0.6}.get(p.venue_type, 0.0)
    if "type:meeting-abstract" in p.subtopics:
        peer, vw = 0.3, min(vw, 0.5)
    cite = citation_signal(p, bool(cfg.get("ranking.citation_age_normalization", True)), today)
    meth = methodological_relevance(p)
    meta = metadata_confidence(p)
    quality = 0.40 * vw + 0.20 * peer + 0.15 * cite + 0.15 * meth + 0.10 * meta
    return quality, {"venue_weight": vw, "peer_review": peer, "citation_signal": cite,
                     "methodological_relevance": meth, "metadata_confidence": meta}


# ---------------------------------------------------------------- general profile: relevance from the query itself

def _term_regex(term: str) -> "re.Pattern[str]":
    words = [w for w in re.split(r"[\s\-]+", term.strip()) if w]
    stems = []
    for w in words:
        w = w.lower()
        for suf in ("ies", "es", "s", "ing", "ed"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        stems.append(re.escape(w) + r"\w*")
    return re.compile(r"(?<!\w)" + r"[\s\-]*".join(stems), re.I)


def query_group_hits(p: Paper, groups: List[List[str]]) -> List[float]:
    """Per query group: 1.0 if a term is in the title, 0.6 if only in the abstract, else 0."""
    out = []
    for g in groups:
        rxs = [_term_regex(t) for t in g if t.strip()]
        if any(r.search(p.title or "") for r in rxs):
            out.append(1.0)
        elif any(r.search(p.abstract or "") for r in rxs):
            out.append(0.6)
        else:
            out.append(0.0)
    return out


def _score_general(p: Paper, cfg: Config, request: Optional[SearchRequest], today: Optional[date]) -> Paper:
    from src.discovery.query import groups_from_text
    C.classify(p)  # only for publication-type checks (editorials, errata, surveys, meeting abstracts)
    p.modalities, p.topics = [], []
    p.subtopics = [t for t in p.subtopics if t.startswith("type:")]
    groups = groups_from_text(request.free_text) if request and request.free_text else []
    hits = query_group_hits(p, groups) if groups else []
    coverage = sum(hits) / len(hits) if hits else 0.5
    all_present = 1.0 if (not hits or all(h > 0 for h in hits)) else 0.0
    title_cov = (sum(1 for h in hits if h == 1.0) / len(hits)) if hits else 0.0
    watch = author_watch_match(p, cfg)
    if watch and not p.author_watch:
        p.author_watch = watch.get("name")
    if request and request.author and any(author_matches(request.author, a) for a in p.authors):
        p.author_watch = p.author_watch or request.author
    aw = 1.0 if p.author_watch else 0.0
    rec = recency(p, today)
    relevance = 0.45 * coverage + 0.25 * all_present + 0.10 * title_cov + 0.10 * aw + 0.10 * rec
    quality, qparts = _quality(p, cfg, today)
    p.relevance_score = round(relevance, 4)
    p.quality_score = round(quality, 4)
    p.final_score = round(0.70 * relevance + 0.30 * quality, 4)
    p.classification["query_groups"] = groups
    p.classification["query_hits"] = hits
    p.score_breakdown = {"query_coverage": round(coverage, 4), "all_terms_present": all_present, "title_coverage": round(title_cov, 4),
                         "author_watch_match": aw, "recency": rec, **qparts}
    return p


def _decide_general(p: Paper, cfg: Config, request: Optional[SearchRequest], allow_editorials: bool = False) -> Paper:
    cls = p.classification or C.classify(p)
    intent = request.intent if request else "topic_search"
    groups, hits = cls.get("query_groups") or [], cls.get("query_hits") or []
    if not p.title:
        p.decision, p.reject_reason = "reject", "missing title"
    elif cls["excluded_type"] and not allow_editorials:
        p.decision, p.reject_reason = "reject", f"excluded publication type ({p.publication_type or 'title pattern'})"
    elif intent == "add_doi":
        p.decision, p.reject_reason = "accept", ""
    elif groups and not all(h > 0 for h in hits):
        missing = [g[0] for g, h in zip(groups, hits) if h == 0]
        p.decision, p.reject_reason = "reject", "does not mention: " + ", ".join(missing[:4])
    elif request and request.venues and p.venue_key not in request.venues:
        p.decision, p.reject_reason = "reject", "not from requested venue " + "/".join(request.venues)
    elif not groups:
        # No query to judge relevance by (expansions, author back-catalogues): never auto-add.
        ok = p.quality_score >= 0.55 and p.final_score >= cfg.review_threshold - 0.1
        p.decision, p.reject_reason = ("review", "no query terms to judge relevance - check manually") if ok else \
            ("reject", f"quality {p.quality_score:.2f} too low without a query")
    elif p.final_score >= cfg.auto_add_threshold:
        blockers = []
        hp = cfg.high_priority_venues
        if hp and p.venue_key not in hp:
            blockers.append("venue not in high-priority list")
        if cfg.get("ranking.prefer_peer_reviewed", True) and not p.is_peer_reviewed:
            blockers.append("not peer reviewed")
        if "type:meeting-abstract" in p.subtopics:
            blockers.append("meeting abstract")
        p.decision, p.reject_reason = ("review", "; ".join(blockers)) if blockers else ("accept", "")
    elif p.final_score >= cfg.review_threshold:
        p.decision, p.reject_reason = "review", ""
    else:
        p.decision, p.reject_reason = "reject", f"score {p.final_score:.2f} below review threshold {cfg.review_threshold:.2f}"
    where = [f"{g[0]} ({'title' if h == 1.0 else 'abstract'})" for g, h in zip(groups, hits) if h > 0]
    p.discovery_reason = ("matches " + ", ".join(where[:6]) if where else "related by citation graph" if not groups else "")
    if p.author_watch:
        p.discovery_reason += f"; author watch: {p.author_watch}"
    if p.relationship:
        p.discovery_reason += f"; {p.relationship}"
    return p


def score_paper(p: Paper, cfg: Config, request: Optional[SearchRequest] = None, today: Optional[date] = None) -> Paper:
    if not cfg.is_radiology:
        return _score_general(p, cfg, request, today)
    cls = p.classification or C.classify(p)
    topics = request.topics if request else []
    mods = request.modalities if request else []

    watch = author_watch_match(p, cfg)
    if watch and not p.author_watch:
        p.author_watch = watch.get("name")
    if request and request.author and any(author_matches(request.author, a) for a in p.authors):
        p.author_watch = p.author_watch or request.author

    semantic = requested_topic_score(cls, topics)
    modality = C.modality_match(cls, mods)
    kw = keyword_match(p, request)
    aw = 1.0 if p.author_watch else 0.0
    rec = recency(p, today)
    relevance = 0.35 * semantic + 0.25 * modality + 0.20 * kw + 0.10 * aw + 0.10 * rec

    quality, qparts = _quality(p, cfg, today)

    p.relevance_score = round(relevance, 4)
    p.quality_score = round(quality, 4)
    p.final_score = round(0.70 * relevance + 0.30 * quality, 4)
    p.score_breakdown = {
        "semantic_topic_match": semantic, "modality_match": modality, "keyword_match": kw,
        "author_watch_match": aw, "recency": rec, **qparts,
    }
    return p


def request_mismatch(p: Paper, request: Optional[SearchRequest]) -> Optional[str]:
    """When a prompt asks for specific topics/modalities/venues, candidates must actually match them."""
    if not request or not request.strict_request_match:
        return None
    cls = p.classification
    if request.modalities:
        if C.modality_match(cls, request.modalities) < C.MODALITY_THRESHOLD:
            return "does not match requested modality " + "/".join(request.modalities)
    if request.topics and requested_topic_score(cls, request.topics) < C.TOPIC_THRESHOLD:
        return "does not match requested topic " + "/".join(request.topics)
    if request.venues and p.venue_key not in request.venues:
        return "not from requested venue " + "/".join(request.venues)
    if request.top_venues_only and p.venue_key not in ("neurips", "icml", "iclr", "cvpr", "iccv", "eccv", "miccai", "midl", "tpami", "tmi",
                                                       "medical_image_analysis", "radiology_ai", "radiology", "jmlr", "tnnls"):
        return "not from a top venue"
    return None


def decide(p: Paper, cfg: Config, request: Optional[SearchRequest] = None, allow_editorials: bool = False) -> Paper:
    """Stage 1 hard filters + thresholds + MVP auto-add restrictions."""
    if request and request.intent == "baselines":
        return _decide_baseline(p, cfg)
    if not cfg.is_radiology:
        return _decide_general(p, cfg, request, allow_editorials)
    reason = C.hard_filter(p, allow_editorials=allow_editorials)
    if reason:
        p.decision, p.reject_reason = "reject", reason
        p.classification["hard_rejected"] = True
        return p
    reason = request_mismatch(p, request)
    if reason:
        p.decision, p.reject_reason = "reject", reason
        return p
    # Topic gate. Strict requests already enforced their own topics in request_mismatch(); every other
    # request (scans, author watches with configured topics, expansions) must show a main-theme signal.
    topics_enforced = bool(request and request.topics and request.strict_request_match)
    if not topics_enforced:
        ok = C.main_topic_score(p.classification) >= C.TOPIC_THRESHOLD or (
            bool(request and request.topics) and requested_topic_score(p.classification, request.topics) >= C.TOPIC_THRESHOLD)
        if not ok:
            p.decision, p.reject_reason = "reject", "no foundation-model / VLM / uncertainty / robustness signal"
            return p

    auto = cfg.auto_add_threshold
    if p.author_watch:
        for a in cfg.authors:
            if a.get("name") == p.author_watch and a.get("auto_add_threshold") is not None:
                auto = min(auto, float(a["auto_add_threshold"]))
    review = cfg.review_threshold

    if p.final_score >= auto:
        blockers = []
        if p.venue_key not in cfg.high_priority_venues:
            blockers.append("venue not in high-priority list")
        if cfg.get("ranking.prefer_peer_reviewed", True) and not p.is_peer_reviewed:
            blockers.append("not peer reviewed")
        if "type:meeting-abstract" in p.subtopics:
            blockers.append("meeting abstract")
        if not any(m in C.PRIMARY_MODALITIES for m in p.modalities) and float(p.classification["general_radiology_relevance"]) < 0.8:
            blockers.append("weak modality evidence")
        if blockers:
            p.decision = "review"
            p.reject_reason = "; ".join(blockers)
        else:
            p.decision = "accept"
    elif p.final_score >= review:
        p.decision = "review"
    else:
        p.decision, p.reject_reason = "reject", f"score {p.final_score:.2f} below review threshold {review:.2f}"
    p.discovery_reason = explain(p)
    return p


def _decide_baseline(p: Paper, cfg: Config) -> Paper:
    """Baselines were explicitly requested for one paper, so the seed defines relevance (a baseline such as
    CLIP need not be a radiology paper). Editorials / errata are still excluded."""
    cls = p.classification or C.classify(p)
    label = (p.graph_evidence or {}).get("label")
    if cls["excluded_type"]:
        p.decision, p.reject_reason = "reject", "excluded publication type"
    elif label in ("seed", "likely baseline"):
        p.decision, p.reject_reason = "accept", ""
    elif label == "possible baseline":
        p.decision, p.reject_reason = "review", "possible baseline - check the evidence"
    else:
        p.decision, p.reject_reason = "reject", f"not a baseline ({label or 'no evidence'})"
    if label not in ("seed", None):
        p.final_score = max(p.final_score, float(p.graph_evidence.get("score", 0)))
    p.discovery_reason = explain(p)
    return p


def explain(p: Paper) -> str:
    mods = {"cxr": "CXR", "ct": "CT", "mammo": "Mammography", "radiology-general": "General radiology"}
    topics = {"foundation-model": "foundation model", "vlm": "VLM", "uncertainty": "uncertainty", "calibration": "calibration",
              "conformal-prediction": "conformal prediction", "ood": "OOD", "distribution-shift": "distribution shift",
              "robustness": "robustness", "selective-prediction": "selective prediction", "report-generation": "report generation",
              "vlm-reliability": "VLM reliability", "image-text-pretraining": "image-text pretraining"}
    parts = [mods.get(m, m) for m in p.modalities] + [topics.get(t, t) for t in p.topics[:4]]
    s = " + ".join(parts) if parts else "keyword match"
    if p.author_watch:
        s += f"; author watch: {p.author_watch}"
    if p.relationship:
        s += f"; {p.relationship}"
    return s


def apply_rate_limit(papers: List[Paper], max_auto: int) -> List[Paper]:
    accepted = sorted([p for p in papers if p.decision == "accept"], key=lambda x: -x.final_score)
    for p in accepted[max_auto:]:
        p.decision = "review"
        p.reject_reason = f"auto-add limit of {max_auto} per run reached"
    return papers
