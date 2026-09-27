"""Radiology profile: deterministic topic / modality classification (CXR / CT / mammography ontology).

Scores are 0-1 per dimension, driven by phrase-aware keyword matching with title hits
weighted above abstract hits. Designed for precision; ambiguous cases can be sent to the
optional LLM adjudicator (src/prompts/relevance.py)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.models import Paper


@dataclass(frozen=True)
class Term:
    pattern: str
    weight: float = 1.0
    case_sensitive: bool = False

    def regex(self) -> "re.Pattern[str]":
        flags = 0 if self.case_sensitive else re.I
        # Terms may follow a hyphen ("PET-CT", "well-calibrated", "RAD-DINO", "CXR-CLIP") but not a letter/digit.
        return re.compile(r"(?<!\w)" + self.pattern + r"(?![\w])", flags)


def T(pattern: str, weight: float = 1.0, cs: bool = False) -> Term:
    return Term(pattern, weight, cs)


# --------------------------------------------------------------------------- modalities

MODALITY_TERMS: Dict[str, List[Term]] = {
    "cxr": [
        T(r"chest\s?x[\s-]?rays?"), T(r"chest\s+radiograph(s|y|ic)?"), T(r"cxrs?"), T(r"mimic[\s-]cxr"),
        T(r"chexpert(\s?plus)?"), T(r"chest\s?x[\s-]?ray\s?14"), T(r"padchest"), T(r"vindr[\s-]cxr"),
        T(r"chest\s+films?", 0.8), T(r"thoracic\s+radiograph(s|y)?"), T(r"frontal\s+radiographs?", 0.6),
        T(r"pneumothorax", 0.4), T(r"cardiomegaly", 0.4),
    ],
    "ct": [
        T(r"computed\s+tomography"), T(r"(chest|thoracic|lung|abdominal|abdomen|head|brain|cardiac|whole[\s-]body|body|non[\s-]contrast|contrast[\s-]enhanced|low[\s-]dose|\d+d|pet|spect|cone[\s-]beam|photon[\s-]counting)[\s/-]?ct"),
        T(r"ct\s+(scans?|images?|imaging|volumes?|data|studies|slices|reports?|examinations?)"), T(r"cect"), T(r"ldct"),
        T(r"ct[\s-]rate"), T(r"totalsegmentator"), T(r"lidc(-idri)?"), T(r"deeplesion"), T(r"cbct", 0.6),
        T(r"CT", 0.6, cs=True),  # bare token: only counted with radiology context (see below)
    ],
    "mammo": [
        T(r"mammograph(y|ic)"), T(r"mammograms?"), T(r"breast\s+imaging"), T(r"digital\s+breast\s+tomosynthesis"),
        T(r"breast\s+tomosynthesis"), T(r"DBT", 0.7, cs=True), T(r"FFDM", 1.0, cs=True), T(r"full[\s-]field\s+digital\s+mammography"),
        T(r"bi[\s-]?rads"), T(r"breast\s+density", 0.8), T(r"breast\s+cancer\s+screening"), T(r"cbis[\s-]ddsm"),
        T(r"inbreast"), T(r"vindr[\s-]mammo"), T(r"embed\s+dataset", 0.6), T(r"(micro)?calcifications?", 0.4),
        T(r"architectural\s+distortion", 0.6),
    ],
    "radiology-general": [
        T(r"radiolog(y|ical|ist|ists)"), T(r"radiograph(s|y|ic)?", 0.8), T(r"medical\s+imag(ing|es?)"),
        T(r"radiology\s+reports?"), T(r"x[\s-]?rays?", 0.6), T(r"imaging\s+modalit(y|ies)", 0.6),
        T(r"diagnostic\s+imaging"), T(r"medical\s+image\s+(analysis|segmentation|classification|understanding)", 0.8),
        T(r"clinical\s+imaging", 0.6), T(r"3d\s+medical\s+images?", 0.8), T(r"volumetric\s+medical", 0.8),
    ],
}

MAMMO_SPECIFIC: List[Term] = [
    T(r"mammograph(y|ic)"), T(r"mammograms?"), T(r"(digital\s+)?breast\s+tomosynthesis"), T(r"DBT", 0.7, cs=True),
    T(r"FFDM", 1.0, cs=True), T(r"cbis[\s-]ddsm"), T(r"inbreast"), T(r"vindr[\s-]mammo"),
]

# Non-radiology modalities: presence without radiology signals triggers rejection.
OTHER_MODALITY_TERMS: List[Term] = [
    T(r"histopatholog(y|ical)"), T(r"whole[\s-]slide\s+images?"), T(r"computational\s+pathology"), T(r"pathology\s+slides?"),
    T(r"dermoscop(y|ic)"), T(r"skin\s+lesions?"), T(r"dermatolog(y|ical)"), T(r"fundus"), T(r"retinal?"),
    T(r"OCT", 1.0, cs=True), T(r"optical\s+coherence\s+tomography"), T(r"endoscop(y|ic)"), T(r"colonoscop(y|ic)"),
    T(r"electrocardiogra(m|phy)"), T(r"ECG", 1.0, cs=True), T(r"EEG", 1.0, cs=True), T(r"cytolog(y|ical)"),
    T(r"surgical\s+videos?"), T(r"electronic\s+health\s+records?", 0.6),
]

# Other radiology modalities: not primary, but still radiology.
SECONDARY_RADIOLOGY_TERMS: List[Term] = [
    T(r"MRI", 1.0, cs=True), T(r"magnetic\s+resonance"), T(r"ultrasound"), T(r"sonograph(y|ic)"), T(r"PET", 0.8, cs=True),
    T(r"positron\s+emission"), T(r"spect"), T(r"fluoroscop(y|ic)"), T(r"angiograph(y|ic)"),
]

# --------------------------------------------------------------------------- topics

TOPIC_TERMS: Dict[str, List[Term]] = {
    "foundation-model": [
        T(r"foundation(al)?\s+models?"), T(r"foundation\s+encoders?"), T(r"generalist\s+(medical\s+)?(ai|models?)"),
        T(r"universal\s+(medical\s+)?models?", 0.8), T(r"self[\s-]supervised\s+(pre[\s-]?train(ing|ed)?|representation\s+learning|visual\s+representations?)"),
        T(r"self[\s-]supervised", 0.6),
        T(r"large[\s-]scale\s+pre[\s-]?train(ing|ed)?"), T(r"(vision|visual)[\s-]language\s+pre[\s-]?train(ing|ed)?"),
        T(r"(image|language)[\s-](text|report|image)\s+pre[\s-]?train(ing|ed)?"), T(r"pre[\s-]?train(ing|ed)", 0.6), T(r"masked\s+(image\s+)?(auto[\s-]?encoders?|modeling)"),
        T(r"contrastive\s+(learning|pre[\s-]?training)", 0.7), T(r"zero[\s-]shot", 0.7), T(r"few[\s-]shot", 0.5),
        T(r"linear\s+prob(e|ing)", 0.6), T(r"frozen\s+(encoder|backbone|features?)", 0.6), T(r"parameter[\s-]efficient", 0.6),
        T(r"prompt[\s-](tuning|learning)", 0.5), T(r"dino(v2|v3)?", 0.6), T(r"segment\s+anything|medsam", 0.7), T(r"SAM2?", 0.6, cs=True),
        T(r"representation\s+learning", 0.5), T(r"backbone\s+models?", 0.3),
    ],
    "vlm": [
        T(r"vision[\s-](and[\s-])?language(\s+models?)?"), T(r"visual[\s-]language(\s+models?)?"), T(r"vlms?"),
        T(r"multimodal\s+large\s+language\s+models?"), T(r"mllms?"), T(r"large\s+multimodal\s+models?"), T(r"LMMs?", 0.8, cs=True),
        T(r"(radiology\s+)?report\s+generation"), T(r"generat(e|ing|ion\s+of)\s+(radiology\s+)?reports?", 0.8),
        T(r"image[\s-](text|report|language)"), T(r"text[\s-]image", 0.7), T(r"language[\s-]image"), T(r"visual\s+question\s+answering"),
        T(r"vqa"), T(r"(phrase|visual)\s+grounding"), T(r"cross[\s-]modal\s+retrieval", 0.8), T(r"CLIP", 0.7, cs=True), T(r"llava(-med)?"),
        T(r"multimodal", 0.3), T(r"large\s+language\s+models?|llms?", 0.3), T(r"medclip|biomedclip|chexzero|gloria|convirt|medklip|maira|chexagent|radfm", 1.0),
    ],
    "uncertainty": [
        T(r"uncertaint(y|ies)"), T(r"uncertain", 0.6), T(r"epistemic"), T(r"aleatoric"), T(r"bayesian", 0.7),
        T(r"monte[\s-]carlo\s+dropout|mc[\s-]dropout"), T(r"deep\s+ensembles?"), T(r"evidential"), T(r"confidence\s+estimation"),
        T(r"predictive\s+confidence", 0.8), T(r"trustworth(y|iness)", 0.5), T(r"reliab(le|ility)", 0.4), T(r"credal", 0.8),
    ],
    "calibration": [
        T(r"calibrat(ion|ed|ing)"), T(r"mis[\s-]?calibrat(ion|ed)"), T(r"expected\s+calibration\s+error"), T(r"ECE", 0.8, cs=True),
        T(r"temperature\s+scaling"), T(r"over[\s-]?confiden(ce|t)", 0.7),
    ],
    "conformal-prediction": [T(r"conformal(\s+prediction|\s+risk\s+control)?"), T(r"prediction\s+sets?", 0.7), T(r"coverage\s+guarantees?", 0.6)],
    "selective-prediction": [
        T(r"selective\s+(prediction|classification)"), T(r"abstention|abstain(ing)?"), T(r"reject\s+option"),
        T(r"learning\s+to\s+defer|deferral", 0.9), T(r"triage", 0.4),
    ],
    "ood": [
        T(r"out[\s-]of[\s-]distribution"), T(r"ood(\s+detection)?"), T(r"novelty\s+detection", 0.8), T(r"failure\s+detection"),
        T(r"misclassification\s+detection"), T(r"anomal(y|ies)\s+detection", 0.3),
    ],
    "distribution-shift": [
        T(r"distribution(al)?\s+shifts?"), T(r"domain\s+shifts?"), T(r"dataset\s+shifts?"), T(r"covariate\s+shifts?"),
        T(r"domain\s+generali[sz]ation", 0.8), T(r"domain\s+adaptation", 0.6), T(r"(scanner|vendor|site|hospital|temporal)\s+(shift|variability|differences)"),
        T(r"external\s+validation", 0.7), T(r"out[\s-]of[\s-]domain", 0.7),
    ],
    "robustness": [
        T(r"robustness"), T(r"robust", 0.4), T(r"fairness"), T(r"(demographic|subgroup)\s+(bias|disparit(y|ies)|performance)"),
        T(r"shortcut\s+learning|shortcuts?", 0.8), T(r"spurious\s+correlations?"), T(r"label\s+noise|noisy\s+labels?"),
        T(r"weak(ly)?\s+supervis(ion|ed)", 0.5), T(r"external\s+validation", 0.7), T(r"adversarial", 0.5), T(r"bias", 0.3),
    ],
    "hallucination": [
        T(r"hallucinat(ion|ions|e|ed|ing)"), T(r"factual(ity|\s+consistency|ly\s+(correct|consistent))"), T(r"faithful(ness)?", 0.6),
        T(r"self[\s-]consistency", 0.8), T(r"answer\s+consistency"), T(r"grounding\s+confidence"),
    ],
    "report-generation": [T(r"report\s+generation"), T(r"generat(e|ing|ion\s+of)\s+(radiology\s+)?reports?", 0.8), T(r"findings\s+generation", 0.8)],
    "image-text-pretraining": [
        T(r"(vision|visual)[\s-]language\s+pre[\s-]?training"), T(r"image[\s-](text|report)\s+(pre[\s-]?training|contrastive|alignment)"),
        T(r"contrastive\s+language[\s-]image"), T(r"CLIP", 0.6, cs=True), T(r"report[\s-]supervis(ed|ion)", 0.8), T(r"medclip|biomedclip|chexzero|gloria|convirt", 1.0),
    ],
}

# Taxonomy tags. Tag string -> terms.
UNCERTAINTY_TAXONOMY: Dict[str, List[Term]] = {
    "uncertainty:epistemic": [T(r"epistemic")],
    "uncertainty:aleatoric": [T(r"aleatoric")],
    "uncertainty:predictive": [T(r"predictive\s+uncertainty"), T(r"predictive\s+entropy")],
    "uncertainty:calibration": TOPIC_TERMS["calibration"],
    "uncertainty:conformal-prediction": TOPIC_TERMS["conformal-prediction"],
    "uncertainty:selective-prediction": TOPIC_TERMS["selective-prediction"],
    "uncertainty:ood-detection": [T(r"out[\s-]of[\s-]distribution\s+detection"), T(r"ood\s+detection"), T(r"failure\s+detection")],
    "uncertainty:distribution-shift": TOPIC_TERMS["distribution-shift"],
    "uncertainty:ensemble-methods": [T(r"(deep\s+)?ensembles?")],
    "uncertainty:bayesian-methods": [T(r"bayesian"), T(r"variational\s+inference"), T(r"monte[\s-]carlo\s+dropout|mc[\s-]dropout"), T(r"laplace\s+approximation")],
    "uncertainty:test-time-uncertainty": [T(r"test[\s-]time\s+(augmentation|uncertainty|adaptation)")],
    "uncertainty:vlm-reliability": TOPIC_TERMS["hallucination"],
}
FM_TAXONOMY: Dict[str, List[Term]] = {
    "fm:image-only": [T(r"image[\s-]only"), T(r"visual\s+foundation\s+models?"), T(r"vision\s+foundation\s+models?")],
    "fm:image-text": TOPIC_TERMS["image-text-pretraining"],
    "fm:vlm": [T(r"vision[\s-]language\s+models?"), T(r"vlms?"), T(r"mllms?"), T(r"multimodal\s+large\s+language\s+models?")],
    "fm:self-supervised": [T(r"self[\s-]supervised"), T(r"masked\s+(image\s+)?(auto[\s-]?encoders?|modeling)"), T(r"dino(v2|v3)?")],
    "fm:contrastive": [T(r"contrastive")],
    "fm:generative": [T(r"generative\s+(pre[\s-]?train\w*|foundation|models?)"), T(r"diffusion\s+models?"), T(r"autoregressive", 0.7), T(r"generative", 0.5)],
    "fm:generalist": [T(r"generalist"), T(r"universal\s+(medical\s+)?models?"), T(r"multi[\s-]task\s+foundation")],
    "fm:zero-shot": [T(r"zero[\s-]shot")],
    "fm:few-shot": [T(r"few[\s-]shot")],
    "fm:prompt-tuning": [T(r"prompt[\s-](tuning|learning)"), T(r"visual\s+prompt")],
    "fm:parameter-efficient-adaptation": [T(r"parameter[\s-]efficient"), T(r"LoRA", 1.0, cs=True), T(r"adapters?", 0.6)],
}

# Searched (not anchored) so PubMed's "published erratum" / "retraction of publication" / "retracted publication" match.
EXCLUDED_TYPE_RE = re.compile(r"\b(editorial|comment|commentary|letter|erratum|correction|retraction|retracted|news|paratext|"
                              r"expression of concern)\b", re.I)
EXCLUDED_TITLE_RE = re.compile(
    r"^(erratum|corrigendum|(author\s+|publisher\s+)?correction\b|retraction|retracted|notice of retraction|withdrawn|"
    r"expression of concern|reply to|response to|in reply|comment on|letter to the editor|"
    r"editorial|preface|front matter|back matter|table of contents|proceedings of)\b", re.I)
SURVEY_RE = re.compile(r"\b(survey|review|overview|systematic review|meta[\s-]analysis|scoping review|tutorial)\b", re.I)
BENCHMARK_RE = re.compile(r"\b(dataset|benchmark|challenge)\b", re.I)
MEETING_ABSTRACT_RE = re.compile(r"^(abstract\s+[\w.\-]+\s*:|[A-Z]{1,3}\d{1,3}[\-.]\d{1,3}[\-.]?\d*\s*:|poster\s+[\w.\-]+\s*:)", re.I)

PRIMARY_MODALITIES = ("cxr", "ct", "mammo")
MODALITY_THRESHOLD = 0.45
TOPIC_THRESHOLD = 0.45


# --------------------------------------------------------------------------- matching

_COMPILED: Dict[Term, "re.Pattern[str]"] = {}


def _compiled(terms: Sequence[Term]) -> List[Tuple[Term, "re.Pattern[str]"]]:
    # Cache per Term (hashable, frozen) - never per list id(), which Python reuses for temporary lists.
    out = []
    for t in terms:
        rx = _COMPILED.get(t)
        if rx is None:
            rx = _COMPILED[t] = t.regex()
        out.append((t, rx))
    return out


STRONG_WEIGHT = 0.9   # a dimension needs at least one term this strong to score above WEAK_CAP
WEAK_CAP = 0.6


def _hits(terms: Sequence[Term], text: str) -> Tuple[float, int, List[str]]:
    """Sum of weights over distinct matching terms, total occurrences, matched patterns."""
    weight, occ, matched = 0.0, 0, []
    for term, rx in _compiled(terms):
        n = len(rx.findall(text))
        if n:
            weight += term.weight
            occ += n
            matched.append(term.pattern)
    return weight, occ, matched


def _strong(terms: Sequence[Term], text: str) -> bool:
    return any(t.weight >= STRONG_WEIGHT and rx.search(text) for t, rx in _compiled(terms))


# Ambiguous acronyms only count when their intended sense is supported by the surrounding text.
TERM_CONTEXT = {
    "SAM2?": re.compile(r"segment", re.I),                                  # vs "spatial attention module"
    "LMMs?": re.compile(r"multi[\s-]?modal|vision|visual|language", re.I),  # vs "linear mixed model"
    "CLIP": re.compile(r"contrastive|language|text|zero[\s-]shot|vision", re.I),
}


def _with_context(terms: Sequence[Term], text: str) -> List[Term]:
    return [t for t in terms if t.pattern not in TERM_CONTEXT or TERM_CONTEXT[t.pattern].search(text)]


def _taxonomy_hit(terms: Sequence[Term], text: str) -> bool:
    """Taxonomy tags (only evaluated once a paper is in the parent topic): one specific term anywhere, or enough weak ones."""
    weight, _, _ = _hits(terms, text)
    return _strong(terms, text) or weight >= 1.0


def dimension_score(terms: Sequence[Term], title: str, abstract: str) -> Tuple[float, List[str]]:
    tw, _, tmatch = _hits(terms, title)
    aw, aocc, amatch = _hits(terms, abstract)
    s_title = min(1.0, tw) if _strong(terms, title) else min(WEAK_CAP, tw)
    s_abs = min(1.0, aw / 2.0) if _strong(terms, abstract) else min(WEAK_CAP, aw / 2.0)
    occ = min(1.0, aocc / 5.0)
    score = max(s_title * 0.85 + s_abs * 0.15, s_abs * 0.7 + occ * 0.1)
    return round(min(1.0, score), 4), sorted(set(tmatch + amatch))


def classify(p: Paper) -> Dict[str, object]:
    """Populate p.topics / p.modalities / p.subtopics / p.classification; returns the classification."""
    title, abstract = p.title or "", p.abstract or ""
    text = f"{title} {abstract}"

    general, gmatch = dimension_score(MODALITY_TERMS["radiology-general"], title, abstract)
    secondary, _ = dimension_score(SECONDARY_RADIOLOGY_TERMS, title, abstract)
    other, omatch = dimension_score(OTHER_MODALITY_TERMS, title, abstract)

    mod_scores: Dict[str, float] = {}
    matched: Dict[str, List[str]] = {"radiology-general": gmatch, "other": omatch}
    for m in PRIMARY_MODALITIES:
        terms = MODALITY_TERMS[m]
        if m == "ct":
            # Bare "CT" only counts with some radiology context; phrase-aware otherwise.
            has_context = general > 0 or secondary > 0 or bool(re.search(
                r"\b(lesions?|tumou?rs?|nodules?|organs?|patients?|clinical|scans?|scanners?|diagnos\w*|segment\w*|"
                r"hounsfield|contrast|slices?|voxels?|volumes?|hospitals?)\b", text, re.I))
            if not has_context:
                terms = [t for t in terms if t.pattern != "CT"]
        score, mt = dimension_score(terms, title, abstract)
        if m == "mammo" and score > 0:
            strong, _ = dimension_score(MAMMO_SPECIFIC, title, abstract)
            other_breast_modality = _hits(SECONDARY_RADIOLOGY_TERMS, title)[0] > 0
            if strong < MODALITY_THRESHOLD and (other_breast_modality or secondary >= MODALITY_THRESHOLD):
                score = round(score * 0.4, 4)  # e.g. breast MRI / breast ultrasound
        mod_scores[m] = score
        matched[m] = mt

    best_primary = max(mod_scores.values()) if mod_scores else 0.0
    radiology_relevance = max(best_primary, general, 0.7 * secondary)

    topic_scores: Dict[str, float] = {}
    for t, terms in TOPIC_TERMS.items():
        terms = _with_context(terms, text)
        score, mt = dimension_score(terms, title, abstract)
        topic_scores[t] = score
        matched["topic:" + t] = mt

    uncertainty_group = max(topic_scores[k] for k in ("uncertainty", "calibration", "conformal-prediction", "selective-prediction", "ood", "hallucination"))
    # A shift paper is only an uncertainty/reliability paper when it also detects / quantifies something.
    uncertainty_group = max(uncertainty_group, 0.8 * min(topic_scores["distribution-shift"], max(topic_scores["ood"], topic_scores["uncertainty"], topic_scores["calibration"]) + 0.3))

    modalities = [m for m in PRIMARY_MODALITIES if mod_scores[m] >= MODALITY_THRESHOLD]
    if not modalities and (general >= MODALITY_THRESHOLD or secondary >= MODALITY_THRESHOLD):
        modalities = ["radiology-general"]
    # Primarily non-radiology imaging with no radiology signal -> other.
    is_other_modality = other >= 0.6 and radiology_relevance < 0.45
    if is_other_modality:
        modalities = []

    topics: List[str] = []
    for t in ("foundation-model", "vlm", "uncertainty", "calibration", "conformal-prediction", "selective-prediction",
              "ood", "distribution-shift", "robustness", "report-generation", "image-text-pretraining"):
        if topic_scores[t] >= TOPIC_THRESHOLD:
            topics.append(t)
    if topic_scores["hallucination"] >= TOPIC_THRESHOLD and topic_scores["vlm"] >= TOPIC_THRESHOLD:
        topics.append("vlm-reliability")
    if uncertainty_group >= TOPIC_THRESHOLD and "uncertainty" not in topics and any(
            t in topics for t in ("calibration", "conformal-prediction", "selective-prediction", "ood", "vlm-reliability")):
        topics.append("uncertainty")

    subtopics: List[str] = []
    if uncertainty_group >= TOPIC_THRESHOLD:
        subtopics += [tag for tag, terms in UNCERTAINTY_TAXONOMY.items() if _taxonomy_hit(terms, text)]
    if topic_scores["foundation-model"] >= TOPIC_THRESHOLD or topic_scores["vlm"] >= TOPIC_THRESHOLD:
        subtopics += [tag for tag, terms in FM_TAXONOMY.items() if _taxonomy_hit(terms, text)]
    if SURVEY_RE.search(title):
        subtopics.append("type:survey")
    if BENCHMARK_RE.search(title):
        subtopics.append("type:benchmark-dataset")
    if MEETING_ABSTRACT_RE.match(title.strip()) or re.search(r"meeting abstract|conference abstract", p.publication_type or "", re.I):
        subtopics.append("type:meeting-abstract")

    title_dimensions = [m for m in PRIMARY_MODALITIES if _hits(MODALITY_TERMS[m], title)[0] > 0]
    title_dimensions += [t for t in ("foundation-model", "vlm", "uncertainty", "calibration", "conformal-prediction",
                                     "selective-prediction", "ood", "distribution-shift", "robustness", "hallucination")
                         if _hits(TOPIC_TERMS[t], title)[0] >= 0.7]

    excluded = bool(EXCLUDED_TYPE_RE.search((p.publication_type or "").strip())) or bool(EXCLUDED_TITLE_RE.match(title.strip()))

    cls: Dict[str, object] = {
        "radiology_relevance": round(radiology_relevance, 4),
        "cxr_relevance": mod_scores["cxr"],
        "ct_relevance": mod_scores["ct"],
        "mammo_relevance": mod_scores["mammo"],
        "general_radiology_relevance": general,
        "secondary_radiology_relevance": secondary,
        "other_modality_relevance": other,
        "foundation_model_relevance": topic_scores["foundation-model"],
        "vlm_relevance": topic_scores["vlm"],
        "uncertainty_relevance": round(uncertainty_group, 4),
        "robustness_relevance": max(topic_scores["robustness"], topic_scores["distribution-shift"]),
        "topic_scores": topic_scores,
        "is_other_modality": is_other_modality,
        "excluded_type": excluded,
        "matched_terms": matched,
        "title_dimensions": title_dimensions,
    }
    p.modalities = modalities
    p.topics = topics
    p.subtopics = subtopics
    p.classification = cls
    return cls


def hard_filter(p: Paper, allow_editorials: bool = False) -> Optional[str]:
    """Stage-1 hard filters. Returns a rejection reason or None."""
    cls = p.classification or classify(p)
    if not p.title:
        return "missing title"
    if cls["excluded_type"] and not allow_editorials:
        return f"excluded publication type ({p.publication_type or 'title pattern'})"
    if cls["is_other_modality"]:
        return "non-radiology imaging modality"
    if float(cls["radiology_relevance"]) < 0.3:
        return "not related to radiology / medical imaging"
    if not p.modalities:
        return "no CXR / CT / mammography / general-radiology relevance"
    return None


def main_topic_score(cls: Dict[str, object]) -> float:
    """Semantic topic match over the primary themes (A-C strong, D weaker)."""
    fm = float(cls["foundation_model_relevance"])
    vlm = float(cls["vlm_relevance"])
    unc = float(cls["uncertainty_relevance"])
    rob = float(cls["robustness_relevance"]) * 0.75
    best = max(fm, vlm, unc, rob)
    strong = sum(1 for s in (fm, vlm, unc) if s >= TOPIC_THRESHOLD)
    return round(min(1.0, best + 0.05 * max(0, strong - 1)), 4)


def modality_match(cls: Dict[str, object], wanted: Sequence[str] = ()) -> float:
    scores = {"cxr": cls["cxr_relevance"], "ct": cls["ct_relevance"], "mammo": cls["mammo_relevance"],
              "radiology-general": cls["general_radiology_relevance"]}
    if wanted:
        # A request for "radiology" / "medical imaging" means any radiology modality, not only papers using those words.
        scores = dict(scores, **{"radiology-general": cls["radiology_relevance"]})
        best = max(float(scores.get(m, 0.0)) for m in wanted)
        return round(best, 4)
    primary = max(float(scores[m]) for m in PRIMARY_MODALITIES)
    general = float(scores["radiology-general"])
    return round(max(primary, 0.6 * general, 0.5 * float(cls["secondary_radiology_relevance"]), 0.2), 4)
