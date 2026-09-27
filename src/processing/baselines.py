"""Which of a paper's references are its baselines?

Deterministic evidence (no LLM):
  * Semantic Scholar citation contexts - the sentences citing each reference - scanned for comparison
    language ("outperforms", "compared with", "state-of-the-art", table rows), plus S2 intents and the
    'influential' flag;
  * when open-access full text exists (arXiv HTML / PMC), where the reference is cited: Experiments
    sections and result tables are strong signals, Related Work alone is not;
  * topic overlap with the seed paper and recency.
Optional LLM pass (llm.tasks.baselines): reads only the Related Work + Experiments sections, labels each
cited reference baseline / dataset / backbone / background, and must quote supporting text - quotes are
checked against the section text and unverifiable answers are discarded."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.fulltext import MARKER_RE, Document, citation_signals, normalize_for_match
from src.models import Paper
from src.processing import classify as C
from src.processing.normalize import normalize_title

STRONG_CUE = re.compile(r"\b(outperform\w*|better (?:than|performance|results?)|superior to|surpass\w*|"
                        r"compar(?:e|ed|ing|ison)\s+(?:to|with|against)|baselines?|state[- ]of[- ]the[- ]art|sota|"
                        r"competing|competitors?|versus|vs\.?)\b", re.I)
MEDIUM_CUE = re.compile(r"\b(table|tab\.|figure|fig\.|results?|performance|accuracy|auc|f1|zero[- ]shot|fine[- ]tun\w*|"
                        r"re-?implement\w*|reproduc\w*|following|in[- ]line with|same setting|evaluated)\b", re.I)
DATASET_TITLE = re.compile(r"\b(dataset|database|benchmark|challenge|corpus|cohort)\b", re.I)
DATASET_CONTEXT = re.compile(r"\b(dataset|database|benchmark|collected from|publicly available|images? from|split)\b", re.I)
BACKBONE_TITLE = re.compile(r"\b(deep residual learning|an image is worth 16x16|attention is all you need|bert: pre-training|"
                            r"adam: a method|batch normalization|dropout: a simple|imagenet)\b", re.I)

LIKELY, POSSIBLE = 0.55, 0.30


def _context_features(contexts: List[str]) -> Dict[str, Any]:
    strong = [c for c in contexts if STRONG_CUE.search(c)]
    medium = [c for c in contexts if MEDIUM_CUE.search(c)]
    table_rows = [c for c in contexts if len(c.split()) <= 25 and c.count("et al") >= 2 and not STRONG_CUE.search(c)]
    background_lists = [c for c in contexts if (c.count("et al") + c.count(";")) >= 4 and not STRONG_CUE.search(c)]
    return {"strong": strong, "medium": medium, "table_rows": table_rows,
            "background_only": bool(contexts) and len(background_lists) == len(contexts)}


def score_reference(ref: Paper, meta: Dict[str, Any], seed: Paper, signal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Deterministic baseline score in [0, 1], a label, and the best evidence snippet."""
    contexts = [c for c in meta.get("contexts") or [] if c]
    f = _context_features(contexts)
    intents = set(meta.get("intents") or [])
    signal = signal or {}
    C.classify(ref)
    shared_topic = bool(set(ref.topics) & set(seed.topics)) or (
        bool({"foundation-model", "vlm"} & set(ref.topics)) and bool({"foundation-model", "vlm"} & set(seed.topics)))
    years_before = (seed.year or 0) - (ref.year or 0) if seed.year and ref.year else None
    score = (0.45 * bool(f["strong"]) + 0.20 * bool(f["medium"]) + 0.30 * bool(f["table_rows"])
             + 0.25 * bool(meta.get("influential")) + 0.20 * ("result" in intents) + 0.10 * ("methodology" in intents)
             + 0.10 * shared_topic + 0.05 * (years_before is not None and 0 <= years_before <= 4))
    if signal:
        score += (0.50 * bool(signal.get("baseline_section")) + 0.35 * bool(signal.get("tables"))
                  + 0.25 * bool(signal.get("experiments")))
        if not signal.get("experiments") and not signal.get("tables"):
            score -= 0.10  # cited, but never where results are discussed
    if f["background_only"] and not signal.get("experiments"):
        score -= 0.15
    score = round(max(0.0, min(1.0, score)), 3)
    evidence = (f["strong"] or (signal.get("snippets") or []) or f["table_rows"] or f["medium"] or contexts or [""])[0]
    dataset_section_only = bool(signal.get("dataset_section")) and not signal.get("baseline_section")
    if DATASET_TITLE.search(ref.title or "") or dataset_section_only or (
            contexts and all(DATASET_CONTEXT.search(c) for c in contexts) and not f["strong"]):
        label = "benchmark/dataset"
    elif BACKBONE_TITLE.search(ref.title or ""):
        label = "backbone/tool"
    elif score >= LIKELY:
        label = "likely baseline"
    elif score >= POSSIBLE:
        label = "possible baseline"
    else:
        label = "background reference"
    return {"score": score, "label": label, "evidence": _clip(evidence), "evidence_source": "citation context",
            "influential": bool(meta.get("influential")), "intents": sorted(intents),
            "cited_in_experiments": int(signal.get("experiments") or 0), "cited_in_tables": int(signal.get("tables") or 0),
            "cited_in_baseline_section": int(signal.get("baseline_section") or 0)}


def _clip(text: str, n: int = 220) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


# ---------------------------------------------------------------- full text <-> trusted references

def map_bib_to_refs(doc: Document, refs: List[Paper]) -> Dict[str, int]:
    """Bibliography marker -> index into the trusted (Semantic Scholar / OpenAlex) reference list."""
    norm_refs = [(i, normalize_title(r.title)) for i, r in enumerate(refs) if r.title]
    out: Dict[str, int] = {}
    for bid, entry in doc.bib.items():
        title = normalize_title(doc.bib_titles.get(bid, ""))
        blob = normalize_title(entry)
        best = None
        for i, nt in norm_refs:
            if len(nt) < 12:
                continue
            if (title and (nt == title or nt in title or (len(title) > 20 and title in nt))) or nt in blob:
                best = i
                break
        if best is not None:
            out[bid] = best
    return out


BASELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ref_id": {"type": "string"},
                    "role": {"type": "string", "enum": ["baseline", "dataset", "backbone", "background"]},
                    "evidence": {"type": "string"},
                },
                "required": ["ref_id", "role", "evidence"],
            },
        },
    },
    "required": ["references"],
}

BASELINE_SYSTEM = """You identify the baselines of a research paper from excerpts of its Related Work and Experiments sections.
Citations appear as markers like [bib12]. After the excerpts, CITATION SITES lists, for each reference, every place it is
cited (section / sub-section and sentence), followed by its bibliography entry.
Decide each role from the Experiments section first: sub-sections such as "Baselines" or "Compared methods", results
tables, and sentences like "we compare with" / "outperforms". A reference that is also mentioned in Related Work is still
a baseline if the experiments compare against it; quote the Experiments sentence in that case. Methods are often named
without a citation marker (e.g. "ConVIRT works on ..." in a Baselines list, or a table row); the "also named" note tells
you which reference such a name belongs to.
For every cited reference, give its role:
- "baseline": a method the paper's experiments compare against (listed in results tables, "we compare with", "outperforms").
- "dataset": a dataset or benchmark used for training or evaluation.
- "backbone": an architecture, encoder, library or optimizer the paper builds on or uses, but does not compare against.
- "background": mentioned only as related or prior work.
"evidence" must be an exact, verbatim quote (8-30 words) copied from the excerpts that supports the role. Use only the
marker ids provided; never invent references. If unsure between baseline and background, choose background."""


def llm_roles(doc: Document, llm, max_chars: int = 80000) -> Tuple[Dict[str, Tuple[str, str]], List[str]]:
    """Ask the LLM for reference roles over the baseline sections. Returns ({bib_id: (role, quote)}, notes)."""
    sections = doc.baseline_sections()
    text, used = "", []
    for s in sections:
        block = f"## {s.title}\n{s.text}\n\n"
        if len(text) + len(block) > max_chars:
            block = block[: max(0, max_chars - len(text))]
        text += block
        used.append(s.title)
        if len(text) >= max_chars:
            break
    cited = sorted(set(MARKER_RE.findall(text)), key=lambda b: (len(b), b))
    if not cited:
        return {}, [f"full text read ({', '.join(used)}) but no linked citations found"]
    signals = citation_signals(doc)
    lines = []
    for b in cited:
        if b not in doc.bib:
            continue
        sig = signals.get(b, {})
        sites = "\n".join(f"    - ({title}) {sentence[:200]}" for title, sentence in (sig.get("sites") or []))
        aka = f"  (also named in the text: {', '.join(sig['aliases'])})" if sig.get("aliases") else ""
        lines.append(f"[{b}] {doc.bib.get(b, '')[:250]}{aka}\n{sites}")
    listing = "\n".join(lines)
    out = llm.complete_json(BASELINE_SYSTEM, f"EXCERPTS\n{text}\nCITATION SITES\n{listing}", BASELINE_SCHEMA, max_output_tokens=8192)
    notes = [f"read sections: {', '.join(used)} ({len(text):,} chars, {len(cited)} cited references) with {llm.model}"]
    if not out:
        return {}, notes + ["LLM gave no usable answer; used citation evidence only"]
    haystack = normalize_for_match(text)
    roles: Dict[str, Tuple[str, str]] = {}
    rejected = 0
    for item in out.get("references") or []:
        bid = str(item.get("ref_id", "")).strip("[] ")
        role, quote = item.get("role"), item.get("evidence") or ""
        q = normalize_for_match(quote)
        # Verify: the marker exists and the quote really occurs in what the model was shown.
        if bid not in cited or role not in ("baseline", "dataset", "backbone", "background") or len(q) < 15 or q[:120] not in haystack:
            rejected += 1
            continue
        roles[bid] = (role, _clip(MARKER_RE.sub("", quote)))
    if rejected:
        notes.append(f"discarded {rejected} LLM answer(s) whose quote could not be verified in the text")
    return roles, notes


def combine(det: Dict[str, Any], llm_role: Optional[Tuple[str, str]], llm_read: bool, model: str = "") -> Dict[str, Any]:
    """Merge the deterministic verdict with a verified LLM role."""
    ev = dict(det)
    if llm_role:
        role, quote = llm_role
        ev["llm_role"] = role
        if role == "baseline" and det["label"] == "benchmark/dataset":
            # e.g. a baseline named after its pretraining data ("ImageNet is a ResNet-50 pretrained on ImageNet")
            ev.update(label="possible baseline", evidence=quote, evidence_source=f"{model} (verified quote; dataset paper)")
        elif role == "baseline":
            ev.update(score=max(ev["score"], 0.8), label="likely baseline", evidence=quote, evidence_source=f"{model} (verified quote)")
        elif role == "dataset":
            ev.update(label="benchmark/dataset", evidence=quote, evidence_source=f"{model} (verified quote)")
        elif role == "backbone":
            ev["label"] = "possible baseline" if det["score"] >= 0.75 and det["label"] == "likely baseline" else "backbone/tool"
        else:  # background
            ev["label"] = "possible baseline" if det["label"] == "likely baseline" else "background reference"
    elif llm_read and ev["label"] == "likely baseline":
        ev["label"] = "possible baseline"  # the model read the experiments and did not name it: send to review
    return ev


def seed_tag(seed: Paper) -> str:
    surname = re.sub(r"[^a-z]", "", (seed.authors[0].split()[-1] if seed.authors else "paper").lower())
    word = next((w for w in re.findall(r"[a-z0-9]+", normalize_title(seed.title)) if len(w) > 3), "paper")
    return f"{surname}{seed.year or ''}-{word}"[:40]
