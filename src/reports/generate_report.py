"""Markdown scan reports, stored in reports/YYYY-MM-DD.md."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional

from src.config import REPORTS_DIR
from src.models import Paper
from src.processing.normalize import venue_display

MOD = {"cxr": "CXR", "ct": "CT", "mammo": "Mammography", "radiology-general": "General Radiology"}
TOP = {"foundation-model": "Foundation Model", "vlm": "VLM", "uncertainty": "Uncertainty", "calibration": "Calibration",
       "conformal-prediction": "Conformal Prediction", "ood": "OOD", "distribution-shift": "Distribution Shift",
       "selective-prediction": "Selective Prediction", "robustness": "Robustness", "report-generation": "Report Generation",
       "image-text-pretraining": "Image-Text Pretraining", "vlm-reliability": "VLM Reliability"}

TITLES = {"scan": "Literature Scan", "topic_search": "Topic Search", "author_search": "Author Search",
          "venue_search": "Venue Search", "similar": "Similar-Paper Expansion", "citations": "Citation Expansion",
          "related": "Related-Paper Expansion", "baselines": "Baseline Papers", "missing_lit": "Missing-Literature Analysis", "add_doi": "Manual Add"}


def _entry(i: int, p: Paper, show_reason_why: bool = False) -> str:
    authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
    link = p.url or (f"https://doi.org/{p.doi}" if p.doi else "")
    lines = [f"{i}. **{p.title}**" + (f" ([link]({link}))" if link else ""),
             f"   - Authors: {authors or '-'}",
             f"   - Venue: {venue_display(p)} ({p.venue_type})",
             *([f"   - Modality: {', '.join(MOD.get(m, m) for m in p.modalities)}"] if p.modalities else []),
             *([f"   - Topics: {', '.join(TOP.get(t, t) for t in p.topics)}"] if p.topics else []),
             f"   - Score: {p.final_score:.2f} (relevance {p.relevance_score:.2f}, quality {p.quality_score:.2f})",
             f"   - Reason: {p.discovery_reason or '-'}"]
    if p.relationship:
        lines.append(f"   - Relationship to seed: {p.relationship}")
    if show_reason_why and p.reject_reason:
        lines.append(f"   - Why not auto-added: {p.reject_reason}")
    if p.doi:
        lines.append(f"   - DOI: {p.doi}")
    if p.subtopics:
        lines.append(f"   - Tags: {', '.join(p.subtopics)}")
    return "\n".join(lines)


def render(result) -> str:
    req = result.request
    today = date.today().isoformat()
    accepted, review = result.by_decision("accept"), result.by_decision("review")
    rejected = result.by_decision("reject")
    title = f"Alert: {req.label}" if getattr(req, "label", None) else TITLES.get(req.intent, "Literature Run")
    out: List[str] = [f"# {title} - {today}", ""]
    mode = "DRY RUN - Zotero not modified" if result.dry_run or req.action == "report_only" else "Zotero writes enabled"
    out.append(f"_Mode: {mode}_  ")
    if result.date_from or result.date_to:
        out.append(f"_Window: {result.date_from or '...'} -> {result.date_to or 'now'}_  ")
    params = {k: v for k, v in req.to_dict().items() if v and k not in ("strict_request_match", "action", "intent")}
    if params:
        out.append(f"_Request: {params}_")
    out.append("")
    if result.seed:
        s = result.seed
        out += [f"**Seed paper:** {s.title} ({venue_display(s)}){' - DOI ' + s.doi if s.doi else ''}", ""]
    if result.author_resolution:
        out.append("**Author resolution candidates:**")
        for c in result.author_resolution[:6]:
            out.append(f"- [{c['source']}] {c['name']} ({c['id']}) - {c['works_count']} works; "
                       f"{', '.join(c['institutions'][:2]) or 'no affiliation listed'}")
        out.append("")
    verb = "Would add" if (result.dry_run or req.action == "report_only") else "Added"
    out += [
        f"Candidates retrieved: {result.retrieved}",
        *( [f"Previously seen (skipped): {result.previously_seen}"] if result.previously_seen else [] ),
        f"After initial filtering: {result.after_filter}",
        f"Already in Zotero: {len(result.existing)}",
        f"Published versions of preprints in Zotero: {len(result.upgrades)}",
        f"New high-confidence papers: {len(accepted)}",
        f"Review queue: {len(review)}",
        f"Rejected: {len(rejected)}",
        "",
    ]
    if result.notes:
        out += ["## Notes", ""] + [f"- {n}" for n in result.notes] + [""]
    out += [f"## {verb} (high confidence)", ""]
    out += [_entry(i, p) for i, p in enumerate(accepted, 1)] or ["_None._"]
    out += ["", "## Needs Review", ""]
    out += [_entry(i, p, show_reason_why=True) for i, p in enumerate(review, 1)] or ["_None._"]
    if result.upgrades:
        out += ["", "## Preprint -> published upgrades", ""]
        for p in result.upgrades:
            out.append(f"- {p.title} -> {venue_display(p)}, DOI {p.doi} (Zotero item {p.related_preprint_key})")
    if result.existing:
        out += ["", "## Already in Zotero", ""]
        for p in sorted(result.existing, key=lambda x: -x.final_score)[:30]:
            out.append(f"- {p.title} (key {p.existing_zotero_key}, matched by {p.duplicate_of})")
        if len(result.existing) > 30:
            out.append(f"- ... and {len(result.existing) - 30} more")
    near = [p for p in rejected if p.final_score >= 0.55 and not p.classification.get("hard_rejected")][:15]
    if near:
        out += ["", "## Near misses (rejected)", ""]
        for p in near:
            out.append(f"- {p.title} - {p.final_score:.2f} - {p.reject_reason}")
    reasons = {}
    for p in rejected:
        r = p.reject_reason.split(" (")[0] if p.reject_reason else "other"
        r = "score below review threshold" if r.startswith("score ") else r
        reasons[r] = reasons.get(r, 0) + 1
    if reasons:
        out += ["", "## Rejection reasons", ""] + [f"- {k}: {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])]
    if result.errors:
        out += ["", "## Source errors / warnings", ""] + [f"- {e}" for e in dict.fromkeys(result.errors)]
    written = {k: len(v) for k, v in result.written.items() if v}
    if written:
        out += ["", f"_Zotero items written: {written}_"]
    return "\n".join(out) + "\n"


def write_report(result, directory: Path = REPORTS_DIR, suffix: Optional[str] = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    base = date.today().isoformat() + (f"-{suffix}" if suffix else "")
    path = directory / f"{base}.md"
    n = 2
    while path.exists():
        path = directory / f"{base}-{n}.md"
        n += 1
    path.write_text(render(result))
    result.report_path = str(path)
    return path
