"""Validation against a manually labeled gold set.

data/gold/gold.jsonl, one JSON object per line:
  {"doi": "...", "label": "relevant|borderline|irrelevant"}                # metadata fetched from trusted sources
  {"title": "...", "abstract": "...", "venue": "...", "year": 2025, "label": "..."}   # offline row
Optional expected metadata for the metadata-error metric: "expect": {"year": 2024, "venue_key": "miccai"}."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.config import ROOT, Config
from src.models import Paper
from src.processing import classify as C
from src.processing import rank as R
from src.processing.normalize import canonical_venue, finalize


def _rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def evaluate(cfg: Config, gold_path: str) -> Dict[str, Any]:
    path = Path(gold_path)
    if not path.is_absolute():
        path = ROOT / path
    rows = _rows(path)
    pipe = None
    results = []
    meta_checked = meta_errors = 0
    for row in rows:
        if row.get("title"):
            key, vtype = canonical_venue(row.get("venue", ""), row.get("venue_type"))
            p = finalize(Paper(title=row["title"], abstract=row.get("abstract", ""), venue=row.get("venue", ""), venue_key=key,
                               venue_type=row.get("venue_type") or vtype, year=row.get("year"), doi=row.get("doi"),
                               authors=row.get("authors", []), citation_count=row.get("citation_count"), source="gold"))
        else:
            if pipe is None:
                from src.pipeline import Pipeline
                pipe = Pipeline(cfg, dry_run=True, use_zotero=False)
            p = pipe.fetch_identifier(doi=row.get("doi"), arxiv_id=row.get("arxiv_id"), pmid=row.get("pmid"))
            if not p:
                results.append({"row": row, "decision": "missing"})
                continue
        C.classify(p)
        R.score_paper(p, cfg)
        R.decide(p, cfg)
        for field, want in (row.get("expect") or {}).items():
            meta_checked += 1
            if getattr(p, field, None) != want:
                meta_errors += 1
        results.append({"row": row, "decision": p.decision, "score": p.final_score, "title": p.title})

    def count(label, decisions):
        return sum(1 for r in results if r["row"]["label"] == label and r["decision"] in decisions)

    accepted = [r for r in results if r["decision"] == "accept"]
    relevant = [r for r in results if r["row"]["label"] == "relevant"]
    irrelevant = [r for r in results if r["row"]["label"] == "irrelevant"]
    out = {
        "n": len(results),
        "precision@auto-add": round(sum(r["row"]["label"] == "relevant" for r in accepted) / len(accepted), 3) if accepted else None,
        "recall@review": round(count("relevant", ("accept", "review")) / len(relevant), 3) if relevant else None,
        "false_positive_rate": round(count("irrelevant", ("accept", "review")) / len(irrelevant), 3) if irrelevant else None,
        "metadata_error_rate": round(meta_errors / meta_checked, 3) if meta_checked else None,
        "target": "precision@auto-add >= 0.95",
        "false_accepts": [r["title"] if "title" in r else r["row"].get("title") for r in accepted if r["row"]["label"] != "relevant"],
        "missed_relevant": [r.get("title") or r["row"].get("title") or r["row"].get("doi") for r in relevant if r["decision"] not in ("accept", "review")],
    }
    return out
