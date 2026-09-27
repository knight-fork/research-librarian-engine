"""Optional LLM steps: relevance adjudication and NL-command parsing fallback.

Provider-agnostic (src/llm.py). Bibliographic metadata is never taken from the model - it only returns
relevance judgments / search parameters."""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.config import Config
from src.llm import get_llm
from src.models import Paper, SearchRequest

RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "radiology_relevance": {"type": "number"},
        "cxr_relevance": {"type": "number"},
        "ct_relevance": {"type": "number"},
        "mammo_relevance": {"type": "number"},
        "foundation_model_relevance": {"type": "number"},
        "vlm_relevance": {"type": "number"},
        "uncertainty_relevance": {"type": "number"},
        "decision": {"type": "string", "enum": ["accept", "review", "reject"]},
        "reason": {"type": "string"},
    },
    "required": ["radiology_relevance", "cxr_relevance", "ct_relevance", "mammo_relevance", "foundation_model_relevance",
                 "vlm_relevance", "uncertainty_relevance", "decision", "reason"],
    "additionalProperties": False,
}

SYSTEM = """You screen papers for a personal radiology-AI literature library. Scope: chest X-ray, CT and mammography;
foundation models, vision-language models, and uncertainty / calibration / reliability. General medical-imaging work
counts only if its methods clearly transfer to those modalities. Score each field from 0 to 1 using only the title and
abstract you are given. Decide "accept" only for clearly in-scope research papers, "review" when relevance is plausible
but uncertain, and "reject" for off-topic work (other imaging domains, keyword coincidences, editorials).
Keep "reason" to one sentence."""


def adjudicate(p: Paper, cfg: Config, req: Optional[SearchRequest] = None) -> Optional[Dict[str, Any]]:
    llm = get_llm(cfg, "relevance")
    if not llm:
        return None
    wanted = ""
    if req and (req.topics or req.modalities):
        wanted = f"\nThe user specifically asked for: topics={req.topics or 'any'}, modalities={req.modalities or 'any'}."
    user = f"Title: {p.title}\nVenue: {p.venue} ({p.venue_type})\nAbstract: {p.abstract[:4000] or '(no abstract)'}{wanted}"
    out = llm.complete_json(SYSTEM, user, RELEVANCE_SCHEMA, max_output_tokens=512)
    if not out or out.get("decision") not in ("accept", "review", "reject"):
        return None
    return out


PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["scan", "topic_search", "author_search", "venue_search", "similar", "citations",
                                              "related", "missing_lit", "baselines"]},
        "author": {"type": ["string", "null"]},
        "topics": {"type": "array", "items": {"type": "string", "enum": ["foundation-model", "vlm", "uncertainty", "calibration",
                   "conformal-prediction", "selective-prediction", "ood", "distribution-shift", "robustness", "vlm-reliability"]}},
        "modalities": {"type": "array", "items": {"type": "string", "enum": ["cxr", "ct", "mammo", "radiology-general"]}},
        "venues": {"type": "array", "items": {"type": "string"}},
        "year_from": {"type": ["integer", "null"]},
        "year_to": {"type": ["integer", "null"]},
        "seed": {"type": ["string", "null"]},
        "collection": {"type": ["string", "null"]},
        "report_only": {"type": "boolean"},
    },
    "required": ["intent", "author", "topics", "modalities", "venues", "year_from", "year_to", "seed", "collection", "report_only"],
    "additionalProperties": False,
}

PARSE_SYSTEM = ("Convert the user's literature-search command into search parameters. Do not invent authors, venues, "
                "identifiers or papers that the command does not mention; use null when absent. 'seed' is the DOI, arXiv id "
                "or exact title of the paper the command is about (for similar / citations / related / baselines). "
                "Venue ids: miccai, midl, neurips, icml, iclr, cvpr, iccv, eccv, tpami, tmi, medical_image_analysis, "
                "radiology_ai, radiology, jmlr, tnnls, aaai, ijcai, isbi.")


def llm_parse(text: str, cfg: Config) -> Optional[SearchRequest]:
    llm = get_llm(cfg, "parse_fallback")
    if not llm:
        return None
    data = llm.complete_json(PARSE_SYSTEM, text, PARSE_SCHEMA, max_output_tokens=512)
    if not data or data.get("intent") not in PARSE_SCHEMA["properties"]["intent"]["enum"]:
        return None
    # Keep only values that literally occur in the prompt, so the model can't smuggle in invented entities.
    lowered = text.lower()
    author = data.get("author") if data.get("author") and data["author"].lower() in lowered else None
    seed = data.get("seed") if data.get("seed") and data["seed"].lower()[:30] in lowered else None
    req = SearchRequest(intent=data["intent"], author=author, topics=data.get("topics") or [],
                        modalities=data.get("modalities") or [], venues=data.get("venues") or [], seed=seed,
                        collection=data.get("collection"),
                        date_from=f"{data['year_from']}-01-01" if data.get("year_from") else None,
                        date_to=f"{data['year_to']}-12-31" if data.get("year_to") else None,
                        action="report_only" if data.get("report_only") else "add_by_threshold",
                        parse_notes=[f"interpreted by {llm.model}"])
    if req.intent == "author_search":
        req.source_tag = "author-watch"
    if req.intent in ("similar", "citations", "related", "missing_lit", "baselines"):
        req.strict_request_match = False
    return req
