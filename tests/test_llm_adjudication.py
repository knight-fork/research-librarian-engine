"""The optional LLM step may only demote candidates (precision first) and never supplies metadata."""
from src import pipeline
from src.models import Paper, SearchRequest
from src.prompts import relevance


def test_llm_can_only_demote(cfg, monkeypatch):
    cfg.raw["llm"]["enabled"] = True
    cfg.raw["llm"]["tasks"]["relevance"] = True
    verdicts = {"a": "reject", "b": "accept", "c": "review"}
    monkeypatch.setattr(relevance, "adjudicate", lambda p, c, r=None: {"decision": verdicts[p.title], "reason": "mock"})
    a = pipeline.Pipeline(cfg, dry_run=True, use_zotero=False)
    papers = []
    for title, decision, score in (("a", "accept", 0.85), ("b", "review", 0.75), ("c", "accept", 0.84)):
        p = Paper(title=title, final_score=score)
        p.decision = decision
        papers.append(p)
    a._llm_adjudicate(papers, SearchRequest())
    assert [p.decision for p in papers] == ["reject", "review", "review"]


def test_llm_disabled_is_noop(cfg):
    cfg.raw["llm"]["tasks"]["relevance"] = False
    a = pipeline.Pipeline(cfg, dry_run=True, use_zotero=False)
    p = Paper(title="x", final_score=0.9)
    p.decision = "accept"
    a._llm_adjudicate([p], SearchRequest())
    assert p.decision == "accept"


def test_schemas_are_strict():
    for schema in (relevance.RELEVANCE_SCHEMA, relevance.PARSE_SCHEMA):
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
