"""General profile: any field, relevance from the query's own terms."""
from datetime import date, datetime

import pytest

from src import alerts as A
from src.discovery.query import groups_from_text
from src.models import Paper, SearchRequest
from src.processing import rank as R
from src.processing.normalize import finalize
from src.prompts.parser import extract_subject, parse
from src.zotero.collections import hierarchy, target_collections

TODAY = date(2026, 9, 27)


def _paper(**kw):
    base = dict(authors=["A B"], publication_date="2026-03-01", doi="10.1234/x", source="openalex")
    base.update(kw)
    return finalize(Paper(**base))


def _run(p, cfg, text, intent="topic_search"):
    req = SearchRequest(intent=intent, free_text=text)
    R.score_paper(p, cfg, req, TODAY)
    return R.decide(p, cfg, req)


def test_subject_extraction():
    assert extract_subject("fetch me recent papers on graph neural networks for drug discovery since 2023") == \
        "graph neural networks for drug discovery"
    assert extract_subject("What are the latest papers about LLM agents in the last 3 months") == "LLM agents"
    r, conf = parse("track new work on protein language models", TODAY, profile="general")
    assert r.intent == "topic_search" and r.free_text == "protein language models" and conf >= 0.6


def test_query_groups():
    assert groups_from_text("graph neural networks for drug discovery") == [["graph"], ["neural"], ["networks"], ["drug"], ["discovery"]]
    assert groups_from_text('(GNN OR "graph neural network") AND "drug discovery"') == [["GNN", "graph neural network"], ["drug discovery"]]
    vlm = groups_from_text("mammography based VLMs")
    assert any("vision-language" in g for g in vlm) and any("mammogram" in g for g in vlm)  # built-in synonyms


def test_general_accept_review_reject(general_cfg):
    q = "graph neural networks for drug discovery"
    top = _paper(title="Graph neural networks for drug discovery: a benchmark of molecular property prediction",
                 abstract="We study graph neural networks for drug discovery.", venue="Neural Information Processing Systems",
                 venue_type="conference")
    assert _run(top, general_cfg, q).decision == "accept"
    journal = _paper(title="Graph neural networks accelerate drug discovery", abstract="Graph neural network models for drug design.",
                     venue="Journal of Cheminformatics", venue_type="journal")
    assert _run(journal, general_cfg, q).decision == "review"   # strong match, venue not high-priority
    partial = _paper(title="Graph neural networks for traffic forecasting", abstract="Road networks.", venue="NeurIPS",
                     venue_type="conference")
    d = _run(partial, general_cfg, q)
    assert d.decision == "reject" and "drug" in d.reject_reason
    assert not top.topics and not top.modalities  # no radiology tags in the general profile


def test_general_explicit_boolean(general_cfg):
    p = _paper(title="GNN-based molecular generation for drug discovery", abstract="", venue="ICML", venue_type="conference")
    assert _run(p, general_cfg, '(GNN OR "graph neural network") AND "drug discovery"').decision in ("accept", "review")


def test_general_without_query_never_auto_adds(general_cfg):
    p = _paper(title="Some highly cited paper", venue="NeurIPS", venue_type="conference", citation_count=5000)
    assert _run(p, general_cfg, None, intent="related").decision in ("review", "reject")
    assert _run(_paper(title="x", venue="NeurIPS", venue_type="conference"), general_cfg, None, intent="add_doi").decision == "accept"


def test_general_collections():
    p = Paper(title="x")
    assert target_collections(p, "accept", "auto-discovery", label="GNNs for drug discovery", profile="general") == ["GNNs for drug discovery"]
    assert target_collections(p, "accept", "manual-prompt", profile="general") == ["Auto Discovery - Accepted"]
    assert target_collections(p, "review", "auto-discovery", label="x", profile="general") == ["Auto Discovery - Review"]
    assert set(hierarchy(profile="general")) == {"Author Watches", "Auto Discovery - Review", "Auto Discovery - Accepted"}


def test_general_alerts(general_cfg, tmp_path):
    store = A.AlertStore(tmp_path / "alerts.yaml")
    a = A.create_alert(query="graph neural networks for drug discovery since 2023", store=store, cfg=general_cfg,
                       now=datetime(2026, 9, 27))
    assert a.free_text == "graph neural networks for drug discovery" and a.since == "2023-01-01" and a.interval_days == 30
    assert a.name == "Graph neural networks for drug discovery" and a.to_request().free_text == a.free_text


def test_scan_is_radiology_only(general_cfg):
    from src.pipeline import Pipeline
    r = Pipeline(general_cfg, dry_run=True, use_zotero=False).scan()
    assert r.retrieved == 0 and any("alerts run-due" in n for n in r.notes)
