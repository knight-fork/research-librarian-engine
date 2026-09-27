"""Regression tests for defects found in the multi-agent code review."""
import json
from datetime import date

import pytest

from src import config as config_mod
from src import pipeline
from src.http import HttpError
from src.models import Paper, SearchRequest
from src.processing import rank as R
from src.processing.classify import classify, hard_filter, modality_match
from src.processing.deduplicate import DedupIndex, Record, dedupe_candidates, same_work_title
from src.processing.normalize import author_matches, canonical_venue, finalize, merge
from src.processing.version_resolution import resolve_against_library
from src.prompts.parser import parse
from src.zotero.collections import CollectionMap, resolve_collection, target_collections

TODAY = date(2026, 9, 26)


# ---------------------------------------------------------------- http
def test_secrets_redacted_from_http_errors():
    e = HttpError(503, "https://api.openalex.org/works?filter=x&api_key=SECRET&mailto=me@x.org", "token=abc")
    assert "SECRET" not in str(e) and "me@x.org" not in str(e) and "abc" not in str(e)


# ---------------------------------------------------------------- normalize / merge
@pytest.mark.parametrize("venue,key", [
    ("2020 19th IEEE International Conference on Machine Learning and Applications (ICMLA)", "other_conference"),
    ("International Conference on Computer Vision Theory and Applications", "other_conference"),
    ("Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society", "other_conference"),
    ("Proceedings of the 37th International Conference on Machine Learning", "icml"),
    ("2021 IEEE/CVF International Conference on Computer Vision (ICCV)", "iccv"),
    ("Proceedings of the AAAI Conference on Artificial Intelligence", "aaai"),
])
def test_venue_lookalikes(venue, key):
    assert canonical_venue(venue, "conference")[0] == key


def test_merge_takes_year_from_published_version_and_drops_preprint_locators():
    pre = finalize(Paper(title="X", arxiv_id="2310.01234", publication_date="2023-10-02", venue="arXiv", volume="abs/2310.01234"))
    pub = finalize(Paper(title="X", year=2024, venue="ICLR 2024", venue_key="iclr", venue_type="conference", source="openreview"))
    m = merge(pre, pub)
    assert m.year == 2024 and m.publication_date is None and m.volume is None and m.venue_key == "iclr"
    pub2 = finalize(Paper(title="X", year=2024, venue="ICLR 2024", venue_key="iclr", venue_type="conference"))
    pre2 = finalize(Paper(title="X", arxiv_id="2310.01234", publication_date="2023-10-02", venue="arXiv", volume="abs/2310.01234"))
    m2 = merge(pub2, pre2)
    assert m2.year == 2024 and m2.publication_date is None and m2.volume is None


def test_medrxiv_is_preprint_and_its_doi_never_displaces_publisher_doi():
    p = finalize(Paper(title="X", venue="medRxiv : the preprint server for health sciences", venue_type="journal", doi="10.1101/2023.01.01.1"))
    assert p.venue_type == "preprint" and not p.is_peer_reviewed
    pub = finalize(Paper(title="X", venue="Radiology", venue_type="journal", doi="10.1148/radiol.231234"))
    assert merge(p, pub).doi == "10.1148/radiol.231234"


def test_author_matching_strict_vs_loose():
    assert author_matches("J. Smith", "John Smith")                 # loose byline guard
    assert not author_matches("Yuyin Zhou", "Yang Zhou")            # different given names
    assert author_matches("Yu-Yin Zhou", "Yuyin Zhou", strict=True)
    assert not author_matches("Yuyin Zhou", "Y. Zhou", strict=True)
    assert not author_matches("Yuyin Zhou", "Zhou", strict=True)


# ---------------------------------------------------------------- dedup
def test_part_i_part_ii_and_2d_3d_never_merge():
    idx = DedupIndex()
    idx.add(Record("P1", "Uncertainty quantification in deep learning for medical image analysis: a review, Part I", {"doi": "10.1016/j.x.1"}))
    assert idx.find({"doi": "10.1016/j.x.2"}, "Uncertainty quantification in deep learning for medical image analysis: a review, Part II")[0] is None
    assert idx.find({}, "Uncertainty quantification in deep learning for medical image analysis: a review, Part II")[0] is None
    idx.add(Record("D2", "A foundation model for 2D medical image segmentation", {}))
    assert idx.find({}, "A foundation model for 3D medical image segmentation")[0] is None
    assert not same_work_title("Model for 2D segmentation", "Model for 3D segmentation")


def test_conflicting_publisher_dois_block_exact_title_match():
    idx = DedupIndex()
    idx.add(Record("A", "Chest X-ray foundation model", {"doi": "10.1148/ryai.1"}))
    assert idx.find({"doi": "10.1148/ryai.2"}, "Chest X-ray foundation model")[0] is None
    assert idx.find({"doi": "10.48550/arxiv.2401.00001"}, "Chest X-ray foundation model")[0] == "A"  # preprint DOI: same work


def test_fuzzy_match_independent_of_library_contents():
    idx = DedupIndex()
    for i in range(10):
        idx.add(Record(f"N{i}", f"A totally unrelated paper number {chr(65 + i)} about knees", {}))
    idx.add(Record("T", "Foundation model for chest radiograph interpretation with calibrated uncertainty estimates", {}))
    assert idx.find({}, "A foundation model for chest radiograph interpretation with calibrated uncertainty estimates")[0] == "T"


def test_dedupe_order_independent():
    def recs():
        return [finalize(Paper(title="Preprint title for CXR model", arxiv_id="2401.01234", source="arxiv")),
                finalize(Paper(title="Published title for the CXR foundation model", arxiv_id="2401.01234", doi="10.1007/abc",
                               venue="MICCAI", venue_type="conference", source="semantic_scholar")),
                finalize(Paper(title="Published title for the CXR foundation model", venue="MIDL", venue_type="conference", source="pmlr"))]
    a = recs()
    assert len(dedupe_candidates(a)) == 1
    b = recs()
    assert len(dedupe_candidates([b[1], b[0], b[2]])) == 1


def test_medrxiv_library_preprint_gets_upgrade():
    idx = DedupIndex()
    idx.add(Record("MEDRX001", "Deep learning uncertainty estimation for chest radiograph triage", {"doi": "10.1101/2023.05.01.23289000"}, "preprint"))
    pub = finalize(Paper(title="Deep learning uncertainty estimation for chest radiograph triage", doi="10.1148/radiol.231234",
                         venue="Radiology", venue_type="journal"))
    resolve_against_library(pub, idx)
    assert pub.related_preprint_key == "MEDRX001"


# ---------------------------------------------------------------- classify / rank
@pytest.mark.parametrize("title,ptype", [
    ("Correction: A vision-language foundation model for chest X-ray report generation", ""),
    ("Author Correction: chest X-ray foundation model", ""),
    ("A chest X-ray vision-language foundation model", "retracted publication"),
    ("A chest X-ray vision-language foundation model", "published erratum"),
])
def test_errata_and_retractions_rejected(title, ptype):
    p = Paper(title=title, publication_type=ptype)
    classify(p)
    assert hard_filter(p) and "excluded" in hard_filter(p)


def test_hyphenated_terms_and_plural_ct_context():
    for t in ("A foundation model for whole-body PET-CT lesion segmentation", "Self-supervised pretraining for 4D-CT lung motion"):
        p = Paper(title=t)
        classify(p)
        assert "ct" in p.modalities, t
    p = Paper(title="Benchmarking vision-language models on CT",
              abstract="We evaluate on CT from 5,000 patients covering tumors, lesions and organs.")
    classify(p)
    assert "ct" in p.modalities
    p = Paper(title="A CXR-CLIP model with well-calibrated uncertainty", abstract="Contrastive image-text pretraining on chest X-rays.")
    classify(p)
    assert "calibration" in p.topics and "vlm" in p.topics


@pytest.mark.parametrize("title", [
    "Deep learning metal artifact reduction for intracranial aneurysm clip imaging on head CT",
    "SAM: a spatial attention module for pneumonia detection on chest X-ray",
    "Longitudinal emphysema progression on chest CT: a linear mixed model (LMM) analysis",
])
def test_ambiguous_acronyms_do_not_assign_topics(title):
    p = Paper(title=title)
    classify(p)
    assert not ({"vlm", "foundation-model"} & set(p.topics))


def test_radiology_general_request_matches_any_modality():
    p = Paper(title="Uncertainty estimation for lung nodule segmentation in chest CT")
    cls = classify(p)
    assert modality_match(cls, ["radiology-general"]) >= 0.45


def test_taxonomy_from_abstract():
    p = Paper(title="Uncertainty quantification for chest X-ray classification",
              abstract="We compare Monte Carlo dropout, deep ensembles and test-time augmentation.")
    classify(p)
    assert {"uncertainty:ensemble-methods", "uncertainty:bayesian-methods", "uncertainty:test-time-uncertainty"} <= set(p.subtopics)


def test_watch_request_with_topics_still_needs_topic_signal(cfg):
    req = SearchRequest(intent="author_search", author="Jane Q. Doe", topics=["foundation-model", "vlm"],
                        strict_request_match=False, source_tag="author-watch")
    p = finalize(Paper(title="Chest CT emphysema quantification: a multicenter external validation", authors=["Jane Doe"],
                       venue="Radiology", venue_type="journal", publication_date="2025-01-01", citation_count=120))
    classify(p)
    R.score_paper(p, cfg, req, TODAY)
    R.decide(p, cfg, req)
    assert p.decision == "reject"


def test_namesake_does_not_get_watch_threshold(cfg):
    p = finalize(Paper(title="Self-supervised pretraining for CT lesion detection", abstract="Self-supervised pretraining on CT scans.",
                       authors=["Jonathan Doe"], venue="IEEE Transactions on Medical Imaging", venue_type="journal"))
    classify(p)
    R.score_paper(p, cfg, None, TODAY)
    assert p.author_watch is None


def test_watch_matches_by_pinned_id(cfg):
    cfg.authors = [{"name": "Some Person", "openalex_id": "A123", "auto_add_threshold": 0.72}]
    p = Paper(title="x", authors=["S. P."], author_ids=["openalex:A123"])
    assert R.author_watch_match(p, cfg)["name"] == "Some Person"
    q = Paper(title="x", authors=["Some Person"], author_ids=["openalex:A999"])
    assert R.author_watch_match(q, cfg) is None  # same name, different pinned person


# ---------------------------------------------------------------- parser
def test_parser_scan_verb_only():
    r, _ = parse("Add papers by Jane Doe on CT scan foundation models since 2024", TODAY)
    assert r.intent == "author_search" and r.author == "Jane Doe"
    r, _ = parse("Find calibration papers for chest CT scan classification since 2024", TODAY)
    assert r.intent == "topic_search"
    r, _ = parse("Scan the last 10 days for CXR papers but don't add anything", TODAY)
    assert r.intent == "scan" and r.action == "report_only"


@pytest.mark.parametrize("text,name", [
    ("Add papers by J. Smith on mammography foundation models since 2024", "J. Smith"),
    ("Find papers by Pranav Rajpurkar et al. on CXR", "Pranav Rajpurkar"),
    ("Add papers by Dr. Jane Smith on CT since 2023", "Jane Smith"),
    ("Find papers by Curtis P. Langlotz on chest X-ray VLMs", "Curtis P. Langlotz"),
])
def test_parser_author_names(text, name):
    assert parse(text, TODAY)[0].author == name


def test_parser_journal_venues():
    assert parse("Find European Radiology papers on mammography foundation models", TODAY)[0].venues == ["european_radiology"]


# ---------------------------------------------------------------- collections
class _Cols:
    def __init__(self, cols):
        self._cols = cols

    def collections(self):
        return self._cols


def _colmap(spec):
    cols = [{"key": k, "data": {"key": k, "name": n, "parentCollection": p or False, **({"deleted": True} if d else {})}}
            for k, n, p, d in spec]
    return CollectionMap(_Cols(cols), root="Radiology AI").load()


def test_dry_run_plans_every_missing_level():
    class DryClient(_Cols):
        def create_collection(self, name, parent=None):
            return None
    cm = CollectionMap(DryClient([]), root="Radiology AI").load()
    cm.ensure_hierarchy()
    assert "Radiology AI" in cm.planned and "Radiology AI/Foundation Models/CXR" in cm.planned
    assert "Radiology AI/Uncertainty & Reliability/OOD / Shift" in cm.planned and len(cm.planned) == 21


def test_trashed_collections_ignored():
    cm = _colmap([("R", "Radiology AI", None, True), ("F", "Foundation Models", "R", False)])
    assert "Radiology AI" not in cm.paths and "Radiology AI/Foundation Models" not in cm.paths


def test_resolve_collection_by_segments_and_ambiguity():
    cm = _colmap([("P", "My Project", None, False), ("R", "Radiology AI", None, False), ("F", "Foundation Models", "R", False),
                  ("V", "Vision-Language Models", "R", False), ("FC", "CT", "F", False), ("VC", "CT", "V", False),
                  ("FM", "Mammography", "F", False)])
    key, err = resolve_collection(cm, "CT")
    assert key is None and "Ambiguous" in err
    assert resolve_collection(cm, "Foundation Models/CT")[0] == "FC"
    assert resolve_collection(cm, "mammography foundation models", ["foundation-model"], ["mammo"])[0] == "FM"
    assert resolve_collection(cm, "")[0] is None


def test_configured_collection_names_used():
    p = Paper(title="x")
    p.modalities, p.topics = ["cxr"], ["vlm"]
    names = {"review": "Needs Review", "accepted": "Accepted", "watch": "Watch"}
    assert target_collections(p, "review", "auto-discovery", names) == ["Needs Review"]
    assert "Accepted" in target_collections(p, "accept", "auto-discovery", names)


# ---------------------------------------------------------------- pipeline
def test_outside_window():
    p = Paper(title="x", publication_date="2025-03-01")
    assert pipeline._outside_window(p, "2024-01-01", "2024-12-31")
    assert pipeline._outside_window(Paper(title="x", year=2026), "2024-01-01", "2024-12-31")
    assert pipeline._outside_window(p, "2025-01-01", None) is None


def test_seen_keys_cover_all_identifiers():
    merged = Paper(title="t", doi="10.1109/tmi.2024.5", arxiv_id="2401.01234")
    raw_arxiv = Paper(title="t", arxiv_id="2401.01234")
    assert pipeline._seen_keys(raw_arxiv) & pipeline._seen_keys(merged)
