from datetime import date

from src.models import Paper, SearchRequest
from src.processing import rank as R
from src.processing.classify import classify
from src.processing.normalize import finalize

TODAY = date(2026, 9, 26)


def _paper(**kw):
    base = dict(authors=["A B"], publication_date="2026-03-01", doi="10.1234/x", source="openalex")
    base.update(kw)
    return finalize(Paper(**base))


def _run(p, cfg, req=None):
    classify(p)
    R.score_paper(p, cfg, req, TODAY)
    return R.decide(p, cfg, req)


def test_clear_relevant_top_venue_is_accepted(cfg):
    p = _paper(title="Uncertainty-aware vision-language foundation model for chest X-ray report generation",
               abstract="We pretrain a vision-language foundation model on MIMIC-CXR chest radiographs and calibrate its "
                        "uncertainty with conformal prediction for report generation.",
               venue="Medical Image Computing and Computer Assisted Intervention – MICCAI 2026", venue_type="conference")
    _run(p, cfg)
    assert p.decision == "accept", (p.final_score, p.reject_reason)
    assert p.final_score >= 0.82
    assert "cxr" in p.modalities and {"vlm", "uncertainty", "foundation-model"} <= set(p.topics)


def test_same_paper_as_preprint_goes_to_review(cfg):
    p = _paper(title="Uncertainty-aware vision-language foundation model for chest X-ray report generation",
               abstract="We pretrain a vision-language foundation model on MIMIC-CXR chest radiographs and calibrate its "
                        "uncertainty.", venue="arXiv", doi=None, arxiv_id="2601.00001")
    _run(p, cfg)
    assert p.decision in ("review", "reject")
    assert p.decision != "accept"


def test_non_radiology_rejected(cfg):
    p = _paper(title="A foundation model for computational pathology", abstract="Self-supervised pretraining on whole-slide images.",
               venue="Nature Medicine", venue_type="journal")
    _run(p, cfg)
    assert p.decision == "reject" and "radiology" in p.reject_reason


def test_bare_ct_token_false_positive(cfg):
    p = _paper(title="CT: a contrastive transformer for code search", abstract="We pretrain a foundation model for source code.",
               venue="ICLR", venue_type="conference")
    _run(p, cfg)
    assert p.decision == "reject"
    assert "ct" not in p.modalities


def test_editorial_rejected(cfg):
    p = _paper(title="Foundation models in chest radiography", abstract="An editorial on chest x-ray foundation models.",
               venue="Radiology", venue_type="journal", publication_type="editorial")
    _run(p, cfg)
    assert p.decision == "reject" and "excluded" in p.reject_reason


def test_request_mismatch(cfg):
    req = SearchRequest(topics=["foundation-model"], modalities=["mammo"])
    p = _paper(title="Foundation model for chest X-ray classification", abstract="Self-supervised pretraining on chest radiographs.",
               venue="Radiology: Artificial Intelligence", venue_type="journal")
    _run(p, cfg, req)
    assert p.decision == "reject" and "modality" in p.reject_reason


def test_low_priority_venue_never_auto_added(cfg):
    p = _paper(title="Vision-language foundation model for mammography screening with calibrated uncertainty",
               abstract="A mammography vision-language foundation model pretrained on mammograms and reports; we study calibration.",
               venue="Some Open Access Journal", venue_type="journal", citation_count=500, publication_date="2024-01-01")
    _run(p, cfg)
    assert p.decision == "review"


def test_citation_age_normalization():
    new = Paper(title="x", publication_date="2026-08-01", citation_count=0)
    old = Paper(title="y", publication_date="2021-01-01", citation_count=0)
    # a brand-new paper with no citations is not punished like an old uncited one
    assert R.citation_signal(new, True, TODAY) > R.citation_signal(old, True, TODAY)


def test_rate_limit(cfg):
    papers = []
    for i in range(5):
        p = Paper(title=str(i), final_score=0.9 - i * 0.01)
        p.decision = "accept"
        papers.append(p)
    R.apply_rate_limit(papers, 3)
    assert [p.decision for p in papers] == ["accept"] * 3 + ["review"] * 2


def test_author_watch_boost(cfg):
    base = dict(title="Self-supervised pretraining for CT lesion detection", abstract="Self-supervised pretraining on CT scans.",
                venue="IEEE Transactions on Medical Imaging", venue_type="journal")
    a = _paper(**base, authors=["Someone Else"])
    b = _paper(**base, authors=["Jane Doe"])
    _run(a, cfg)
    _run(b, cfg)
    assert b.author_watch == "Jane Q. Doe"
    assert b.relevance_score > a.relevance_score
