from datetime import date

from src.prompts.parser import parse

TODAY = date(2026, 9, 26)


def test_author_search():
    r, conf = parse("Add papers by Jane Doe related to mammography foundation models since 2024.", TODAY)
    assert r.intent == "author_search" and r.author == "Jane Doe"
    assert r.topics == ["foundation-model"] and r.modalities == ["mammo"] and r.date_from == "2024-01-01"
    assert conf >= 0.8


def test_topic_search_recent():
    r, _ = parse("Find recent uncertainty papers for chest X-ray VLMs and add the strongest ones.", TODAY)
    assert r.intent == "topic_search" and set(r.topics) == {"uncertainty", "vlm"} and r.modalities == ["cxr"]
    assert r.date_from == "2024-09-26"


def test_venue_search():
    r, _ = parse("Add relevant MICCAI 2026 papers on CT foundation models.", TODAY)
    assert r.intent == "venue_search" and r.venues == ["miccai"] and r.modalities == ["ct"]
    assert (r.date_from, r.date_to) == ("2026-01-01", "2026-12-31")


def test_expansion_intents():
    assert parse('Find papers similar to "MedCLIP: Contrastive Learning" that are not already in Zotero.', TODAY)[0].seed == "MedCLIP: Contrastive Learning"
    r, _ = parse("Find important references cited by 10.1038/s41551-022-00936-9 that I do not already have.", TODAY)
    assert r.intent == "citations" and r.seed == "10.1038/s41551-022-00936-9"
    r, _ = parse("Find papers related to this Zotero item ABCD1234 that I am missing.", TODAY)
    assert r.intent == "related" and r.seed == "ABCD1234"


def test_scan_and_missing():
    r, _ = parse("Scan the last 10 days for CXR, CT, and mammography papers on foundation models, VLMs, and uncertainty.", TODAY)
    assert r.intent == "scan" and r.date_from == "2026-09-16"
    r, _ = parse("Analyze my Zotero collection on Mammography and identify important missing papers.", TODAY)
    assert r.intent == "missing_lit" and r.collection == "Mammography"


def test_venue_not_mistaken_for_author():
    r, _ = parse("Find papers from MICCAI on CXR report generation", TODAY)
    assert r.intent == "venue_search" and r.author is None
