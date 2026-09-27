from src.models import Paper
from src.processing.normalize import finalize
from src.zotero.collections import split_path, target_collections
from src.zotero.items import build_tags, record_from_item, split_name, to_zotero_item


def _p(**kw):
    p = finalize(Paper(**kw))
    p.modalities, p.topics = ["mammo"], ["foundation-model", "vlm", "calibration", "uncertainty"]
    p.final_score = 0.91
    return p


def test_journal_mapping():
    p = _p(title="T", authors=["Hong-Yu Zhou", "Ludwig van Beethoven"], doi="10.1148/ryai.240646", venue="Radiology: Artificial Intelligence",
           venue_type="journal", volume="7", issue="2", pages="e240646", publication_date="2025-03-01", arxiv_id="2501.00001", pmid="40000000")
    item = to_zotero_item(p, ["COLL0001"], ["modality:mammo"])
    assert item["itemType"] == "journalArticle"
    assert item["publicationTitle"] == "Radiology: Artificial Intelligence" and item["DOI"] == "10.1148/ryai.240646"
    assert item["volume"] == "7" and item["issue"] == "2" and item["pages"] == "e240646"
    assert item["creators"][1] == {"creatorType": "author", "firstName": "Ludwig", "lastName": "van Beethoven"}
    assert "arXiv: 2501.00001" in item["extra"] and "PMID: 40000000" in item["extra"]
    assert item["collections"] == ["COLL0001"] and item["tags"] == [{"tag": "modality:mammo"}]


def test_conference_and_preprint_mapping():
    c = to_zotero_item(_p(title="T", venue="MICCAI 2026", venue_type="conference", year=2026, doi="10.1007/x"), [], [])
    assert c["itemType"] == "conferencePaper" and c["conferenceName"] == "MICCAI 2026" and c["proceedingsTitle"] == "MICCAI 2026"
    pr = to_zotero_item(_p(title="T", arxiv_id="2601.12345", venue="arXiv"), [], [])
    assert pr["itemType"] == "preprint" and pr["archiveID"] == "arXiv:2601.12345" and pr["repository"] == "arXiv"


def test_tags():
    p = _p(title="T", venue="MICCAI", venue_type="conference")
    tags = build_tags(p, "accept", "auto-discovery")
    for t in ("modality:mammo", "topic:foundation-model", "topic:vlm", "source:auto-discovery", "status:unread",
              "status:high-priority", "venue:miccai"):
        assert t in tags
    assert "status:review-required" in build_tags(p, "review", "auto-discovery")


def test_collections():
    p = _p(title="T", venue="MICCAI", venue_type="conference")
    paths = target_collections(p, "accept", "auto-discovery")
    assert "Foundation Models/Mammography" in paths and "Vision-Language Models/Mammography" in paths
    assert "Uncertainty & Reliability/Calibration" in paths and "Auto Discovery - Accepted" in paths
    assert target_collections(p, "review", "auto-discovery") == ["Auto Discovery - Review"]
    assert split_path("Uncertainty & Reliability/OOD / Shift") == ["Uncertainty & Reliability", "OOD / Shift"]


def test_split_name_single_and_comma():
    assert split_name("Doe, Jane") == {"creatorType": "author", "firstName": "Jane", "lastName": "Doe"}
    assert split_name("Plato") == {"creatorType": "author", "name": "Plato"}


def test_record_from_item_reads_identifiers():
    rec = record_from_item({"key": "K1", "itemType": "preprint", "title": "X", "DOI": "10.48550/arXiv.2401.00001",
                            "extra": "PMID: 123456\nOpenAlex: W123", "url": "https://arxiv.org/abs/2401.00001v2", "date": "2024-01-05"})
    assert rec.ids["arxiv_id"] == "2401.00001" and rec.ids["pmid"] == "123456" and rec.ids["openalex_id"] == "W123"
    assert rec.year == 2024
    assert record_from_item({"key": "N", "itemType": "note"}) is None
