from src.models import Paper
from src.processing.normalize import (canonical_venue, finalize, merge, normalize_arxiv_id, normalize_doi, normalize_title)


def test_doi_normalization():
    assert normalize_doi("https://doi.org/10.1148/RYAI.240646") == "10.1148/ryai.240646"
    assert normalize_doi("doi:10.1007/978-3-031-72390-2_12.") == "10.1007/978-3-031-72390-2_12"
    assert normalize_doi("not a doi") is None


def test_arxiv_normalization():
    assert normalize_arxiv_id("https://arxiv.org/abs/2505.10579v3") == "2505.10579"
    assert normalize_arxiv_id("10.48550/arXiv.2401.01234") == "2401.01234"
    assert normalize_arxiv_id("arXiv:2609.29156") == "2609.29156"
    assert normalize_arxiv_id("cs/0112017") == "cs/0112017"


def test_title_normalization():
    assert normalize_title("  Médical <i>CLIP</i>: Zero-Shot!  ") == "medical clip zero shot"
    assert normalize_title("A Model (Extended Version)") == normalize_title("A model")


def test_venue_canonicalization():
    assert canonical_venue("Radiology: Artificial Intelligence") == ("radiology_ai", "journal")
    assert canonical_venue("Radiology") == ("radiology", "journal")
    assert canonical_venue("IEEE Transactions on Medical Imaging") == ("tmi", "journal")
    assert canonical_venue("Medical Image Analysis") == ("medical_image_analysis", "journal")
    assert canonical_venue("Medical Image Computing and Computer Assisted Intervention – MICCAI 2026") == ("miccai", "conference")
    assert canonical_venue("2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)")[0] == "cvpr"
    assert canonical_venue("IEEE Transactions on Pattern Analysis and Machine Intelligence")[0] == "tpami"
    assert canonical_venue("Neural Information Processing Systems") == ("neurips", "conference")
    assert canonical_venue("NeurIPS 2024 Workshop on GenAI for Health") == ("neurips", "workshop")
    assert canonical_venue("arXiv (Cornell University)") == ("arxiv", "preprint")
    assert canonical_venue("Lecture notes in computer science")[1] == "conference"
    # PMLR is a platform, not a venue; JMLR is a journal
    assert canonical_venue("Journal of Machine Learning Research")[0] == "jmlr"
    assert canonical_venue("Proceedings of Machine Learning Research")[0] != "jmlr"


def test_merge_prefers_published_version():
    pre = finalize(Paper(title="X", arxiv_id="2401.00001", venue="arXiv", venue_key="arxiv", venue_type="preprint",
                         abstract="short", source="arxiv"))
    pub = finalize(Paper(title="X", doi="10.1109/tmi.2024.1", venue="IEEE Transactions on Medical Imaging", venue_type="journal",
                         abstract="a much longer abstract", volume="43", pages="1-10", source="crossref", citation_count=5))
    m = merge(pre, pub)
    assert m.venue_key == "tmi" and m.venue_type == "journal"
    assert m.doi == "10.1109/tmi.2024.1" and m.arxiv_id == "2401.00001"
    assert m.abstract == "a much longer abstract" and m.volume == "43"
    assert set(m.sources) == {"arxiv", "crossref"}


def test_arxiv_doi_does_not_displace_publisher_doi():
    a = finalize(Paper(title="X", doi="10.1007/abc", venue_type="conference", venue="MICCAI"))
    b = finalize(Paper(title="X", doi="10.48550/arXiv.2401.00001", venue="arXiv"))
    assert merge(a, b).doi == "10.1007/abc"


def test_series_name_replaced_by_volume_title():
    a = finalize(Paper(title="X", doi="10.1007/abc", venue="Lecture notes in computer science", venue_type="conference"))
    b = finalize(Paper(title="X", doi="10.1007/abc", venue="Information Processing in Medical Imaging", venue_type="conference"))
    assert merge(a, b).venue == "Information Processing in Medical Imaging"
