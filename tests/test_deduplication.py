from src.models import Paper
from src.processing.deduplicate import DedupIndex, Record, dedupe_candidates, title_similarity
from src.processing.normalize import finalize
from src.processing.version_resolution import resolve_against_library


def _index():
    idx = DedupIndex(0.96)
    idx.add(Record("AAAA1111", "Chest X-ray Foundation Models: A Benchmark", {"doi": "10.1/abc"}, "journalArticle"))
    idx.add(Record("BBBB2222", "Uncertainty-aware Mammography VLMs", {"arxiv_id": "2401.00001"}, "preprint"))
    idx.add(Record("CCCC3333", "Conformal Prediction for CT Triage", {"pmid": "12345678"}, "journalArticle"))
    return idx


def test_match_order_ids_then_titles():
    idx = _index()
    assert idx.find({"doi": "10.1/ABC"}, "totally different") == ("AAAA1111", "doi")
    assert idx.find({"pmid": "12345678"}, "") == ("CCCC3333", "pmid")
    assert idx.find({"arxiv_id": "2401.00001"}, "") == ("BBBB2222", "arxiv_id")
    assert idx.find({}, "chest x-ray foundation models - a benchmark")[1] == "title"


def test_fuzzy_title_threshold():
    idx = _index()
    key, how = idx.find({}, "Chest X-ray Foundation Models: A Benchmarks")
    assert key == "AAAA1111" and how.startswith("fuzzy")
    assert idx.find({}, "Chest X-ray Foundation Models for Segmentation")[0] is None
    assert title_similarity("abc def", "abc def") == 1.0


def test_candidate_dedup_merges_sources():
    a = finalize(Paper(title="MedCLIP-X: zero-shot CXR", doi="10.1234/x", source="openalex"))
    b = finalize(Paper(title="MedCLIP-X: Zero-Shot CXR", arxiv_id="2402.01234", source="arxiv"))
    c = finalize(Paper(title="Other paper", source="pubmed"))
    out = dedupe_candidates([a, b, c])
    assert len(out) == 2
    merged = next(p for p in out if p.doi)
    assert merged.arxiv_id == "2402.01234" and set(merged.sources) == {"openalex", "arxiv"}


def test_preprint_upgrade_detection():
    idx = _index()
    pub = finalize(Paper(title="Uncertainty-aware mammography VLMs", doi="10.1109/tmi.2025.9", arxiv_id="2401.00001",
                         venue="IEEE Transactions on Medical Imaging", venue_type="journal"))
    resolve_against_library(pub, idx)
    assert pub.existing_zotero_key == "BBBB2222"
    assert pub.related_preprint_key == "BBBB2222"
    # Already-published duplicates are not upgrades
    dup = finalize(Paper(title="Chest X-ray Foundation Models: A Benchmark", doi="10.1/abc", venue_type="journal", venue="Radiology"))
    resolve_against_library(dup, idx)
    assert dup.existing_zotero_key == "AAAA1111" and dup.related_preprint_key is None
