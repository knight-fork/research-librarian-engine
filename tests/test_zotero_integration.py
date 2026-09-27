"""End-to-end ingestion against an in-memory fake of the Zotero Web API."""
import copy

import pytest

from src import config as config_mod
from src import pipeline
from src.models import Paper, SearchRequest
from src.processing.normalize import finalize
from src.zotero.client import ZoteroClient


class FakeZotero(ZoteroClient):
    def __init__(self):  # noqa: D401 - bypass credential checks
        self.prefix = "https://api.zotero.org/users/0"
        self.dry_run = False
        self.library_version = 1
        self.items_db = {}
        self.collections_db = {}
        self.patches = []
        self._n = 0

    def _key(self):
        self._n += 1
        return f"K{self._n:07d}"

    def collections(self):
        return [{"key": k, "data": v} for k, v in self.collections_db.items()]

    def items_top(self, since=None):
        return [{"key": k, "data": v} for k, v in self.items_db.items()
                if v["itemType"] not in ("note", "attachment") and "parentItem" not in v]

    def deleted_since(self, since):
        return {"items": []}

    def item(self, key):
        return {"key": key, "version": 5, "data": self.items_db[key]}

    def create_items(self, items):
        keys = []
        for it in items:
            k = self._key()
            self.items_db[k] = {**copy.deepcopy(it), "key": k}
            keys.append(k)
        self.library_version += 1
        return keys

    def create_collection(self, name, parent=None):
        k = self._key()
        self.collections_db[k] = {"key": k, "name": name, "parentCollection": parent or False}
        return k

    def patch_item(self, key, version, data):
        self.patches.append((key, data))
        self.items_db[key].update(data)
        return True


@pytest.fixture
def pipe(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "CACHE_DIR", tmp_path)
    for name in ("AUDIT_PATH", "PROPOSALS_PATH", "STATE_PATH"):
        monkeypatch.setattr(pipeline, name, tmp_path / name)
    monkeypatch.setattr(pipeline, "_SEEN_PATH", tmp_path / "seen.json")
    cfg.raw["sources"] = {k: False for k in cfg.raw["sources"]}  # no network
    cfg.raw["safety"]["writes_enabled"] = True
    cfg.raw["zotero"]["attach_open_access_pdf"] = False
    cfg.raw["zotero"]["root_collection"] = "Radiology AI"
    a = pipeline.Pipeline(cfg, dry_run=False, use_zotero=False)
    a.zotero, a.dry_run, a.notes = FakeZotero(), False, []
    return a


def _papers():
    good = finalize(Paper(title="Calibrated vision-language foundation model for chest X-ray report generation",
                          abstract="A vision-language foundation model pretrained on MIMIC-CXR chest radiographs with uncertainty "
                                   "calibration for radiology report generation.", authors=["Ann Lee", "Bo Chan"],
                          publication_date="2026-06-01", doi="10.1007/978-3-032-00000-0_1",
                          venue="Medical Image Computing and Computer Assisted Intervention – MICCAI 2026", venue_type="conference",
                          source="openalex"))
    borderline = finalize(Paper(title="Self-supervised pretraining for mammography density estimation",
                                abstract="We use self-supervised pretraining on mammograms.", authors=["C D"],
                                publication_date="2026-05-01", arxiv_id="2605.01234", venue="arXiv", source="arxiv"))
    off = finalize(Paper(title="Foundation model for histopathology whole-slide images", abstract="Pathology slides.",
                         publication_date="2026-05-01", doi="10.1234/path", venue="Nature", venue_type="journal", source="openalex"))
    return [good, borderline, off]


def _req():
    return SearchRequest(intent="scan", source_tag="auto-discovery", strict_request_match=False)


def test_ingest_then_no_duplicates(pipe):
    r1 = pipe.run(_req(), raw=_papers())
    z = pipe.zotero
    parents = [v for v in z.items_db.values() if v["itemType"] not in ("note", "attachment")]
    notes = [v for v in z.items_db.values() if v["itemType"] == "note"]
    assert len(r1.written["accept"]) == 1 and len(r1.written["review"]) == 1
    assert len(parents) == 2 and len(notes) == 2
    accepted = next(v for v in parents if v["itemType"] == "conferencePaper")
    tags = {t["tag"] for t in accepted["tags"]}
    assert {"modality:cxr", "topic:vlm", "source:auto-discovery", "venue:miccai", "status:unread"} <= tags
    assert accepted["collections"]  # filed into topic collections
    assert "Auto Discovery - Accepted" in {z.collections_db[k]["name"] for k in accepted["collections"]}

    # second run with the same (re-fetched) records creates nothing new
    r2 = pipe.run(_req(), raw=_papers())
    parents2 = [v for v in z.items_db.values() if v["itemType"] not in ("note", "attachment")]
    assert len(parents2) == 2
    assert len(r2.existing) == 2 and not r2.written["accept"] and not r2.written["review"]


def test_preprint_upgrade_is_annotation_only(pipe):
    z = pipe.zotero
    k = z.create_items([{"itemType": "preprint", "title": "Conformal prediction for CT triage with foundation models",
                         "archiveID": "arXiv:2501.01234", "url": "https://arxiv.org/abs/2501.01234", "DOI": "", "extra": "",
                         "tags": [], "collections": []}])[0]
    pub = finalize(Paper(title="Conformal prediction for CT triage with foundation models",
                         abstract="Conformal prediction sets over a CT foundation model for chest CT scans triage.",
                         doi="10.1109/tmi.2026.1234567", arxiv_id="2501.01234", venue="IEEE Transactions on Medical Imaging",
                         venue_type="journal", publication_date="2026-07-01", source="crossref"))
    r = pipe.run(_req(), raw=[pub])
    assert r.upgrades and r.upgrades[0].related_preprint_key == k
    parents = [v for v in z.items_db.values() if v["itemType"] not in ("note", "attachment")]
    assert len(parents) == 1  # no second independent item
    assert "Published DOI: 10.1109/tmi.2026.1234567" in z.items_db[k]["extra"]
    assert any(t["tag"] == "status:published-version-available" for t in z.items_db[k]["tags"])
    assert z.items_db[k]["title"] == "Conformal prediction for CT triage with foundation models"  # nothing replaced


def test_dry_run_writes_nothing(pipe):
    pipe.dry_run = True
    pipe.zotero.dry_run = True
    r = pipe.run(_req(), raw=_papers())
    assert r.by_decision("accept") and not pipe.zotero.items_db


def test_target_collection_files_new_and_existing(pipe):
    z = pipe.zotero
    first = pipe.run(_req(), raw=_papers())            # accepted paper now exists in the library
    assert first.written["accept"]
    req = SearchRequest(intent="scan", source_tag="auto-discovery", strict_request_match=False, target_collection="Mammo/Selected")
    r = pipe.run(req, raw=_papers())
    target = next(k for k, c in z.collections_db.items() if c["name"] == "Selected")
    parent = z.collections_db[target]["parentCollection"]
    assert z.collections_db[parent]["name"] == "Mammo" and not z.collections_db[parent]["parentCollection"]  # top level
    existing_key = first.written["accept"][0]
    assert target in z.items_db[existing_key]["collections"] and r.written.get("filed") == [existing_key]


def test_cli_parser_builds_and_profile_override():
    from src.cli import build_parser
    ap = build_parser()
    args = ap.parse_args(["--profile", "radiology", "search", "--topic", "vlm", "--collection", "Mammo", "--dry-run"])
    assert args.profile == "radiology" and args.collection == "Mammo"
    assert ap.parse_args(["gaps", "--collection", "Mammo"]).collection == "Mammo"


def test_target_collection_files_kept_preprint_but_not_pending_review(pipe):
    z = pipe.zotero
    kept = z.create_items([{"itemType": "preprint", "title": "Conformal prediction for CT triage with foundation models",
                            "archiveID": "arXiv:2501.01234", "url": "https://arxiv.org/abs/2501.01234", "DOI": "", "extra": "",
                            "tags": [], "collections": []}])[0]
    pending = z.create_items([{"itemType": "journalArticle", "title": "Calibrated vision-language foundation model for chest X-ray report generation",
                               "DOI": "10.1007/978-3-032-00000-0_1", "tags": [{"tag": "status:review-required"}], "collections": []}])[0]
    pub = finalize(Paper(title="Conformal prediction for CT triage with foundation models",
                         abstract="Conformal prediction sets over a CT foundation model for chest CT scans triage.",
                         doi="10.1109/tmi.2026.1234567", arxiv_id="2501.01234", venue="IEEE Transactions on Medical Imaging",
                         venue_type="journal", publication_date="2026-07-01", source="crossref"))
    req = SearchRequest(intent="scan", source_tag="auto-discovery", strict_request_match=False, target_collection="CT")
    r = pipe.run(req, raw=[pub, _papers()[0]])
    target = next(k for k, c in z.collections_db.items() if c["name"] == "CT")
    assert target in z.items_db[kept]["collections"]            # published version found -> your preprint is filed
    assert target not in z.items_db[pending]["collections"]     # still pending review -> not promoted
    assert kept in r.written.get("filed", [])
