"""Alert management: CRUD, cadence / due logic, independent runs, NL commands."""
from datetime import datetime, timedelta

import pytest

from src import alerts as A
from src.models import Paper
from src.pipeline import RunResult

NOW = datetime(2026, 9, 26, 7, 30)


@pytest.fixture
def store(tmp_path):
    return A.AlertStore(tmp_path / "alerts.yaml")


class FakePipeline:
    """Stands in for pipeline.Pipeline: records calls, returns a canned RunResult."""
    calls = []
    fail_for = set()

    def __init__(self, cfg, dry_run):
        self.dry_run = dry_run

    def run(self, req, date_from, date_to, skip_seen=False, seen_path=None):
        FakePipeline.calls.append({"req": req, "from": date_from, "skip_seen": skip_seen, "seen_path": seen_path, "dry": self.dry_run})
        if req.label in FakePipeline.fail_for:
            raise RuntimeError("source exploded")
        r = RunResult(req, date_from, date_to, dry_run=self.dry_run)
        p = Paper(title="x", final_score=0.9)
        p.decision = "accept"
        r.papers, r.retrieved = [p], 3
        if not self.dry_run:
            r.written["accept"].append("K1")
        return r


@pytest.fixture(autouse=True)
def reset_fake():
    FakePipeline.calls, FakePipeline.fail_for = [], set()


def run_now(ref, cfg, store, dry=False, now=NOW):
    return A.run_alert_now(ref, cfg, dry_run=dry, store=store, now=now, pipeline_factory=FakePipeline, write_reports=False)


# ---------------------------------------------------------------- create / list / get
def test_create_defaults_to_30_days_and_is_due_immediately(cfg, store):
    a = A.create_alert(query="mammography foundation models", store=store, cfg=cfg, now=NOW)
    assert a.interval_days == 30 and a.status == "active"
    assert a.topics == ["foundation-model"] and a.modalities == ["mammo"] and a.intent == "topic_search"
    assert a.name == "Mammography foundation models" and a.id == "mammography-foundation-models"
    assert a.is_due(NOW)


def test_create_from_query_variants(cfg, store):
    a = A.create_alert(query="papers by Jane Doe on radiology foundation models since 2024", store=store, cfg=cfg, now=NOW)
    assert a.intent == "author_search" and a.author == "Jane Doe" and a.since == "2024-01-01"
    b = A.create_alert("CT uncertainty", query="CT uncertainty estimation every 2 weeks", store=store, cfg=cfg, now=NOW)
    assert b.interval_days == 14 and b.topics == ["uncertainty"] and b.modalities == ["ct"]
    c = A.create_alert(query="MICCAI chest x-ray vision-language models", store=store, cfg=cfg, now=NOW)
    assert c.intent == "venue_search" and c.venues == ["miccai"]


def test_explicit_fields_override_query_and_validation(cfg, store):
    a = A.create_alert("CXR VLMs", query="chest x-ray vlm", modalities=["ct"], interval_days="1m", store=store, cfg=cfg)
    assert a.modalities == ["ct"] and a.interval_days == 30
    with pytest.raises(A.AlertError):
        A.create_alert("Empty", store=store, cfg=cfg)
    with pytest.raises(A.AlertError):
        A.create_alert("Bad", topics=["knitting"], store=store, cfg=cfg)
    with pytest.raises(A.AlertError):
        A.create_alert("CXR VLMs", topics=["vlm"], store=store, cfg=cfg)  # duplicate name
    with pytest.raises(A.AlertError):
        A.create_alert("Too often", topics=["vlm"], interval_days=0, store=store, cfg=cfg)


def test_list_and_get(cfg, store):
    A.create_alert("Mammography foundation models", topics=["foundation-model"], modalities=["mammo"], store=store, cfg=cfg, now=NOW)
    A.create_alert("CXR vision-language models", topics=["vlm"], modalities=["cxr"], store=store, cfg=cfg, now=NOW)
    assert [a.id for a in A.list_alerts(store=store)] == ["cxr-vision-language-models", "mammography-foundation-models"]
    assert A.get_alert("cxr-vision-language-models", store).modalities == ["cxr"]
    assert A.get_alert("Mammography Foundation Models", store).id == "mammography-foundation-models"  # by name
    assert A.get_alert("mammo", store).id == "mammography-foundation-models"                         # prefix
    assert A.get_alert("the CXR VLM", store).id == "cxr-vision-language-models"                      # semantic
    with pytest.raises(A.AlertNotFound):
        A.get_alert("knee MRI", store)
    A.pause_alert("mammo", store)
    assert [a.id for a in A.list_alerts(status="paused", store=store)] == ["mammography-foundation-models"]


# ---------------------------------------------------------------- update / pause / resume / delete
def test_update_interval_and_definition(cfg, store):
    a = A.create_alert("CT", topics=["uncertainty"], modalities=["ct"], store=store, cfg=cfg, now=NOW)
    run_now(a.id, cfg, store)
    b = A.update_alert(a.id, store=store, interval_days="2w", now=NOW)
    assert b.interval_days == 14 and b.next_run_at == (NOW + timedelta(days=14)).isoformat()
    assert b.last_success_at  # cadence change keeps coverage
    c = A.update_alert(a.id, store=store, modalities=["cxr"], now=NOW)
    assert c.modalities == ["cxr"] and c.last_success_at is None and c.is_due(NOW)  # new definition backfills
    with pytest.raises(A.AlertError):
        A.update_alert(a.id, store=store, id="hacked")


def test_pause_resume_delete(cfg, store, tmp_path):
    a = A.create_alert("CT", topics=["uncertainty"], modalities=["ct"], store=store, cfg=cfg, now=NOW)
    assert not A.pause_alert(a.id, store).is_due(NOW)
    assert A.resume_alert(a.id, store).is_due(NOW)
    run_now(a.id, cfg, store)
    store.seen_path(a.id).parent.mkdir(parents=True, exist_ok=True)
    store.seen_path(a.id).write_text("[]")
    A.delete_alert(a.id, store)
    assert A.list_alerts(store=store) == [] and not store.seen_path(a.id).exists()
    with pytest.raises(A.AlertNotFound):
        A.delete_alert(a.id, store)


# ---------------------------------------------------------------- running
def test_run_now_writing_advances_schedule_and_coverage(cfg, store):
    a = A.create_alert("CXR VLM", topics=["vlm"], modalities=["cxr"], since="2025", store=store, cfg=cfg, now=NOW)
    alert, result = run_now(a.id, cfg, store)
    call = FakePipeline.calls[-1]
    assert call["from"] == "2025-01-01" and not call["skip_seen"]          # first run: from `since`
    assert call["req"].extra_tags == ["alert:cxr-vlm"] and call["req"].strict_request_match
    assert call["seen_path"] == store.seen_path("cxr-vlm")
    assert alert.last_status == "ok" and alert.last_success_at == NOW.isoformat()
    assert alert.next_run_at == (NOW + timedelta(days=30)).isoformat()
    assert alert.last_result["accepted"] == 1 and alert.last_result["written"] == {"accept": 1}
    later = NOW + timedelta(days=30)
    run_now(a.id, cfg, store, now=later)
    call = FakePipeline.calls[-1]
    assert call["skip_seen"] and call["from"] == (NOW - timedelta(days=10)).date().isoformat()  # last run - overlap


def test_first_run_window_without_since(cfg, store):
    a = A.create_alert("CT", topics=["uncertainty"], modalities=["ct"], store=store, cfg=cfg, now=NOW)
    run_now(a.id, cfg, store)
    assert FakePipeline.calls[-1]["from"] == (NOW - timedelta(days=365 * 3)).date().isoformat()


def test_dry_run_keeps_coverage_and_manual_schedule(cfg, store):
    a = A.create_alert("CT", topics=["uncertainty"], modalities=["ct"], store=store, cfg=cfg, now=NOW)
    alert, _ = run_now(a.id, cfg, store, dry=True)
    assert alert.last_status == "dry-run" and alert.last_success_at is None and alert.last_dry_run_at
    assert alert.next_run_at == NOW.isoformat()  # a manual dry run doesn't push the schedule


def test_run_due_runs_only_due_active_alerts_independently(cfg, store):
    A.create_alert("Due one", topics=["vlm"], modalities=["cxr"], store=store, cfg=cfg, now=NOW)
    A.create_alert("Broken", topics=["uncertainty"], modalities=["ct"], store=store, cfg=cfg, now=NOW)
    A.create_alert("Paused", topics=["foundation-model"], modalities=["mammo"], store=store, cfg=cfg, now=NOW, status="paused")
    A.create_alert("Future", topics=["calibration"], modalities=["cxr"], store=store, cfg=cfg, now=NOW + timedelta(days=5))
    FakePipeline.fail_for = {"Broken"}
    out = A.run_due_alerts(cfg, store=store, now=NOW, pipeline_factory=FakePipeline, write_reports=False)
    ran = {a.id: err for a, _, err in out}
    assert set(ran) == {"due-one", "broken"}
    assert ran["due-one"] is None and "source exploded" in ran["broken"]
    broken = A.get_alert("broken", store)
    assert broken.last_status == "error" and broken.next_run_at == (NOW + timedelta(days=1)).isoformat()  # retried tomorrow
    assert A.get_alert("due-one", store).next_run_at == (NOW + timedelta(days=30)).isoformat()
    # nothing is due again immediately
    assert A.run_due_alerts(cfg, store=store, now=NOW, pipeline_factory=FakePipeline, write_reports=False) == []


def test_scheduled_dry_runs_follow_cadence(cfg, store):
    A.create_alert("CT", topics=["uncertainty"], modalities=["ct"], store=store, cfg=cfg, now=NOW)
    A.run_due_alerts(cfg, dry_run=True, store=store, now=NOW, pipeline_factory=FakePipeline, write_reports=False)
    a = A.get_alert("ct", store)
    assert a.next_run_at == (NOW + timedelta(days=30)).isoformat() and a.last_success_at is None


def test_store_is_plain_yaml(cfg, store):
    A.create_alert("CT", topics=["uncertainty"], modalities=["ct"], store=store, cfg=cfg, now=NOW)
    text = store.path.read_text()
    assert "interval_days: 30" in text and text.startswith("# Saved literature alerts")


# ---------------------------------------------------------------- NL commands
@pytest.mark.parametrize("text,op,extra", [
    ("Create an alert for mammography foundation models every 2 weeks", "create", {"query": "mammography foundation models every 2 weeks"}),
    ("Set up an alert on papers by Jane Doe about radiology foundation models", "create", {"query": "papers by Jane Doe about radiology foundation models"}),
    ("List my alerts", "list", {}),
    ("Pause the CXR VLM alert", "pause", {"ref": "CXR VLM"}),
    ("Resume alert mammography-foundation-models", "resume", {"ref": "mammography-foundation-models"}),
    ("Delete the 'CT uncertainty' alert", "delete", {"ref": "CT uncertainty"}),
    ("Run the mammography alert now", "run", {"ref": "mammography"}),
    ("Change the CT uncertainty alert to every 2 weeks", "update", {"ref": "CT uncertainty", "interval_days": 14}),
])
def test_parse_alert_command(text, op, extra):
    cmd = A.parse_alert_command(text)
    assert cmd["op"] == op
    for k, v in extra.items():
        assert cmd[k] == v


def test_non_alert_prompts_pass_through():
    assert A.parse_alert_command("Find recent uncertainty papers for chest X-ray VLMs") is None


def test_parse_interval():
    assert [A.parse_interval(x) for x in (30, "30", "30d", "2w", "1m", "weekly", "every 2 weeks", "monthly")] == [30, 30, 30, 14, 30, 7, 14, 30]
