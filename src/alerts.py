"""Saved literature alerts: independent, individually scheduled searches.

An alert is a saved search ("graph neural networks for drug discovery", "papers by <author> on protein language
models", or with the radiology profile "mammography foundation models") with its own cadence (default every 30 days), its own coverage
watermark and its own seen-set. Alerts live in alerts.yaml (safe to edit by hand while nothing is running).

Operations: create_alert, list_alerts, get_alert, update_alert, pause_alert, resume_alert, delete_alert,
run_alert_now, plus run_due_alerts for the scheduler. Deleting an alert never touches Zotero items."""
from __future__ import annotations

import fcntl
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import yaml

from src import config as config_mod
from src.config import Config
from src.models import SearchRequest
from src.prompts.parser import MODALITY_SYNONYMS, TOPIC_SYNONYMS, VENUE_SYNONYMS, _collect
from src.prompts.parser import parse as parse_prompt

DEFAULT_INTERVAL_DAYS = 30
ALERT_INTENTS = ("topic_search", "author_search", "venue_search")
STATUSES = ("active", "paused")
ACTIONS = ("add_by_threshold", "report_only")
VALID_TOPICS = {"foundation-model", "vlm", "uncertainty", "calibration", "conformal-prediction", "selective-prediction",
                "ood", "distribution-shift", "robustness", "vlm-reliability", "report-generation", "image-text-pretraining"}
VALID_MODALITIES = {"cxr", "ct", "mammo", "radiology-general"}
# Changing any of these changes what the alert searches for, so its coverage restarts from scratch.
SEARCH_FIELDS = ("query", "intent", "topics", "modalities", "venues", "author", "author_openalex_id", "author_s2_id",
                 "free_text", "top_venues_only", "since")
EDITABLE_FIELDS = SEARCH_FIELDS + ("name", "interval_days", "action", "limit", "status")

TOPIC_LABEL = {"foundation-model": "foundation models", "vlm": "vision-language models", "uncertainty": "uncertainty",
               "calibration": "calibration", "conformal-prediction": "conformal prediction", "ood": "OOD detection",
               "selective-prediction": "selective prediction", "distribution-shift": "distribution shift",
               "robustness": "robustness", "vlm-reliability": "VLM reliability", "report-generation": "report generation",
               "image-text-pretraining": "image-text pretraining"}
MODALITY_LABEL = {"cxr": "CXR", "ct": "CT", "mammo": "Mammography", "radiology-general": "Radiology"}


class AlertError(ValueError):
    """Invalid alert definition or operation."""


class AlertNotFound(AlertError):
    pass


# ============================================================================ model

def _now() -> datetime:
    return datetime.now().replace(microsecond=0)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


@dataclass
class Alert:
    id: str
    name: str
    query: Optional[str] = None                 # original natural-language description, if given
    intent: str = "topic_search"                # topic_search | author_search | venue_search
    topics: List[str] = field(default_factory=list)
    modalities: List[str] = field(default_factory=list)
    venues: List[str] = field(default_factory=list)
    author: Optional[str] = None
    author_openalex_id: Optional[str] = None
    author_s2_id: Optional[str] = None
    free_text: Optional[str] = None
    top_venues_only: bool = False
    since: Optional[str] = None                 # earliest publication date ever searched (YYYY-MM-DD)
    interval_days: int = DEFAULT_INTERVAL_DAYS
    action: str = "add_by_threshold"            # add_by_threshold | report_only
    limit: Optional[int] = None
    status: str = "active"                      # active | paused
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    next_run_at: Optional[str] = None
    last_run_at: Optional[str] = None
    last_success_at: Optional[str] = None       # last run that could write to Zotero (coverage watermark)
    last_dry_run_at: Optional[str] = None
    last_status: Optional[str] = None           # ok | dry-run | report-only | error
    last_error: Optional[str] = None
    last_result: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Alert":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_request(self) -> SearchRequest:
        return SearchRequest(
            intent=self.intent, topics=list(self.topics), modalities=list(self.modalities), venues=list(self.venues),
            author=self.author, author_openalex_id=self.author_openalex_id, author_s2_id=self.author_s2_id,
            free_text=self.free_text, top_venues_only=self.top_venues_only, limit=self.limit, action=self.action,
            source_tag="author-watch" if self.intent == "author_search" else "auto-discovery",
            strict_request_match=True, extra_tags=[f"alert:{self.id}"], label=self.name,
        )

    def is_due(self, now: Optional[datetime] = None) -> bool:
        if self.status != "active":
            return False
        nxt = _parse_dt(self.next_run_at)
        return nxt is None or nxt <= (now or _now())

    def describe(self) -> str:
        parts = []
        if self.author:
            parts.append(f"author: {self.author}")
        if self.topics:
            parts.append("topics: " + ", ".join(self.topics))
        if self.modalities:
            parts.append("modalities: " + ", ".join(self.modalities))
        if self.venues:
            parts.append("venues: " + ", ".join(self.venues))
        if self.top_venues_only:
            parts.append("top venues only")
        if self.free_text:
            parts.append(f"keywords: {self.free_text}")
        if self.since:
            parts.append(f"since {self.since}")
        if self.action == "report_only":
            parts.append("report only")
        return " | ".join(parts)


# ============================================================================ storage

class AlertStore:
    """alerts.yaml with an exclusive lock around every read-modify-write and atomic replacement."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else config_mod.ROOT / "alerts.yaml"
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.seen_dir = config_mod.CACHE_DIR / "alerts" if path is None else self.path.parent / "alert-seen"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _read(self) -> List[Alert]:
        if not self.path.exists():
            return []
        data = yaml.safe_load(self.path.read_text()) or {}
        return [Alert.from_dict(d) for d in data.get("alerts") or []]

    def _write(self, alerts: List[Alert]) -> None:
        header = ("# Saved literature alerts - managed by `python -m src.cli alerts ...`.\n"
                  "# Hand edits are fine while no alert is running; fields below 'status' are run state.\n")
        body = yaml.safe_dump({"alerts": [a.to_dict() for a in alerts]}, sort_keys=False, allow_unicode=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".alerts-", suffix=".yaml")
        with os.fdopen(fd, "w") as fh:
            fh.write(header + body)
        os.replace(tmp, self.path)

    def all(self) -> List[Alert]:
        with self._locked():
            return self._read()

    def mutate(self, fn: Callable[[List[Alert]], Any]) -> Any:
        with self._locked():
            alerts = self._read()
            out = fn(alerts)
            self._write(alerts)
            return out

    def seen_path(self, alert_id: str) -> Path:
        return self.seen_dir / f"{alert_id}.seen.json"


def _store(store: Optional[AlertStore]) -> AlertStore:
    return store or AlertStore()


# ============================================================================ parsing / validation

def parse_interval(value: Any) -> int:
    """30 | '30' | '30d' | '2w' | '1m' | 'every 2 weeks' | 'monthly' | 'weekly' | 'biweekly' | 'daily' -> days."""
    if value is None or value == "":
        raise AlertError("interval is empty")
    if isinstance(value, int):
        days = value
    else:
        s = str(value).strip().lower()
        named = {"daily": 1, "weekly": 7, "biweekly": 14, "fortnightly": 14, "monthly": 30, "quarterly": 91}
        m = re.fullmatch(r"(?:every\s+)?(\d+|a|an|one|two|three|four|six)?\s*(d|day|days|w|wk|week|weeks|m|mo|month|months)", s)
        if s in named:
            days = named[s]
        elif s.isdigit():
            days = int(s)
        elif m:
            words = {None: 1, "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "six": 6}
            n = int(m.group(1)) if (m.group(1) or "").isdigit() else words[m.group(1)]
            unit = m.group(2)[0]
            days = n * {"d": 1, "w": 7, "m": 30}[unit]
        else:
            raise AlertError(f"could not understand interval {value!r} (examples: 30, 30d, 2w, 1m, monthly)")
    if not 1 <= days <= 366:
        raise AlertError("interval must be between 1 and 366 days")
    return days


_EVERY_RE = re.compile(r"\b(?:run\s+|check\s+)?(?:every|each)\s+((?:\d+|a|an|one|two|three|four|six)?\s*(?:days?|weeks?|months?))\b"
                       r"|\b(daily|weekly|biweekly|fortnightly|monthly|quarterly)\b", re.I)


def _extract_interval(text: str) -> Tuple[Optional[int], str]:
    m = _EVERY_RE.search(text or "")
    if not m:
        return None, text
    days = parse_interval(m.group(1) or m.group(2))
    return days, (text[:m.start()] + text[m.end():]).strip(" ,.;")


def _norm_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = str(value).strip()
    if re.fullmatch(r"(19|20)\d{2}", s):
        return f"{s}-01-01"
    if re.fullmatch(r"(19|20)\d{2}-\d{2}", s):
        return f"{s}-01"
    try:
        return datetime.fromisoformat(s[:10]).date().isoformat()
    except ValueError:
        raise AlertError(f"invalid date {value!r} (use YYYY, YYYY-MM or YYYY-MM-DD)") from None


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in re.split(r"[,;]", value) if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def normalize_topics(values: Any) -> List[str]:
    out: List[str] = []
    for v in _as_list(values):
        found = [v] if v in VALID_TOPICS else _collect(v.lower(), TOPIC_SYNONYMS)
        if not found:
            raise AlertError(f"unknown topic {v!r}; known: {', '.join(sorted(VALID_TOPICS))}")
        out += [t for t in found if t not in out]
    return out


def normalize_modalities(values: Any) -> List[str]:
    aliases = {"cxr": "cxr", "chest": "cxr", "ct": "ct", "mammo": "mammo", "mammography": "mammo", "breast": "mammo",
               "general": "radiology-general", "radiology": "radiology-general", "radiology-general": "radiology-general"}
    out: List[str] = []
    for v in _as_list(values):
        m = aliases.get(v.lower()) or next(iter(_collect(v.lower(), MODALITY_SYNONYMS)), None)
        if not m:
            raise AlertError(f"unknown modality {v!r}; known: cxr, ct, mammo, general")
        if m not in out:
            out.append(m)
    return out


def normalize_venues(values: Any) -> List[str]:
    from src.processing.normalize import canonical_venue
    generic = ("unknown", "other_journal", "other_conference", "workshop", "preprint")
    out: List[str] = []
    for v in _as_list(values):
        found = _collect(v.lower(), VENUE_SYNONYMS) or [k for k in [canonical_venue(v)[0]] if k not in generic]
        if not found:
            raise AlertError(f"unknown venue {v!r}")
        out += [x for x in found if x not in out]
    return out


def _validate(a: Alert) -> Alert:
    a.name = (a.name or "").strip()
    if not a.name:
        raise AlertError("alert name is required")
    if a.intent not in ALERT_INTENTS:
        raise AlertError(f"alerts support {', '.join(ALERT_INTENTS)}; got {a.intent!r}")
    bad_t = [t for t in a.topics if t not in VALID_TOPICS]
    bad_m = [m for m in a.modalities if m not in VALID_MODALITIES]
    if bad_t or bad_m:
        raise AlertError(f"unknown topics/modalities: {bad_t + bad_m}")
    if a.intent == "author_search" and not (a.author or a.author_openalex_id or a.author_s2_id):
        raise AlertError("an author alert needs an author name or id")
    if a.intent == "venue_search" and not a.venues:
        raise AlertError("a venue alert needs at least one venue")
    if a.intent == "topic_search" and not (a.topics or a.modalities or a.free_text):
        raise AlertError("an alert needs at least one topic, modality, author, venue or keyword")
    if a.status not in STATUSES:
        raise AlertError(f"status must be one of {STATUSES}")
    if a.action not in ACTIONS:
        raise AlertError(f"action must be one of {ACTIONS}")
    a.interval_days = parse_interval(a.interval_days)
    a.since = _norm_date(a.since)
    if a.limit is not None:
        a.limit = int(a.limit)
        if a.limit < 1:
            raise AlertError("limit must be positive")
    return a


def _infer_intent(author: Optional[str], ids: bool, venues: List[str]) -> str:
    if author or ids:
        return "author_search"
    if venues:
        return "venue_search"
    return "topic_search"


def _default_name(a: Alert) -> str:
    mods = " / ".join(MODALITY_LABEL.get(m, m) for m in a.modalities)
    tops = ", ".join(TOPIC_LABEL.get(t, t) for t in a.topics)
    base = " ".join(x for x in (mods, tops) if x) or (a.free_text or "")
    if a.author:
        base = f"{a.author}" + (f" - {base}" if base else "")
    if a.venues:
        base += (" @ " if base else "") + ", ".join(v.upper() for v in a.venues)
    return base[:1].upper() + base[1:] if base else "Literature alert"


def _clean_description(query: str) -> str:
    """Alert name from its description: drop cadence, date and filler clauses ('since 2025', 'every 2 weeks')."""
    _, text = _extract_interval(query or "")
    text = re.sub(r"\b(?:since|from|after|in)\s+(?:19|20)\d{2}\b|\bbetween\s+(?:19|20)\d{2}\s+and\s+(?:19|20)\d{2}\b", " ", text, flags=re.I)
    text = re.sub(r"^\s*(?:new\s+)?(?:papers?|work|publications?|articles?)\s+(?:on|about)\s+", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:-")
    return text[:1].upper() + text[1:] if text else ""


def _slug(name: str, taken: set) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "alert"
    if len(base) > 40:  # cut at a word boundary
        base = base[:41].rsplit("-", 1)[0] if "-" in base[:41] else base[:40]
    slug, n = base, 2
    while slug in taken:
        slug, n = f"{base}-{n}", n + 1
    return slug


def _fields_from_query(query: str, profile: str = "general") -> Dict[str, Any]:
    """Compile a natural-language alert description into search fields (deterministic parser)."""
    interval, rest = _extract_interval(query)
    req, _conf = parse_prompt(rest, profile=profile)
    out: Dict[str, Any] = {"topics": req.topics, "modalities": req.modalities, "venues": req.venues,
                           "author": req.author, "top_venues_only": req.top_venues_only, "free_text": req.free_text}
    if req.date_from and re.search(r"\b(since|from|after|between|in)\s+(19|20)\d{2}", rest, re.I):
        out["since"] = req.date_from  # explicit years only; "recent" is what the cadence is for
    if req.action == "report_only":
        out["action"] = "report_only"
    if interval:
        out["interval_days"] = interval
    return out


# ============================================================================ operations

def create_alert(name: Optional[str] = None, *, query: Optional[str] = None, intent: Optional[str] = None,
                 topics: Any = None, modalities: Any = None, venues: Any = None, author: Optional[str] = None,
                 author_openalex_id: Optional[str] = None, author_s2_id: Optional[str] = None,
                 free_text: Optional[str] = None, top_venues_only: Optional[bool] = None, since: Optional[str] = None,
                 interval_days: Any = None, action: Optional[str] = None, limit: Optional[int] = None,
                 status: str = "active", store: Optional[AlertStore] = None, cfg: Optional[Config] = None,
                 now: Optional[datetime] = None) -> Alert:
    """Save a new alert. Explicit arguments override what is parsed from `query`. First run is due immediately."""
    now = now or _now()
    profile = cfg.profile if cfg else "general"
    parsed = _fields_from_query(query, profile) if query else {}
    explicit = {"free_text": free_text,
                "topics": normalize_topics(topics) if topics is not None else None,
                "modalities": normalize_modalities(modalities) if modalities is not None else None,
                "venues": normalize_venues(venues) if venues is not None else None,
                "author": author, "top_venues_only": top_venues_only, "since": since,
                "interval_days": interval_days, "action": action}
    merged = {k: (explicit[k] if explicit.get(k) is not None else parsed.get(k)) for k in set(parsed) | set(explicit)}
    default_interval = int(cfg.get("alerts.default_interval_days", DEFAULT_INTERVAL_DAYS)) if cfg else DEFAULT_INTERVAL_DAYS
    a = Alert(
        id="", name=name or "", query=query, topics=merged.get("topics") or [], modalities=merged.get("modalities") or [],
        venues=merged.get("venues") or [], author=merged.get("author"), author_openalex_id=author_openalex_id,
        author_s2_id=author_s2_id, free_text=merged.get("free_text"), top_venues_only=bool(merged.get("top_venues_only")),
        since=merged.get("since"),
        interval_days=merged["interval_days"] if merged.get("interval_days") is not None else default_interval,
        action=merged.get("action") or "add_by_threshold", limit=limit, status=status,
        created_at=_iso(now), updated_at=_iso(now), next_run_at=_iso(now),
    )
    a.intent = intent or _infer_intent(a.author, bool(author_openalex_id or author_s2_id), a.venues)
    if not a.name:
        a.name = (_clean_description(query) if query else "") or _default_name(a)
    _validate(a)

    def add(alerts: List[Alert]) -> Alert:
        if any(x.name.lower() == a.name.lower() for x in alerts):
            raise AlertError(f"an alert named {a.name!r} already exists")
        a.id = _slug(a.name, {x.id for x in alerts})
        alerts.append(a)
        return a
    return _store(store).mutate(add)


def list_alerts(status: Optional[str] = None, store: Optional[AlertStore] = None) -> List[Alert]:
    if status and status not in STATUSES:
        raise AlertError(f"status must be one of {STATUSES}")
    alerts = [a for a in _store(store).all() if not status or a.status == status]
    return sorted(alerts, key=lambda a: (a.status != "active", a.next_run_at or "", a.name.lower()))


def _find(alerts: List[Alert], ref: str) -> Alert:
    r = (ref or "").strip()
    if not r:
        raise AlertNotFound("no alert given")
    rl = r.lower()
    for test in (lambda a: a.id == r, lambda a: a.name.lower() == rl):
        hits = [a for a in alerts if test(a)]
        if hits:
            return hits[0]
    for test in (lambda a: a.id.startswith(rl), lambda a: rl in a.name.lower() or rl in a.id):
        hits = [a for a in alerts if test(a)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise AlertError(f"{ref!r} matches several alerts: " + ", ".join(a.id for a in hits))
    # Semantic fallback: "the CXR VLM alert" -> same topics / modalities / author.
    words = re.sub(r"\b(the|my|alert|alerts|one|for|on|about|papers?)\b", " ", rl)
    req, _ = parse_prompt(words)
    t, m = set(req.topics), set(req.modalities)
    if t or m or req.author:
        hits = [a for a in alerts if (not t or t <= set(a.topics)) and (not m or m <= set(a.modalities))
                and (not req.author or (a.author or "").lower() == req.author.lower())]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise AlertError(f"{ref!r} matches several alerts: " + ", ".join(a.id for a in hits))
    raise AlertNotFound(f"no alert matches {ref!r}" + (f"; alerts: {', '.join(a.id for a in alerts)}" if alerts else ""))


def get_alert(ref: str, store: Optional[AlertStore] = None) -> Alert:
    """Look an alert up by id, name, unique id prefix / name fragment, or its topics ('CXR VLM')."""
    return _find(_store(store).all(), ref)


def update_alert(ref: str, store: Optional[AlertStore] = None, now: Optional[datetime] = None, cfg: Optional[Config] = None,
                 **changes: Any) -> Alert:
    """Edit alert fields. Changing what it searches for restarts its coverage (the next run backfills)."""
    now = now or _now()
    unknown = set(changes) - set(EDITABLE_FIELDS)
    if unknown:
        raise AlertError(f"cannot edit {sorted(unknown)}; editable: {', '.join(EDITABLE_FIELDS)}")
    st = _store(store)
    changes = {k: v for k, v in changes.items()}
    if "query" in changes and changes["query"]:
        parsed = _fields_from_query(changes["query"], cfg.profile if cfg else "general")
        for k, v in parsed.items():
            changes.setdefault(k, v)
    if "topics" in changes:
        changes["topics"] = normalize_topics(changes["topics"])
    if "modalities" in changes:
        changes["modalities"] = normalize_modalities(changes["modalities"])
    if "venues" in changes:
        changes["venues"] = normalize_venues(changes["venues"])
    if "interval_days" in changes:
        changes["interval_days"] = parse_interval(changes["interval_days"])

    def apply(alerts: List[Alert]) -> Alert:
        a = _find(alerts, ref)
        before = a.to_dict()
        for k, v in changes.items():
            setattr(a, k, v)
        if "intent" not in changes and any(k in changes for k in ("author", "author_openalex_id", "author_s2_id", "venues")):
            a.intent = _infer_intent(a.author, bool(a.author_openalex_id or a.author_s2_id), a.venues)
        if "name" in changes and any(x is not a and x.name.lower() == a.name.lower() for x in alerts):
            raise AlertError(f"an alert named {a.name!r} already exists")
        _validate(a)
        if a.interval_days != before["interval_days"]:
            last = _parse_dt(a.last_run_at)
            a.next_run_at = _iso(last + timedelta(days=a.interval_days)) if last else (a.next_run_at or _iso(now))
        if any(before[k] != getattr(a, k) for k in SEARCH_FIELDS):
            a.last_success_at = None  # new definition -> backfill on the next run
            a.next_run_at = _iso(min(_parse_dt(a.next_run_at) or now, now))
            st.seen_path(a.id).unlink(missing_ok=True)
        a.updated_at = _iso(now)
        return a
    return st.mutate(apply)


def pause_alert(ref: str, store: Optional[AlertStore] = None) -> Alert:
    return update_alert(ref, store=store, status="paused")


def resume_alert(ref: str, store: Optional[AlertStore] = None) -> Alert:
    """Reactivate; an alert whose next run passed while paused runs on the next scheduler pass."""
    return update_alert(ref, store=store, status="active")


def delete_alert(ref: str, store: Optional[AlertStore] = None) -> Alert:
    """Remove the alert and its seen-set. Papers it already added to Zotero are left untouched."""
    st = _store(store)

    def remove(alerts: List[Alert]) -> Alert:
        a = _find(alerts, ref)
        alerts.remove(a)
        return a
    a = st.mutate(remove)
    try:
        st.seen_path(a.id).unlink()
    except FileNotFoundError:
        pass
    return a


# ============================================================================ running

def window_start(a: Alert, cfg: Config, now: datetime) -> str:
    """First run: `since` or first_run_years back. Later: last writing run minus overlap (>= min lookback)."""
    since = _parse_dt(a.since)
    last = _parse_dt(a.last_success_at)
    if last:
        overlap = int(cfg.get("scan.overlap_days", 10))
        min_lb = int(cfg.get("scan.min_lookback_days", 20))
        start = min(last - timedelta(days=overlap), now - timedelta(days=min_lb))
    else:
        start = since or now - timedelta(days=365 * int(cfg.get("alerts.first_run_years", cfg.get("scan.first_run_years", 3))))
    if since and start < since:
        start = since
    return start.date().isoformat()


def _default_pipeline(cfg: Config, dry_run: bool):
    from src.pipeline import Pipeline
    return Pipeline(cfg, dry_run=dry_run)


def _execute(ref: str, cfg: Config, dry_run: bool, store: AlertStore, now: datetime, scheduled: bool,
             pipeline_factory: Optional[Callable[[Config, bool], Any]] = None, write_reports: bool = True):
    from src.reports.generate_report import write_report
    alert = get_alert(ref, store)
    pipe = (pipeline_factory or _default_pipeline)(cfg, dry_run)
    req = alert.to_request()
    start = window_start(alert, cfg, now)
    try:
        result = pipe.run(req, start, None, skip_seen=bool(alert.last_success_at), seen_path=store.seen_path(alert.id))
        result.notes.insert(0, f"Alert '{alert.name}' ({alert.id}), every {alert.interval_days} days: {alert.describe()}")
        report = str(write_report(result, suffix=f"alert-{alert.id}")) if write_reports else None
    except Exception as exc:  # noqa: BLE001 - record the failure on the alert, then surface it
        def failed(alerts: List[Alert]) -> None:
            try:
                a = _find(alerts, alert.id)
            except AlertNotFound:
                return
            a.last_run_at, a.last_status, a.last_error = _iso(now), "error", f"{type(exc).__name__}: {exc}"[:500]
            if scheduled:  # retry on the next daily pass rather than waiting a full interval
                a.next_run_at = _iso(now + timedelta(days=1))
        store.mutate(failed)
        raise
    writing = not result.dry_run and alert.action != "report_only"
    succeeded = result.retrieved > 0 or not result.errors

    def record(alerts: List[Alert]) -> Optional[Alert]:
        try:
            a = _find(alerts, alert.id)
        except AlertNotFound:
            return None  # deleted while it was running
        a.last_run_at = _iso(now)
        a.last_error = None if succeeded else "; ".join(dict.fromkeys(result.errors))[:500]
        a.last_status = ("ok" if writing else "report-only" if alert.action == "report_only" else "dry-run") if succeeded else "error"
        a.last_result = {
            "window_from": start, "retrieved": result.retrieved, "accepted": len(result.by_decision("accept")),
            "review": len(result.by_decision("review")), "rejected": len(result.by_decision("reject")),
            "already_in_zotero": len(result.existing), "written": {k: len(v) for k, v in result.written.items() if v},
            "dry_run": result.dry_run, "report": report,
        }
        if writing and succeeded:
            a.last_success_at = _iso(now)
        elif not writing:
            a.last_dry_run_at = _iso(now)
        # Scheduled runs always move to the next slot; a manual run resets the timer only when it wrote.
        if scheduled or (writing and succeeded):
            a.next_run_at = _iso(now + timedelta(days=a.interval_days))
        a.updated_at = _iso(now)
        return a
    return store.mutate(record) or alert, result


def run_alert_now(ref: str, cfg: Config, dry_run: bool = False, store: Optional[AlertStore] = None,
                  now: Optional[datetime] = None, pipeline_factory=None, write_reports: bool = True):
    """Run one alert immediately (even if paused or not due). Returns (alert, RunResult)."""
    return _execute(ref, cfg, dry_run, _store(store), now or _now(), scheduled=False,
                    pipeline_factory=pipeline_factory, write_reports=write_reports)


def run_due_alerts(cfg: Config, dry_run: bool = False, store: Optional[AlertStore] = None, now: Optional[datetime] = None,
                   pipeline_factory=None, write_reports: bool = True) -> List[Tuple[Alert, Any, Optional[str]]]:
    """Run every active alert whose next run is due, each independently. Returns [(alert, result|None, error|None)]."""
    st, now = _store(store), now or _now()
    out: List[Tuple[Alert, Any, Optional[str]]] = []
    for a in [x for x in list_alerts(store=st) if x.is_due(now)]:
        try:
            alert, result = _execute(a.id, cfg, dry_run, st, now, scheduled=True, pipeline_factory=pipeline_factory,
                                     write_reports=write_reports)
            out.append((alert, result, None))
        except AlertNotFound:
            continue  # deleted by another process meanwhile
        except Exception as exc:  # noqa: BLE001 - one failing alert must not stop the others
            out.append((a, None, f"{type(exc).__name__}: {exc}"))
    return out


# ============================================================================ natural-language commands

_ALERT_WORD = r"\balerts?\b"


def parse_alert_command(text: str) -> Optional[Dict[str, Any]]:
    """Recognise alert-management prompts. Returns {op, ref?, query?, interval_days?} or None."""
    t = (text or "").strip()
    tl = t.lower()
    if not re.search(_ALERT_WORD, tl):
        return None
    if re.search(r"\b(list|show|what are)\b.*\balerts\b|\bmy alerts\b|^alerts$", tl) and not re.search(r"\b(create|add|new)\b", tl):
        return {"op": "list"}
    m = re.search(r"\b(create|add|set\s*up|setup|make|save|new)\b.*?\balert\b\s*(?:for|on|about|to track|tracking|:)?\s*(.*)$", t, re.I)
    if m:
        return {"op": "create", "query": m.group(2).strip(" .") or None}

    def ref_for(verbs: str) -> Optional[str]:
        q = re.search(r"[\"“']([^\"”']+)[\"”']", t)
        if q:
            return q.group(1).strip()
        after = re.search(rf"\b(?:{verbs})\b\s+(?:the\s+|my\s+)?alert\s+(?:called\s+|named\s+)?(.+?)(?:\s+(?:now|to|every|alert)\b.*)?[.?!]?$", t, re.I)
        if after and after.group(1).strip():
            return after.group(1).strip()
        before = re.search(rf"\b(?:{verbs})\b\s+(?:the\s+|my\s+)?(.+?)\s+alert\b", t, re.I)
        return before.group(1).strip() if before else None

    for op, verbs in (("pause", "pause|stop|disable|suspend"), ("resume", "resume|unpause|restart|enable|reactivate"),
                      ("delete", "delete|remove|drop"), ("run", "run|execute|trigger")):
        if re.search(rf"\b(?:{verbs})\b", tl):
            return {"op": op, "ref": ref_for(verbs)}
    if re.search(r"\b(change|set|update|edit|make|switch)\b", tl):
        interval, _ = _extract_interval(t)
        return {"op": "update", "ref": ref_for("change|set|update|edit|make|switch"), "interval_days": interval}
    return None
