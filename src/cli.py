"""research-librarian command-line interface.

    python -m src.cli search -q "graph neural networks for drug discovery" --since 2024
    python -m src.cli author --name "Author Name" -q "protein language models" --since 2023
    python -m src.cli venue --venue neurips --year 2025 -q "diffusion models for proteins"
    python -m src.cli add --doi 10.xxxx/xxxxx
    python -m src.cli related --seed <DOI | arXiv id | Zotero key | title>
    python -m src.cli baselines --seed <DOI | arXiv id | Zotero key | title>
    python -m src.cli gaps --collection "<collection name>"
    python -m src.cli ask "fetch me recent papers on protein language models since 2025"
    python -m src.cli alerts create --query "mammography foundation models"   (list/show/edit/pause/resume/delete/run/run-due)
    python -m src.cli review | check | scan (radiology profile)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from typing import List, Optional

from src.config import load_config
from src.models import SearchRequest
from src.pipeline import Pipeline, RunResult, load_proposals, load_state
from src.prompts.parser import MODALITY_SYNONYMS, TOPIC_SYNONYMS, VENUE_SYNONYMS, _collect, parse
from src.reports.generate_report import write_report

MOD_ALIASES = {"cxr": "cxr", "chest": "cxr", "xray": "cxr", "ct": "ct", "mammo": "mammo", "mammography": "mammo",
               "breast": "mammo", "general": "radiology-general", "radiology": "radiology-general"}


def _topics(values: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for v in values or []:
        found = _collect(v.lower(), TOPIC_SYNONYMS) or [v.lower()]
        out += [t for t in found if t not in out]
    return out


def _modalities(values: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for v in values or []:
        for part in v.replace(",", " ").split():
            m = MOD_ALIASES.get(part.lower()) or next(iter(_collect(part.lower(), MODALITY_SYNONYMS)), None)
            if m and m not in out:
                out.append(m)
    return out


def _venues(values: Optional[List[str]]) -> List[str]:
    from src.processing.normalize import canonical_venue
    generic = ("unknown", "other_journal", "other_conference", "workshop", "preprint")
    out: List[str] = []
    for v in values or []:
        found = _collect(v.lower(), VENUE_SYNONYMS) or [k for k in [canonical_venue(v)[0]] if k not in generic] or [v.lower()]
        out += [x for x in found if x not in out]
    return out


def _since(args) -> Optional[str]:
    if getattr(args, "since", None):
        s = str(args.since)
        return f"{s}-01-01" if len(s) == 4 else s
    return None


def summarize(result: RunResult, path) -> None:
    acc, rev = result.by_decision("accept"), result.by_decision("review")
    verb = "would add" if result.dry_run or result.request.action == "report_only" else "added"
    print(f"\nRetrieved {result.retrieved} | after filtering {result.after_filter} | already in Zotero {len(result.existing)} | "
          f"{verb} {len(acc)} high-confidence | review {len(rev)} | rejected {len(result.by_decision('reject'))}"
          + (f" | published versions of Zotero preprints {len(result.upgrades)}" if result.upgrades else ""))
    for p in result.upgrades[:10]:
        print(f"  [UPGRADE] {p.title[:90]} -> {p.venue_key} {p.year}, DOI {p.doi} (annotates Zotero item {p.related_preprint_key})")
    for p in result.existing[:5]:
        print(f"  [EXISTS ] {p.title[:90]} (Zotero item {p.existing_zotero_key}, matched by {p.duplicate_of})")
    filed = result.written.get("filed") or []
    if filed:
        print(f"  filed {len(filed)} matching paper(s) you already keep into '{result.request.target_collection}'")
    for p in acc[:15]:
        print(f"  [ACCEPT {p.final_score:.2f}] {p.title[:100]}  ({p.venue_key} {p.year})")
    for p in rev[:15]:
        print(f"  [REVIEW {p.final_score:.2f}] {p.title[:100]}  ({p.venue_key} {p.year})")
    for n in result.notes:
        print(f"  note: {n}")
    errs = list(dict.fromkeys(result.errors))
    for e in errs[:8]:
        print(f"  warning: {e[:200]}")
    if len(errs) > 8:
        print(f"  ... {len(errs) - 8} more warnings in the report")
    print(f"Report: {path}")


def run_request(pipe: Pipeline, req: SearchRequest, default_years: int = 3) -> RunResult:
    if req.intent == "scan":
        days = None
        if req.date_from:
            days = (date.today() - date.fromisoformat(req.date_from)).days
        return pipe.scan(days=days, action=req.action)
    if req.intent in ("similar", "citations", "related"):
        kinds = {"similar": ("recommendations", "related"), "citations": ("references",),
                 "related": ("references", "citations", "recommendations", "related")}[req.intent]
        return pipe.expand(req, kinds)
    if req.intent == "missing_lit":
        return pipe.missing_literature(req)
    if req.intent == "baselines":
        return pipe.baselines(req)
    date_from = req.date_from or (date.today() - timedelta(days=365 * default_years)).isoformat()
    return pipe.run(req, date_from, req.date_to)


def cmd_scan(args, cfg) -> int:
    if not cfg.is_radiology:
        print("The broad scan belongs to the radiology profile (profile: radiology in config.yaml). "
              "In the general profile, scheduled discovery runs through alerts: python -m src.cli alerts run-due")
        return 0
    if args.if_due:
        state = load_state()
        # While running dry (writes disabled / no credentials) a recent dry-run scan also counts, so the daily
        # scheduler doesn't repeat a full first-run scan every day; it never delays the first real scan.
        dry = args.dry_run or not cfg.writes_enabled or not cfg.secrets.zotero_configured
        keys = ["last_successful_scan"] + (["last_dry_run_scan"] if dry else [])
        stamps = [state[k][:10] for k in keys if state.get(k)]
        last = max(stamps) if stamps else None
        interval = int(cfg.get("scan.interval_days", 10))
        if last and (date.today() - date.fromisoformat(last)).days < interval:
            print(f"Not due: last scan {last}, interval {interval} days.")
            return 0
    pipe = Pipeline(cfg, dry_run=args.dry_run)
    result = pipe.scan(days=args.days)
    summarize(result, write_report(result))
    return 0


def cmd_search(args, cfg) -> int:
    req = SearchRequest(intent="topic_search", topics=_topics(args.topic), modalities=_modalities(args.modality),
                        date_from=_since(args), venues=_venues(args.venue), top_venues_only=args.top_venues,
                        limit=args.limit, free_text=args.query or args.text, target_collection=args.collection,
                        action="report_only" if args.report_only else "add_by_threshold")
    if not cfg.is_radiology and not req.free_text:
        req.free_text = " ".join((args.topic or []) + (args.modality or [])) or None
    if req.venues:
        req.intent = "venue_search"
    pipe = Pipeline(cfg, dry_run=args.dry_run)
    result = run_request(pipe, req)
    summarize(result, write_report(result, suffix=req.intent.replace("_", "-")))
    return 0


def cmd_venue(args, cfg) -> int:
    date_from = f"{args.year}-01-01" if args.year else _since(args) or (date.today() - timedelta(days=3 * 365)).isoformat()
    date_to = f"{args.year}-12-31" if args.year else None
    req = SearchRequest(intent="venue_search", topics=_topics(args.topic), modalities=_modalities(args.modality),
                        venues=_venues(args.venue), date_from=date_from, date_to=date_to, limit=args.limit,
                        target_collection=args.collection,
                        free_text=args.query or (None if cfg.is_radiology else " ".join((args.topic or []) + (args.modality or [])) or None),
                        action="report_only" if args.report_only else "add_by_threshold")
    pipe = Pipeline(cfg, dry_run=args.dry_run)
    result = pipe.run(req, date_from, date_to)
    summarize(result, write_report(result, suffix="venue"))
    return 0


def cmd_author(args, cfg) -> int:
    req = SearchRequest(intent="author_search", author=args.name, topics=_topics(args.topic), modalities=_modalities(args.modality),
                        date_from=_since(args), author_openalex_id=args.openalex_id, author_s2_id=args.s2_id,
                        target_collection=args.collection,
                        free_text=args.query or (None if cfg.is_radiology else " ".join((args.topic or []) + (args.modality or [])) or None),
                        source_tag="author-watch", limit=args.limit,
                        action="report_only" if args.report_only else "add_by_threshold")
    req.strict_request_match = bool(req.topics or req.modalities)
    pipe = Pipeline(cfg, dry_run=args.dry_run)
    result = pipe.run(req, req.date_from, None)
    summarize(result, write_report(result, suffix="author"))
    return 0


def cmd_watch(args, cfg) -> int:
    """Run every author in authors.yaml (useful for scheduling)."""
    if not cfg.authors:
        print("authors.yaml has no authors.")
        return 0
    since = _since(args) or (date.today() - timedelta(days=int(cfg.get("scan.min_lookback_days", 20)) * 2)).isoformat()
    for a in cfg.authors:
        req = SearchRequest(intent="author_search", author=a["name"], topics=a.get("topics") or [], modalities=a.get("modalities") or [],
                            free_text=a.get("query"),
                            author_openalex_id=a.get("openalex_id"), author_s2_id=a.get("semantic_scholar_id"),
                            date_from=since, source_tag="author-watch", strict_request_match=False)
        try:
            pipe = Pipeline(cfg, dry_run=args.dry_run)
            result = pipe.run(req, since, None)
        except Exception as exc:  # noqa: BLE001 - one failing author must not stop the rest of the watchlist
            print(f"Author watch for {a['name']} failed: {type(exc).__name__}: {exc}")
            continue
        summarize(result, write_report(result, suffix="author-watch"))
    return 0


def cmd_add(args, cfg) -> int:
    pipe = Pipeline(cfg, dry_run=args.dry_run)
    result = pipe.add_identifier(doi=args.doi, arxiv_id=args.arxiv, pmid=args.pmid, title=args.title)
    if result.papers:
        p = result.papers[0]
        print(json.dumps({k: p.to_dict()[k] for k in ("title", "authors", "year", "publication_date", "venue", "venue_type", "doi",
                                                       "arxiv_id", "pmid", "url", "modalities", "topics", "final_score", "decision")},
                         indent=2))
    summarize(result, write_report(result, suffix="add"))
    return 0 if (result.papers or result.existing or result.upgrades) else 1


def cmd_expand(args, cfg, intent: str) -> int:
    req = SearchRequest(intent=intent, seed=args.seed, topics=_topics(args.topic), modalities=_modalities(args.modality),
                        limit=args.limit, free_text=getattr(args, "query", None),
                        action="report_only" if args.report_only else "add_by_threshold")
    pipe = Pipeline(cfg, dry_run=args.dry_run)
    result = run_request(pipe, req)
    summarize(result, write_report(result, suffix=intent))
    return 0


def cmd_baselines(args, cfg) -> int:
    req = SearchRequest(intent="baselines", seed=args.seed, include_seed=args.include_seed, limit=args.limit,
                        target_collection=args.collection,
                        strict_request_match=False, action="report_only" if args.report_only else "add_by_threshold")
    pipe = Pipeline(cfg, dry_run=args.dry_run)
    result = pipe.baselines(req)
    summarize(result, write_report(result, suffix="baselines"))
    for p in result.by_decision("accept") + result.by_decision("review"):
        ev = p.graph_evidence or {}
        if ev.get("evidence"):
            print(f"    {p.title[:70]}: \"{ev['evidence'][:150]}\" [{ev.get('evidence_source')}]")
    return 0


def cmd_gaps(args, cfg) -> int:
    req = SearchRequest(intent="missing_lit", collection=args.collection, topics=_topics(args.topic),
                        modalities=_modalities(args.modality), limit=args.limit, free_text=getattr(args, "query", None),
                        action="report_only" if args.report_only else "add_by_threshold")
    pipe = Pipeline(cfg, dry_run=args.dry_run)
    result = pipe.missing_literature(req, min_cocitations=args.min_cocitations)
    summarize(result, write_report(result, suffix="gaps"))
    return 0


def cmd_ask(args, cfg) -> int:
    text = " ".join(args.prompt)
    from src.alerts import parse_alert_command
    alert_cmd = parse_alert_command(text)
    if alert_cmd:
        return cmd_ask_alert(alert_cmd, text, cfg, args)
    req, conf = parse(text, profile=cfg.profile)
    if conf < 0.6 and cfg.get("llm.enabled", False):
        from src.prompts.relevance import llm_parse
        llm_req = llm_parse(text, cfg)
        if llm_req:
            req, conf = llm_req, 0.7
    print("Parsed request:", json.dumps({k: v for k, v in req.to_dict().items()
                                         if v not in (None, [], False) and k != "parse_notes"}, indent=None))
    for n in req.parse_notes:
        print(f"  note: {n}")
    if conf < 0.6:
        print(f"Low parse confidence ({conf:.2f}); refine the prompt or use an explicit subcommand.")
        return 2
    if req.intent == "baselines" and not req.seed:
        print("Which paper? Give its DOI, arXiv id, Zotero item key, or its title in quotes.")
        return 2
    if req.intent == "author_search" and not req.author:
        print("Could not identify the author name.")
        return 2
    pipe = Pipeline(cfg, dry_run=args.dry_run)
    result = run_request(pipe, req)
    summarize(result, write_report(result, suffix=req.intent.replace("_", "-")))
    return 0


def cmd_review(args, cfg) -> int:
    if args.from_zotero:
        from src.zotero.client import ZoteroClient
        from src.zotero.collections import CollectionMap
        z = ZoteroClient(cfg.secrets)
        cols = CollectionMap(z, cfg.get("zotero.root_collection", "Literature")).load()
        key = cols.paths.get(f"{cols.root}/{cfg.get('zotero.auto_review_collection', 'Auto Discovery - Review')}")
        if not key:
            print("Review collection does not exist yet.")
            return 0
        for it in z.collection_items(key):
            d = it["data"]
            print(f"- [{d['key']}] {d.get('title')} ({d.get('date', '')[:4]})")
        return 0
    data = load_proposals()
    if not data:
        print("No proposals yet - run scan/search first.")
        return 0
    print(f"Proposals from {data['generated']} ({'dry run' if data['dry_run'] else 'written'}) - request: {data['request']['intent']}")
    for p in data["papers"]:
        print(f"  [{p['decision'].upper():6} {p['final_score']:.2f}] {p['title'][:100]}")
        print(f"      {p.get('venue') or '-'} {p.get('year') or ''} | {', '.join(p['modalities'])} | {', '.join(p['topics'])}"
              + (f" | {p['reject_reason']}" if p.get("reject_reason") else ""))
    return 0


def cmd_check(args, cfg) -> int:
    """Phase-1 checks: credentials, key permissions, library read, collections, duplicate lookup."""
    from src.zotero.client import LibraryCache, ZoteroClient, ZoteroError
    from src.zotero.collections import CollectionMap
    from src.zotero.items import build_library_index
    try:
        z = ZoteroClient(cfg.secrets, dry_run=True)
    except ZoteroError as exc:
        print(f"Zotero: {exc}")
        return 1
    info = z.key_info()
    access = info.get("access", {})
    print(f"Zotero key OK for user {info.get('username')} (id {info.get('userID')}); access: {json.dumps(access)}")
    lib = access.get("user", {}) if cfg.secrets.zotero_library_type == "user" else access.get("groups", {})
    if cfg.secrets.zotero_library_type == "user" and not lib.get("write"):
        print("WARNING: key has no write access to the personal library - writes will fail.")
    items = LibraryCache(z).load(refresh=True)
    index = build_library_index(items)
    print(f"Library: {len(items)} top-level items; {len(index.by_id)} identifiers indexed (library version {z.library_version})")
    cols = CollectionMap(z, cfg.get("zotero.root_collection", "Literature")).load()
    root = cols.root
    print(f"Collections: {len(cols.paths)} total; '{root}' {'exists' if root in cols.paths else 'missing (will be created on first write)'}")
    if items:
        sample = items[0]
        from src.zotero.items import record_from_item
        rec = record_from_item(sample)
        if rec:
            key, how = index.find(rec.ids, rec.title)
            print(f"Duplicate lookup self-test on '{rec.title[:60]}': matched {key} via {how}")
    print(f"Writes enabled in config: {cfg.writes_enabled}")
    return 0


def cmd_init_collections(args, cfg) -> int:
    from src.zotero.client import ZoteroClient
    from src.zotero.collections import CollectionMap, collection_names
    dry = args.dry_run or not cfg.writes_enabled
    z = ZoteroClient(cfg.secrets, dry_run=dry)
    cols = CollectionMap(z, cfg.get("zotero.root_collection", "Literature")).load()
    cols.ensure_hierarchy(collection_names(cfg), cfg.profile)
    if dry:
        print("Dry run. Would create:" if cols.planned else "Hierarchy already complete.")
        for p in cols.planned:
            print("  " + p)
    else:
        print("Collection hierarchy ensured.")
    return 0


def cmd_evaluate(args, cfg) -> int:
    from src.evaluate import evaluate
    print(json.dumps(evaluate(cfg, args.gold), indent=2))
    return 0


# ============================================================================ alerts

def _fmt_dt(v: Optional[str]) -> str:
    return (v or "-").replace("T", " ")[:16]


def _print_alert(a, full: bool = False) -> None:
    print(f"{a.id}  [{a.status}]  {a.name}")
    print(f"    every {a.interval_days} days | next run {_fmt_dt(a.next_run_at)} | last run {_fmt_dt(a.last_run_at)}"
          + (f" ({a.last_status})" if a.last_status else ""))
    print(f"    {a.intent}: {a.describe() or '-'}")
    if full:
        if a.query:
            print(f"    query: {a.query}")
        print(f"    created {_fmt_dt(a.created_at)} | updated {_fmt_dt(a.updated_at)} | "
              f"last writing run {_fmt_dt(a.last_success_at)} | last dry run {_fmt_dt(a.last_dry_run_at)}")
        if a.last_result:
            r = a.last_result
            print(f"    last result: retrieved {r.get('retrieved')} | accepted {r.get('accepted')} | review {r.get('review')} | "
                  f"already in Zotero {r.get('already_in_zotero')} | written {r.get('written') or {}} | from {r.get('window_from')}")
            if r.get("report"):
                print(f"    report: {r['report']}")
        if a.last_error:
            print(f"    last error: {a.last_error}")


def _alert_fields(args) -> dict:
    """CLI flags -> alert fields (only flags the user actually gave)."""
    out = {}
    for flag, fld in (("query", "query"), ("author", "author"), ("openalex_id", "author_openalex_id"), ("s2_id", "author_s2_id"),
                      ("text", "free_text"), ("since", "since"), ("every", "interval_days"), ("limit", "limit")):
        v = getattr(args, flag, None)
        if v is not None:
            out[fld] = v
    for flag, fld in (("topic", "topics"), ("modality", "modalities"), ("venue", "venues")):
        v = getattr(args, flag, None)
        if v is not None:
            out[fld] = [x for item in v for x in item.split(",")]
    if getattr(args, "top_venues", None):
        out["top_venues_only"] = True
    if getattr(args, "report_only", None):
        out["action"] = "report_only"
    if getattr(args, "add", None):
        out["action"] = "add_by_threshold"
    return out


def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _run_alert_and_report(ref: str, cfg, dry_run: bool) -> int:
    from src.alerts import run_alert_now
    alert, result = run_alert_now(ref, cfg, dry_run=dry_run)
    summarize(result, alert.last_result.get("report") or "-")
    print(f"Next scheduled run: {_fmt_dt(alert.next_run_at)}")
    return 0


def cmd_alerts(args, cfg) -> int:
    from src import alerts as A
    try:
        if args.alert_cmd == "create":
            f = _alert_fields(args)
            a = A.create_alert(args.name, cfg=cfg, status="paused" if args.paused else "active",
                               **{k: v for k, v in f.items()})
            print("Created alert:")
            _print_alert(a, full=True)
            print("It runs on the next scheduler pass (`alerts run-due`), or now with "
                  f"`python -m src.cli alerts run {a.id} --dry-run`.")
        elif args.alert_cmd == "list":
            items = A.list_alerts(status=args.status)
            if not items:
                print("No alerts yet. Create one with: python -m src.cli alerts create --query \"mammography foundation models\"")
            for a in items:
                _print_alert(a)
        elif args.alert_cmd == "show":
            _print_alert(A.get_alert(args.ref), full=True)
        elif args.alert_cmd == "edit":
            f = _alert_fields(args)
            if args.name:
                f["name"] = args.name
            if args.clear_author:
                f.update(author=None, author_openalex_id=None, author_s2_id=None)
            if args.clear_since:
                f["since"] = None
            if not f:
                print("Nothing to change - pass e.g. --every 14, --topic, --modality, --query, --name.")
                return 2
            a = A.update_alert(args.ref, cfg=cfg, **f)
            print("Updated alert:")
            _print_alert(a, full=True)
        elif args.alert_cmd == "pause":
            _print_alert(A.pause_alert(args.ref))
        elif args.alert_cmd == "resume":
            _print_alert(A.resume_alert(args.ref))
        elif args.alert_cmd == "delete":
            a = A.get_alert(args.ref)
            if not _confirm(f"Delete alert '{a.name}' ({a.id})? Papers it already added stay in Zotero.", args.yes):
                print("Not deleted.")
                return 1
            A.delete_alert(a.id)
            print(f"Deleted alert {a.id}.")
        elif args.alert_cmd == "run":
            return _run_alert_and_report(args.ref, cfg, args.dry_run)
        elif args.alert_cmd == "run-due":
            results = A.run_due_alerts(cfg, dry_run=args.dry_run)
            if not results:
                print("No alerts due.")
            for alert, result, err in results:
                if err:
                    print(f"[{alert.id}] FAILED: {err}")
                else:
                    print(f"[{alert.id}] {alert.name}")
                    summarize(result, alert.last_result.get("report") or "-")
            return 1 if any(err for _, _, err in results) else 0
    except A.AlertError as exc:
        print(f"Alert error: {exc}")
        return 2
    return 0


def cmd_ask_alert(cmd: dict, text: str, cfg, args) -> int:
    from src import alerts as A
    try:
        op = cmd["op"]
        if op == "list":
            items = A.list_alerts()
            print("No alerts yet." if not items else "")
            for a in items:
                _print_alert(a)
            return 0
        if op == "create":
            if not cmd.get("query"):
                print("What should the alert track? e.g. 'Create an alert for mammography foundation models every 2 weeks'")
                return 2
            a = A.create_alert(query=cmd["query"], cfg=cfg)
            print("Created alert:")
            _print_alert(a, full=True)
            return 0
        if not cmd.get("ref"):
            print("Which alert? Name it, e.g. 'pause the mammography foundation models alert'.")
            return 2
        a = A.get_alert(cmd["ref"])
        if op == "pause":
            _print_alert(A.pause_alert(a.id))
        elif op == "resume":
            _print_alert(A.resume_alert(a.id))
        elif op == "delete":
            if not _confirm(f"Delete alert '{a.name}' ({a.id})? Papers it already added stay in Zotero.", args.yes):
                print("Not deleted.")
                return 1
            A.delete_alert(a.id)
            print(f"Deleted alert {a.id}.")
        elif op == "run":
            return _run_alert_and_report(a.id, cfg, args.dry_run)
        elif op == "update":
            if not cmd.get("interval_days"):
                print("From a prompt I can change the cadence ('every 2 weeks'); use `alerts edit` for other fields.")
                return 2
            _print_alert(A.update_alert(a.id, interval_days=cmd["interval_days"]), full=True)
        return 0
    except A.AlertError as exc:
        print(f"Alert error: {exc}")
        return 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="research-librarian", description="Automated literature discovery -> Zotero")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--profile", choices=["general", "radiology"], help="override config.yaml's profile for this command")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, topic=True):
        p.add_argument("--dry-run", action="store_true", help="discover and rank without changing Zotero")
        if p.prog.split()[-1] != "gaps":  # gaps uses --collection for the collection it analyses
            p.add_argument("--collection", help="file accepted papers into this collection, from the library's top level "
                                                "(e.g. 'Mammo' or 'CXR/Longitudinal'); created if missing")
        if topic and p.prog.split()[-1] in ("venue", "author", "similar", "citations", "related", "gaps"):
            p.add_argument("--query", "-q", help="what the papers should be about (general profile)")
        p.add_argument("--report-only", action="store_true", help="never write, even when writes are enabled")
        p.add_argument("--limit", type=int)
        if topic:
            p.add_argument("--topic", action="append", help="(radiology profile) e.g. 'foundation models', 'VLM', uncertainty")
            p.add_argument("--modality", action="append", help="cxr / ct / mammo / general")

    p = sub.add_parser("scan", help="scheduled-style discovery")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--days", type=int, help="override the lookback window")
    p.add_argument("--if-due", action="store_true", help="exit unless interval_days have passed since the last scan")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("search", help="search for papers on a topic")
    common(p)
    p.add_argument("--query", "-q", help='what to search for, e.g. "graph neural networks for drug discovery" '
                                           'or \'(GNN OR "graph neural network") AND "drug discovery"\'')
    p.add_argument("--since")
    p.add_argument("--venue", action="append")
    p.add_argument("--top-venues", action="store_true")
    p.add_argument("--text", help="(radiology profile) extra keywords that should appear")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("venue", help="venue search, e.g. MICCAI 2026")
    common(p)
    p.add_argument("--venue", action="append", required=True)
    p.add_argument("--year", type=int)
    p.add_argument("--since")
    p.set_defaults(fn=cmd_venue)

    p = sub.add_parser("author", help="author search")
    common(p)
    p.add_argument("--name", required=True)
    p.add_argument("--since")
    p.add_argument("--openalex-id")
    p.add_argument("--s2-id")
    p.set_defaults(fn=cmd_author)

    p = sub.add_parser("watch", help="run all authors in authors.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--since")
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("add", help="add a known paper")
    p.add_argument("--doi")
    p.add_argument("--arxiv")
    p.add_argument("--pmid")
    p.add_argument("--title")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_add)

    for name, intent, helptext in (("similar", "similar", "papers similar to a seed"),
                                   ("citations", "citations", "important references of a seed"),
                                   ("related", "related", "references + citations + neighbours of a seed / Zotero item")):
        p = sub.add_parser(name, help=helptext)
        common(p)
        p.add_argument("--seed", required=True, help="DOI, arXiv id, Zotero item key, or title")
        p.set_defaults(fn=lambda a, c, i=intent: cmd_expand(a, c, i))

    p = sub.add_parser("baselines", help="find (and optionally add) the baseline papers a paper compares against")
    p.add_argument("--seed", required=True, help="DOI, arXiv id, Zotero item key, or title of the paper")
    p.add_argument("--include-seed", action="store_true", help="also add the paper itself")
    p.add_argument("--collection", help="file accepted baselines into this collection (from the library's top level)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_baselines)

    p = sub.add_parser("gaps", help="missing-literature analysis for a Zotero collection")
    common(p)
    p.add_argument("--collection", required=True)
    p.add_argument("--min-cocitations", type=int, default=2)
    p.set_defaults(fn=cmd_gaps)

    p = sub.add_parser("ask", help="natural-language command (searches and alert management)")
    p.add_argument("prompt", nargs="+")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true", help="skip confirmation (alert deletion)")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("alerts", help="saved literature alerts (default cadence: every 30 days)")
    asub = p.add_subparsers(dest="alert_cmd", required=True)

    def alert_def(q, creating):
        q.add_argument("--query", help="natural-language description, e.g. 'papers by Jane Doe on mammography foundation models'")
        q.add_argument("--topic", action="append", help="foundation-model, vlm, uncertainty, calibration, ... (repeatable)")
        q.add_argument("--modality", action="append", help="cxr / ct / mammo / general (repeatable)")
        q.add_argument("--venue", action="append", help="e.g. neurips, icml, miccai, tmi")
        q.add_argument("--author")
        q.add_argument("--openalex-id", help="pin the author's OpenAlex id")
        q.add_argument("--s2-id", help="pin the author's Semantic Scholar id")
        q.add_argument("--text", help="extra keywords that should appear")
        q.add_argument("--top-venues", action="store_true", help="only top venues")
        q.add_argument("--since", help="earliest publication date searched (YYYY or YYYY-MM-DD)")
        q.add_argument("--every", help="cadence: 30, 30d, 2w, 1m, weekly, monthly (default 30 days)" if creating else "new cadence")
        q.add_argument("--limit", type=int, help="max papers kept per run")
        q.add_argument("--report-only", action="store_true", help="never write to Zotero; just report")
        if not creating:
            q.add_argument("--add", action="store_true", help="switch a report-only alert back to adding papers")

    q = asub.add_parser("create", help="create an alert")
    q.add_argument("name", nargs="?", help="display name (generated from the definition if omitted)")
    alert_def(q, True)
    q.add_argument("--paused", action="store_true", help="create it paused")
    q = asub.add_parser("list", help="list alerts")
    q.add_argument("--status", choices=["active", "paused"])
    q = asub.add_parser("show", help="show one alert with its last result")
    q.add_argument("ref", help="alert id, name, or a unique part of either")
    q = asub.add_parser("edit", help="change an alert")
    q.add_argument("ref")
    q.add_argument("--name")
    alert_def(q, False)
    q.add_argument("--clear-author", action="store_true")
    q.add_argument("--clear-since", action="store_true")
    for name, helptext in (("pause", "pause an alert"), ("resume", "resume a paused alert")):
        q = asub.add_parser(name, help=helptext)
        q.add_argument("ref")
    q = asub.add_parser("delete", help="delete an alert (Zotero items are kept)")
    q.add_argument("ref")
    q.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    q = asub.add_parser("run", help="run one alert now (even if paused or not due)")
    q.add_argument("ref")
    q.add_argument("--dry-run", action="store_true")
    q = asub.add_parser("run-due", help="run every active alert that is due (for the scheduler)")
    q.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_alerts)

    p = sub.add_parser("review", help="show proposed additions from the last run")
    p.add_argument("--from-zotero", action="store_true", help="list the Zotero review collection instead")
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("check", help="verify Zotero connectivity (phase 1)")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("init-collections", help="create the collection hierarchy")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_init_collections)

    p = sub.add_parser("evaluate", help="measure precision / recall on a labeled gold set")
    p.add_argument("--gold", default="data/gold/gold.jsonl")
    p.set_defaults(fn=cmd_evaluate)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    from src.zotero.client import ZoteroError
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    cfg = load_config()
    if args.profile:
        cfg.raw["profile"] = args.profile
    try:
        return args.fn(args, cfg)
    except ZoteroError as exc:
        print(f"Zotero: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
