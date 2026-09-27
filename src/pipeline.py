"""Orchestration: retrieve -> normalize -> classify -> deduplicate -> rank -> ingest -> report."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.config import AUDIT_PATH, CACHE_DIR, PROPOSALS_PATH, STATE_PATH, Config
from src.discovery import arxiv, crossref, openalex, openreview, pmlr, pubmed
from src.discovery import semantic_scholar as s2
from src.discovery.base import SourceContext
from src.discovery.query import BoolQuery, build_queries, scan_queries, text_query
from src.http import HttpClient, HttpError
from src.models import Paper, SearchRequest
from src.processing import classify as C
from src.processing import rank as R
from src.processing.deduplicate import DedupIndex, dedupe_candidates, same_work_title
from src.processing.normalize import (author_matches, finalize, is_preprint_doi, merge, normalize_arxiv_id, normalize_doi,
                                      normalize_pmid, normalize_title)
from src.processing.version_resolution import resolve_against_library, upgrade_note
from src.zotero import attachments, items as zitems
from src.zotero.client import LibraryCache, ZoteroClient, ZoteroError
from src.zotero.collections import CollectionMap, collection_names, resolve_collection, target_collections

log = logging.getLogger("zotero_tool")

S2_VENUE_NAMES = {
    "miccai": "MICCAI", "midl": "Medical Imaging with Deep Learning", "tmi": "IEEE Transactions on Medical Imaging",
    "medical_image_analysis": "Medical Image Analysis", "neurips": "Neural Information Processing Systems",
    "icml": "International Conference on Machine Learning", "iclr": "International Conference on Learning Representations",
    "cvpr": "Computer Vision and Pattern Recognition", "iccv": "IEEE International Conference on Computer Vision",
    "eccv": "European Conference on Computer Vision", "tpami": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    "radiology_ai": "Radiology: Artificial Intelligence", "radiology": "Radiology", "aaai": "AAAI Conference on Artificial Intelligence",
    "ijcai": "International Joint Conference on Artificial Intelligence", "isbi": "IEEE International Symposium on Biomedical Imaging",
    "tnnls": "IEEE Transactions on Neural Networks and Learning Systems", "jmlr": "Journal of machine learning research",
    "european_radiology": "European Radiology",
}
JOURNAL_KEYS = {"tmi", "medical_image_analysis", "tpami", "radiology_ai", "radiology", "tnnls", "jmlr", "european_radiology", "jdi", "aim"}
WEAK_VENUES = {"unknown", "other_conference", "other_journal", "preprint", "workshop"}


@dataclass
class RunResult:
    request: SearchRequest
    date_from: Optional[str]
    date_to: Optional[str]
    retrieved: int = 0
    after_filter: int = 0
    papers: List[Paper] = field(default_factory=list)
    existing: List[Paper] = field(default_factory=list)
    upgrades: List[Paper] = field(default_factory=list)
    previously_seen: int = 0
    written: Dict[str, List[str]] = field(default_factory=lambda: {"accept": [], "review": []})
    dry_run: bool = True
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    report_path: Optional[str] = None
    author_resolution: List[Dict[str, Any]] = field(default_factory=list)
    seed: Optional[Paper] = None

    def by_decision(self, d: str) -> List[Paper]:
        return sorted([p for p in self.papers if p.decision == d], key=lambda p: -p.final_score)


class Pipeline:
    def __init__(self, cfg: Config, dry_run: bool = True, use_zotero: bool = True):
        self.cfg = cfg
        self.http = HttpClient(CACHE_DIR, float(cfg.get("http.cache_ttl_hours", 24)), float(cfg.get("http.timeout_seconds", 30)),
                               contact_email=cfg.secrets.contact_email)
        self.ctx = SourceContext(cfg, self.http)
        self.dry_run = dry_run or not cfg.writes_enabled
        self.zotero: Optional[ZoteroClient] = None
        self.library_items: List[Dict[str, Any]] = []
        self.index = DedupIndex(float(cfg.get("ranking.title_fuzzy_threshold", 0.96)))
        self.collections: Optional[CollectionMap] = None
        self.notes: List[str] = []
        if not cfg.writes_enabled and not dry_run:
            self.notes.append("Zotero writes are disabled (safety.writes_enabled: false) - ran as dry-run.")
        if use_zotero and cfg.secrets.zotero_configured:
            self.zotero = ZoteroClient(cfg.secrets, HttpClient(timeout=60), dry_run=self.dry_run)
        elif use_zotero:
            self.notes.append("Zotero credentials not configured - library deduplication skipped, nothing written.")
            self.dry_run = True

    # ================================================================ library
    def load_library(self) -> None:
        if not self.zotero:
            return
        self.library_items = LibraryCache(self.zotero).load(refresh=True)
        self.index = zitems.build_library_index(self.library_items, self.index.fuzzy_threshold)
        self.collections = CollectionMap(self.zotero, self.cfg.get("zotero.root_collection", "Literature")).load()
        log.info("Zotero library: %d top-level items indexed", len(self.index.records))

    # ================================================================ retrieval
    def _run_tasks(self, tasks: List[Tuple[str, Callable[[], List[Paper]]]]) -> List[Paper]:
        out: List[Paper] = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(fn): name for name, fn in tasks}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    res = fut.result()
                    log.info("  %-45s %4d results", name, len(res))
                    out.extend(res)
                except Exception as exc:  # noqa: BLE001 - one failing source must not abort the run
                    log.debug("source task %s failed", name, exc_info=True)
                    self.ctx.warn(name.split(":")[0], f"{name} failed: {type(exc).__name__}: {exc}")
        return out

    def _query_tasks(self, queries: List[BoolQuery], date_from: Optional[str], date_to: Optional[str],
                     limit: Optional[int] = None, venue: Optional[str] = None) -> List[Tuple[str, Callable[[], List[Paper]]]]:
        t: List[Tuple[str, Callable[[], List[Paper]]]] = []
        for q in queries:
            if self.cfg.source_enabled("openalex") and not venue:
                t.append((f"openalex:{q.label}", lambda q=q: openalex.search(self.ctx, q, date_from, date_to, limit)))
            if self.cfg.source_enabled("semantic_scholar"):
                t.append((f"semantic_scholar:{q.label}", lambda q=q: s2.search(self.ctx, q, date_from, date_to, limit, venue=venue)))
            if self.cfg.source_enabled("pubmed") and not venue:
                t.append((f"pubmed:{q.label}", lambda q=q: pubmed.search(self.ctx, q, date_from, date_to, limit)))
            if self.cfg.source_enabled("arxiv") and not venue:
                t.append((f"arxiv:{q.label}", lambda q=q: arxiv.search(self.ctx, q, date_from, date_to, limit)))
        return t

    def _proceedings_tasks(self, date_from: Optional[str], date_to: Optional[str] = None,
                           venues: Optional[List[str]] = None) -> List[Tuple[str, Callable[[], List[Paper]]]]:
        y0 = int(date_from[:4]) if date_from else 0
        y1 = int(date_to[:4]) if date_to else 9999
        t: List[Tuple[str, Callable[[], List[Paper]]]] = []
        if self.cfg.source_enabled("pmlr"):
            for vol, meta in (self.cfg.get("pmlr_volumes", {}) or {}).items():
                if y0 <= meta["year"] <= y1 and (not venues or meta["venue"] in venues):
                    t.append((f"pmlr:v{vol}", lambda v=int(vol), m=meta: pmlr.volume_papers(self.ctx, v, m["venue"], m["year"])))
        if self.cfg.source_enabled("openreview"):
            for v in self.cfg.get("openreview_venues", []) or []:
                if y0 <= v["year"] <= y1 and (not venues or v["venue"] in venues):
                    t.append((f"openreview:{v['id']}", lambda v=v: openreview.venue_papers(self.ctx, v["id"], v["venue"], v["year"])))
        return t

    def _queries(self, req: SearchRequest) -> List[BoolQuery]:
        """General profile: the request's own text. Radiology profile: built-in modality x topic queries."""
        if self.cfg.is_radiology and (req.topics or req.modalities or not req.free_text):
            return build_queries(req.modalities, req.topics, None)
        if not req.free_text:
            self.notes.append("Nothing to search for: give a query (e.g. --query \"graph neural networks for drug discovery\").")
            return []
        return [text_query(req.free_text, re.sub(r"[^a-z0-9]+", "-", req.free_text.lower())[:40])]

    def retrieve(self, req: SearchRequest, date_from: Optional[str], date_to: Optional[str]) -> List[Paper]:
        if req.intent == "scan":
            queries = scan_queries()
            tasks = self._query_tasks(queries, date_from, date_to) + self._proceedings_tasks(date_from, date_to)
            return self._run_tasks(tasks)
        if req.intent == "author_search":
            return self.retrieve_author(req, date_from, date_to)
        if req.intent == "venue_search":
            return self.retrieve_venue(req, date_from, date_to)
        queries = self._queries(req)
        tasks = self._query_tasks(queries, date_from, date_to, req.limit and max(req.limit * 3, 50))
        if req.top_venues_only or req.venues:
            tasks += self._proceedings_tasks(date_from, date_to, req.venues or None)
        return self._run_tasks(tasks)

    def retrieve_author(self, req: SearchRequest, date_from: Optional[str], date_to: Optional[str] = None) -> List[Paper]:
        name = req.author or ""
        watch = next((a for a in self.cfg.authors if a.get("name", "").lower() == name.lower()), {})
        oa_id = req.author_openalex_id or watch.get("openalex_id")
        s2_id = req.author_s2_id or watch.get("semantic_scholar_id")
        resolution: List[Dict[str, Any]] = []
        if not oa_id and self.cfg.source_enabled("openalex"):
            try:
                cands = openalex.resolve_author(self.ctx, name)
            except HttpError as exc:
                self.ctx.warn("openalex", f"author search failed: {exc}")
                cands = []
            resolution += cands[:3]
            if cands:
                oa_id = cands[0]["id"]
                if len(cands) > 1 and cands[1]["score"] >= 0.8 * cands[0]["score"] and cands[1]["affinity"] > 0:
                    self.notes.append(f"Author name '{name}' is ambiguous in OpenAlex; using {cands[0]['name']} ({oa_id}, "
                                      f"{', '.join(cands[0]['institutions'][:2]) or 'no affiliation'}). Pin with --openalex-id if wrong.")
        if not s2_id and self.cfg.source_enabled("semantic_scholar"):
            try:
                cands = s2.resolve_author(self.ctx, name)
            except HttpError as exc:
                self.ctx.warn("semantic_scholar", f"author search failed: {exc}")
                cands = []
            resolution += cands[:3]
            if cands:
                s2_id = cands[0]["id"]
        self._author_resolution = resolution
        tasks: List[Tuple[str, Callable[[], List[Paper]]]] = []
        oa_ids = [oa_id] if oa_id else []
        if oa_id and not (req.author_openalex_id or watch.get("openalex_id")):
            # OpenAlex often splits one person across several ids: add same-name ids sharing an institution.
            top = next((c for c in resolution if c["source"] == "openalex" and c["id"] == oa_id), None)
            for c in resolution:
                if c["source"] == "openalex" and c["id"] != oa_id and top and author_matches(name, c["name"] or "") \
                        and set(c["institutions"]) & set(top["institutions"]):
                    oa_ids.append(c["id"])
        for aid in oa_ids:
            tasks.append((f"openalex:author:{aid}", lambda aid=aid: openalex.author_works(self.ctx, aid, date_from, date_to=date_to)))
        if s2_id:
            tasks.append((f"semantic_scholar:author:{s2_id}", lambda: s2.author_papers(self.ctx, s2_id, date_from, date_to=date_to)))
        if not tasks:
            self.notes.append(f"Could not resolve author '{name}' in any source.")
            return []
        papers = self._run_tasks(tasks)
        # Guard against disambiguation errors: the resolved author must appear on the byline.
        return [p for p in papers if any(author_matches(name, a) for a in p.authors) or not p.authors]

    def retrieve_venue(self, req: SearchRequest, date_from: Optional[str], date_to: Optional[str]) -> List[Paper]:
        queries = self._queries(req)
        tasks: List[Tuple[str, Callable[[], List[Paper]]]] = []
        for v in req.venues:
            if v in S2_VENUE_NAMES:
                tasks += self._query_tasks(queries, date_from, date_to, venue=S2_VENUE_NAMES[v])
            if v in JOURNAL_KEYS and self.cfg.source_enabled("openalex"):
                try:
                    sids = openalex.source_ids_for_venue(self.ctx, S2_VENUE_NAMES.get(v, v))
                except HttpError:
                    sids = []
                if sids:
                    for q in queries:
                        tasks.append((f"openalex:{v}:{q.label}", lambda q=q, s=sids: openalex.search_in_sources(self.ctx, q, s, date_from, date_to)))
        # Conference papers in OpenAlex often sit under LNCS; general queries + Crossref enrichment resolve them.
        if any(v not in JOURNAL_KEYS for v in req.venues) and self.cfg.source_enabled("openalex"):
            for q in queries:
                tasks.append((f"openalex:{q.label}", lambda q=q: openalex.search(self.ctx, q, date_from, date_to)))
        tasks += self._proceedings_tasks(date_from, date_to, req.venues)
        return self._run_tasks(tasks)

    # ================================================================ enrichment
    def enrich(self, papers: List[Paper]) -> None:
        """Fill abstracts / canonical venues for promising candidates (S2 batch, then Crossref)."""
        promising = []
        for p in papers:
            C.classify(p)
            if not self.cfg.is_radiology:
                if not p.classification["excluded_type"]:
                    promising.append(p)
            elif C.hard_filter(p) is None or (not p.abstract and float(p.classification["radiology_relevance"]) >= 0.3):
                promising.append(p)
        need_s2 = [p for p in promising if (not p.abstract or p.venue_key in WEAK_VENUES or p.venue_type == "preprint")
                   and (p.doi or p.arxiv_id or p.pmid)
                   and "semantic_scholar" not in p.sources]
        if need_s2 and self.cfg.source_enabled("semantic_scholar"):
            ids = ["DOI:" + p.doi if p.doi else "ARXIV:" + p.arxiv_id if p.arxiv_id else "PMID:" + p.pmid for p in need_s2]
            for p, got in zip(need_s2, s2.batch(self.ctx, ids)):
                if got:
                    merge(p, got)
        if self.cfg.source_enabled("crossref"):
            need_cr = [p for p in promising if p.doi and not is_preprint_doi(p.doi) and
                       (p.venue_key in WEAK_VENUES or p.venue_type == "preprint" or not p.pages or not p.abstract)][:200]
            with ThreadPoolExecutor(max_workers=4) as pool:
                for p, got in zip(need_cr, pool.map(lambda x: crossref.get_doi(self.ctx, x.doi), need_cr)):
                    if got and got.title and normalize_title(got.title)[:40] == normalize_title(p.title)[:40]:
                        merge(p, got)
            # Published version lookup for papers that only carry an arXiv DOI / no DOI.
            need_pub = [p for p in promising if p.venue_type == "preprint" and (p.arxiv_id or is_preprint_doi(p.doi))
                        and (not p.doi or is_preprint_doi(p.doi))][:100]
            with ThreadPoolExecutor(max_workers=4) as pool:
                for p, hits in zip(need_pub, pool.map(lambda x: crossref.find_by_title(self.ctx, x.title, rows=3), need_pub)):
                    for got in hits:
                        if got.venue_type in ("journal", "conference") and same_work_title(got.title, p.title) and \
                                (not p.year or not got.year or got.year >= p.year):
                            merge(p, got)
                            break
        for p in papers:
            p.classification = {}

    # ================================================================ processing
    def process(self, raw: List[Paper], req: SearchRequest, today: Optional[date] = None,
                window: Tuple[Optional[str], Optional[str]] = (None, None)) -> Tuple[List[Paper], List[Paper], List[Paper]]:
        """Returns (candidates, existing_in_zotero, preprint_upgrades). `window` = (from, to) publication-date bounds."""
        papers = dedupe_candidates(raw, self.index.fuzzy_threshold)
        self.enrich(papers)
        papers = dedupe_candidates(papers, self.index.fuzzy_threshold)  # enrichment may reveal shared DOIs
        allow_editorials = bool(req.free_text and "editorial" in req.free_text.lower())
        candidates, existing, upgrades = [], [], []
        for p in papers:
            C.classify(p)
            R.score_paper(p, self.cfg, req, today)
            cocited = max((int(q.split(":", 1)[1]) for q in p.queries if q.startswith("cocitation:")), default=1)
            if cocited > 1:  # missing-literature analysis: frequently co-cited references rank higher
                p.final_score = round(min(1.0, p.final_score + 0.02 * (cocited - 1)), 4)
            R.decide(p, self.cfg, req, allow_editorials=allow_editorials)
            outside = _outside_window(p, *window)
            if outside and p.decision != "reject":
                p.decision, p.reject_reason = "reject", outside
            resolve_against_library(p, self.index)
            if p.related_preprint_key and p.decision != "reject":
                upgrades.append(p)
            elif p.existing_zotero_key:
                existing.append(p)
            else:
                candidates.append(p)
        self._llm_adjudicate(candidates, req)
        R.apply_rate_limit(candidates, int(self.cfg.get("safety.max_auto_add_per_scan", 25)))
        if req.limit:
            kept = sorted([p for p in candidates if p.decision in ("accept", "review")], key=lambda p: -p.final_score)
            for p in kept[req.limit:]:
                p.decision, p.reject_reason = "reject", f"beyond requested limit of {req.limit}"
        return candidates, existing, upgrades

    def _llm_adjudicate(self, papers: List[Paper], req: SearchRequest) -> None:
        if not self.cfg.get("llm.enabled", False) or not self.cfg.get("llm.tasks.relevance", False) or req.intent == "baselines" \
                or not self.cfg.is_radiology:
            return
        try:
            from src.prompts.relevance import adjudicate
        except ImportError as exc:
            self.notes.append(f"LLM adjudication unavailable: {exc}")
            return
        band = [p for p in papers if p.decision == "review"
                or (p.decision == "accept" and p.final_score - self.cfg.auto_add_threshold < 0.1)]
        band = sorted(band, key=lambda p: -p.final_score)[: int(self.cfg.get("llm.max_calls_per_run", 40))]
        for p in band:
            verdict = adjudicate(p, self.cfg, req)
            if not verdict:
                continue
            p.classification["llm"] = verdict
            # Precision first: the LLM may only demote, never promote.
            if verdict["decision"] == "reject":
                p.decision, p.reject_reason = "reject", "LLM adjudication: " + verdict.get("reason", "")[:200]
            elif verdict["decision"] == "review" and p.decision == "accept":
                p.decision, p.reject_reason = "review", "LLM adjudication: " + verdict.get("reason", "")[:200]

    # ================================================================ ingestion
    def ingest(self, result: RunResult) -> None:
        req = result.request
        if req.action == "report_only":
            result.notes.append("Report-only request - Zotero not modified.")
            return
        to_write = [p for p in result.papers if p.decision in ("accept", "review")]
        if not self.zotero or not self.collections:
            return
        names = collection_names(self.cfg)
        if self.dry_run:
            for p in to_write:
                for path in target_collections(p, p.decision, req.source_tag, names, label=req.label, profile=self.cfg.profile):
                    self.collections.ensure(path)
            return
        created: List[Dict[str, Any]] = []
        payloads, notes_for = [], []
        try:
            self.collections.ensure("")
            for p in to_write:
                paths = target_collections(p, p.decision, req.source_tag, names, label=req.label, profile=self.cfg.profile)
                keys = [k for k in (self.collections.ensure(path) for path in paths) if k]
                tags = zitems.build_tags(p, p.decision, req.source_tag, float(self.cfg.get("ranking.high_priority_tag_score", 0.9)))
                tags += [t for t in req.extra_tags + p.extra_tags if t not in tags]
                payloads.append(zitems.to_zotero_item(p, keys, tags))
                notes_for.append(p)
        except (HttpError, ZoteroError) as exc:
            result.errors.append(f"Zotero collection setup failed, nothing written: {exc}")
            return
        # Write in chunks of 50 and record audit + provenance per chunk, so a failure part-way through never
        # leaves created items without their audit trail.
        for start in range(0, len(payloads), 50):
            chunk, chunk_papers = payloads[start:start + 50], notes_for[start:start + 50]
            try:
                new_keys = self.zotero.create_items(chunk)
            except (HttpError, ZoteroError) as exc:
                result.errors.append(f"Zotero create failed for items {start + 1}-{start + len(chunk)} of {len(payloads)} "
                                     f"(this chunk may be partially written; remaining items not sent): {exc}")
                break
            note_payloads = []
            for p, key, payload in zip(chunk_papers, new_keys, chunk):
                if not key:
                    result.errors.append(f"Zotero rejected: {p.title}")
                    continue
                p.existing_zotero_key = key
                result.written[p.decision].append(key)
                created.append({**payload, "key": key})
                note_payloads.append(zitems.note_item(key, zitems.provenance_note(p)))
                self.index.add(zitems.record_from_item({**payload, "key": key}))
                self._audit("create", p, key, req)
            if note_payloads:
                try:
                    self.zotero.create_items(note_payloads)
                except (HttpError, ZoteroError) as exc:
                    result.errors.append(f"Provenance notes failed for items {start + 1}-{start + len(chunk)} (audit log has them): {exc}")
        if self.cfg.get("zotero.attach_open_access_pdf", False):
            for p in notes_for:
                if p.existing_zotero_key:
                    try:
                        attachments.attach(self.zotero, self.http, p.existing_zotero_key, p, self.cfg.get("zotero.pdf_mode", "link"))
                    except (HttpError, ZoteroError, KeyError) as exc:
                        result.errors.append(f"PDF attach failed for {p.title[:60]}: {exc}")
        LibraryCache(self.zotero).remember(created)
        if self.cfg.get("safety.upgrade_preprints", True):
            for p in result.upgrades:
                try:
                    self._upgrade_preprint(p, result)
                except (HttpError, ZoteroError) as exc:
                    result.errors.append(f"preprint annotation failed for {p.related_preprint_key}: {exc}")

    def _upgrade_preprint(self, p: Paper, result: RunResult) -> None:
        """Conservative: annotate the existing preprint (Extra line, tag, note). Never replaces fields."""
        key = p.related_preprint_key
        try:
            item = self.zotero.item(key)
        except HttpError as exc:
            result.errors.append(f"could not load preprint {key}: {exc}")
            return
        data = item["data"]
        extra = data.get("extra") or ""
        if p.doi and p.doi in extra.lower():
            return  # already annotated
        new_extra = (extra + "\n" if extra else "") + f"Published DOI: {p.doi}\nPublished venue: {p.venue} {p.year or ''}".rstrip()
        tags = data.get("tags", []) + [{"tag": "status:published-version-available"}]
        if self.zotero.patch_item(key, item["version"], {"extra": new_extra, "tags": tags}):
            self.zotero.create_items([zitems.note_item(key, upgrade_note(p) or "")])
            self._audit("annotate_preprint_upgrade", p, key, result.request)
            result.written.setdefault("upgrade", []).append(key)

    def _audit(self, action: str, p: Paper, key: str, req: SearchRequest) -> None:
        entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), "action": action, "zotero_key": key,
                 "title": p.title, "doi": p.doi, "decision": p.decision, "final_score": p.final_score,
                 "relevance_score": p.relevance_score, "sources": p.sources, "queries": p.queries[:10],
                 "reason": p.discovery_reason, "request": req.to_dict()}
        with AUDIT_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    # ================================================================ runs
    def run(self, req: SearchRequest, date_from: Optional[str] = None, date_to: Optional[str] = None,
            raw: Optional[List[Paper]] = None, skip_seen: bool = False, seen_path: Optional[Path] = None) -> RunResult:
        """seen_path: separate seen-set file (each alert keeps its own); defaults to the scheduled-scan set."""
        self.load_library()
        result = RunResult(req, date_from, date_to, dry_run=self.dry_run)
        log.info("Retrieving candidates (%s, %s -> %s)", req.intent, date_from or "-", date_to or "now")
        raw = raw if raw is not None else self.retrieve(req, date_from, date_to)
        result.retrieved = len(raw)
        if skip_seen:
            seen = _load_seen(seen_path)
            before = len(raw)
            raw = [p for p in raw if not (_seen_keys(p) & seen)]
            result.previously_seen = before - len(raw)
        window = (None, None) if req.intent in ("similar", "citations", "related", "missing_lit", "add_doi", "baselines") else \
            (date_from, date_to if req.intent != "scan" else None)
        candidates, existing, upgrades = self.process(raw, req, window=window)
        result.papers, result.existing, result.upgrades = candidates, existing, upgrades
        result.after_filter = sum(1 for p in candidates + existing + upgrades if not p.classification.get("hard_rejected"))
        result.author_resolution = getattr(self, "_author_resolution", [])
        self.ingest(result)
        result.notes = [f"Prompt: {n}" for n in req.parse_notes] + self.notes + result.notes
        result.errors += self.ctx.errors
        if self.collections and self.collections.planned:
            result.notes.append("Collections that would be created: " + ", ".join(self.collections.planned))
        save_proposals(result)
        if skip_seen and not self.dry_run and req.action != "report_only":
            # Only papers this run actually settled: rejected, written, or already in the library. Accepted /
            # review papers whose write failed stay unseen so the next scan retries them.
            settled = [p for p in candidates if p.decision == "reject" or p.existing_zotero_key] + existing + upgrades
            _save_seen(set().union(*(_seen_keys(p) for p in settled)) if settled else set(), seen_path)
        return result

    def scan(self, days: Optional[int] = None, now: Optional[date] = None, action: str = "add_by_threshold") -> RunResult:
        if not self.cfg.is_radiology:
            result = RunResult(SearchRequest(intent="scan", action=action), None, None, dry_run=self.dry_run)
            result.notes.append("The broad scan is part of the radiology profile. In the general profile, scheduled discovery "
                                "runs through alerts: `python -m src.cli alerts run-due`.")
            return result
        now = now or date.today()
        state = load_state()
        last = state.get("last_successful_scan")
        overlap = int(self.cfg.get("scan.overlap_days", 10))
        min_lb = int(self.cfg.get("scan.min_lookback_days", 20))
        if days:
            start = now - timedelta(days=days)
        elif last:
            start = min(date.fromisoformat(last[:10]) - timedelta(days=overlap), now - timedelta(days=min_lb))
        else:
            start = now - timedelta(days=365 * int(self.cfg.get("scan.first_run_years", 3)))
        req = SearchRequest(intent="scan", source_tag="auto-discovery", strict_request_match=False, action=action)
        result = self.run(req, start.isoformat(), now.isoformat(), skip_seen=bool(last) and not days)
        succeeded = result.retrieved > 0 or not result.errors
        writing = not self.dry_run and action != "report_only"
        # An ad-hoc --days window only advances the coverage watermark if it reaches back to it (minus overlap).
        covers_gap = (not last) or (not days) or start <= date.fromisoformat(last[:10]) - timedelta(days=overlap)
        if writing and succeeded and covers_gap:
            state["last_successful_scan"] = now.isoformat()
        elif writing and succeeded:
            state["last_adhoc_scan"] = now.isoformat()
            result.notes.append(f"Ad-hoc {days}-day scan did not reach the last scheduled scan ({last[:10]}); "
                                "the scheduled-scan watermark was left unchanged so the gap is still covered next time.")
        elif not writing:
            state["last_dry_run_scan"] = now.isoformat()
        save_state(state)
        return result

    # ------------------------------------------------------------- single paper
    def fetch_identifier(self, doi: Optional[str] = None, arxiv_id: Optional[str] = None, pmid: Optional[str] = None,
                         title: Optional[str] = None) -> Optional[Paper]:
        """Resolve one paper from trusted sources and merge their metadata."""
        doi, arxiv_id, pmid = normalize_doi(doi), normalize_arxiv_id(arxiv_id), normalize_pmid(pmid)
        found: List[Paper] = []

        def add(source: str, fn: Callable[[], Any]) -> None:
            # One failing source must not prevent the others from supplying metadata.
            try:
                got = fn()
            except (HttpError, ET.ParseError, ValueError, KeyError) as exc:
                self.ctx.warn(source, f"lookup failed: {exc}")
                return
            for p in (got if isinstance(got, list) else [got]):
                if p:
                    found.append(p)

        if doi:
            add("crossref", lambda: crossref.get_doi(self.ctx, doi))
            add("openalex", lambda: openalex.get_work(self.ctx, doi=doi))
            add("semantic_scholar", lambda: s2.get_paper(self.ctx, f"DOI:{doi}"))
        if arxiv_id:
            add("arxiv", lambda: arxiv.get(self.ctx, arxiv_id))
            add("semantic_scholar", lambda: s2.get_paper(self.ctx, f"ARXIV:{arxiv_id}"))
            add("openalex", lambda: openalex.get_work(self.ctx, doi=f"10.48550/arxiv.{arxiv_id}"))
        if pmid:
            add("pubmed", lambda: pubmed.fetch(self.ctx, [pmid]))
            add("semantic_scholar", lambda: s2.get_paper(self.ctx, f"PMID:{pmid}"))
        if title and not found:
            hit: List[Paper] = []
            add("semantic_scholar", lambda: hit.append(s2.search_title(self.ctx, title)))
            p = hit[0] if hit else None
            if p and normalize_title(p.title) and (_similar(p.title, title) or _short_name_match(p.title, title)):
                found.append(p)
                if p.doi:
                    add("crossref", lambda: crossref.get_doi(self.ctx, p.doi))
        if not found:
            return None
        # Crossref is canonical for bibliographic fields when present.
        found.sort(key=lambda x: {"crossref": 0, "pubmed": 1, "openalex": 2, "semantic_scholar": 3, "arxiv": 4}.get(x.source, 5))
        base = found[0]
        for other in found[1:]:
            merge(base, other)
        if not base.is_peer_reviewed and base.title and self.cfg.source_enabled("crossref"):
            # Look for the peer-reviewed version of a preprint.
            try:
                hits = crossref.find_by_title(self.ctx, base.title, rows=3)
            except HttpError:
                hits = []
            for got in hits:
                if got.venue_type in ("journal", "conference") and same_work_title(got.title, base.title):
                    merge(base, got)
                    break
        return finalize(base)

    def add_identifier(self, **ids) -> RunResult:
        """Manual add (`cli add --doi`): user-requested, so thresholds decide placement, not admission."""
        req = SearchRequest(intent="add_doi", source_tag="manual-prompt", strict_request_match=False)
        self.load_library()
        result = RunResult(req, None, None, dry_run=self.dry_run)
        p = self.fetch_identifier(**ids)
        if not p:
            result.errors.append(f"No trusted metadata found for {ids}")
            return result
        C.classify(p)
        R.score_paper(p, self.cfg, req)
        R.decide(p, self.cfg, req)
        if p.decision == "reject":
            p.decision, p.reject_reason = "review", f"manually requested (would be rejected: {p.reject_reason})"
        elif p.decision == "review" and p.reject_reason == "":
            p.reject_reason = "manual add below auto-add threshold"
        if p.decision == "review" and p.topics and p.modalities:
            p.decision = "accept"  # explicit user request: file it in its topic collections
        p.discovery_reason = R.explain(p)
        resolve_against_library(p, self.index)
        if p.existing_zotero_key and not p.related_preprint_key:
            result.existing = [p]
        elif p.related_preprint_key:
            result.upgrades = [p]
        else:
            result.papers = [p]
        result.retrieved = 1
        result.after_filter = 0 if p.classification.get("hard_rejected") else 1
        self.ingest(result)
        result.notes = self.notes + result.notes
        result.errors += self.ctx.errors
        save_proposals(result)
        return result

    # ------------------------------------------------------------- citation graph
    def resolve_seed(self, seed: str) -> Optional[Paper]:
        """Seed may be a DOI, arXiv id, Zotero item key, or title."""
        if normalize_doi(seed):
            return self.fetch_identifier(doi=seed)
        if normalize_arxiv_id(seed) and len(seed) < 40:
            return self.fetch_identifier(arxiv_id=seed)
        if self.zotero and len(seed) == 8 and seed.isalnum() and seed.isupper():
            try:
                data = self.zotero.item(seed)["data"]
            except HttpError:
                data = None
            if data:
                rec = zitems.record_from_item(data)
                if rec:
                    return self.fetch_identifier(doi=rec.ids.get("doi"), arxiv_id=rec.ids.get("arxiv_id"),
                                                 pmid=rec.ids.get("pmid"), title=rec.title)
        return self.fetch_identifier(title=seed)

    def expand(self, req: SearchRequest, kinds=("references", "citations", "recommendations", "related")) -> RunResult:
        seed = self.resolve_seed(req.seed or "")
        if not seed:
            self.load_library()
            r = RunResult(req, None, None, dry_run=self.dry_run)
            r.errors.append(f"Could not resolve seed paper: {req.seed}")
            return r
        C.classify(seed)  # relationship labels compare against the seed's own topics
        s2_id = seed.semantic_scholar_id or (f"DOI:{seed.doi}" if seed.doi else f"ARXIV:{seed.arxiv_id}" if seed.arxiv_id else None)
        tasks: List[Tuple[str, Callable[[], List[Paper]]]] = []
        if s2_id:
            if "references" in kinds:
                tasks.append(("semantic_scholar:references", lambda: _tag(s2.references(self.ctx, s2_id), "reference")))
            if "citations" in kinds:
                tasks.append(("semantic_scholar:citations", lambda: _tag(s2.citations(self.ctx, s2_id, 500), "citing")))
            if "recommendations" in kinds:
                tasks.append(("semantic_scholar:recommendations", lambda: _tag(s2.recommendations(self.ctx, s2_id), "similar")))
        oa = seed.openalex_id
        if not oa and seed.doi:
            w = openalex.get_work(self.ctx, doi=seed.doi)
            oa = w.openalex_id if w else None
        if oa and "related" in kinds:
            tasks.append(("openalex:related", lambda: _tag(openalex.related_works(self.ctx, oa), "similar")))
        if oa and "references" in kinds and seed.references:
            tasks.append(("openalex:references", lambda: _tag(openalex.works_by_ids(self.ctx, seed.references[:200]), "reference")))
        raw = self._run_tasks(tasks)
        seed_key = normalize_title(seed.title)
        raw = [p for p in raw if normalize_title(p.title) != seed_key]
        req.strict_request_match = bool(req.topics or req.modalities)
        result = self.run(req, None, None, raw=raw)
        result.seed = seed
        for p in result.papers + result.existing:
            p.relationship = classify_relationship(p, seed)
            p.discovery_reason = R.explain(p)
        save_proposals(result)
        return result

    def baselines(self, req: SearchRequest) -> RunResult:
        """The seed paper's baselines: references it compares against (see src/processing/baselines.py)."""
        from src import fulltext
        from src.llm import get_llm
        from src.processing import baselines as B
        seed = self.resolve_seed(req.seed or "")
        if not seed:
            self.load_library()
            r = RunResult(req, None, None, dry_run=self.dry_run)
            r.errors.append(f"Could not resolve the paper: {req.seed!r} (give a DOI, arXiv id, Zotero key or exact title)")
            return r
        C.classify(seed)
        notes: List[str] = []
        # Try every identifier Semantic Scholar might know the paper by (venue DOIs are sometimes missing).
        pids = [x for x in (seed.semantic_scholar_id, f"ARXIV:{seed.arxiv_id}" if seed.arxiv_id else None,
                            f"DOI:{seed.doi}" if seed.doi else None, f"PMID:{seed.pmid}" if seed.pmid else None) if x]
        edges = []
        for pid in pids:
            edges = s2.reference_edges(self.ctx, pid)
            if edges:
                break
        if not edges and seed.doi:
            w = openalex.get_work(self.ctx, doi=seed.doi)
            ids = (w.references if w else [])[:300]
            edges = [(p, {}) for p in openalex.works_by_ids(self.ctx, ids)] if ids else []
            if edges:
                notes.append("Semantic Scholar has no citation contexts for this paper; used the OpenAlex reference list "
                             "(ranking relies on full text / topic overlap only).")
        if not edges:
            self.load_library()
            r = RunResult(req, None, None, dry_run=self.dry_run)
            r.seed = seed
            r.errors.append("No reference list is available for this paper from Semantic Scholar or OpenAlex.")
            return r
        refs = [p for p, _ in edges]
        # Section-structured open-access full text: where each reference is cited.
        doc = fulltext.fetch_document(self.ctx, seed) if self.cfg.get("fulltext.enabled", True) else None
        signals, bib_map, roles, llm_read, model = {}, {}, {}, False, ""
        if doc:
            bib_map = B.map_bib_to_refs(doc, refs)
            signals = fulltext.citation_signals(doc)
            notes.append(f"Full text: {doc.source} ({doc.url}); sections used: "
                         f"{', '.join(s.title for s in doc.baseline_sections())}; {len(bib_map)}/{len(doc.bib)} bibliography "
                         f"entries matched to trusted reference records.")
            llm = get_llm(self.cfg, "baselines")
            if llm:
                roles, llm_notes = B.llm_roles(doc, llm, int(self.cfg.get("fulltext.max_chars", 80000)))
                llm_read, model = bool(roles), llm.model
                notes += llm_notes + [llm.usage_note()]
        else:
            has_contexts = any((meta or {}).get("contexts") for _, meta in edges)
            notes.append("No open-access section-structured full text (arXiv HTML / PMC); " +
                         ("ranked from Semantic Scholar citation contexts only." if has_contexts else
                          "no citation contexts either, so baselines can't be identified reliably - none are proposed. "
                          f"`python -m src.cli citations --seed {req.seed}` lists the full reference list."))
        if doc:
            # Baselines cited in the paper but missing from the trusted reference list: look them up by title.
            wanted = [b for b in doc.bib if b not in bib_map and ((roles.get(b) or ("",))[0] == "baseline"
                      or (signals.get(b) or {}).get("baseline_section") or (signals.get(b) or {}).get("tables"))]
            for bid in wanted[:15]:
                found = self._lookup_title(doc.bib_titles.get(bid) or "")
                if found:
                    bib_map[bid] = len(edges)
                    edges.append((found, {}))
        by_ref = {i: bid for bid, i in bib_map.items()}
        tag = B.seed_tag(seed)
        raw: List[Paper] = []
        counts: Dict[str, int] = {}
        for i, (ref, meta) in enumerate(edges):
            bid = by_ref.get(i)
            det = B.score_reference(ref, meta, seed, signals.get(bid) if bid else None)
            ev = B.combine(det, roles.get(bid) if bid else None, llm_read, model)
            counts[ev["label"]] = counts.get(ev["label"], 0) + 1
            if ev["label"] not in ("likely baseline", "possible baseline"):
                continue
            ref.graph_evidence = ev
            ref.relationship = f"{ev['label']} ({ev['score']:.2f}): \"{ev['evidence']}\" [{ev['evidence_source']}]"
            ref.extra_tags = ["relation:baseline", f"baseline-of:{tag}"]
            ref.queries.append(f"baselines-of:{tag}")
            raw.append(ref)
        if req.include_seed:
            seed.graph_evidence = {"label": "seed", "score": 1.0}
            seed.relationship = "requested paper"
            seed.extra_tags = [f"baseline-seed:{tag}"]
            raw.append(seed)
        notes.append("Reference roles: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
        result = self.run(req, None, None, raw=raw)
        result.seed = seed
        result.notes = notes + result.notes
        save_proposals(result)
        return result

    def _lookup_title(self, title: str) -> Optional[Paper]:
        """Trusted metadata for a bibliography title (Semantic Scholar match, then OpenAlex search)."""
        if len(normalize_title(title)) < 12:
            return None
        try:
            p = s2.search_title(self.ctx, title)
        except HttpError:
            p = None
        if p and same_work_title(p.title, title, 0.9):
            return p
        try:
            hits = openalex.search(self.ctx, BoolQuery([[title.replace('"', "")]], "title"), limit=3)
        except HttpError:
            hits = []
        return next((h for h in hits if same_work_title(h.title, title, 0.9)), None)

    def missing_literature(self, req: SearchRequest, min_cocitations: int = 2) -> RunResult:
        """Papers frequently referenced by a Zotero collection but absent from the library."""
        self.load_library()
        if not self.zotero:
            r = RunResult(req, None, None)
            r.errors.append("Missing-literature analysis needs Zotero credentials.")
            return r
        cols = CollectionMap(self.zotero, self.cfg.get("zotero.root_collection", "Literature")).load()
        key, problem = resolve_collection(cols, req.collection, req.topics, req.modalities)
        if not key:
            r = RunResult(req, None, None)
            r.errors.append(problem)
            return r
        req.collection = next((path for path, k in cols.paths.items() if k == key), req.collection)
        members = [zitems.record_from_item(it["data"]) for it in self.zotero.collection_items(key)]
        ids = [("DOI:" + m.ids["doi"]) if m.ids.get("doi") else ("ARXIV:" + m.ids["arxiv_id"]) if m.ids.get("arxiv_id") else None
               for m in members if m]
        counts: Dict[str, int] = {}
        pool: Dict[str, Paper] = {}
        for pid in [i for i in ids if i][:100]:
            for ref in s2.references(self.ctx, pid, 500):
                k = ref.doi or ref.arxiv_id or normalize_title(ref.title)
                counts[k] = counts.get(k, 0) + 1
                pool.setdefault(k, ref)
        raw = []
        for k, p in pool.items():
            if counts[k] >= min_cocitations:
                p.relationship = f"cited by {counts[k]} papers in '{req.collection}'"
                p.queries.append(f"cocitation:{counts[k]}")
                raw.append(p)
        req.strict_request_match = bool(req.topics or req.modalities)
        result = self.run(req, None, None, raw=raw)  # co-citation bonus is applied inside process(), before decisions
        result.notes.append(f"Analyzed {len(members)} items in '{req.collection}'; {len(raw)} references cited >= {min_cocitations} times.")
        save_proposals(result)
        return result


# ==================================================================== helpers

def _tag(papers: List[Paper], rel: str) -> List[Paper]:
    for p in papers:
        p.relationship = rel
        p.queries.append(f"graph:{rel}")
    return papers


def classify_relationship(p: Paper, seed: Paper) -> str:
    """Relationship to a seed paper: foundational predecessor / direct competitor / follow-up / benchmark / reliability extension."""
    base = p.relationship or ""
    if "type:benchmark-dataset" in p.subtopics:
        return "benchmark/dataset"
    if base == "citing" and "uncertainty" in p.topics and "uncertainty" not in (seed.topics or []):
        return "uncertainty/reliability extension"
    if base == "reference":
        return "foundational predecessor" if (p.year or 0) < (seed.year or 9999) - 1 or (p.citation_count or 0) > 200 else "direct competitor"
    if base == "citing":
        return "methodological follow-up"
    if base == "similar":
        return "direct competitor" if abs((p.year or 0) - (seed.year or 0)) <= 1 else "related work"
    return base or "related"


def _short_name_match(title: str, query: str) -> bool:
    """'MedCLIP' matches 'MedCLIP: Contrastive Learning from Unpaired Medical Images and Text'."""
    q = normalize_title(query)
    head = normalize_title(title.split(":")[0])
    return bool(q) and len(q) >= 3 and (head == q or (len(q.split()) >= 3 and normalize_title(title).startswith(q)))


def _similar(a: str, b: str) -> bool:
    from src.processing.deduplicate import title_similarity
    return title_similarity(a, b) >= 0.9


def load_state() -> Dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, ValueError):
        return {}


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


_SEEN_PATH = CACHE_DIR / "seen.json"


def _seen_keys(p: Paper) -> set:
    """One key per identifier, so a raw record matches whichever identifier the merged record was saved under."""
    ids = [p.doi, p.arxiv_id, p.pmid]
    if p.doi and p.doi.startswith("10.48550/arxiv."):
        ids.append(normalize_arxiv_id(p.doi))
    vals = [v for v in ids if v] or [normalize_title(p.title)]
    return {hashlib.sha1(v.encode()).hexdigest()[:16] for v in vals if v}


def _outside_window(p: Paper, date_from: Optional[str], date_to: Optional[str]) -> Optional[str]:
    """Reason string if the paper's publication date / year lies outside [date_from, date_to]."""
    if not date_from and not date_to:
        return None
    d = (p.publication_date or "")[:10]
    if len(d) == 10:
        if date_from and d < date_from[:10]:
            return f"published {d}, before requested window ({date_from[:10]})"
        if date_to and d > date_to[:10]:
            return f"published {d}, after requested window ({date_to[:10]})"
        return None
    if p.year:
        if date_from and p.year < int(date_from[:4]):
            return f"published {p.year}, before requested window ({date_from[:10]})"
        if date_to and p.year > int(date_to[:4]):
            return f"published {p.year}, after requested window ({date_to[:10]})"
    return None


def _load_seen(path: Optional[Path] = None) -> set:
    try:
        return set(json.loads((path or _SEEN_PATH).read_text()))
    except (OSError, ValueError):
        return set()


def _save_seen(keys: set, path: Optional[Path] = None) -> None:
    path = path or _SEEN_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(_load_seen(path) | keys)))


def save_proposals(result: RunResult) -> None:
    data = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "dry_run": result.dry_run,
        "request": result.request.to_dict(),
        "window": [result.date_from, result.date_to],
        "papers": [p.to_dict() for p in result.by_decision("accept") + result.by_decision("review")],
    }
    PROPOSALS_PATH.write_text(json.dumps(data, indent=1, default=str))


def load_proposals() -> Dict[str, Any]:
    try:
        return json.loads(PROPOSALS_PATH.read_text())
    except (OSError, ValueError):
        return {}
