"""Shared HTTP layer: polite per-host rate limiting, retries with backoff, and a disk cache."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

warnings.filterwarnings("ignore", message=".*OpenSSL.*")
import requests  # noqa: E402

log = logging.getLogger(__name__)

USER_AGENT = "research-librarian/0.1 (personal literature curator)"

# Minimum seconds between requests to a host.
HOST_INTERVALS = {
    "export.arxiv.org": 3.1,
    "api.semanticscholar.org": 1.1,
    "eutils.ncbi.nlm.nih.gov": 0.35,
    "api.openalex.org": 0.12,
    "api.crossref.org": 0.1,
    "proceedings.mlr.press": 0.3,
    "api2.openreview.net": 0.3,
}


_SECRET_RE = re.compile(r"((?:api_key|apikey|api-key|key|token|access_token|password|email|mailto)=)[^&\s#'\"]+", re.I)


def redact(text: object) -> str:
    """Remove credentials (api_key=..., token=..., mailto=...) from URLs / messages before they are logged or reported."""
    return _SECRET_RE.sub(r"\1***", str(text or ""))


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        url, body = redact(url), redact(body)
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.status = status
        self.url = url
        self.body = body


class HttpClient:
    def __init__(self, cache_dir: Optional[Path] = None, cache_ttl_hours: float = 24, timeout: float = 30,
                 contact_email: Optional[str] = None):
        self.session = requests.Session()
        ua = USER_AGENT + (f" mailto:{contact_email}" if contact_email else "")
        self.session.headers["User-Agent"] = ua
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl_hours * 3600
        self.timeout = timeout
        self._last: Dict[str, float] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._glock = threading.Lock()
        if cache_dir:
            (cache_dir / "http").mkdir(parents=True, exist_ok=True)

    def _host_lock(self, host: str) -> threading.Lock:
        with self._glock:
            return self._locks.setdefault(host, threading.Lock())

    def _throttle(self, host: str, interval: Optional[float] = None) -> None:
        interval = HOST_INTERVALS.get(host, 0.2) if interval is None else interval
        wait = self._last.get(host, 0) + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.monotonic()

    def _cache_path(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / "http" / (hashlib.sha1(key.encode()).hexdigest() + ".json")

    def request(self, method: str, url: str, *, params: Optional[Dict[str, Any]] = None,
                headers: Optional[Dict[str, str]] = None, json_body: Any = None, data: Any = None,
                use_cache: bool = False, cache_ttl: Optional[float] = None, max_retries: int = 5,
                expect: str = "json", ok_statuses=(200, 201, 204)) -> Any:
        """Perform a request. expect: json | text | bytes | response."""
        cache_key = f"{method} {url} {json.dumps(params, sort_keys=True, default=str)}"
        cpath = self._cache_path(cache_key) if use_cache and method == "GET" else None
        ttl = self.cache_ttl if cache_ttl is None else cache_ttl
        if cpath and cpath.exists() and time.time() - cpath.stat().st_mtime < ttl:
            try:
                cached = json.loads(cpath.read_text())
                return cached["body"]
            except (ValueError, KeyError):
                pass

        host = urlparse(url).netloc
        backoff = 2.0
        for attempt in range(max_retries + 1):
            with self._host_lock(host):
                self._throttle(host)
                try:
                    resp = self.session.request(method, url, params=params, headers=headers, json=json_body,
                                                data=data, timeout=self.timeout)
                except requests.RequestException as exc:
                    if attempt >= max_retries:
                        raise HttpError(0, url, str(exc)) from None
                    log.debug("network error %s (attempt %d): %s", redact(url), attempt, redact(exc))
                    time.sleep(backoff)
                    backoff *= 2
                    continue
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                retry_after = resp.headers.get("Retry-After") or resp.headers.get("Backoff")
                try:
                    delay = float(retry_after) if retry_after else backoff
                except ValueError:
                    delay = backoff
                log.debug("HTTP %s from %s, retrying in %.1fs", resp.status_code, host, delay)
                time.sleep(min(delay, 60))
                backoff *= 2
                continue
            # Zotero asks clients to pause after a Backoff header even on success.
            if resp.headers.get("Backoff"):
                try:
                    self._last[host] = time.monotonic() + float(resp.headers["Backoff"])
                except ValueError:
                    pass
            if resp.status_code not in ok_statuses:
                raise HttpError(resp.status_code, resp.url, resp.text)
            if expect == "response":
                return resp
            if expect == "bytes":
                return resp.content
            body = resp.text if expect == "text" else (resp.json() if resp.content else None)
            if cpath:
                try:
                    cpath.write_text(json.dumps({"url": url, "body": body}))
                except (TypeError, OSError):
                    pass
            return body
        raise HttpError(0, url, "retries exhausted")

    def get(self, url: str, **kw) -> Any:
        return self.request("GET", url, **kw)
