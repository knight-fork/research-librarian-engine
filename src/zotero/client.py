"""Minimal Zotero Web API v3 client (read, create, conservative update; never deletes)."""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src import config as _config
from src.config import Secrets
from src.http import HttpClient, HttpError

log = logging.getLogger("zotero_tool.zotero")
API = "https://api.zotero.org"


class ZoteroError(RuntimeError):
    pass


class ZoteroClient:
    def __init__(self, secrets: Secrets, http: Optional[HttpClient] = None, dry_run: bool = True):
        if not secrets.zotero_configured:
            raise ZoteroError("ZOTERO_LIBRARY_ID and ZOTERO_API_KEY must be set (see .env.example)")
        lt = secrets.zotero_library_type
        if lt not in ("user", "group"):
            raise ZoteroError("ZOTERO_LIBRARY_TYPE must be 'user' or 'group'")
        self.prefix = f"{API}/{lt}s/{secrets.zotero_library_id}"
        self.api_key = secrets.zotero_api_key
        self.http = http or HttpClient(timeout=60)
        self.dry_run = dry_run
        self.library_version: Optional[int] = None

    # ------------------------------------------------------------------ basics
    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        h = {"Zotero-API-Key": self.api_key, "Zotero-API-Version": "3"}
        if extra:
            h.update(extra)
        return h

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None):
        resp = self.http.request("GET", self.prefix + path, params=params, headers=self._headers(), expect="response")
        lv = resp.headers.get("Last-Modified-Version")
        if lv:
            self.library_version = int(lv)
        return resp

    def _get_all(self, path: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = dict(params or {})
        params.setdefault("limit", 100)
        start, out = 0, []
        while True:
            params["start"] = start
            resp = self._get(path, params)
            batch = resp.json() if resp.content else []
            out.extend(batch)
            total = int(resp.headers.get("Total-Results", len(out)))
            start += len(batch)
            if not batch or start >= total:
                return out

    def key_info(self) -> Dict[str, Any]:
        """Verify the API key and its permissions."""
        return self.http.get(f"{API}/keys/current", headers=self._headers())

    # ------------------------------------------------------------------ reads
    def collections(self) -> List[Dict[str, Any]]:
        return self._get_all("/collections")

    def live_items_top(self) -> List[Dict[str, Any]]:
        """Top-level items excluding the trash (use this for organizing / listing)."""
        return [it for it in self.items_top() if not it["data"].get("deleted")]

    def items_top(self, since: Optional[int] = None) -> List[Dict[str, Any]]:
        # includeTrashed so incremental syncs see items moved to the trash (data.deleted) and drop them.
        params: Dict[str, Any] = {"include": "data", "includeTrashed": 1}
        if since is not None:
            params["since"] = since
        return self._get_all("/items/top", params)

    def item(self, key: str) -> Dict[str, Any]:
        """One item (may be in the trash: check data['deleted'] before editing it)."""
        return self._get(f"/items/{key}").json()

    def children(self, key: str) -> List[Dict[str, Any]]:
        return self._get_all(f"/items/{key}/children")

    def collection_items(self, collection_key: str) -> List[Dict[str, Any]]:
        return self._get_all(f"/collections/{collection_key}/items/top", {"include": "data"})

    def deleted_since(self, since: int) -> Dict[str, Any]:
        return self._get("/deleted", {"since": since}).json()

    # ------------------------------------------------------------------ writes
    def _write_guard(self, what: str) -> bool:
        if self.dry_run:
            log.info("[dry-run] would %s", what)
            return False
        return True

    def create_items(self, items: List[Dict[str, Any]]) -> List[Optional[str]]:
        """Create up to 50 items per request. Returns the new keys (None for failures)."""
        keys: List[Optional[str]] = []
        for i in range(0, len(items), 50):
            chunk = items[i:i + 50]
            if not self._write_guard(f"create {len(chunk)} item(s)"):
                keys.extend([None] * len(chunk))
                continue
            res = self.http.request("POST", self.prefix + "/items", json_body=chunk,
                                    headers=self._headers({"Zotero-Write-Token": uuid.uuid4().hex, "Content-Type": "application/json"}))
            for idx in range(len(chunk)):
                ok = (res.get("successful") or {}).get(str(idx))
                if ok:
                    keys.append(ok.get("key") or ok.get("data", {}).get("key"))
                else:
                    fail = (res.get("failed") or {}).get(str(idx))
                    log.error("Zotero rejected item %d: %s", idx, fail)
                    keys.append(None)
        return keys

    def create_collection(self, name: str, parent: Optional[str] = None) -> Optional[str]:
        body = [{"name": name, "parentCollection": parent or False}]
        if not self._write_guard(f"create collection {name!r}"):
            return None
        res = self.http.request("POST", self.prefix + "/collections", json_body=body,
                                headers=self._headers({"Zotero-Write-Token": uuid.uuid4().hex, "Content-Type": "application/json"}))
        ok = (res.get("successful") or {}).get("0")
        if not ok:
            raise ZoteroError(f"collection create failed: {res.get('failed')}")
        return ok.get("key")

    def patch_item(self, key: str, version: int, data: Dict[str, Any]) -> bool:
        """Partial update guarded by the item version (fails on concurrent edits)."""
        if not self._write_guard(f"update item {key} fields {sorted(data)}"):
            return False
        self.http.request("PATCH", f"{self.prefix}/items/{key}", json_body=data,
                          headers=self._headers({"If-Unmodified-Since-Version": str(version), "Content-Type": "application/json"}),
                          ok_statuses=(204,))
        return True

    # ------------------------------------------------------------------ file upload
    def upload_attachment_file(self, attachment_key: str, path: Path, md5: str, mtime_ms: int) -> bool:
        if not self._write_guard(f"upload file {path.name} to {attachment_key}"):
            return False
        size = path.stat().st_size
        auth = self.http.request("POST", f"{self.prefix}/items/{attachment_key}/file",
                                 data={"md5": md5, "filename": path.name, "filesize": size, "mtime": mtime_ms},
                                 headers=self._headers({"If-None-Match": "*", "Content-Type": "application/x-www-form-urlencoded"}))
        if auth.get("exists"):
            return True
        body = auth["prefix"].encode() + path.read_bytes() + auth["suffix"].encode()
        self.http.request("POST", auth["url"], data=body, headers={"Content-Type": auth["contentType"]}, expect="response",
                          ok_statuses=(200, 201, 204))
        self.http.request("POST", f"{self.prefix}/items/{attachment_key}/file", data={"upload": auth["uploadKey"]},
                          headers=self._headers({"If-None-Match": "*", "Content-Type": "application/x-www-form-urlencoded"}),
                          expect="response", ok_statuses=(204,))
        return True


class LibraryCache:
    """Local cache of top-level item data, refreshed incrementally with ?since=version."""

    def __init__(self, client: ZoteroClient, path: Optional[Path] = None):
        self.client = client
        self.path = path or _config.CACHE_DIR / "zotero_library.json"

    def load(self, refresh: bool = True) -> List[Dict[str, Any]]:
        cached: Dict[str, Any] = {"version": None, "prefix": None, "items": {}}
        if self.path.exists():
            try:
                cached = json.loads(self.path.read_text())
            except ValueError:
                pass
        if cached.get("prefix") != self.client.prefix:
            cached = {"version": None, "prefix": self.client.prefix, "items": {}}
        if refresh:
            since = cached.get("version")
            try:
                changed = self.client.items_top(since=since)
                for it in changed:
                    if it["data"].get("deleted"):
                        cached["items"].pop(it["key"], None)
                    else:
                        cached["items"][it["key"]] = it["data"]
                if since is not None:
                    for key in (self.client.deleted_since(since).get("items") or []):
                        cached["items"].pop(key, None)
                cached["version"] = self.client.library_version
                self.path.write_text(json.dumps(cached))
            except HttpError as exc:
                if not cached["items"]:
                    raise
                log.warning("Zotero refresh failed, using cached library: %s", exc)
        return list(cached["items"].values())

    def remember(self, items: Iterable[Dict[str, Any]]) -> None:
        """Add freshly created items so later dedup within this run sees them."""
        if not self.path.exists():
            return
        cached = json.loads(self.path.read_text())
        for it in items:
            if it.get("key"):
                cached["items"][it["key"]] = it
        self.path.write_text(json.dumps(cached))
