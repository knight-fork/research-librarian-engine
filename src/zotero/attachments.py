"""Conservative, legal-only PDF attachment. Never bypasses paywalls."""
from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.http import HttpClient, HttpError
from src.models import Paper
from src.zotero.client import ZoteroClient

log = logging.getLogger("zotero_tool.zotero")

# Priority: official OA -> arXiv -> PubMed Central -> proceedings.
PRIORITY = ("official_oa", "arxiv", "pmc", "proceedings")


def choose_pdf(p: Paper) -> Optional[Tuple[str, str]]:
    for kind in PRIORITY:
        url = p.pdf_candidates.get(kind)
        if url:
            return kind, url
    return None


def _attachment_item(parent: str, url: str, kind: str, link_mode: str) -> Dict:
    title = {"official_oa": "Full Text PDF (open access)", "arxiv": "arXiv PDF", "pmc": "PubMed Central PDF",
             "proceedings": "Proceedings PDF"}[kind]
    item = {"itemType": "attachment", "parentItem": parent, "linkMode": link_mode, "title": title, "url": url,
            "contentType": "application/pdf", "tags": [], "relations": {}}
    if link_mode == "imported_url":
        item.update({"filename": _filename(url), "charset": ""})
    return item


def _filename(url: str) -> str:
    base = re.sub(r"[^\w.\-]", "_", url.rstrip("/").rsplit("/", 1)[-1]) or "paper"
    return base if base.lower().endswith(".pdf") else base + ".pdf"


def attach(client: ZoteroClient, http: HttpClient, parent_key: str, p: Paper, mode: str = "link") -> Optional[str]:
    """Attach an open-access PDF. mode 'link' stores a linked URL; 'upload' downloads and stores the file."""
    choice = choose_pdf(p)
    if not choice:
        return None
    kind, url = choice
    if mode != "upload":
        keys = client.create_items([_attachment_item(parent_key, url, kind, "linked_url")])
        return keys[0]
    try:
        data = http.get(url, expect="bytes", max_retries=2)
    except HttpError as exc:
        log.info("PDF download failed (%s), falling back to link: %s", url, exc)
        return client.create_items([_attachment_item(parent_key, url, kind, "linked_url")])[0]
    if not data.startswith(b"%PDF"):
        log.info("URL did not return a PDF (%s); linking instead", url)
        return client.create_items([_attachment_item(parent_key, url, kind, "linked_url")])[0]
    att_key = client.create_items([_attachment_item(parent_key, url, kind, "imported_url")])[0]
    if not att_key:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / _filename(url)
        path.write_bytes(data)
        client.upload_attachment_file(att_key, path, hashlib.md5(data).hexdigest(), int(path.stat().st_mtime * 1000))
    return att_key
