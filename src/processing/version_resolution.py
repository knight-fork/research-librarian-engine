"""Preprint -> published version resolution."""
from __future__ import annotations

from typing import Optional

from src.models import Paper
from src.processing.deduplicate import DedupIndex, Record
from src.processing.normalize import is_preprint_doi


def is_preprint_record(rec: Record) -> bool:
    doi = rec.ids.get("doi")
    if rec.item_type == "preprint" or is_preprint_doi(doi):
        return True
    return bool(rec.ids.get("arxiv_id")) and not (doi and not is_preprint_doi(doi))


def resolve_against_library(p: Paper, index: DedupIndex) -> Paper:
    """Mark p as duplicate / upgrade of an existing library record.

    Sets p.existing_zotero_key when the paper already exists. If the existing record is a
    preprint and p is the peer-reviewed version, also sets p.related_preprint_key so the
    caller can conservatively link / annotate the existing item instead of adding a new one."""
    key, method = index.find_paper(p)
    if not key:
        return p
    rec = index.records[key]
    p.existing_zotero_key = key
    p.duplicate_of = method
    if p.is_peer_reviewed and is_preprint_record(rec):
        # The published paper carries a publisher DOI the preprint item doesn't have (it has none, or a
        # preprint-server DOI such as arXiv / medRxiv / Research Square) -> upgrade candidate.
        rec_doi = rec.ids.get("doi")
        if p.doi and not is_preprint_doi(p.doi) and (not rec_doi or rec_doi.lower() != p.doi.lower()):
            p.related_preprint_key = key
    return p


def preferred_version(a: Paper, b: Paper) -> Paper:
    """Prefer the peer-reviewed / final version."""
    rank = {"journal": 3, "conference": 3, "workshop": 2, "preprint": 1}
    ra, rb = rank.get(a.venue_type, 0), rank.get(b.venue_type, 0)
    if ra != rb:
        return a if ra > rb else b
    return a if (a.year or 0) >= (b.year or 0) else b


def upgrade_note(p: Paper) -> Optional[str]:
    if not p.related_preprint_key:
        return None
    return (f"<p><b>Published version available</b> (detected by research-librarian)</p>"
            f"<p>{p.title}<br/>Venue: {p.venue} {p.year or ''}<br/>DOI: "
            f"<a href=\"https://doi.org/{p.doi}\">{p.doi}</a></p>")
