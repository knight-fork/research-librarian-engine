"""Collection hierarchy: create-or-reuse, never delete."""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from src.models import Paper
from src.zotero.client import ZoteroClient

log = logging.getLogger("zotero_tool.zotero")

MODALITY_CHILDREN = ["CXR", "CT", "Mammography", "General Radiology"]
HIERARCHY: Dict[str, List[str]] = {
    "Foundation Models": MODALITY_CHILDREN,
    "Vision-Language Models": MODALITY_CHILDREN,
    "Uncertainty & Reliability": ["Calibration", "Conformal Prediction", "OOD / Shift", "Selective Prediction", "VLM Reliability"],
    "Robustness & Evaluation": [],
    "Author Watches": [],
    "Auto Discovery - Review": [],
    "Auto Discovery - Accepted": [],
}
MODALITY_NAME = {"cxr": "CXR", "ct": "CT", "mammo": "Mammography", "radiology-general": "General Radiology"}


DEFAULT_NAMES = {"review": "Auto Discovery - Review", "accepted": "Auto Discovery - Accepted", "watch": "Author Watches"}


def collection_names(cfg) -> Dict[str, str]:
    """Names of the review / accepted / author-watch collections (configurable in config.yaml)."""
    return {
        "review": cfg.get("zotero.auto_review_collection", DEFAULT_NAMES["review"]),
        "accepted": cfg.get("zotero.auto_accept_collection", DEFAULT_NAMES["accepted"]),
        "watch": cfg.get("zotero.author_watch_collection", DEFAULT_NAMES["watch"]),
    }


def hierarchy(names: Optional[Dict[str, str]] = None, profile: str = "radiology") -> Dict[str, List[str]]:
    names = names or DEFAULT_NAMES
    if profile != "radiology":
        return {names["watch"]: [], names["review"]: [], names["accepted"]: []}
    h = {k: v for k, v in HIERARCHY.items() if k not in DEFAULT_NAMES.values()}
    for role in ("watch", "review", "accepted"):
        h[names[role]] = []
    return h


class CollectionMap:
    """Resolves paths like 'Foundation Models/CXR' to collection keys under the root collection."""

    def __init__(self, client: Optional[ZoteroClient], root: str = "Literature"):
        self.client = client
        self.root = root
        self.paths: Dict[str, str] = {}   # "<root>/Foundation Models/CXR" -> key
        self.by_key: Dict[str, Tuple[str, Optional[str]]] = {}
        self.planned: List[str] = []      # paths that would be created in dry-run

    def load(self) -> "CollectionMap":
        if not self.client:
            return self
        cols = self.client.collections()
        by_key = {c["key"]: c["data"] for c in cols}

        def path_of(key: str) -> str:
            d = by_key[key]
            parent = d.get("parentCollection")
            return (path_of(parent) + "/" if parent and parent in by_key else "") + d["name"]

        trashed: Dict[str, bool] = {}

        def is_trashed(key: str, depth: int = 0) -> bool:
            # /collections also returns collections in the trash (deleted: true); their subtrees are trashed too.
            if key not in trashed:
                d = by_key[key]
                parent = d.get("parentCollection")
                trashed[key] = bool(d.get("deleted")) or bool(parent and parent in by_key and depth < 50 and is_trashed(parent, depth + 1))
            return trashed[key]

        for key in by_key:
            if not is_trashed(key):
                self.paths.setdefault(path_of(key), key)
        self.by_key = {k: (path, None) for path, k in self.paths.items()}
        return self

    def ensure(self, relpath: str) -> Optional[str]:
        """Return the key for root/relpath, creating missing levels. In dry-run every missing level is
        recorded in `planned` (in creation order) and None is returned."""
        parts = [self.root] + split_path(relpath)
        parent: Optional[str] = None
        path = ""
        missing = False
        for name in parts:
            path = f"{path}/{name}" if path else name
            if not missing and path in self.paths:
                parent = self.paths[path]
                continue
            already_planned = path in self.planned  # dry-run: don't ask the client again for a level already planned
            key = None if (missing or already_planned) else (self.client.create_collection(name, parent) if self.client else None)
            if key is None:
                missing = True
                if path not in self.planned:
                    self.planned.append(path)
                continue
            log.info("created collection %s", path)
            self.paths[path] = key
            parent = key
        return None if missing else parent

    def ensure_hierarchy(self, names: Optional[Dict[str, str]] = None, profile: str = "radiology") -> None:
        self.ensure("")
        for top, children in hierarchy(names, profile).items():
            self.ensure(top)
            for c in children:
                self.ensure(f"{top}/{c}")


def split_path(relpath: str) -> List[str]:
    """Split on '/' path separators but not on ' / ' inside names like 'OOD / Shift'."""
    return [p for p in re.split(r"(?<! )/(?! )", relpath or "") if p]


TOPIC_BRANCH = {"foundation-model": "Foundation Models", "vlm": "Vision-Language Models",
                "uncertainty": "Uncertainty & Reliability", "robustness": "Robustness & Evaluation"}


def resolve_collection(cols: CollectionMap, arg: Optional[str], topics=(), modalities=()) -> Tuple[Optional[str], str]:
    """Find a collection by full path or trailing path segments (never raw substring). Returns (key, error)."""
    arg = (arg or "").strip().strip("/")
    if not arg:
        return None, "Collection not found: a collection name is required (e.g. 'Foundation Models/Mammography')."
    want = [x.lower() for x in split_path(arg)]
    root = cols.root.lower()

    def segs(path: str) -> List[str]:
        return [x.lower() for x in split_path(path)]

    exact = [k for path, k in cols.paths.items() if segs(path) in (want, [root] + want)]
    if len(exact) == 1:
        return exact[0], ""
    cands = [(path, k) for path, k in cols.paths.items() if segs(path)[-len(want):] == want]
    if not cands:
        # Free text such as "mammography foundation models": map topics / modalities onto the hierarchy.
        mods = [MODALITY_NAME[m] for m in modalities if m in MODALITY_NAME]
        branches = [TOPIC_BRANCH[t] for t in topics if t in TOPIC_BRANCH]
        guesses = [f"{b}/{m}" for b in branches for m in mods] or branches
        cands = [(path, k) for path, k in cols.paths.items() for g in guesses if segs(path) == [root] + segs(g)]
    under_root = [(path, k) for path, k in cands if segs(path)[:1] == [root]]
    cands = under_root or cands
    if len(cands) == 1:
        return cands[0][1], ""
    if not cands:
        return None, f"Collection not found: {arg}"
    return None, f"Ambiguous collection '{arg}': " + ", ".join(sorted(p for p, _ in cands)) + " - give a longer path."


def target_collections(p: Paper, decision: str, source_tag: str, names: Optional[Dict[str, str]] = None,
                       label: Optional[str] = None, profile: str = "radiology") -> List[str]:
    """Relative collection paths for a paper (topic collections only for accepted items).

    General profile: accepted papers go into a collection named after the alert (or the accepted collection)."""
    names = names or DEFAULT_NAMES
    if profile != "radiology":
        watch = [names["watch"]] if (source_tag == "author-watch" or p.author_watch) else []
        if decision == "review":
            return list(dict.fromkeys([names["review"]] + watch))
        own = label.replace("/", "-").strip() if label else names["accepted"]
        return list(dict.fromkeys([own] + watch))
    if decision == "review":
        paths = [names["review"]]
        if source_tag == "author-watch":
            paths.append(names["watch"])
        return paths
    if not p.modalities:
        # Not a radiology paper (e.g. a general-ML baseline such as CLIP): no modality branch, just tags.
        return list(dict.fromkeys([names["accepted"]] + ([names["watch"]] if source_tag == "author-watch" else [])))
    paths: List[str] = []
    mods = [MODALITY_NAME[m] for m in p.modalities if m in MODALITY_NAME] or ["General Radiology"]
    if "foundation-model" in p.topics or "image-text-pretraining" in p.topics:
        paths += [f"Foundation Models/{m}" for m in mods]
    if "vlm" in p.topics or "report-generation" in p.topics:
        paths += [f"Vision-Language Models/{m}" for m in mods]
    unc = []
    if "calibration" in p.topics:
        unc.append("Calibration")
    if "conformal-prediction" in p.topics:
        unc.append("Conformal Prediction")
    if "ood" in p.topics or ("distribution-shift" in p.topics and "uncertainty" in p.topics):
        unc.append("OOD / Shift")
    if "selective-prediction" in p.topics:
        unc.append("Selective Prediction")
    if "vlm-reliability" in p.topics:
        unc.append("VLM Reliability")
    if unc:
        paths += [f"Uncertainty & Reliability/{u}" for u in unc]
    elif "uncertainty" in p.topics:
        paths.append("Uncertainty & Reliability")
    if "robustness" in p.topics or ("distribution-shift" in p.topics and "uncertainty" not in p.topics):
        paths.append("Robustness & Evaluation")
    if source_tag == "author-watch" or p.author_watch:
        paths.append(names["watch"])
    if source_tag == "auto-discovery":
        paths.append(names["accepted"])
    if not paths:
        paths.append(names["accepted"])
    return list(dict.fromkeys(paths))
