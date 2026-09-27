"""Section-structured open-access full text (arXiv HTML / ar5iv, PubMed Central JATS XML).

Returns a Document with ordered sections whose citations are rewritten as [bibN] markers that resolve to
bibliography entries, and tables wrapped in [TABLE] ... [/TABLE]. Nothing is fetched from paywalled sources."""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.discovery.base import SourceContext
from src.http import HttpError
from src.models import Paper

RELATED_RE = re.compile(r"related|prior work|previous work|background|literature", re.I)
EXPERIMENT_RE = re.compile(r"experiment|result|evaluat|comparison|compar|benchmark|implementation|ablation|performance|"
                           r"quantitative|analysis|setup|study", re.I)
SKIP_RE = re.compile(r"^(abstract|introduction|conclusions?|acknowledg|references|bibliography|appendix|supplement|limitations|"
                     r"broader impact|ethics)", re.I)
MARKER_RE = re.compile(r"\[(bib[\w.:-]+)\]")
BASELINE_SECTION_RE = re.compile(r"baselines?|compared (?:methods?|models?|approaches)|comparison|competing|"
                                 r"state[- ]of[- ]the[- ]art|sota", re.I)
DATASET_SECTION_RE = re.compile(r"datasets?|data\b|benchmarks?|cohorts?", re.I)


@dataclass
class Section:
    title: str
    text: str


@dataclass
class Document:
    source: str                     # arxiv_html | ar5iv | pmc
    url: str
    sections: List[Section] = field(default_factory=list)
    bib: Dict[str, str] = field(default_factory=dict)          # marker id -> bibliography entry text
    bib_titles: Dict[str, str] = field(default_factory=dict)   # marker id -> reference title (when structured)

    def select(self, kind: str) -> List[Section]:
        if kind == "related":
            return [s for s in self.sections if RELATED_RE.search(s.title)]
        if kind == "experiments":
            return [s for s in self.sections if EXPERIMENT_RE.search(s.title) and not RELATED_RE.search(s.title)
                    and not SKIP_RE.match(_strip_number(s.title))]
        return list(self.sections)

    def baseline_sections(self) -> List[Section]:
        """Related Work + Experiments; if headings are unusual, every body section except intro/conclusion."""
        chosen = self.select("related") + self.select("experiments")
        if not self.select("experiments"):
            chosen = [s for s in self.sections if not SKIP_RE.match(_strip_number(s.title))]
        seen, out = set(), []
        for s in chosen:
            if id(s) not in seen:
                seen.add(id(s))
                out.append(s)
        return out


def _strip_number(title: str) -> str:
    return re.sub(r"^[\dIVX.\s]+", "", title or "").strip()


def _text(fragment: str) -> str:
    fragment = re.sub(r"<(script|style)\b.*?</\1>", " ", fragment, flags=re.S | re.I)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(fragment)).strip()


# ---------------------------------------------------------------- arXiv HTML (LaTeXML)

def _parse_latexml(page: str, source: str, url: str) -> Optional[Document]:
    doc = Document(source=source, url=url)
    bib_start = page.find('class="ltx_bibliography"')
    body, bib = (page[:bib_start], page[bib_start:]) if bib_start > 0 else (page, "")
    for m in re.finditer(r'<li id="(bib\.bib\d+)" class="ltx_bibitem">(.*?)</li>', bib, re.S):
        bid = m.group(1).replace("bib.", "")
        blocks = [_text(b) for b in re.findall(r'<span class="ltx_bibblock">(.*?)</span>\s*(?=<span class="ltx_bibblock">|$)', m.group(2), re.S)]
        doc.bib[bid] = _text(m.group(2))
        if len(blocks) >= 2:
            doc.bib_titles[bid] = blocks[1].rstrip(". ")
    # math -> its TeX alttext (short) ; citations -> [bibN] markers ; tables -> [TABLE] blocks
    body = re.sub(r"<math\b[^>]*alttext=\"([^\"]{0,80})\"[^>]*>.*?</math>", lambda m: " " + html.unescape(m.group(1)) + " ", body, flags=re.S)
    body = re.sub(r"<math\b.*?</math>", " ", body, flags=re.S)
    body = re.sub(r'<a href="#(bib\.bib\d+)"[^>]*>(.*?)</a>', lambda m: f" {_text(m.group(2))} [{m.group(1).replace('bib.', '')}] ", body, flags=re.S)
    body = re.sub(r'<figure\b[^>]*class="[^"]*ltx_table[^"]*"[^>]*>(.*?)</figure>', lambda m: f" [TABLE] {_text(m.group(1))} [/TABLE] ", body, flags=re.S)
    starts = [(m.start(), m.group(1)) for m in re.finditer(r'<section id="(S\d+)" class="ltx_section">', body)]
    for i, (pos, _sid) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(body)
        chunk = body[pos:end]
        t = re.search(r"<h2[^>]*>(.*?)</h2>", chunk, re.S)
        title = _text(t.group(1)) if t else ""
        rest = chunk[t.end():] if t else chunk
        subs = [(m.start(), m.end()) for m in re.finditer(r'<section id="S\d+\.SS\d+" class="ltx_subsection">', rest)]
        lead = _text(rest[: subs[0][0]] if subs else rest)
        if lead:
            doc.sections.append(Section(title, lead))
        for j, (a, b) in enumerate(subs):
            sub = rest[a: subs[j + 1][0] if j + 1 < len(subs) else len(rest)]
            h = re.search(r"<h3[^>]*>(.*?)</h3>", sub, re.S)
            sub_title = _text(h.group(1)) if h else ""
            text = _text(sub[h.end():] if h else sub)
            if text:
                doc.sections.append(Section(f"{title} / {sub_title}" if sub_title else title, text))
    return doc if doc.sections else None


def fetch_arxiv(ctx: SourceContext, arxiv_id: str) -> Optional[Document]:
    for source, url in (("arxiv_html", f"https://arxiv.org/html/{arxiv_id}"), ("ar5iv", f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}")):
        try:
            page = ctx.http.get(url, expect="text", use_cache=True, cache_ttl=30 * 86400, max_retries=2)
        except HttpError:
            continue
        if "ltx_section" not in page:
            continue  # no HTML rendering (ar5iv redirects to the abstract page)
        doc = _parse_latexml(page, source, url)
        if doc:
            return doc
    return None


# ---------------------------------------------------------------- PubMed Central (JATS XML via Europe PMC)

def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _jats_text(el: ET.Element) -> str:
    """Flatten a JATS element: <xref ref-type="bibr" rid=".."> -> [bib:rid] markers, tables -> [TABLE] blocks."""
    parts: List[str] = []

    def children(node: ET.Element) -> None:
        for child in node:
            walk(child)
            if child.tail:
                parts.append(child.tail)

    def walk(node: ET.Element) -> None:
        tag = _local(node.tag)
        if tag == "xref" and node.get("ref-type") == "bibr":
            label = "".join(node.itertext()).strip()
            parts.append(f" {label} " + " ".join(f"[bib:{r}]" for r in (node.get("rid") or "").split()) + " ")
        elif tag == "sec" and node is not el:
            return  # nested sections are emitted separately
        elif tag == "table-wrap":
            parts.append(" [TABLE] ")
            children(node)
            parts.append(" [/TABLE] ")
        else:
            if node.text:
                parts.append(node.text)
            children(node)
            if tag in ("title", "p"):
                parts.append(" ")
    walk(el)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def fetch_pmc(ctx: SourceContext, pmcid: str) -> Optional[Document]:
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        xml_text = ctx.http.get(url, expect="text", use_cache=True, cache_ttl=30 * 86400, max_retries=2)
        root = ET.fromstring(xml_text)
    except (HttpError, ET.ParseError):
        return None
    doc = Document(source="pmc", url=url)
    for ref in root.iter():
        if _local(ref.tag) != "ref" or not ref.get("id"):
            continue
        rid = f"bib:{ref.get('id')}"
        doc.bib[rid] = re.sub(r"\s+", " ", " ".join(ref.itertext())).strip()
        title = next((t for t in ref.iter() if t.tag.split("}")[-1] == "article-title"), None)
        if title is not None:
            doc.bib_titles[rid] = re.sub(r"\s+", " ", "".join(title.itertext())).strip()
    body = next((b for b in root.iter() if b.tag.split("}")[-1] == "body"), None)
    if body is None:
        return None
    def emit(sec: ET.Element, parent: str, depth: int) -> None:
        title_el = next((c for c in sec if _local(c.tag) == "title"), None)
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        full = f"{parent} / {title}" if parent and title else (title or parent)
        text = _jats_text(sec)
        if title and text.startswith(title):
            text = text[len(title):].strip()
        if text:
            doc.sections.append(Section(full, text))
        if depth < 3:
            for child in sec:
                if _local(child.tag) == "sec":
                    emit(child, full, depth + 1)

    for sec in [c for c in body if _local(c.tag) == "sec"]:
        emit(sec, "", 1)
    return doc if doc.sections else None


def fetch_document(ctx: SourceContext, p: Paper) -> Optional[Document]:
    """Best available section-structured open-access full text for a paper, or None."""
    if p.arxiv_id:
        doc = fetch_arxiv(ctx, p.arxiv_id)
        if doc:
            return doc
    if p.pmcid:
        return fetch_pmc(ctx, p.pmcid)
    return None


# ---------------------------------------------------------------- citation signals

_ALIAS_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)\s*[(,]?\s*"          # method name: >= 2 capitals (CLIP, GLoRIA, ConVIRT)
    r"[A-Z][\w'\-]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][\w'\-]+)?,?\s*\(?\d{4}[a-z]?\)?\s*\[(bib[\w.:-]+)\]")


def method_aliases(doc: Document) -> Dict[str, str]:
    """Method / dataset names attached to a citation anywhere in the paper ("ConVIRT Zhang et al. (2020) [bib45]"),
    so later name-only mentions (baseline lists, result tables) can be linked to the right reference."""
    seen: Dict[str, Dict[str, int]] = {}
    for s in doc.sections:
        for m in _ALIAS_RE.finditer(s.text):
            name, bid = m.group(1), m.group(2)
            if 3 <= len(name) <= 24 and not name.isdigit():
                seen.setdefault(name, {}).setdefault(bid, 0)
                seen[name][bid] += 1
    # keep unambiguous names only
    return {name: next(iter(b)) for name, b in seen.items() if len(b) == 1}


def citation_signals(doc: Document) -> Dict[str, Dict[str, object]]:
    """Per bibliography id: where it is cited (experiments / tables / related work) with a sample sentence."""
    out: Dict[str, Dict[str, object]] = {}
    exp = {id(s) for s in doc.select("experiments")}
    rel = {id(s) for s in doc.select("related")}
    for s in doc.sections:
        # table spans inside this section
        tables = [(m.start(), m.end()) for m in re.finditer(r"\[TABLE\].*?\[/TABLE\]", s.text, re.S)]
        for m in MARKER_RE.finditer(s.text):
            bid = m.group(1)
            sig = out.setdefault(bid, {"experiments": 0, "tables": 0, "related": 0, "other": 0, "baseline_section": 0,
                                       "dataset_section": 0, "snippets": [], "sites": []})
            in_table = any(a <= m.start() < b for a, b in tables)
            if in_table:
                sig["tables"] += 1
            if id(s) in exp:
                sig["experiments"] += 1
            elif id(s) in rel:
                sig["related"] += 1
            else:
                sig["other"] += 1
            leaf = s.title.split(" / ")[-1]
            if BASELINE_SECTION_RE.search(leaf):
                sig["baseline_section"] += 1
            elif DATASET_SECTION_RE.search(leaf):
                sig["dataset_section"] += 1
            sentence = _sentence_around(s.text, m.start())
            if id(s) in exp and len(sig["snippets"]) < 2:
                sig["snippets"].append(sentence)
            if len(sig["sites"]) < 4 and (s.title, sentence) not in sig["sites"]:
                sig["sites"].append((s.title, sentence))
    # Name-only mentions ("ConVIRT works on ...", a table row "GLoRIA 0.44") in experiments / tables.
    for name, bid in method_aliases(doc).items():
        rx = re.compile(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])")
        for s in doc.sections:
            if id(s) not in exp:
                continue
            tables = [(m.start(), m.end()) for m in re.finditer(r"\[TABLE\].*?\[/TABLE\]", s.text, re.S)]
            for m in rx.finditer(s.text):
                follow = s.text[m.end(): m.end() + 60]
                if f"[{bid}]" in follow:
                    continue  # already counted through its citation marker
                sig = out.setdefault(bid, {"experiments": 0, "tables": 0, "related": 0, "other": 0, "baseline_section": 0,
                                           "dataset_section": 0, "snippets": [], "sites": []})
                sig["experiments"] += 1
                if any(a <= m.start() < b for a, b in tables):
                    sig["tables"] += 1
                leaf = s.title.split(" / ")[-1]
                if BASELINE_SECTION_RE.search(leaf):
                    sig["baseline_section"] += 1
                sentence = _sentence_around(s.text, m.start())
                sig.setdefault("aliases", [])
                if name not in sig["aliases"]:
                    sig["aliases"].append(name)
                if len(sig["sites"]) < 5 and (s.title, sentence) not in sig["sites"]:
                    sig["sites"].append((s.title, f"[named '{name}'] {sentence}"))
                if len(sig["snippets"]) < 2:
                    sig["snippets"].append(sentence)
    return out


_BOUNDARY = re.compile(r"(?<!\bal)(?<!e\.g)(?<!i\.e)(?<!\bFig)(?<!\bTab)(?<!\bvs)(?<!\bEq)[.!?]\s+(?=[A-Z(\[])")


def _sentence_around(text: str, pos: int, width: int = 260) -> str:
    """The sentence containing position `pos` (citation-aware: 'et al.' / 'e.g.' don't end sentences)."""
    starts = [m.end() for m in _BOUNDARY.finditer(text, max(0, pos - width), pos)]
    start = starts[-1] if starts else max(0, pos - width)
    m = _BOUNDARY.search(text, pos)
    end = m.start() + 1 if m and m.start() - pos < width else min(len(text), pos + width)
    return re.sub(r"\s+", " ", MARKER_RE.sub("", text[start:end])).strip()


def normalize_for_match(text: str) -> str:
    text = MARKER_RE.sub(" ", text or "")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
