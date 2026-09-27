"""Typo tolerance, baselines intent, section-structured full text, baseline scoring and LLM verification."""
from datetime import date

import pytest

from src import fulltext as F
from src.models import Paper, SearchRequest
from src.processing import baselines as B
from src.processing import rank as R
from src.prompts.parser import correct_typos, extract_seed, parse
from src.zotero.collections import target_collections

TODAY = date(2026, 9, 27)


# ---------------------------------------------------------------- typo tolerance
@pytest.mark.parametrize("text,modalities,topics,note", [
    ("fetch me papers on mammograpghy based VLMs", ["mammo"], ["vlm"], "mammograpghy"),
    ("find uncertainity papers for chest xray vison-langauge models", ["cxr"], ["vlm", "uncertainty"], "uncertainity"),
    ("mammographic foundation models", ["mammo"], ["foundation-model"], None),
])
def test_typos_corrected_and_reported(text, modalities, topics, note):
    r, conf = parse(text, TODAY)
    assert r.modalities == modalities and set(r.topics) == set(topics) and conf >= 0.6
    if note:
        assert any(note in n for n in r.parse_notes)


def test_unreadable_domain_word_stops_instead_of_widening():
    r, conf = parse("papers on mammoxyz foundation models", TODAY)
    assert conf < 0.6 and any("mammoxyz" in n for n in r.parse_notes)


def test_names_titles_and_ids_never_corrected():
    fixed, notes, _ = correct_typos('papers by Jane Calibra on "Uncertainity in CXR" 10.1148/uncertainity.1')
    assert fixed == 'papers by Jane Calibra on "Uncertainity in CXR" 10.1148/uncertainity.1' and not notes


# ---------------------------------------------------------------- baselines intent
@pytest.mark.parametrize("text,seed,include", [
    ("Fetch 10.1148/ryai.240646 and get its baselines papers too", "10.1148/ryai.240646", True),
    ('Fetch "MedCLIP: Contrastive Learning from Unpaired Medical Images and Text" and get its baseline papers',
     "MedCLIP: Contrastive Learning from Unpaired Medical Images and Text", True),
    ("What are the baselines of MedCLIP?", "MedCLIP", False),
    ("get baselines for arXiv:2210.10163", "2210.10163", False),
    ("Add MedCLIP and its baselines", "MedCLIP", True),
])
def test_baselines_prompts(text, seed, include):
    r, conf = parse(text, TODAY)
    assert r.intent == "baselines" and r.seed == seed and r.include_seed == include and conf >= 0.8


def test_baselines_without_paper_asks():
    r, conf = parse("Fetch this paper and get its baselines papers too", TODAY)
    assert r.intent == "baselines" and r.seed is None and conf < 0.6 and any("which paper" in n for n in r.parse_notes)
    assert extract_seed("baselines of this paper") is None


# ---------------------------------------------------------------- full text
HTML = """<html><body>
<section id="S1" class="ltx_section"><h2 class="ltx_title ltx_title_section">1 Introduction</h2>
<p>Figure 1: Zero-shot performance of ours, ConVIRT <cite><a href="#bib.bib3" class="ltx_ref">Zhang et al. (2020)</a></cite>.</p></section>
<section id="S2" class="ltx_section"><h2 class="ltx_title ltx_title_section">2 Related Work</h2>
<p>Contrastive learning <cite><a href="#bib.bib3" class="ltx_ref">Zhang et al. (2020)</a></cite> and CLIP
<cite><a href="#bib.bib1" class="ltx_ref">Radford et al. (2021)</a></cite> are popular. Adam <cite><a href="#bib.bib2" class="ltx_ref">Kingma (2015)</a></cite>.</p></section>
<section id="S3" class="ltx_section"><h2 class="ltx_title ltx_title_section">3 Experiments</h2><p>We evaluate on CXR.</p>
<section id="S3.SS1" class="ltx_subsection"><h3 class="ltx_title ltx_title_subsection">3.1 Baselines</h3>
<p>CLIP <cite><a href="#bib.bib1" class="ltx_ref">Radford et al. (2021)</a></cite> is a contrastive model. ConVIRT works on medical image-text pairs.</p></section>
<section id="S3.SS2" class="ltx_subsection"><h3 class="ltx_title ltx_title_subsection">3.2 Results</h3>
<figure id="S3.T1" class="ltx_table"><table><tr><td>ConVIRT</td><td>0.41</td></tr><tr><td>CLIP</td><td>0.30</td></tr></table>
<figcaption>Table 1: zero-shot accuracy.</figcaption></figure><p>We train with Adam <cite><a href="#bib.bib2" class="ltx_ref">Kingma (2015)</a></cite>.</p></section>
</section>
<section id="bib" class="ltx_bibliography"><ul>
<li id="bib.bib1" class="ltx_bibitem"><span class="ltx_tag">Radford et al. (2021)</span> <span class="ltx_bibblock">A. Radford. 2021. </span> <span class="ltx_bibblock">Learning transferable visual models from natural language supervision. </span> <span class="ltx_bibblock">In ICML.</span></li>
<li id="bib.bib2" class="ltx_bibitem"><span class="ltx_tag">Kingma (2015)</span> <span class="ltx_bibblock">D. Kingma. 2015. </span> <span class="ltx_bibblock">Adam: a method for stochastic optimization. </span> <span class="ltx_bibblock">In ICLR.</span></li>
<li id="bib.bib3" class="ltx_bibitem"><span class="ltx_tag">Zhang et al. (2020)</span> <span class="ltx_bibblock">Y. Zhang. 2020. </span> <span class="ltx_bibblock">Contrastive learning of medical visual representations from paired images and text. </span> <span class="ltx_bibblock">arXiv.</span></li>
</ul></section></body></html>"""


@pytest.fixture
def doc():
    return F._parse_latexml(HTML, "arxiv_html", "https://arxiv.org/html/x")


def test_sections_subsections_and_bibliography(doc):
    assert [s.title for s in doc.sections] == ["1 Introduction", "2 Related Work", "3 Experiments",
                                              "3 Experiments / 3.1 Baselines", "3 Experiments / 3.2 Results"]
    assert "[TABLE]" in doc.sections[-1].text and doc.bib_titles["bib1"].startswith("Learning transferable")
    assert [s.title for s in doc.baseline_sections()][0] == "2 Related Work"


def test_aliases_link_name_only_mentions(doc):
    assert F.method_aliases(doc) == {"ConVIRT": "bib3", "CLIP": "bib1"}
    sig = F.citation_signals(doc)
    assert sig["bib3"]["baseline_section"] >= 1 and sig["bib3"]["tables"] >= 1   # "ConVIRT works on..." + table row
    assert sig["bib1"]["baseline_section"] >= 1
    assert sig["bib2"]["baseline_section"] == 0 and sig["bib2"]["tables"] == 0    # optimizer: experiments text only


def test_bib_mapping(doc):
    refs = [Paper(title="Adam: A Method for Stochastic Optimization"),
            Paper(title="Learning Transferable Visual Models From Natural Language Supervision"),
            Paper(title="Contrastive Learning of Medical Visual Representations from Paired Images and Text")]
    assert B.map_bib_to_refs(doc, refs) == {"bib2": 0, "bib1": 1, "bib3": 2}


# ---------------------------------------------------------------- scoring
SEED = Paper(title="A vision-language foundation model for chest X-ray", year=2022, topics=["vlm", "foundation-model"])


def test_score_uses_comparison_language_and_sections():
    strong = B.score_reference(Paper(title="GLoRIA: global-local representation learning", year=2021),
                               {"contexts": ["Our method obtains better performances than the state-of-the-art GLoRIA (Huang et al., 2021)."],
                                "influential": True}, SEED)
    background = B.score_reference(Paper(title="VirTex: learning visual representations from textual annotations", year=2020),
                                   {"contexts": ["Vision-text learning was studied (Joulin et al., 2016; Li et al., 2017; "
                                                 "Sariyildiz et al., 2020; Desai and Johnson, 2021; Kim et al., 2021)."]}, SEED)
    dataset = B.score_reference(Paper(title="CheXpert: a large chest radiograph dataset", year=2019),
                                {"contexts": ["CheXpert (Irvin et al., 2019) is a large dataset of chest X-rays."]}, SEED)
    assert strong["label"] == "likely baseline" and "better performances" in strong["evidence"]
    assert background["label"] == "background reference"
    assert dataset["label"] == "benchmark/dataset"
    listed = B.score_reference(Paper(title="Some method", year=2021), {}, SEED, {"baseline_section": 1, "experiments": 1})
    assert listed["label"] == "likely baseline"


class FakeLLM:
    model = "fake-lite"

    def __init__(self, answer):
        self.answer = answer

    def complete_json(self, system, user, schema, max_output_tokens=4096):
        assert "CITATION SITES" in user and "also named in the text: ConVIRT" in user
        return self.answer


def test_llm_roles_are_verified(doc):
    answer = {"references": [
        {"ref_id": "bib3", "role": "baseline", "evidence": "ConVIRT works on medical image-text pairs."},
        {"ref_id": "bib1", "role": "baseline", "evidence": "CLIP is famous for beating every model on earth"},   # not in text
        {"ref_id": "bib9", "role": "baseline", "evidence": "We evaluate on CXR and more words here"},            # unknown id
        {"ref_id": "bib2", "role": "backbone", "evidence": "We train with Adam Kingma (2015)"},
    ]}
    roles, notes = B.llm_roles(doc, FakeLLM(answer))
    assert roles["bib3"][0] == "baseline" and roles["bib2"][0] == "backbone"
    assert "bib1" not in roles and "bib9" not in roles
    assert any("discarded 2" in n for n in notes)


def test_combine_rules():
    det_likely = {"score": 0.9, "label": "likely baseline", "evidence": "x", "evidence_source": "citation context"}
    det_bg = {"score": 0.1, "label": "background reference", "evidence": "y", "evidence_source": "citation context"}
    assert B.combine(det_bg, ("baseline", "quote"), True, "m")["label"] == "likely baseline"
    assert B.combine(det_likely, ("background", "quote"), True, "m")["label"] == "possible baseline"   # conflict -> review
    assert B.combine(det_likely, None, True, "m")["label"] == "possible baseline"                     # LLM didn't name it
    assert B.combine(det_likely, None, False, "m")["label"] == "likely baseline"                      # no LLM read
    assert B.combine(det_bg, ("dataset", "quote"), True, "m")["label"] == "benchmark/dataset"
    det_ds = {"score": 0.6, "label": "benchmark/dataset", "evidence": "z", "evidence_source": "citation context"}
    assert B.combine(det_ds, ("baseline", "quote"), True, "m")["label"] == "possible baseline"      # dataset named as baseline


def test_baseline_decisions_ignore_radiology_filter(cfg):
    req = SearchRequest(intent="baselines", strict_request_match=False)
    clip = Paper(title="Learning transferable visual models from natural language supervision", venue_type="conference")
    clip.graph_evidence = {"label": "likely baseline", "score": 0.9}
    maybe = Paper(title="Representation learning with contrastive predictive coding")
    maybe.graph_evidence = {"label": "possible baseline", "score": 0.4}
    erratum = Paper(title="Correction: some method")
    erratum.graph_evidence = {"label": "likely baseline", "score": 0.9}
    assert [R.decide(p, cfg, req).decision for p in (clip, maybe, erratum)] == ["accept", "review", "reject"]
    assert target_collections(clip, "accept", "manual-prompt") == ["Auto Discovery - Accepted"]  # no radiology branch


def test_seed_tag():
    assert B.seed_tag(Paper(title="MedCLIP: Contrastive Learning", authors=["Zifeng Wang"], year=2022)) == "wang2022-medclip"


# ---------------------------------------------------------------- LLM provider
def test_gemini_resolves_latest_stable_flash_lite(cfg):
    from src.llm import GeminiLLM

    class Http:
        def get(self, url, **kw):
            names = ["gemini-2.5-flash-lite", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash-lite-tts",
                     "gemini-3.1-flash-lite-preview", "gemini-3.5-flash"]
            return {"models": [{"name": f"models/{n}", "supportedGenerationMethods": ["generateContent"]} for n in names]}
    assert GeminiLLM(cfg, "k", http=Http()).model == "gemini-3.5-flash-lite"


def test_llm_parse_drops_invented_entities(cfg, monkeypatch):
    from src.prompts import relevance

    class Fake:
        model = "fake"

        def complete_json(self, *a, **k):
            return {"intent": "author_search", "author": "Geoffrey Hinton", "topics": ["vlm"], "modalities": [], "venues": [],
                    "year_from": None, "year_to": None, "seed": None, "collection": None, "report_only": False}
    monkeypatch.setattr(relevance, "get_llm", lambda cfg, task: Fake())
    req = relevance.llm_parse("papers from that famous deep learning person on vlm", cfg)
    assert req.author is None  # not in the prompt -> not trusted
