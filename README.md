# zotero-tool — automated literature discovery for Zotero

Keeps a Zotero library up to date with the papers you care about, in **any field**:

1. It searches scholarly APIs: OpenAlex, Semantic Scholar, PubMed, arXiv, Crossref, PMLR and OpenReview.
2. It merges each paper's records across those sources and removes duplicates against your library.
3. It ranks every candidate and files the strongest into Zotero, with tags and a note explaining why it was added.

Saved **alerts** re-run each search on a schedule. A **baselines** command pulls in the methods a paper compares
against.

```
retrieve → normalize → deduplicate → rank → conservative Zotero ingestion → report
```

It is built for precision rather than bulk. Papers are added automatically only when they clearly match
and come from a venue you trust; everything else plausible lands in a review collection.

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                                   # fill in ZOTERO_LIBRARY_ID and ZOTERO_API_KEY
sh scripts/install_hooks.sh                            # pre-commit hook that blocks committing secrets
.venv/bin/python -m src.cli check                      # verify the key, read the library, dedup self-test
.venv/bin/python -m src.cli search -q "graph neural networks for drug discovery" --since 2025 --dry-run
```

To get a Zotero key, go to <https://www.zotero.org/settings/keys> and create one with library read and
write access. The numeric *userID* shown on that page is your `ZOTERO_LIBRARY_ID`.

### What you need

| Service | Needed? | Cost | What it's for |
|---|---|---|---|
| **Zotero API key** | Required to read and write your library. Dry runs work without it, but can't check what you already have. | Free with any Zotero account | Dedup, adding papers, collections |
| OpenAlex, PubMed, arXiv, Crossref, PMLR | No key | Free | Discovery and metadata |
| Semantic Scholar key | Optional, recommended | Free | Higher rate limits |
| OpenReview login | Optional | Free account | ICLR / NeurIPS papers |
| `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` | Optional | Pay-per-use | The optional LLM steps (see below) |

Nothing depends on a paid subscription. Only open-access PDFs are downloaded. Every item keeps its DOI,
so Zotero desktop's "Find Available PDF" can use your institution's access.

### Safety

* Zotero is **never modified** unless `safety.writes_enabled: true` is set in `config.yaml` (default `false`) **and** the command runs without `--dry-run`.
* Nothing in Zotero is ever deleted. The only change made to an existing item is annotating a preprint when its published version appears.
* Secrets live only in `.env`, which git ignores. `scripts/check_secrets.py` refuses any commit containing a `.env` value or something that looks like a key.

---

## Profiles: any field, or the built-in radiology ontology

Set `profile` in `config.yaml`:

| | `general` (default) | `radiology` |
|---|---|---|
| Works for | any field | chest X-ray, CT and mammography papers on foundation models, vision-language models and uncertainty |
| What you search with | a query, as free text or explicit `AND` / `OR` groups | built-in topics and modalities, with synonyms |
| What counts as relevant | the paper mentions every part of the query; title hits weigh more than abstract hits | an ontology classifier (phrase-aware, avoids false positives such as "CT" in unrelated text) |
| Where accepted papers go | a collection named after the alert (or *Auto Discovery - Accepted*) | a hierarchy: *Foundation Models / VLMs / Uncertainty & Reliability*, each split by CXR / CT / Mammography |
| Scheduled discovery | alerts | alerts, plus a broad `scan` over every modality and theme |

Query syntax in the general profile:

```text
graph neural networks for drug discovery          every significant word must appear
"protein language model" antibody                 quoted phrases stay together
(GNN OR "graph neural network") AND "drug discovery"
```

Terms the built-in vocabulary knows expand into their synonyms automatically. For example, "VLM" also matches
"vision-language" and "report generation", and "mammography" also matches "mammogram" and "breast imaging".

---

## Alerts — saved searches on their own schedule

Alerts start empty. Each one is a saved search that runs **independently**, with its own cadence (**every 30
days** by default), coverage window, report and last-run status. One alert failing never stops the others.

### Example: a radiology alert

```bash
python -m src.cli alerts create --query "mammography foundation models since 2024"
```

```text
Created alert:
mammography-foundation-models  [active]  Mammography foundation models
    every 30 days | next run <now> | last run -
    topic_search: topics: foundation-model | modalities: mammo | keywords: mammography foundation models | since 2024-01-01
```

What happens next depends on the profile:

* **Radiology profile:** the query compiles into built-in topics and modalities. Accepted papers are filed under *Foundation Models / Mammography* and tagged `alert:mammography-foundation-models`.
* **General profile:** the same alert matches the query's terms and their synonyms (mammography / mammogram / breast imaging; foundation model / self-supervised / pretraining). Accepted papers go into a *Mammography foundation models* collection.

Preview it before anything is written:

```bash
python -m src.cli alerts run mammography --dry-run
```

### Managing alerts

```bash
python -m src.cli alerts create --query "graph neural networks for drug discovery" --every 2w
python -m src.cli alerts create "CXR VLMs" --topic vlm --modality cxr                     # radiology profile flags
python -m src.cli alerts create --query "papers by Jane Doe on protein language models" --openalex-id A5023888391
python -m src.cli alerts list                        # --status active|paused
python -m src.cli alerts show mammography            # one alert + last result + report path
python -m src.cli alerts edit mammography --every 14 # cadence, --query, --name, --topic, --venue, --since, ...
python -m src.cli alerts pause mammography
python -m src.cli alerts resume mammography
python -m src.cli alerts run mammography --dry-run   # run now, even if paused or not due
python -m src.cli alerts run-due                     # run every active alert that is due (the scheduler calls this)
python -m src.cli alerts delete mammography          # asks for confirmation; papers already added stay in Zotero
```

You can refer to an alert by its id, its name, or a unique part of either. For cadence, use `30`, `30d`, `2w`,
`1m`, `weekly`, `monthly` and so on, anywhere from 1 to 366 days.

The same operations work in plain language:

```bash
python -m src.cli ask "Create an alert for mammography foundation models every 2 weeks"
python -m src.cli ask "List my alerts"
python -m src.cli ask "Pause the mammography alert"
python -m src.cli ask "Change the mammography alert to every 2 weeks"
python -m src.cli ask "Run the mammography alert now" --dry-run
python -m src.cli ask "Delete the 'mammography foundation models' alert"
```

The same operations from Python:

```python
from src.config import load_config
from src.alerts import (create_alert, list_alerts, get_alert, update_alert, pause_alert,
                        resume_alert, delete_alert, run_alert_now, run_due_alerts)

cfg = load_config()
a = create_alert(query="mammography foundation models since 2024", cfg=cfg)   # every 30 days by default
update_alert(a.id, interval_days="2w", cfg=cfg)
alert, result = run_alert_now(a.id, cfg, dry_run=True)
delete_alert(a.id)
```

How alert runs are scheduled:

* **Storage:** alerts are kept in `alerts.yaml`, which is git-ignored because it's personal.
* **Windows:** the first run backfills from `since`, or `alerts.first_run_years` (3) back. Later runs start at the last writing run minus a 10-day overlap, never less than 20 days back.
* **Timing:**
  * Scheduled runs set the next run to now + the cadence.
  * A failed run is retried the next day.
  * A manual `run` resets the timer only if it wrote to Zotero.
  * Dry runs never advance coverage.
* **Edits:** changing *what* an alert searches for makes the next run backfill again.
* **Limits:** the same thresholds, dedup and per-run auto-add limit (25) apply as for every other search.

---

## Baselines of a paper

This finds the methods a paper compares against, and optionally adds them (and the paper itself):

```bash
python -m src.cli baselines --seed 2210.10163 --dry-run          # DOI, arXiv id, Zotero item key or title
python -m src.cli ask "Fetch 10.1148/ryai.240646 and get its baselines papers too"
python -m src.cli ask "What are the baselines of MedCLIP?"       # "what are / show" = report only
```

How it decides:

1. **Trusted reference list.** Candidates come from Semantic Scholar (or OpenAlex), with the sentences citing each reference.
2. **Open-access full text.** When arXiv HTML or PubMed Central has the paper, only its **Related Work and Experiments sections** are read, including sub-sections and results tables.
   * Every citation is resolved to its bibliography entry.
   * Method names written without a citation are linked through aliases learned elsewhere in the paper. For example, a caption citing "ConVIRT Zhang et al. (2020)" links a later bare "ConVIRT" in the Baselines list.
3. **Deterministic scoring.** Strong signals are comparison language ("outperforms", "compared with"), a "Baselines" sub-section and results tables. Datasets, backbones and tools are separated out.
4. **Optional LLM pass** (`llm.tasks.baselines`). The model sees only those sections, about 10–15k tokens per paper. It labels each reference *baseline / dataset / backbone / background* and must quote verbatim evidence; unverifiable answers are discarded.

What happens to each label:

| Label | Result |
|---|---|
| *likely baseline* | Added |
| *possible baseline* | Review collection |
| Conflict between the rules and the model | Review collection |
| Everything else | Rejected |

Baselines are tagged `relation:baseline` and `baseline-of:<firstauthorYEAR-word>`. If a paper has neither open
full text nor citation contexts, nothing is proposed.

---

## Other commands

| Task | Example |
|---|---|
| Search | `python -m src.cli search -q "CRISPR off-target prediction" --since 2024` |
| Venue search | `python -m src.cli venue --venue neurips --year 2025 -q "diffusion models for proteins"` |
| Author search | `python -m src.cli author --name "Jane Doe" -q "protein language models" --since 2023` |
| Author watchlist | `python -m src.cli watch` (everyone in `authors.yaml`; give each a `query:`) |
| Add a known paper | `python -m src.cli add --doi 10.xxxx/xxxxx` (or `--arxiv`, `--pmid`, `--title`) |
| Similar / references / related | `python -m src.cli related --seed <DOI · arXiv id · Zotero key · title>` (`similar`, `citations`) |
| Missing literature | `python -m src.cli gaps --collection "Graph neural networks for drug discovery"` |
| Natural language | `python -m src.cli ask "fetch me recent papers on protein language models since 2025"` |
| Last run's proposals | `python -m src.cli review` (`--from-zotero` lists the review collection) |
| Broad scan (radiology profile) | `python -m src.cli scan` (`--dry-run`, `--days 20`, `--if-due`) |
| Validation | `python -m src.cli evaluate --gold data/gold/gold.jsonl` |

`--dry-run` previews any command. `--report-only` never writes, even with writes enabled.

**Filing into your own collections.** Pass `--collection` to `search`, `venue`, `author` or `baselines`:

```bash
python -m src.cli search -q '("chest x-ray" OR CXR) AND ("vision-language" OR "report generation") AND (longitudinal OR "prior study" OR temporal)' \
       --collection "CXR/Longitudinal VLMs (prior studies)"
```

* Accepted papers go into that collection, a path from your library's top level such as `Mammo` or `CXR/Longitudinal`. It is created if missing.
* Matching papers you **already keep** are filed there too. That includes a preprint whose published version was just found.
* Borderline papers still go to the review collection, and items waiting for review are never promoted this way.

**One-off profile.** `--profile radiology` (or `general`) overrides `config.yaml` for a single command, e.g.
`python -m src.cli --profile radiology search --topic "foundation models" --modality mammo --since 2025 --collection Mammo`.

**Query tip.** A bare word like "prior" also matches phrases such as "prior work" and "prior knowledge". To mean
prior *studies*, use phrases: `"prior study"`, `"prior report"`, `"previous exam"`, `longitudinal`, `temporal`.

**How prompts are understood.** A deterministic parser turns the prompt into a search request:

* It extracts the subject ("papers on **X** since 2024"), plus author, venue, dates and intent.
* **Misspelled domain words are corrected**, and the assumption is printed, e.g. `note: interpreted 'mammograpghy' as 'mammography'`.
* A domain-looking word it can't read stops the run instead of silently widening the search.
* Names, quoted titles and identifiers are never "corrected".
* Only when confidence is low does the optional LLM fallback (`llm.tasks.parse_fallback`) interpret the prompt. Names and identifiers it returns are kept only if they literally appear in your prompt.

---

## Configuration

| File | Purpose |
|---|---|
| `.env` | Secrets only: `ZOTERO_*`, optional `OPENREVIEW_USERNAME/PASSWORD`, `SEMANTIC_SCHOLAR_API_KEY`, `OPENALEX_API_KEY`, `NCBI_API_KEY`, `CONTACT_EMAIL`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`. Git-ignored. |
| `config.yaml` | `profile`, thresholds, safety switches, alert defaults (`alerts.default_interval_days: 30`), venue weights, `high_priority_venues`, PMLR volumes, OpenReview venues, LLM settings. |
| `alerts.yaml` | Your alerts (created by `alerts create`; git-ignored). |
| `authors.yaml` | Author watchlist. Pin `openalex_id` / `semantic_scholar_id` for exact matching. |

**Set `high_priority_venues` for your field.** Only papers from those venues are auto-added; everything else
that matches goes to review. The default list covers ML, computer-vision and medical-imaging venues.

**LLM (optional).** Configured under `llm:` in `config.yaml`:

* `provider: gemini` with `model: latest-flash-lite` resolves to the newest stable `gemini-*-flash-lite` at run time. `provider: anthropic` uses Claude instead.
* Each task is switched separately: `baselines`, `parse_fallback`, and `relevance` (radiology profile only).
* The LLM never supplies bibliographic metadata, and its outputs are validated.
* Every report shows the model used and its token count.

## Outputs

* `reports/YYYY-MM-DD[-kind].md` — one report per run. It lists counts, the papers added and in review with reasons and scores, near misses, rejection reasons and source warnings.
* `data/proposals.json` — what the last run would add (shown by `review`).
* `data/audit.jsonl` — one line per Zotero write: source, query, timestamp, scores, reason.
* On each added Zotero item — tags (`source:*`, `status:*`, `venue:*`, `alert:<id>`, and in the radiology profile `modality:*` / `topic:*`) and a provenance note.

## How ranking works

* **Relevance:**
  * General profile: query coverage (title hits count more than abstract hits) and whether every term appears, plus author watch and recency.
  * Radiology profile: the topic, modality, keyword, author-watch and recency components.
* **Quality:** venue prior, peer review, age-normalized citations, methodological signals and metadata completeness.
* **Final score** = 0.7 × relevance + 0.3 × quality.
* **Decisions:**
  * ≥ 0.82 is auto-added only if the venue is high-priority and peer reviewed.
  * ≥ 0.68 goes to *Auto Discovery - Review*; anything lower is rejected.
  * Editorials, errata, retractions and meeting abstracts are filtered out.
  * A search with no query terms never auto-adds.
* **Dedup:** matches in this order: DOI → PMID → arXiv id → OpenAlex / S2 id → normalized title → fuzzy title (≥ 0.96).
  * It checks both other candidates and your library, which is cached locally and refreshed incrementally.
  * Records with different publisher DOIs or arXiv ids are never merged, and neither are titles differing only in "Part I/II", "2D/3D" or a year.
* **Preprint → published:** a preprint already in your library gets an Extra line, a `status:published-version-available` tag and a note. No duplicate is created and no field is overwritten.

## Scheduling

`scripts/run_scan.sh` is meant to run daily. It runs `alerts run-due` and the author watchlist; in the radiology
profile it also runs the broad `scan --if-due` (every 10 days). Each part decides for itself whether it's due.

On macOS, first set this repository's path inside `scripts/zotero-tool-scan.plist` (the file explains how),
then install it:

```bash
cp scripts/zotero-tool-scan.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/zotero-tool-scan.plist
```

## Validation

Label a gold set in `data/gold/gold.jsonl`; `data/gold/gold.example.jsonl` shows the format. Then run
`evaluate`, which reports precision@auto-add, recall@review, the false-positive rate and the metadata error
rate.

## Project layout

```
src/cli.py            command-line interface (python -m src.cli ...)
src/pipeline.py       retrieve → enrich → classify → dedup → rank → ingest → report
src/alerts.py         saved alerts: storage, scheduling, CRUD, plain-language alert commands
src/fulltext.py       arXiv HTML / PMC section extraction, citation resolution, method-name aliases
src/llm.py            optional LLM providers (Gemini REST, Anthropic SDK) behind one JSON-output call
src/config.py, models.py, http.py, evaluate.py
src/discovery/        openalex, semantic_scholar, pubmed, crossref, arxiv, openreview, pmlr, query
src/processing/       normalize, classify (radiology ontology), rank, deduplicate, version_resolution, baselines
src/zotero/           client, collections, items, attachments
src/prompts/          parser (prompt → search request, typo tolerance), relevance (optional LLM steps)
src/reports/          generate_report
scripts/              run_scan.sh, launchd plist, check_secrets.py, install_hooks.sh
tests/                unit tests + fake-Zotero integration tests: .venv/bin/python -m pytest -q
```

## Known limitations

* OpenReview blocks anonymous API access; set `OPENREVIEW_USERNAME` / `OPENREVIEW_PASSWORD` to include ICLR / NeurIPS.
* Semantic Scholar without an API key shares a public rate limit (429s). Runs carry on and list the warnings in the report.
* The general profile matches the words you write. Add synonyms with `OR`, e.g. `(LLM OR "large language model")`.
* New PMLR volumes must be added to `pmlr_volumes` in `config.yaml`.
* API keys are redacted from every error message, log line and report.
