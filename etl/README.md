# etl — Ingestion & medallion

Target (Phase 3, deferred to the end of the roadmap): Lakeflow Connect pipelines (**SharePoint / OneDrive** documents → **Unity Catalog**
(`structural_biology`)), bronze → silver → gold, with **PHI scanning on ingest**. Gold docs are
embedded into **Databricks AI Search** (the global governed document index); structured tables
feed **Genie**.

## Today: the `mygpt_etl` job (incremental)
Source PDFs live in `/Volumes/structural_biology/mygpt/docs/<corpus>/`. One top-level folder is
one **corpus** (`biomed`, `health_policies_fr`, `health_policies_es`, `health_policies_hi`,
`kinase_literature`), which the agent filters on via `custom_inputs.corpus`.
`health_policies_hi` (Phase 6) holds public Hindi health-policy PDFs that exercise the
parse fallback and the Hindi config. Upload them with `make corpus-hi HI_PDFS=<dir> PROFILE=mygpt`.

| Task | What |
|---|---|
| `landing` (`sql/01_landing_documents.sql`) | `documents`: one pointer row per file. `id = xxhash64(path)`, so ids are stable across runs |
| `parse_chunk_index` (`sql/02_parse_chunk_index.sql`) | Parses only **new or changed** files with `ai_parse_document` into section-tagged staging elements |
| `parse_fallback` (`parse_fallback.py`, Phase 6) | Flags documents whose text is empty, has 3+ `(cid:N)` markers, has more than 2% U+FFFD/NUL characters, or (Hindi) has no Devanagari. Re-extracts them with PyMuPDF and keeps that text when it's better. Records the check and parser in `doc_parse_quality`. Does nothing when no document is flagged |
| `chunk_merge` (`sql/02b_chunk_merge.sql`) | **MERGEs** ~1000-char section chunks into `doc_chunks` (steps 4–9 of the old `02`, unchanged) |
| `sync_index` (`sync_index.py`) | Syncs the TRIGGERED Delta Sync index `doc_chunks_corpus_index`, waits for it to finish, and checks the indexed row count |
| `sections_and_config` (`sql/03_sections_and_config.sql`) | Per-corpus scoring config tables |

**Incremental semantics**
- A file is re-parsed only when its `(size, modification_time)` differs from its
  `doc_parse_log` row. To force a re-parse, delete that row.
- `doc_chunks` is MERGEd on `chunk_id` (path hash + section hash + bucket). Unchanged chunks
  emit no change-feed rows, so the index only re-embeds what changed.
- Removing a file, or dropping a corpus from the allow-list in `02`, deletes its elements,
  chunks, and log row on the next run.
- To add a corpus: create the volume folder, add it to the allow-list in `02`, and add a
  `dataset_config` row in `03` (plus `etl/config/dataset_config.csv`). For a non-English corpus
  you also need to:
  - add it to `CORPUS_LANGUAGE` in `agents/rag_agent/agent.py` (and `config.py`,
    `parse_fallback.py`);
  - add a corpus library in the app migrations.
- Parse quality: `SELECT corpus, parser, reason, count(*) FROM doc_parse_quality GROUP BY ALL`.
  To force a document through the fallback again, delete its `doc_parse_log` row.

```bash
databricks bundle deploy -p mygpt --var warehouse_id=<id>
databricks bundle run mygpt_etl -p mygpt --var warehouse_id=<id>
```

Index **creation** is a one-time API step; see the tail of `sql/02b_chunk_merge.sql`.

**Gotchas (both hit on 2026-10-12)**
- **Never `CREATE OR REPLACE` `doc_chunks`.** A replace gives the Delta table a new id, which
  invalidates the index's change-feed checkpoint. `sync_index` then fails with
  `DIFFERENT_DELTA_TABLE_READ_BY_STREAMING_SOURCE`, and the only fix is to rebuild the index.
- **After deleting and recreating the index, redeploy the agent** (`mygpt_deploy_agent`). The
  agent endpoint's SELECT grant on the index is provisioned by `agents.deploy`, and a recreated
  index is a new object without it. Until the redeploy, every query returns
  `retrieve_error: PermissionDenied`.

## Phase 4: the `mygpt_upload_ingest` job (upload libraries)
User upload libraries live in Lakebase, not AI Search. The app writes uploads to
`/Volumes/structural_biology/mygpt/uploads/<library_id>/` and triggers this job with
`library_id`. The job, `upload_ingest.py`:
- parses the library's pending documents with `ai_parse_document`;
- chunks them with the **same section SQL as `02` steps 3 and 5** (inlined, so keep the two in
  sync);
- embeds the chunks via `mygpt-embeddings`;
- writes each document's chunks to Lakebase `mygpt.upload_chunks` in one transaction (admin RLS
  context).

A failed document is marked `failed` with its error; the others still index. Re-running the job
for a library resumes, because documents left in `parsing` are picked up again.

