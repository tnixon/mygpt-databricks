# etl — Ingestion & medallion

Target (Phase 3): Lakeflow Connect pipelines (**SharePoint / OneDrive** documents → **Unity Catalog**
(`structural_biology`)), bronze → silver → gold, with **PHI scanning on ingest**. Gold docs are
embedded into **Databricks AI Search** (the global governed document index); structured tables
feed **Genie**.

## Today: the `mygpt_etl` job (incremental)
Source PDFs live in `/Volumes/structural_biology/mygpt/docs/<corpus>/`. One top-level folder is
one **corpus** (`biomed`, `health_policies_fr`, `health_policies_es`, `kinase_literature`), which
the agent filters on via `custom_inputs.corpus`.

| Task | What |
|---|---|
| `landing` (`sql/01_landing_documents.sql`) | `documents`: one pointer row per file. `id = xxhash64(path)`, so ids are stable across runs |
| `parse_chunk_index` (`sql/02_parse_chunk_index.sql`) | Parses only **new or changed** files with `ai_parse_document`, then **MERGEs** section chunks into `doc_chunks` |
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
  `dataset_config` row in `03` (plus `etl/config/dataset_config.csv`).

```bash
databricks bundle deploy -p mygpt --var warehouse_id=<id>
databricks bundle run mygpt_etl -p mygpt --var warehouse_id=<id>
```

Index **creation** is a one-time API step; see the tail of `sql/02_parse_chunk_index.sql`.

**Gotchas (both hit on 2026-10-12)**
- **Never `CREATE OR REPLACE` `doc_chunks`.** A replace gives the Delta table a new id, which
  invalidates the index's change-feed checkpoint. `sync_index` then fails with
  `DIFFERENT_DELTA_TABLE_READ_BY_STREAMING_SOURCE`, and the only fix is to rebuild the index.
- **After deleting and recreating the index, redeploy the agent** (`mygpt_deploy_agent`). The
  agent endpoint's SELECT grant on the index is provisioned by `agents.deploy`, and a recreated
  index is a new object without it. Until the redeploy, every query returns
  `retrieve_error: PermissionDenied`.
