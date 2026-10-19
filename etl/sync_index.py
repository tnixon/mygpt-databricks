"""Sync the MyGPT Vector Search index after the ETL, and wait until it's caught up.

Final task of the `mygpt_etl` DABs job (after chunk_merge). The index is a
TRIGGERED Delta Sync index over `doc_chunks`: it only picks up the chunks 02 MERGEd in
when a sync is requested. This task:
  1. triggers the sync — if one is already running, waits for it and triggers again so
     this run's MERGE is included;
  2. waits for the index's backing pipeline update to reach a terminal state (the index
     status API has no "sync finished" field), failing on FAILED / CANCELED / timeout;
  3. checks indexed_row_count == count(doc_chunks), which catches silent embedding
     failures (e.g. the `mygpt-bge-m3` endpoint being down).

If the index doesn't exist yet (first ETL run), it is CREATED here instead: a TRIGGERED Delta
Sync index on --endpoint (a bundle resource), embedding chunk_text through the
--embedding-endpoint MODEL endpoint (AI Search rejects gateway endpoints). Its initial sync
then plays the role of step 1, and steps 2-3 are unchanged.

Usage:
    python etl/sync_index.py --index structural_biology.mygpt.doc_chunks_corpus_index \
        --source-table structural_biology.mygpt.doc_chunks --endpoint mygpt-vs --embedding-endpoint mygpt-bge-m3
"""

from __future__ import annotations

import argparse
import sys
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
    VectorIndexType,
)

_TERMINAL = {"COMPLETED", "FAILED", "CANCELED"}
_POLL_S = 15
# Columns the agent reads back (agents/rag_agent/agent.py _retrieve) and filters on.
_COLUMNS_TO_SYNC = ["chunk_id", "doc_id", "source_name", "corpus", "section", "page", "elem_type", "chunk_text"]


def _create_index(w: WorkspaceClient, index: str, source_table: str, endpoint: str, embedding_endpoint: str,
                  deadline: float) -> str:
    """Create the index (waiting for its VS endpoint to be ONLINE first); returns its pipeline id."""
    while (state := w.vector_search_endpoints.get_endpoint(endpoint).endpoint_status.state.value) != "ONLINE":
        if time.time() > deadline:
            raise TimeoutError(f"VS endpoint {endpoint} still {state}")
        print(f"[sync] VS endpoint {endpoint} is {state}; waiting")
        time.sleep(_POLL_S)
    w.vector_search_indexes.create_index(
        name=index, endpoint_name=endpoint, primary_key="chunk_id", index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=source_table, pipeline_type=PipelineType.TRIGGERED, columns_to_sync=_COLUMNS_TO_SYNC,
            embedding_source_columns=[EmbeddingSourceColumn(name="chunk_text", embedding_model_endpoint_name=embedding_endpoint)],
        ),
    )
    print(f"[sync] created {index} on {endpoint} (embeddings: {embedding_endpoint}); waiting for its initial sync")
    # The backing pipeline id appears once the index has been provisioned.
    while not (pipeline_id := w.vector_search_indexes.get_index(index).delta_sync_index_spec.pipeline_id):
        if time.time() > deadline:
            raise TimeoutError(f"index {index} has no pipeline yet")
        time.sleep(_POLL_S)
    return pipeline_id


def _latest_update(w: WorkspaceClient, pipeline_id: str):
    updates = w.pipelines.get(pipeline_id).latest_updates or []
    return updates[0] if updates else None  # newest first


def _state(update) -> str:
    return update.state.value if update and update.state else ""


def _wait_terminal(w: WorkspaceClient, pipeline_id: str, after_id: str | None, deadline: float):
    """Wait for the newest pipeline update (one newer than `after_id`, if given) to finish."""
    while time.time() < deadline:
        update = _latest_update(w, pipeline_id)
        if update and update.update_id != after_id and _state(update) in _TERMINAL:
            return update
        time.sleep(_POLL_S)
    raise TimeoutError(f"index sync did not finish (pipeline {pipeline_id})")


def _latest_error(w: WorkspaceClient, pipeline_id: str) -> str:
    """First line of the pipeline's most recent ERROR event, so the job failure says why."""
    try:
        for event in w.pipelines.list_pipeline_events(pipeline_id, filter="level='ERROR'", max_results=5):
            for exc in (event.error.exceptions if event.error else None) or []:
                if exc.message:
                    return exc.message.strip().splitlines()[0][:500]
            if event.message:
                return event.message[:500]
    except Exception as exc:  # noqa: BLE001  (diagnostics only)
        return f"(could not read pipeline events: {type(exc).__name__})"
    return ""


def _source_count(table: str) -> int:
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate().table(table).count()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="catalog.schema.index")
    ap.add_argument("--source-table", required=True, help="catalog.schema.table the index syncs from")
    ap.add_argument("--endpoint", required=True, help="VS endpoint to create the index on, if it's missing")
    ap.add_argument("--embedding-endpoint", required=True, help="model serving endpoint embedding chunk_text")
    ap.add_argument("--timeout-min", type=int, default=60)
    args = ap.parse_args()

    w = WorkspaceClient()
    deadline = time.time() + args.timeout_min * 60
    started = time.time()
    try:
        pipeline_id = w.vector_search_indexes.get_index(args.index).delta_sync_index_spec.pipeline_id
    except NotFound:
        # First run: the initial sync of the new index covers everything 02b MERGEd.
        pipeline_id = _create_index(w, args.index, args.source_table, args.endpoint, args.embedding_endpoint, deadline)
        update = _wait_terminal(w, pipeline_id, None, deadline)
    else:
        print(f"[sync] {args.index} (pipeline {pipeline_id})")
        # A sync still running from an earlier trigger may predate this run's MERGE: let it
        # finish, then trigger our own.
        current = _latest_update(w, pipeline_id)
        if current and _state(current) not in _TERMINAL:
            print(f"[sync] update {current.update_id} already {_state(current)}; waiting for it first")
            current = _wait_terminal(w, pipeline_id, None, deadline)
        started = time.time()
        w.vector_search_indexes.sync_index(args.index)
        update = _wait_terminal(w, pipeline_id, current.update_id if current else None, deadline)
    print(f"[sync] update {update.update_id} {_state(update)} in {time.time() - started:.0f}s")
    if _state(update) != "COMPLETED":
        # e.g. DIFFERENT_DELTA_TABLE_READ_BY_STREAMING_SOURCE: the source table was replaced
        # (CREATE OR REPLACE) after the index was built, which invalidates the index's
        # change-feed checkpoint. The index then needs a full rebuild.
        sys.exit(f"index sync {_state(update)} (pipeline {pipeline_id}): {_latest_error(w, pipeline_id)}")

    # indexed_row_count can trail the pipeline's completion briefly; give it a few polls.
    expected = _source_count(args.source_table)
    indexed = None
    for _ in range(8):
        indexed = w.vector_search_indexes.get_index(args.index).status.indexed_row_count
        if indexed == expected:
            break
        time.sleep(_POLL_S)
    print(f"[sync] indexed_row_count={indexed} source_rows={expected}")
    if indexed != expected:
        sys.exit(f"index has {indexed} rows but {args.source_table} has {expected} — embedding failures?")


if __name__ == "__main__":
    main()
