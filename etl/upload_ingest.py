"""Index an upload library's new documents into the Lakebase upload index (Phase 4, gap 7).

Run by the `mygpt_upload_ingest` DABs job, which the app triggers after a user uploads files
(app/server/ingest.ts; one active run per library). For the library's documents in status
`uploaded` (or `parsing`, left over by a run that died) this job:
  1. parses them with ai_parse_document (serverless Spark), straight from the uploads volume;
  2. section-chunks them with the SAME SQL as etl/sql/02_parse_chunk_index.sql step 3 + 02b_chunk_merge.sql step 5
     (titles / section headers start a section; ~1000-char section-bounded buckets) —
     inlined below because a serverless spark_python_task can't import sibling modules;
     keep the two in sync;
     Phase 6: documents whose parsed text is garbled or lacks its script (Hindi without
     Devanagari) are re-extracted with PyMuPDF first, as in the mygpt_etl job;
  3. embeds the chunks with bge-m3 through the `mygpt-embeddings` gateway endpoint (driver-side
     batches: upload libraries are small, and this is the same REST path the agent uses);
  4. writes each document's chunks to Lakebase in ONE transaction per document (delete old,
     insert new, mark `ready`), so a failure costs only that document and re-runs resume.
It connects as the job's run-as identity with the admin RLS context (mygpt.role = admin).

Usage:
    python etl/upload_ingest.py --library-id <uuid> \
        --pg-endpoint projects/mygpt/branches/production/endpoints/primary
"""

from __future__ import annotations

import argparse
import re
import sys
import time

from databricks.sdk import WorkspaceClient

EMBED_BATCH = 32
SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

# Postgres text-search configs for MyGPT's document languages (mirror of app/server/db/store.ts).
TS_CONFIGS = {
    "english": "english", "french": "french", "spanish": "spanish", "portuguese": "portuguese",
    "german": "german", "italian": "italian", "dutch": "dutch", "arabic": "arabic",
    "hungarian": "hungarian", "hindi": "hindi",
}

# --- Parse fallback heuristics: a COPY of etl/parse_fallback.py (a serverless
# spark_python_task can't import sibling modules). etl/test_parse_fallback.py checks
# the two copies agree; change both together.
CID_RE = re.compile(r"\(cid:\d+\)")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
SCRIPT_RE = {"hindi": DEVANAGARI_RE}
HEADER_FONT_RATIO = 1.2
HEADER_MAX_CHARS = 120


def is_probably_garbled(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if len(CID_RE.findall(stripped)) >= 3:
        return True
    bad = stripped.count("\ufffd") + stripped.count("\x00")
    return bad / max(len(stripped), 1) > 0.02


def fallback_reason(text: str, language: str) -> str | None:
    stripped = (text or "").strip()
    if not stripped:
        return "empty"
    if len(CID_RE.findall(stripped)) >= 3:
        return "cid_markers"
    if is_probably_garbled(stripped):
        return "replacement_chars"
    script = SCRIPT_RE.get(language)
    if script and not script.search(stripped):
        return "missing_script"
    return None


def prefer_fitz(original: str, fitz_text: str, language: str) -> bool:
    if not (fitz_text or "").strip():
        return False
    if is_probably_garbled(original) and not is_probably_garbled(fitz_text):
        return True
    script = SCRIPT_RE.get(language, DEVANAGARI_RE)
    if not script.search(original or "") and script.search(fitz_text):
        return True
    return not (original or "").strip()


def blocks_to_elements(pages: list[list[tuple[str, float]]]) -> list[dict]:
    import statistics

    sizes = [size for page in pages for _, size in page if size]
    median = statistics.median(sizes) if sizes else 0.0
    rows, section, idx = [], "(no section)", 0
    for page_no, page in enumerate(pages):
        for text, size in page:
            text = " ".join(text.split())
            if not text:
                continue
            is_header = bool(median) and len(text) <= HEADER_MAX_CHARS and size >= HEADER_FONT_RATIO * median
            if is_header:
                section = text
            rows.append({
                "idx": idx, "elem_type": "section_header" if is_header else "text",
                "page": page_no, "content": text,
                "header_text": text if is_header else None, "section": section,
            })
            idx += 1
    return rows


def pdf_blocks(path: str) -> list[list[tuple[str, float]]]:
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(path) as doc:
        for page in doc:
            blocks = []
            for b in page.get_text("dict").get("blocks", []):
                spans = [s for line in b.get("lines", []) for s in line.get("spans", [])]
                text = "\n".join("".join(s.get("text", "") for s in line.get("spans", [])) for line in b.get("lines", []))
                if spans and text.strip():
                    blocks.append((text, max(float(s.get("size") or 0) for s in spans)))
            pages.append(blocks)
    return pages
# --- end of the parse_fallback.py copy


def with_parse_fallback(elements: list[dict], docs: list[tuple], language: str) -> list[dict]:
    """Replace the elements of garbled / script-less documents with PyMuPDF's when better."""
    by_doc: dict[str, list[dict]] = {}
    for e in elements:
        by_doc.setdefault(e["doc_id"], []).append(e)
    out: list[dict] = []
    for doc_id, path, filename in docs:
        rows = sorted(by_doc.get(doc_id, []), key=lambda e: e["idx"])
        text = "\n".join(e["content"] or "" for e in rows)
        reason = fallback_reason(text, language)
        if reason:
            try:
                fitz_rows = blocks_to_elements(pdf_blocks(path))
                if prefer_fitz(text, "\n".join(r["content"] for r in fitz_rows), language):
                    rows = [{"doc_id": doc_id, **{k: r[k] for k in ("idx", "elem_type", "page", "content", "section")}}
                            for r in fitz_rows]
                    reason += " -> pymupdf"
            except Exception as exc:  # noqa: BLE001  (keep the ai_parse_document output)
                reason += f"; pymupdf failed: {type(exc).__name__}: {exc}"
            print(f"[ingest] {filename}: parse fallback ({reason})")
        out.extend(rows)
    return out

# ai_parse_document -> section-tagged elements (mirror of 02 step 3).
ELEMENTS_SQL = """
WITH files AS (
  SELECT regexp_replace(path, '^dbfs:', '') AS path, content
  FROM read_files('{folder}', format => 'binaryFile')
),
parsed AS (
  SELECT d.doc_id, ai_parse_document(f.content) AS doc
  FROM files f JOIN _upload_docs d ON d.volume_path = f.path
),
elems AS (
  SELECT doc_id, idx, e
  FROM parsed
  LATERAL VIEW posexplode(from_json(to_json(doc:document.elements), 'array<variant>')) t AS idx, e
),
tagged AS (
  SELECT
    doc_id, idx,
    e:type::string                        AS elem_type,
    try_cast(e:bbox[0]:page_id AS INT)    AS page,
    trim(e:content::string)               AS content,
    CASE WHEN lower(e:type::string) IN ('title', 'section_header')
         THEN trim(e:content::string) END AS header_text
  FROM elems
)
SELECT
  doc_id, idx, elem_type, page, content,
  coalesce(
    last_value(header_text) IGNORE NULLS
      OVER (PARTITION BY doc_id ORDER BY idx ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    '(no section)') AS section
FROM tagged
"""

# Elements -> ~1000-char section-bounded chunks (mirror of 02b step 5).
CHUNKS_SQL = """
WITH body AS (
  SELECT doc_id, idx, page, section, content
  FROM _upload_elements
  WHERE length(content) >= 80
    AND lower(elem_type) NOT IN ('page_header', 'page_footer', 'page_number')
),
bucketed AS (
  SELECT doc_id, idx, page, section, content,
    cast(floor(
      (sum(length(content) + 1)
         OVER (PARTITION BY doc_id, section ORDER BY idx
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) - 1)
      / 1000.0) AS INT) AS bucket
  FROM body
),
chunks AS (
  SELECT doc_id, section, bucket, min(page) AS page,
         concat_ws(' ', array_sort(collect_list(struct(idx, content))).content) AS chunk_text
  FROM bucketed
  GROUP BY doc_id, section, bucket
)
SELECT concat(doc_id, '-', substr(md5(section), 1, 16), '-', bucket) AS chunk_id,
       doc_id, section, page, chunk_text
FROM chunks
WHERE length(chunk_text) >= 80
"""


def _parse_embeddings(resp, n: int) -> list[list[float]]:
    """OpenAI envelope (gateway) or a bare list of vectors (mirror of the agent's parser)."""
    if isinstance(resp, dict):
        if "data" in resp:
            return [d["embedding"] for d in resp["data"]]
        for key in ("predictions", "embeddings", "outputs"):
            if key in resp:
                resp = resp[key]
                break
    if isinstance(resp, list):
        if resp and isinstance(resp[0], (int, float)) and n == 1:
            return [list(resp)]
        return [list(v) for v in resp]
    raise ValueError(f"Unrecognized embeddings response: {type(resp)}")


def embed(w: WorkspaceClient, endpoint: str, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        for attempt in range(3):
            try:
                resp = w.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations", body={"input": batch})
                vectors.extend(_parse_embeddings(resp, len(batch)))
                break
            except Exception:  # noqa: BLE001  (retry transient serving errors)
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
    return vectors


def connect(w: WorkspaceClient, endpoint: str, database: str):
    import psycopg

    host = w.postgres.get_endpoint(endpoint).status.hosts.host
    token = w.postgres.generate_database_credential(endpoint=endpoint).token
    user = w.current_user.me().user_name
    return psycopg.connect(host=host, dbname=database, user=user, password=token, sslmode="require")


def _admin(cur) -> None:
    cur.execute("SELECT set_config('mygpt.role', 'admin', true)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library-id", required=True)
    ap.add_argument("--pg-endpoint", required=True, help="projects/<p>/branches/<b>/endpoints/<e>")
    ap.add_argument("--pg-database", default="databricks_postgres")
    ap.add_argument("--pg-schema", default="mygpt")
    ap.add_argument("--embed-endpoint", default="mygpt-embeddings")
    args = ap.parse_args()
    if not SCHEMA_RE.match(args.pg_schema):
        sys.exit(f"invalid --pg-schema {args.pg_schema}")
    s = args.pg_schema

    w = WorkspaceClient()
    conn = connect(w, args.pg_endpoint, args.pg_database)

    # 1) Claim the library's pending documents. `parsing` = left over by a run that died
    #    (the app allows one active run per library, so nothing else holds them).
    with conn.transaction(), conn.cursor() as cur:
        _admin(cur)
        cur.execute(f"SELECT documents_language FROM {s}.libraries WHERE id = %s AND kind = 'upload'", (args.library_id,))
        row = cur.fetchone()
        if row is None:
            sys.exit(f"upload library {args.library_id} not found")
        language = row[0].lower()
        ts_config = TS_CONFIGS.get(language, "simple")
        cur.execute(
            f"""UPDATE {s}.library_documents SET status = 'parsing', error = NULL, updated_at = now()
                WHERE library_id = %s AND status IN ('uploaded', 'parsing')
                RETURNING id::text, volume_path, filename""",
            (args.library_id,),
        )
        docs = cur.fetchall()
        cur.execute(f"UPDATE {s}.libraries SET status = 'ingesting', updated_at = now() WHERE id = %s", (args.library_id,))
    print(f"[ingest] library {args.library_id}: {len(docs)} document(s) to index")

    if docs:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        spark.createDataFrame(docs, "doc_id string, volume_path string, filename string").createOrReplaceTempView("_upload_docs")
        folder = docs[0][1].rsplit("/", 1)[0] + "/"  # every upload of a library shares its folder
        # Collected once (upload libraries are small; no .cache() on serverless), so
        # ai_parse_document runs once while the parse fallback and CHUNKS_SQL both read it.
        elements = [r.asDict() for r in spark.sql(ELEMENTS_SQL.format(folder=folder.replace("'", "''"))).collect()]
        elements = with_parse_fallback(elements, docs, language)
        spark.createDataFrame(
            elements, "doc_id string, idx int, elem_type string, page int, content string, section string"
        ).createOrReplaceTempView("_upload_elements")
        chunks = [r.asDict() for r in spark.sql(CHUNKS_SQL).collect()]
        print(f"[ingest] {len(chunks)} chunk(s) from {len({c['doc_id'] for c in chunks})} document(s)")
        vectors = embed(w, args.embed_endpoint, [c["chunk_text"] for c in chunks]) if chunks else []
        if len(vectors) != len(chunks):
            sys.exit(f"embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks")

        by_doc: dict[str, list[tuple[dict, list[float]]]] = {}
        for c, v in zip(chunks, vectors):
            by_doc.setdefault(c["doc_id"], []).append((c, v))

        # 2) One transaction per document: replace its chunks, mark it ready (or failed).
        for doc_id, _path, filename in docs:
            items = by_doc.get(doc_id, [])
            try:
                with conn.transaction(), conn.cursor() as cur:
                    _admin(cur)
                    cur.execute(f"DELETE FROM {s}.upload_chunks WHERE document_id = %s", (doc_id,))
                    if not items:
                        cur.execute(
                            f"""UPDATE {s}.library_documents SET status = 'failed', n_chunks = 0,
                                  error = 'no text could be extracted', updated_at = now() WHERE id = %s""",
                            (doc_id,),
                        )
                        print(f"[ingest] {filename}: no chunks")
                        continue
                    cur.executemany(
                        f"""INSERT INTO {s}.upload_chunks
                              (chunk_id, library_id, document_id, source_name, section, page, chunk_text, embedding, ts_config)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s::regconfig)""",
                        [
                            (c["chunk_id"], args.library_id, doc_id, filename, c["section"], c["page"], c["chunk_text"],
                             "[" + ",".join(f"{x:.7g}" for x in v) + "]", ts_config)
                            for c, v in items
                        ],
                    )
                    cur.execute(
                        f"""UPDATE {s}.library_documents SET status = 'ready', n_chunks = %s, error = NULL,
                              updated_at = now() WHERE id = %s""",
                        (len(items), doc_id),
                    )
                print(f"[ingest] {filename}: {len(items)} chunk(s)")
            except Exception as exc:  # noqa: BLE001  (fail this document only)
                with conn.transaction(), conn.cursor() as cur:
                    _admin(cur)
                    cur.execute(
                        f"UPDATE {s}.library_documents SET status = 'failed', error = %s, updated_at = now() WHERE id = %s",
                        (f"{type(exc).__name__}: {exc}"[:1000], doc_id),
                    )
                print(f"[ingest] {filename}: FAILED {exc}")

    # 3) Library status from its documents.
    with conn.transaction(), conn.cursor() as cur:
        _admin(cur)
        cur.execute(
            f"""UPDATE {s}.libraries l SET updated_at = now(), status = CASE
                  WHEN EXISTS (SELECT 1 FROM {s}.library_documents d WHERE d.library_id = l.id AND d.status = 'ready') THEN 'ready'
                  WHEN EXISTS (SELECT 1 FROM {s}.library_documents d WHERE d.library_id = l.id AND d.status = 'failed') THEN 'failed'
                  ELSE 'empty' END
                WHERE l.id = %s
                RETURNING status""",
            (args.library_id,),
        )
        status = cur.fetchone()[0]
        cur.execute(
            f"SELECT count(*) FILTER (WHERE status = 'failed') FROM {s}.library_documents WHERE id = ANY(%s::uuid[])",
            ([d[0] for d in docs],),
        )
        n_failed = cur.fetchone()[0]
    conn.close()
    print(f"[ingest] library status: {status}; {n_failed} of {len(docs)} document(s) failed")
    if docs and n_failed == len(docs):
        sys.exit("every document failed to index")


if __name__ == "__main__":
    main()
