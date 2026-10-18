"""Garbled / Devanagari parse fallback for the mygpt_etl job (Phase 6, gap 10).

Runs between `parse` (02_parse_chunk_index.sql: ai_parse_document -> _doc_elements_stage)
and `chunk_merge` (02b_chunk_merge.sql). Port of MyGPT's pypdf -> PyMuPDF fallback
(backend/testdb/views/document_processing.py), at document level:

  1. checks every freshly parsed document's text: flagged when it is empty, has 3+
     `(cid:N)` markers (font-to-Unicode mapping missing), more than 2% U+FFFD / NUL
     replacement characters, or — for a Hindi corpus — no Devanagari at all;
  2. re-extracts flagged documents with PyMuPDF and keeps that text when it is better
     (MyGPT's three cases: original garbled and PyMuPDF has text; PyMuPDF has Devanagari
     the original lacks; original empty and PyMuPDF has text);
  3. replaces those documents' rows in _doc_elements_stage with PyMuPDF elements in the
     same shape (one element per text block; short, large-font blocks become section
     headers, so 02b's section-aware chunking still applies);
  4. records the check per document version in `doc_parse_quality` (parser =
     ai_parse_document | pymupdf). A no-op when nothing is flagged.

etl/upload_ingest.py carries a copy of the heuristics (serverless spark_python_tasks can't
import sibling modules); test_parse_fallback.py checks the two copies agree.

Usage (job task):
    python etl/parse_fallback.py --catalog structural_biology --schema mygpt
"""

from __future__ import annotations

import argparse
import re
import statistics

CID_RE = re.compile(r"\(cid:\d+\)")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
# Languages whose documents must contain their script (else the text layer is broken).
SCRIPT_RE = {"hindi": DEVANAGARI_RE}
# Governed corpora's document language (mirror of dataset_config / the agent's CORPUS_LANGUAGE),
# used when dataset_config has no row yet (03 runs after this task on a corpus' first run).
CORPUS_LANGUAGE = {"health_policies_fr": "french", "health_policies_es": "spanish", "health_policies_hi": "hindi"}
HEADER_FONT_RATIO = 1.2
HEADER_MAX_CHARS = 120


# --------------------------------------------------------------------------- heuristics
def is_probably_garbled(text: str) -> bool:
    """MyGPT's _is_probably_garbled."""
    stripped = (text or "").strip()
    if not stripped:
        return True
    if len(CID_RE.findall(stripped)) >= 3:
        return True
    bad = stripped.count("�") + stripped.count("\x00")
    return bad / max(len(stripped), 1) > 0.02


def missing_script(text: str, language: str) -> bool:
    script = SCRIPT_RE.get(language)
    return bool(script) and not script.search(text or "")


def fallback_reason(text: str, language: str) -> str | None:
    """Why a document's parsed text should be re-extracted, or None if it looks fine."""
    stripped = (text or "").strip()
    if not stripped:
        return "empty"
    if len(CID_RE.findall(stripped)) >= 3:
        return "cid_markers"
    if is_probably_garbled(stripped):
        return "replacement_chars"
    if missing_script(stripped, language):
        return "missing_script"
    return None


def prefer_fitz(original: str, fitz_text: str, language: str) -> bool:
    """MyGPT's three cases for switching to the PyMuPDF text (document level)."""
    if not (fitz_text or "").strip():
        return False
    if is_probably_garbled(original) and not is_probably_garbled(fitz_text):
        return True
    script = SCRIPT_RE.get(language, DEVANAGARI_RE)
    if not script.search(original or "") and script.search(fitz_text):
        return True
    return not (original or "").strip()


def blocks_to_elements(pages: list[list[tuple[str, float]]]) -> list[dict]:
    """PyMuPDF text blocks [(text, max font size)] per page -> element rows (idx, elem_type,
    page, content, header_text, section) in _doc_elements_stage's shape. A block is a
    section header when it is short and its font is >= 1.2x the document's median size."""
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
                # 0-based, like ai_parse_document's bbox page_id
                "page": page_no, "content": text,
                "header_text": text if is_header else None, "section": section,
            })
            idx += 1
    return rows


def pdf_blocks(path: str) -> list[list[tuple[str, float]]]:
    """Text blocks per page with their largest font size (PyMuPDF `dict` extraction)."""
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


# --------------------------------------------------------------------------- job
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="structural_biology")
    ap.add_argument("--schema", default="mygpt")
    ap.add_argument("--base-dir", default="")  # accepted for job parity; unused here
    args = ap.parse_args()
    s = f"{args.catalog}.{args.schema}"

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {s}.doc_parse_quality (
          doc_id BIGINT, path STRING, corpus STRING, size_bytes BIGINT, modification_time TIMESTAMP,
          language STRING, n_chars BIGINT, reason STRING, parser STRING, checked_at TIMESTAMP)""")

    languages = dict(CORPUS_LANGUAGE)
    if spark.catalog.tableExists(f"{s}.dataset_config"):
        languages.update({r.dataset_name: r.documents_language for r in
                          spark.sql(f"SELECT dataset_name, documents_language FROM {s}.dataset_config").collect()})

    docs = spark.sql(f"""
        SELECT d.doc_id, d.path, d.corpus, d.size_bytes, d.modification_time,
               concat_ws('\\n', transform(array_sort(collect_list(struct(e.idx, e.content))), x -> x.content)) AS text
        FROM {s}._docs_to_parse d
        LEFT JOIN {s}._doc_elements_stage e ON e.doc_id = d.doc_id
        GROUP BY d.doc_id, d.path, d.corpus, d.size_bytes, d.modification_time""").collect()

    quality, replaced = [], []
    for d in docs:
        language = languages.get(d.corpus, "english")
        text = d.text or ""
        reason = fallback_reason(text, language)
        parser = "ai_parse_document"
        if reason:
            local = d.path.replace("dbfs:", "", 1)
            try:
                rows = blocks_to_elements(pdf_blocks(local))
                fitz_text = "\n".join(r["content"] for r in rows)
                if prefer_fitz(text, fitz_text, language):
                    parser = "pymupdf"
                    replaced += [{"doc_id": d.doc_id, "path": d.path, "corpus": d.corpus, **r} for r in rows]
                    text = fitz_text
            except Exception as exc:  # noqa: BLE001  (keep the ai_parse_document output)
                reason = f"{reason}; pymupdf failed: {type(exc).__name__}: {exc}"[:500]
            print(f"[parse_fallback] {d.path}: {reason} -> {parser}")
        quality.append((d.doc_id, d.path, d.corpus, d.size_bytes, d.modification_time, language,
                        len(text), reason, parser))

    if replaced:
        ids = sorted({r["doc_id"] for r in replaced})
        spark.sql(f"DELETE FROM {s}._doc_elements_stage WHERE doc_id IN ({','.join(str(i) for i in ids)})")
        (spark.createDataFrame(replaced, "doc_id bigint, path string, corpus string, idx int, elem_type string, "
                                         "page int, content string, header_text string, section string")
         .select("doc_id", "path", "corpus", "idx", "elem_type", "page", "content", "header_text", "section")
         .write.mode("append").saveAsTable(f"{s}._doc_elements_stage"))
    if quality:
        spark.createDataFrame(
            quality, "doc_id bigint, path string, corpus string, size_bytes bigint, modification_time timestamp, "
                     "language string, n_chars bigint, reason string, parser string",
        ).createOrReplaceTempView("_quality")
        spark.sql(f"""
            MERGE INTO {s}.doc_parse_quality t
            USING (SELECT *, current_timestamp() AS checked_at FROM _quality) q ON t.doc_id = q.doc_id
            WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *""")
    n_flagged = sum(1 for q in quality if q[7])
    print(f"[parse_fallback] {len(quality)} document(s) checked, {n_flagged} flagged, "
          f"{len({r['doc_id'] for r in replaced})} re-extracted with PyMuPDF")


if __name__ == "__main__":
    main()
