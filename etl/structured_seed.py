"""Phase 5: seed the structured tables Genie answers over (mygpt_etl task `structured_seed`).

MyGPT never made spreadsheets SQL-queryable (it chunks their rows into RAG), so this is the
minimal loader PLAN.md calls for until Phase 3 lands spreadsheet ingestion at scale. It reads
St. Jude's MyGPT-evaluations kinase_literature CSVs from the reference volume (deploy.sh step 6),
falling back to the bundle-synced reference/ copy of a dev checkout, and writes:

  kinome            the human kinome: one row per kinase (UniProt entry, synonyms, group, family)
  kinase_papers     the BlueSky library: one row per (paper, kinase) pair, keyed by PubMed id
  corpus_documents  view: every governed document with its corpus and chunk count

Table and column comments are part of the contract: Genie (and the Supervisor's routing
prompt) read them. A serverless spark_python_task rather than SQL, because SQL warehouses
can't read workspace files and a CSV-via-volume hop was brittle in Phase 1.

    python etl/structured_seed.py --base-dir ${workspace.file_path} \
        --reference-volume /Volumes/<catalog>/<schema>/<reference_volume>
"""

from __future__ import annotations

import argparse
import csv
import os

KINASE_INPUTS = os.path.join("MyGPT-evaluations", "kinase_literature", "inputs")


def kinase_inputs(reference_volume: str, base_dir: str) -> str:
    """The kinase_literature inputs folder: the reference volume, else the synced repo copy."""
    candidates = [os.path.join(base_dir, "reference", KINASE_INPUTS)]
    if reference_volume:
        candidates.insert(0, os.path.join(reference_volume, KINASE_INPUTS))
    for path in candidates:
        if os.path.isfile(os.path.join(path, "human_kinome_synonyms.csv")):
            return path
    raise FileNotFoundError(
        f"human_kinome_synonyms.csv not found in {' or '.join(candidates)}. Upload the "
        "MyGPT-evaluations CSVs to the reference volume (deploy.sh step 6 with EVAL_DIR set, "
        "or `make reference-csvs EVAL_INPUTS=<dir>`), then re-run mygpt_etl.")


def read_kinome(path: str) -> list[dict]:
    """human_kinome_synonyms.csv -> rows. The file starts with a UTF-8 BOM, and synonyms
    are one comma-separated cell that includes the kinase's own id."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            kinase_id = (r.get("kinase_ID") or "").strip()
            if not kinase_id:
                continue
            synonyms = [s.strip() for s in (r.get("synonyms") or "").split(",") if s.strip()]
            rows.append({
                "kinase_id": kinase_id,
                "uniprot_id": (r.get("entry") or "").strip() or None,
                "synonyms": synonyms,
                "kinase_group": (r.get("kinase_group") or "").strip() or None,
                "kinase_family": (r.get("kinase_family") or "").strip() or None,
            })
    return rows


def read_kinase_papers(path: str) -> list[dict]:
    """kinase_pubmed_pairs.csv -> rows; source_name is the PDF name the ETL indexes
    (docs/kinase_literature/<pubmed_id>.pdf), so papers join to documents / doc_chunks."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            pmid = (r.get("pubmed_id") or "").strip()
            if not pmid:
                continue
            rows.append({
                "pubmed_id": pmid,
                "kinase_id": (r.get("kinase_name") or "").strip() or None,
                "doi": (r.get("doi") or "").strip() or None,
                "source_name": f"{pmid}.pdf",
            })
    return rows


TABLE_COMMENTS = {
    "kinome": (
        "The human kinome: one row per human protein kinase, with its UniProt accession, "
        "alternative names and its Manning classification (group and family). Source: St. Jude "
        "MyGPT-evaluations human_kinome_synonyms.csv.",
        {
            "kinase_id": "Kinase gene symbol, e.g. ABL1, MAPK14, EPHA8 (HGNC symbol; the key other tables use)",
            "uniprot_id": "UniProt accession of the kinase, e.g. P00519",
            "synonyms": "Alternative names and aliases, including protein names (array of strings)",
            "kinase_group": "Manning kinase group: AGC, CAMK, CK1, CMGC, RGC, STE, TK, TKL, Atypical or Other",
            "kinase_family": "Kinase family within the group, e.g. Abl, MAPK, Eph",
        },
    ),
    "kinase_papers": (
        "BlueSky kinase literature library: one row per (paper, kinase) pair studied by St. Jude's "
        "kinase evaluation. Each paper is indexed for document Q&A under the kinase_literature "
        "corpus as <pubmed_id>.pdf.",
        {
            "pubmed_id": "PubMed id of the paper (string of digits)",
            "kinase_id": "Gene symbol of the kinase the paper is paired with; joins kinome.kinase_id",
            "doi": "DOI of the paper",
            "source_name": "File name of the paper in the document library; joins corpus_documents.source_name",
        },
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="structural_biology")
    ap.add_argument("--schema", default="mygpt")
    ap.add_argument("--base-dir", default="")
    ap.add_argument("--reference-volume", default="")
    args = ap.parse_args()

    inputs = kinase_inputs(args.reference_volume, args.base_dir or os.getcwd())
    print(f"[structured_seed] reading {inputs}")
    fq = f"{args.catalog}.{args.schema}"

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    data = {
        "kinome": (read_kinome(os.path.join(inputs, "human_kinome_synonyms.csv")),
                   "kinase_id STRING, uniprot_id STRING, synonyms ARRAY<STRING>, kinase_group STRING, kinase_family STRING"),
        "kinase_papers": (read_kinase_papers(os.path.join(inputs, "kinase_pubmed_pairs.csv")),
                          "pubmed_id STRING, kinase_id STRING, doi STRING, source_name STRING"),
    }
    for table, (rows, schema) in data.items():
        spark.createDataFrame(rows, schema).write.mode("overwrite").option("overwriteSchema", "true") \
            .saveAsTable(f"{fq}.{table}")
        comment, columns = TABLE_COMMENTS[table]
        spark.sql(f"COMMENT ON TABLE {fq}.{table} IS {_sql_str(comment)}")
        for col, text in columns.items():
            spark.sql(f"ALTER TABLE {fq}.{table} ALTER COLUMN {col} COMMENT {_sql_str(text)}")
        print(f"[structured_seed] {fq}.{table}: {len(rows)} rows")

    spark.sql(f"""
        CREATE OR REPLACE VIEW {fq}.corpus_documents (
          source_name COMMENT 'File name of the document, e.g. 10498895.pdf',
          corpus COMMENT 'Document collection: biomed (BioASQ papers), kinase_literature (BlueSky kinase papers), health_policies_fr / health_policies_es (SJ Global health policies)',
          size_bytes COMMENT 'File size in bytes',
          modified_at COMMENT 'When the file last changed in the document volume',
          n_chunks COMMENT 'Number of indexed text chunks (0 = not parsed yet)'
        )
        COMMENT 'Every governed document in the MyGPT library with its corpus and how many chunks are indexed for document Q&A.'
        AS SELECT regexp_extract(d.path, '([^/]+)$', 1), d.corpus, d.size_bytes, d.modification_time,
                  coalesce(c.n, 0)
        FROM {fq}.documents d
        LEFT JOIN (SELECT doc_id, count(*) AS n FROM {fq}.doc_chunks GROUP BY doc_id) c ON c.doc_id = d.id
        WHERE d.corpus <> ''
    """)
    print(f"[structured_seed] {fq}.corpus_documents: view")


def _sql_str(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


if __name__ == "__main__":
    main()
