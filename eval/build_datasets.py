"""Build / refresh the Phase-2 UC eval datasets (DABs job: mygpt_eval_build_datasets).

Reads St. Jude's MyGPT-evaluations CSVs from the bundle-synced submodule (and, for gold,
the private ground truths in the restricted eval_private volume) and merges the records
into UC-backed MLflow eval datasets:

    structural_biology.mygpt.eval_health_policies_fr   (21 questions x 10 FR policies)
    structural_biology.mygpt.eval_health_policies_es   (21 questions x 10 ES policies)
    structural_biology.mygpt.eval_kinase_literature    (3 questions x 94 paper/kinase pairs)
    structural_biology.mygpt.eval_bioasq               (163 BioASQ questions, public gold)

`merge_records` upserts on `inputs`, so re-running is idempotent (e.g. after the
kinase expression-system vocabulary lands, or the private gold is uploaded, rows are
refreshed in place). The tables then hold St. Jude's gold in `expectations`: keep SELECT
on them limited to the project team (see eval/README.md).

Usage:
    python eval/build_datasets.py --catalog structural_biology --schema mygpt \
        [--datasets health_policies_fr,kinase_literature] [--source-sha <submodule sha>]
"""

from __future__ import annotations

import argparse
import os
import sys


def _base_dir(base_dir: str) -> str:
    """Repo root. On a Databricks job `__file__` is undefined, so the job passes
    --base-dir ${workspace.file_path}."""
    if base_dir:
        return base_dir
    try:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        return os.getcwd()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="structural_biology")
    ap.add_argument("--schema", default="mygpt")
    ap.add_argument("--experiment", default="/Shared/mygpt/mygpt-rag-agent")
    ap.add_argument("--datasets", default="health_policies_fr,health_policies_es,kinase_literature,bioasq")
    ap.add_argument("--source-sha", default="", help="MyGPT-evaluations submodule commit, recorded per row")
    ap.add_argument("--base-dir", default="")
    args = ap.parse_args()

    base = _base_dir(args.base_dir)
    sys.path.insert(0, os.path.join(base, "eval"))
    import eval_datasets
    import mlflow
    from mlflow.genai import datasets

    experiment_id = mlflow.set_experiment(args.experiment).experiment_id
    if eval_datasets.private_dir(base) is None:
        print("[build] St. Jude's private ground truths not found (make eval-private) — "
              "SJ Global / BlueSky rows carry guidelines only")
    specs = eval_datasets.specs(args.catalog, args.schema)
    for name in [n.strip() for n in args.datasets.split(",") if n.strip()]:
        spec = specs[name]
        records = eval_datasets.build_records(name, base, args.source_sha)
        try:
            ds = datasets.get_dataset(name=spec.uc_table)
        except Exception:  # noqa: BLE001  (not created yet)
            # (No dataset-level tags: on Databricks they're managed through Unity Catalog;
            # use_case / corpus / source_sha are recorded per row instead.)
            ds = datasets.create_dataset(name=spec.uc_table, experiment_id=experiment_id)
        ds = ds.merge_records(records)
        print(f"{spec.uc_table}: merged {len(records)} records (corpus={spec.corpus})")


if __name__ == "__main__":
    main()
