"""Smoke test for the deployed Supervisor (`mygpt-supervisor`): one question per route.

    python agents/supervisor/smoke_test.py --profile mygpt [--only documents]

Checks custom_outputs.route and the per-route payload (RAG scores passed through verbatim,
Genie SQL + rows, tool results, library selection across two libraries). Prints each answer.
"""

from __future__ import annotations

import argparse
import json

CASES = [
    ("documents", "What biomarkers predict effective temozolomide dosing in glioblastoma?", {},
     lambda co: co["scores_scope"] == "answer" and co["qrs"] is not None and co["sources"]),
    ("genie", "How many kinases are in each kinase group?", {},
     lambda co: (co.get("genie") or {}).get("status") == "COMPLETED" and co["genie"]["rows"]),
    ("tools", "Which PDB structures exist for human ABL1?", {},
     lambda co: any(c["name"] == "search_pdb_by_gene" and c["ok"] for c in co["tool_calls"])),
    # Two selected libraries: a kinase question should search the kinase one (alone, verbatim).
    ("documents", "Which expression system was used to produce recombinant EPHA8 kinase?", {"libraries": [
        {"id": "biomed", "name": "Biomedical seed (BioASQ)", "kind": "corpus", "corpus": "biomed", "dataset": "biomed",
         "description": "BioASQ biomedical seed corpus."},
        {"id": "kinase", "name": "BlueSky kinase literature", "kind": "corpus", "corpus": "kinase_literature",
         "dataset": "kinase_literature", "description": "BlueSky kinase papers."}]},
     lambda co: "kinase" in [x["library_id"] for x in co["libraries_searched"]]
     and all(s.get("library_id") for s in co["sources"])),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="mygpt")
    ap.add_argument("--endpoint", default="mygpt-supervisor")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=args.profile)
    failed = []
    for route, question, ci, check in CASES:
        if args.only and route != args.only:
            continue
        resp = w.api_client.do("POST", f"/serving-endpoints/{args.endpoint}/invocations", body={
            "input": [{"role": "user", "content": question}], "custom_inputs": ci})
        co = resp.get("custom_outputs") or {}
        text = "".join(c.get("text", "") for item in resp.get("output") or [] for c in item.get("content") or [])
        ok = route in (co.get("route") or []) and bool(check(co))
        print(f"[{'ok' if ok else 'FAIL'}] {route}: route={co.get('route')} tools="
              f"{[(c['name'], c['ok'], c['latency_ms']) for c in co.get('tool_calls') or []]}")
        print(f"       {text[:200]!r}")
        if not ok:
            failed.append(route)
            print("       custom_outputs:", json.dumps({k: co.get(k) for k in ("error", "retrieve_error", "genie", "libraries_searched")}, default=str)[:600])
    if failed:
        raise SystemExit(f"FAILED: {failed}")
    print("OK")


if __name__ == "__main__":
    main()
