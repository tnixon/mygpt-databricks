"""Register the Phase-2 scorers for production monitoring of mygpt-rag-agent.

The agent endpoint logs its traces to the shared experiment it was logged under
(/Shared/mygpt/mygpt-rag-agent). Registering scorers there scores a sample of live
traffic with the same definitions the offline evals use. Only app traffic is
sampled: the agent tags traces `mygpt.corpus = all` when no corpus is requested,
which eval runs always set.

Idempotent — re-running replaces already-registered scorers (delete + register), so a
changed sampling config or judge model (--judge-model, Opus; see scorers.judge_model)
reaches the live monitors.

Usage:
    python eval/monitor.py [--sample-rate 0.2]
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="/Shared/mygpt/mygpt-rag-agent")
    ap.add_argument("--sample-rate", type=float, default=0.2)
    ap.add_argument("--judge-sample-rate", type=float, default=0.1, help="LLM judges cost more; sample less")
    ap.add_argument("--judge-model", default="databricks:/databricks-claude-opus-5-5")
    ap.add_argument("--base-dir", default="")
    args = ap.parse_args()

    base = args.base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base, "eval"))
    import mlflow
    import scorers as sc
    from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, Safety, ScorerSamplingConfig, delete_scorer

    sc.check_judge_outranks("sonnet", args.judge_model)  # app traffic: the Sonnet default

    experiment_id = mlflow.set_experiment(args.experiment).experiment_id
    app_traffic = "tags.`mygpt.corpus` = 'all'"
    plan = [
        (sc.retrieval_ok, args.sample_rate),
        (sc.mygpt_trust, args.sample_rate),
        (Safety(model=args.judge_model), args.judge_sample_rate),
        (RelevanceToQuery(model=args.judge_model), args.judge_sample_rate),
        (RetrievalGroundedness(model=args.judge_model), args.judge_sample_rate),
    ]
    for scorer, rate in plan:
        config = ScorerSamplingConfig(sample_rate=rate, filter_string=app_traffic)
        try:
            # update() only changes sampling; re-register so the scorer definition (e.g.
            # its judge model) is replaced too.
            delete_scorer(name=scorer.name, experiment_id=experiment_id, version="all")
            print(f"deleted previous {scorer.name}")
        except Exception:  # noqa: BLE001  (not registered yet)
            pass
        registered = scorer.register(name=scorer.name, experiment_id=experiment_id)
        registered.start(sampling_config=config)
        print(f"registered + started {scorer.name} (sample_rate={rate})")


if __name__ == "__main__":
    main()
