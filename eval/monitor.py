"""Register the Phase-2 scorers for production monitoring of mygpt-rag-agent.

The agent endpoint logs its traces to the shared experiment it was logged under
(/Shared/mygpt/mygpt-rag-agent). Registering scorers there scores a sample of live
traffic with the same definitions the offline evals use. Only app traffic is
sampled: the agent tags traces `mygpt.corpus = all` when no corpus is requested,
which eval runs always set.

Idempotent — re-running updates the sampling config of already-registered scorers.

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
    ap.add_argument("--base-dir", default="")
    args = ap.parse_args()

    base = args.base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base, "eval"))
    import mlflow
    import scorers as sc
    from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, Safety, ScorerSamplingConfig, get_scorer

    experiment_id = mlflow.set_experiment(args.experiment).experiment_id
    app_traffic = "tags.`mygpt.corpus` = 'all'"
    plan = [
        (sc.retrieval_ok, args.sample_rate),
        (sc.mygpt_trust, args.sample_rate),
        (Safety(), args.judge_sample_rate),
        (RelevanceToQuery(), args.judge_sample_rate),
        (RetrievalGroundedness(), args.judge_sample_rate),
    ]
    for scorer, rate in plan:
        config = ScorerSamplingConfig(sample_rate=rate, filter_string=app_traffic)
        try:
            registered = scorer.register(name=scorer.name, experiment_id=experiment_id)
            registered.start(sampling_config=config)
            print(f"registered + started {scorer.name} (sample_rate={rate})")
        except Exception as exc:  # noqa: BLE001  (already registered)
            get_scorer(name=scorer.name, experiment_id=experiment_id).update(sampling_config=config)
            print(f"updated {scorer.name} (sample_rate={rate}) [{type(exc).__name__}]")


if __name__ == "__main__":
    main()
