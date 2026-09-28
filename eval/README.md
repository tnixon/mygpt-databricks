# eval — MLflow GenAI evaluation

Evaluation harness for the RAG / agent pipeline, seeded from St. Jude's **MyGPT-evaluations** gold data (`kinase_literature` → "BlueSky", `health_policies` → "SJ Global", plus PubMedQA / BioASQ benchmarks). Uses **MLflow 3** `mlflow.genai.evaluate()` with scorers — `Correctness`, `RetrievalGroundedness`, `Guidelines` + custom (kinase extraction accuracy, structured-output validity). Datasets registered in Unity Catalog.

_Status: not started._
