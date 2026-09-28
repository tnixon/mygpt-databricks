# etl — Ingestion & medallion

Lakeflow Connect pipelines: **SharePoint / OneDrive** documents → **Unity Catalog** (`structural_biology`), bronze → silver → gold, with **PHI scanning on ingest**. Gold docs are embedded into **Databricks AI Search** (the global governed document index); structured tables feed **Genie**.

_Status: not started._
