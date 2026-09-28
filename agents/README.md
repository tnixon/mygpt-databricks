# agents — MyGPT multi-agent layer

The **Supervisor** agent (Mosaic AI Agent Framework) that routes each query to the right specialist — **Databricks AI Search** (governed docs), **Genie** (structured), **Hosted MCPs** (tools) — and calls models through the **Unity AI Gateway** (FMAPI primary + external Azure AI Foundry route). Also carries the QRS / ARS / Hallucination-style scoring the app surfaces. AI side. Python.

_Status: not started._
