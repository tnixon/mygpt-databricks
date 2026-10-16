-- MyGPT Phase 5 — governed HTTP connections for the domain tools (mygpt_deploy_tools).
-- The UC functions in 02 reach external APIs only through these connections
-- (`http_request(conn => ...)`), so egress is governed and auditable in Unity Catalog.
-- All three APIs are public and unauthenticated; HTTP connections require a bearer token,
-- so a placeholder is set (the APIs ignore it).
-- IF NOT EXISTS: rcsb_search was created by the 2026-10-15 feasibility spike.

CREATE CONNECTION IF NOT EXISTS rcsb_search TYPE HTTP
  OPTIONS (host 'https://search.rcsb.org', port '443', base_path '/rcsbsearch/v2', bearer_token 'public')
  COMMENT 'MyGPT tools: RCSB PDB Search API v2 (public, no auth)';

CREATE CONNECTION IF NOT EXISTS rcsb_data TYPE HTTP
  OPTIONS (host 'https://data.rcsb.org', port '443', base_path '/', bearer_token 'public')
  COMMENT 'MyGPT tools: RCSB PDB Data API (GraphQL; public, no auth)';

CREATE CONNECTION IF NOT EXISTS ncbi_eutils TYPE HTTP
  OPTIONS (host 'https://eutils.ncbi.nlm.nih.gov', port '443', base_path '/entrez/eutils', bearer_token 'public')
  COMMENT 'MyGPT tools: NCBI E-utilities (PubMed esummary; public, no key = 3 requests/s)';

-- The Supervisor's serving identity gets USE CONNECTION / EXECUTE through the resources
-- declared in agents/supervisor/driver.py (DatabricksUCConnection / DatabricksFunction).
