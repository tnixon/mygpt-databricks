-- MyGPT Phase 5 — domain tools as Unity Catalog functions, served by managed MCP:
--   /api/2.0/mcp/functions/<catalog>/<tools_schema>
-- Ports MyGPT's FastMCP tools (reference/MyGPT/MCP-server/mygpt_backend.py). Every function
-- in this schema becomes an MCP tool named <catalog>__<tools_schema>__<fn>; its
-- description and argument schema come from the COMMENTs, so write them for the LLM.
-- A dedicated schema keeps helper / ETL objects in `<schema>` out of the tool list.
-- Job parameters (databricks.yml sql_task.parameters): catalog, schema, tools_schema.
--
-- Outputs are compact JSON strings: the Supervisor passes them to the LLM verbatim, and
-- raw API payloads waste its tokens. RCSB answers HTTP 204 (empty body) on zero hits.
-- MyGPT's `get_mygpt_datasets` / `get_mygpt_documents` are split: the governed corpora are
-- list_corpora / list_corpus_documents below; upload libraries live in Lakebase under RLS,
-- so the app hands them to the Supervisor instead (custom_inputs.libraries).

USE CATALOG IDENTIFIER(:catalog);
CREATE SCHEMA IF NOT EXISTS IDENTIFIER(:tools_schema)
  COMMENT 'MyGPT Phase 5: domain tools exposed to the Supervisor through managed MCP';
USE SCHEMA IDENTIFIER(:tools_schema);

-- --------------------------------------------------------------------------- PubMed
-- Port of get_pubmed_id_from_question. MyGPT matched bare 8-digit numbers only, which
-- misses older 7-digit PMIDs; explicit "PMID"/"PubMed" mentions accept 1-8 digits.
-- agents/tools/test_tools.py executes this body.
CREATE OR REPLACE FUNCTION pubmed_ids_from_text(
  text STRING COMMENT 'Free text that may mention PubMed ids, e.g. a user question or a citation list')
RETURNS ARRAY<STRING>
LANGUAGE PYTHON
COMMENT 'Extract PubMed ids (PMIDs) mentioned in a piece of text, in order of first mention, without duplicates. Use it before pubmed_summary when a question mentions papers by PubMed id.'
AS $$
import re

if not text:
    return []
found = []
for m in re.finditer(r"(?:\bPMID\s*:?\s*|\bPubMed(?:\s+ID)?\s*:?\s*|pubmed\.ncbi\.nlm\.nih\.gov/|/pubmed/)(\d{1,8})\b|\b(\d{7,8})\b", text, re.IGNORECASE):
    pmid = (m.group(1) or m.group(2)).lstrip("0")
    if pmid and pmid not in found:
        found.append(pmid)
return found
$$;

CREATE OR REPLACE FUNCTION pubmed_summary(
  pmids STRING COMMENT 'One or more PubMed ids separated by commas, e.g. 10498895,10838079 (at most 20)')
RETURNS STRING
COMMENT 'Look up papers in PubMed by id: returns JSON [{pubmed_id, title, journal, pubdate, authors (first 6), doi}]. Ids PubMed does not know are omitted. Use it to name, date or cite a paper; it does not return the paper text.'
RETURN (
  WITH r AS (
    SELECT from_json(
      http_request(conn => 'ncbi_eutils', method => 'GET', path => '/esummary.fcgi',
        params => map('db', 'pubmed', 'retmode', 'json',
                      'id', array_join(slice(split(regexp_replace(pmids, '[^0-9,]', ''), ','), 1, 20), ','))).text,
      'result MAP<STRING, STRING>').result AS m
  )
  SELECT to_json(transform(
    filter(coalesce(from_json(m['uids'], 'ARRAY<STRING>'), array()),
           u -> m[u] IS NOT NULL AND get_json_object(m[u], '$.error') IS NULL),
    u -> named_struct(
      'pubmed_id', u,
      'title', get_json_object(m[u], '$.title'),
      'journal', get_json_object(m[u], '$.fulljournalname'),
      'pubdate', get_json_object(m[u], '$.pubdate'),
      'authors', slice(transform(coalesce(from_json(get_json_object(m[u], '$.authors'), 'ARRAY<STRUCT<name: STRING>>'), array()), a -> a.name), 1, 6),
      'doi', element_at(transform(filter(coalesce(from_json(get_json_object(m[u], '$.articleids'), 'ARRAY<STRUCT<idtype: STRING, value: STRING>>'), array()), x -> x.idtype = 'doi'), x -> x.value), 1))))
  FROM r
);

-- --------------------------------------------------------------------------- RCSB PDB
-- Port of search_pdb, split in two after the spike: full-text hits are mentions, not
-- structures of the protein ("ABL1 kinase" = 4,448 text matches vs 74 human ABL1 entries),
-- and the LLM reported the text count as a structure count. The gene search is the
-- precise one; full text says what its count means.
CREATE OR REPLACE FUNCTION search_pdb_by_gene(
  gene STRING COMMENT 'Gene symbol of the protein, e.g. ABL1, MAPK14, EGFR',
  organism STRING DEFAULT 'Homo sapiens' COMMENT 'Source organism scientific name; default Homo sapiens')
RETURNS STRING
COMMENT 'Find experimentally solved 3D structures in the RCSB Protein Data Bank that contain the protein encoded by a gene: returns JSON {total, entries: [{rcsb_id, title, method, resolution_A, released}]} with the 10 best-resolved entries. Prefer this over search_pdb for "which structures exist for <protein>".'
RETURN (
  WITH hits AS (
    SELECT from_json(
      http_request(conn => 'rcsb_search', method => 'POST', path => '/query',
        headers => map('Content-Type', 'application/json'),
        json => to_json(named_struct(
          'query', named_struct('type', 'group', 'logical_operator', 'and', 'nodes', array(
            named_struct('type', 'terminal', 'service', 'text', 'parameters', named_struct(
              'attribute', 'rcsb_entity_source_organism.rcsb_gene_name.value', 'operator', 'exact_match', 'value', upper(trim(gene)))),
            named_struct('type', 'terminal', 'service', 'text', 'parameters', named_struct(
              'attribute', 'rcsb_entity_source_organism.scientific_name', 'operator', 'exact_match', 'value', coalesce(organism, 'Homo sapiens'))))),
          'return_type', 'entry',
          'request_options', named_struct(
            'paginate', named_struct('start', 0, 'rows', 10),
            'sort', array(named_struct('sort_by', 'rcsb_entry_info.resolution_combined', 'direction', 'asc')))))).text,
      'total_count BIGINT, result_set ARRAY<STRUCT<identifier: STRING, score: DOUBLE>>') AS r
  )
  SELECT CASE WHEN coalesce(size(r.result_set), 0) = 0 THEN '{"total":0,"entries":[]}'
    ELSE to_json(named_struct('total', r.total_count, 'entries', transform(
      from_json(get_json_object(
        http_request(conn => 'rcsb_data', method => 'POST', path => '/graphql',
          headers => map('Content-Type', 'application/json'),
          json => to_json(named_struct('query', concat(
            '{entries(entry_ids:[', array_join(transform(r.result_set, x -> concat('"', x.identifier, '"')), ','), ']){',
            'rcsb_id struct{title} exptl{method} rcsb_entry_info{resolution_combined} rcsb_accession_info{initial_release_date}}}')))).text,
        '$.data.entries'),
        'ARRAY<STRUCT<rcsb_id: STRING, `struct`: STRUCT<title: STRING>, exptl: ARRAY<STRUCT<method: STRING>>, rcsb_entry_info: STRUCT<resolution_combined: ARRAY<DOUBLE>>, rcsb_accession_info: STRUCT<initial_release_date: STRING>>>'),
      e -> named_struct('rcsb_id', e.rcsb_id, 'title', e.`struct`.title, 'method', e.exptl[0].method,
                        'resolution_A', e.rcsb_entry_info.resolution_combined[0],
                        'released', substr(e.rcsb_accession_info.initial_release_date, 1, 10)))))
    END
  FROM hits
);

CREATE OR REPLACE FUNCTION search_pdb(
  query STRING COMMENT 'Free text, e.g. a ligand, drug, disease or method: imatinib, cryo-EM spliceosome')
RETURNS STRING
COMMENT 'Full-text search of the RCSB Protein Data Bank: returns JSON {text_matches, entries: [{rcsb_id, title, method, resolution_A, released}]} for the 10 most relevant entries. text_matches counts entries whose text MENTIONS the query, not structures of that protein; use search_pdb_by_gene for structures of a protein.'
RETURN (
  WITH hits AS (
    SELECT from_json(
      http_request(conn => 'rcsb_search', method => 'POST', path => '/query',
        headers => map('Content-Type', 'application/json'),
        json => to_json(named_struct(
          'query', named_struct('type', 'terminal', 'service', 'full_text', 'parameters', named_struct('value', query)),
          'return_type', 'entry',
          'request_options', named_struct('paginate', named_struct('start', 0, 'rows', 10))))).text,
      'total_count BIGINT, result_set ARRAY<STRUCT<identifier: STRING, score: DOUBLE>>') AS r
  )
  SELECT CASE WHEN coalesce(size(r.result_set), 0) = 0 THEN '{"text_matches":0,"entries":[]}'
    ELSE to_json(named_struct('text_matches', r.total_count, 'entries', transform(
      from_json(get_json_object(
        http_request(conn => 'rcsb_data', method => 'POST', path => '/graphql',
          headers => map('Content-Type', 'application/json'),
          json => to_json(named_struct('query', concat(
            '{entries(entry_ids:[', array_join(transform(r.result_set, x -> concat('"', x.identifier, '"')), ','), ']){',
            'rcsb_id struct{title} exptl{method} rcsb_entry_info{resolution_combined} rcsb_accession_info{initial_release_date}}}')))).text,
        '$.data.entries'),
        'ARRAY<STRUCT<rcsb_id: STRING, `struct`: STRUCT<title: STRING>, exptl: ARRAY<STRUCT<method: STRING>>, rcsb_entry_info: STRUCT<resolution_combined: ARRAY<DOUBLE>>, rcsb_accession_info: STRUCT<initial_release_date: STRING>>>'),
      e -> named_struct('rcsb_id', e.rcsb_id, 'title', e.`struct`.title, 'method', e.exptl[0].method,
                        'resolution_A', e.rcsb_entry_info.resolution_combined[0],
                        'released', substr(e.rcsb_accession_info.initial_release_date, 1, 10)))))
    END
  FROM hits
);

-- Port of get_pdb_details, plus the molecules, ligands and primary citation (whose PubMed
-- id chains into pubmed_summary).
CREATE OR REPLACE FUNCTION get_pdb_details(
  pdb_id STRING COMMENT 'Four-character PDB id, e.g. 2HYY')
RETURNS STRING
COMMENT 'Details of one RCSB Protein Data Bank entry: returns JSON with title, experimental method, resolution, release date, the macromolecules (name, organism, gene), bound ligands (id, name) and the primary citation (title, PubMed id, DOI). Returns null when the id does not exist.'
RETURN get_json_object(
  http_request(conn => 'rcsb_data', method => 'POST', path => '/graphql',
    headers => map('Content-Type', 'application/json'),
    json => to_json(named_struct('query', concat(
      '{entry(entry_id:"', upper(regexp_replace(pdb_id, '[^A-Za-z0-9]', '')), '"){',
      'rcsb_id struct{title} exptl{method} rcsb_entry_info{resolution_combined} rcsb_accession_info{initial_release_date} ',
      'rcsb_primary_citation{title pdbx_database_id_PubMed pdbx_database_id_DOI} ',
      'polymer_entities{rcsb_polymer_entity{pdbx_description} rcsb_entity_source_organism{ncbi_scientific_name rcsb_gene_name{value}}} ',
      'nonpolymer_entities{nonpolymer_comp{chem_comp{id name}}}}}')))).text,
  '$.data.entry');

-- --------------------------------------------------------------------------- governed corpora
-- Governed side of get_mygpt_datasets / get_mygpt_documents (corpus_documents is built by
-- etl/structured_seed.py). A SQL function body resolves unqualified names against the catalog /
-- schema current when it is CREATEd, so switch to the data schema and name the functions into
-- the tools schema explicitly.
USE SCHEMA IDENTIFIER(:schema);

CREATE OR REPLACE FUNCTION IDENTIFIER(:tools_schema || '.list_corpora')()
RETURNS TABLE (corpus STRING, documents BIGINT, indexed_documents BIGINT)
COMMENT 'List the governed document collections (corpora) MyGPT can search, with how many documents each holds and how many are indexed. biomed = BioASQ biomedical papers, kinase_literature = BlueSky kinase papers, health_policies_fr / health_policies_es = SJ Global health policies.'
RETURN
  SELECT corpus, count(*), count_if(n_chunks > 0)
  FROM corpus_documents
  GROUP BY corpus
  ORDER BY corpus;

CREATE OR REPLACE FUNCTION IDENTIFIER(:tools_schema || '.list_corpus_documents')(
  corpus STRING COMMENT 'Corpus name from list_corpora, e.g. kinase_literature')
RETURNS TABLE (source_name STRING, indexed_chunks BIGINT)
COMMENT 'List the documents (file names) in one governed corpus, with their indexed chunk count, at most 500. Use the file names to focus document search on specific papers.'
RETURN
  SELECT d.source_name, d.n_chunks
  FROM corpus_documents d
  WHERE d.corpus = list_corpus_documents.corpus
  ORDER BY d.source_name
  LIMIT 500;

-- Sanity (through managed MCP the same calls are tools/call on <catalog>__<tools_schema>__<fn>):
--   SELECT <catalog>.<tools_schema>.search_pdb_by_gene('ABL1');
--   SELECT <catalog>.<tools_schema>.pubmed_summary('10498895,10838079');
--   SELECT * FROM <catalog>.<tools_schema>.list_corpora();
