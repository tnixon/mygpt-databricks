// Hybrid retrieval over an upload library in Lakebase: dense bge-m3 vectors (pgvector, cosine)
// + full-text BM25 over tsvector lexemes, fused with RRF. Runs inside withUser(), so RLS limits it
// to libraries the caller can read. The fused hits go to the agent as custom_inputs.sources,
// where they are reranked and scored exactly like AI Search hits.
//
// Score mapping onto the agent's AI Search conventions:
//   similarity = cosine similarity (1 - pgvector cosine distance), so the per-model bge-m3
//                cosine-distance cutoffs (embedding_model_cutoffs) still normalize it;
//   bm25_score = Okapi BM25 (computed in SQL, IDF over the library) scaled to [0, 1]
//                against the best keyword hit.

import { createWorkspaceClient } from '@databricks/appkit';
import type { ProvidedSource } from '../agent';
import { config } from '../config';
import type { Queryable } from '../db/context';
import { tsConfigFor, type Library } from '../db/store';
import { normalizeKeywordScores, rrfFuse } from './rrf';

const CANDIDATES = 30;

/** Parse an embeddings response: OpenAI envelope, or a bare list of vectors. The SDK's
 *  apiClient.request turns a top-level JSON array into an index-keyed object ({"0": [...]}),
 *  which is what `mygpt-embeddings` returns through it. */
export function parseEmbedding(resp: unknown): number[] {
  const r = resp as { data?: { embedding?: number[] }[]; predictions?: number[][]; 0?: number[] } | number[][];
  const vec = Array.isArray(r) ? r[0] : (r.data?.[0]?.embedding ?? r.predictions?.[0] ?? r[0]);
  if (!Array.isArray(vec) || vec.length === 0 || typeof vec[0] !== 'number') {
    throw new Error('unexpected embeddings response');
  }
  return vec;
}

/** Embed the query through the Unity AI Gateway endpoint (app service principal). */
export async function embedQuery(text: string): Promise<number[]> {
  const wc = createWorkspaceClient();
  const raw: unknown = await wc.apiClient.request({
    path: `/serving-endpoints/${config.embedEndpoint}/invocations`,
    method: 'POST',
    headers: new Headers({ 'Content-Type': 'application/json' }),
    raw: false,
    payload: { input: [text] },
  });
  return parseEmbedding(raw);
}

export interface SearchFocus {
  documents: string[];
  section?: string;
}

export async function searchUploadLibrary(
  q: Queryable,
  schema: string,
  library: Library,
  query: string,
  queryVector: number[],
  focus: SearchFocus,
  k: number,
): Promise<ProvidedSource[]> {
  const vec = `[${queryVector.join(',')}]`;
  const docs = focus.documents.length > 0 ? focus.documents : null;
  const section = focus.section && focus.section !== 'all' ? focus.section : null;
  const tsConfig = tsConfigFor(library.documents_language);
  await q.query("SELECT set_config('hnsw.ef_search', '200', true)");

  const filters = `library_id = $1 AND ($4::text[] IS NULL OR source_name = ANY($4)) AND ($5::text IS NULL OR section = $5)`;
  const lists = await q.query<{ list: 'v' | 'k'; chunk_id: string; rank: number; score: number }>(
    `WITH vec AS (
       SELECT chunk_id, (embedding <=> $2::vector) AS dist
       FROM ${schema}.upload_chunks WHERE ${filters}
       ORDER BY embedding <=> $2::vector LIMIT $6
     ), lib AS (
       SELECT chunk_id, tsv FROM ${schema}.upload_chunks WHERE ${filters}
     ), stats AS (
       SELECT greatest(count(*), 1)::float8 AS n, greatest(avg(length(tsv)), 1)::float8 AS avgdl FROM lib
     ), qlex AS (
       -- the question's lexemes (stemmed, stop words dropped) in the library's language
       SELECT DISTINCT lexeme FROM unnest(to_tsvector($3::regconfig, $7))
     ), terms AS (
       -- GIN-prefiltered to chunks containing ANY query lexeme, then per-term frequencies
       SELECT l.chunk_id, u.lexeme, coalesce(array_length(u.positions, 1), 1)::float8 AS tf,
              length(l.tsv)::float8 AS dl
       FROM lib l
       CROSS JOIN LATERAL unnest(l.tsv) u
       WHERE l.tsv @@ (SELECT nullif(replace(plainto_tsquery($3::regconfig, $7)::text, ' & ', ' | '), '')::tsquery)
         AND u.lexeme IN (SELECT lexeme FROM qlex)
     ), df AS (
       SELECT lexeme, count(*)::float8 AS df FROM terms GROUP BY lexeme
     ), kw AS (
       -- Okapi BM25 (k1 = 1.2, b = 0.75) with IDF over the library — MyGPT's bm25s
       -- equivalent. Plain ts_rank_cd has no IDF, so common words ("cancer", "plan") swamp
       -- the rare ones that actually discriminate.
       SELECT t.chunk_id,
              sum(ln(1 + (s.n - d.df + 0.5) / (d.df + 0.5))
                  * (t.tf * 2.2) / (t.tf + 1.2 * (0.25 + 0.75 * t.dl / s.avgdl))) AS kscore
       FROM terms t JOIN df d USING (lexeme) CROSS JOIN stats s
       GROUP BY t.chunk_id
       ORDER BY kscore DESC LIMIT $6
     )
     SELECT 'v' AS list, chunk_id, (row_number() OVER (ORDER BY dist))::int AS rank, 1 - dist AS score FROM vec
     UNION ALL
     SELECT 'k', chunk_id, (row_number() OVER (ORDER BY kscore DESC))::int, kscore FROM kw`,
    [library.id, vec, tsConfig, docs, section, CANDIDATES, query],
  );
  const vectorHits = lists.rows.filter((r) => r.list === 'v');
  const keywordHits = lists.rows.filter((r) => r.list === 'k');
  const fused = rrfFuse(vectorHits, keywordHits, k);
  if (fused.length === 0) return [];

  const keyword = normalizeKeywordScores(new Map(keywordHits.map((h) => [h.chunk_id, Number(h.score)])));
  const rows = await q.query<{
    chunk_id: string;
    source_name: string;
    section: string;
    page: number | null;
    chunk_text: string;
    similarity: number;
  }>(
    `SELECT chunk_id, source_name, section, page, chunk_text, 1 - (embedding <=> $2::vector) AS similarity
     FROM ${schema}.upload_chunks WHERE chunk_id = ANY($1)`,
    [fused.map((h) => h.chunk_id), vec],
  );
  const byId = new Map(rows.rows.map((r) => [r.chunk_id, r]));
  return fused.flatMap((h) => {
    const r = byId.get(h.chunk_id);
    if (!r) return [];
    return [{
      chunk_id: r.chunk_id,
      source_name: r.source_name,
      corpus: `upload:${library.id}`,
      section: r.section,
      page: r.page,
      chunk_text: r.chunk_text,
      similarity: Math.round(Number(r.similarity) * 1e4) / 1e4,
      bm25_score: keyword.get(r.chunk_id) ?? 0,
    }];
  });
}
