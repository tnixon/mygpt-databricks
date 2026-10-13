// Reciprocal Rank Fusion of the dense (pgvector) and keyword (full-text) result lists —
// the "Lakebase Search" hybrid for upload libraries (gap 7).

export interface RankedHit {
  chunk_id: string;
  /** 1-based rank in the list it came from */
  rank: number;
}

export interface FusedHit {
  chunk_id: string;
  rrf: number;
  vector_rank: number | null;
  keyword_rank: number | null;
}

export const RRF_K = 60;

export function rrfFuse(vectorHits: RankedHit[], keywordHits: RankedHit[], k: number, rrfK = RRF_K): FusedHit[] {
  const fused = new Map<string, FusedHit>();
  const add = (hits: RankedHit[], key: 'vector_rank' | 'keyword_rank') => {
    for (const h of hits) {
      const cur = fused.get(h.chunk_id) ?? { chunk_id: h.chunk_id, rrf: 0, vector_rank: null, keyword_rank: null };
      if (cur[key] === null || h.rank < cur[key]) {
        cur.rrf += 1 / (rrfK + h.rank) - (cur[key] === null ? 0 : 1 / (rrfK + cur[key]));
        cur[key] = h.rank;
      }
      fused.set(h.chunk_id, cur);
    }
  };
  add(vectorHits, 'vector_rank');
  add(keywordHits, 'keyword_rank');
  return [...fused.values()]
    .sort((a, b) => b.rrf - a.rrf || (a.vector_rank ?? Infinity) - (b.vector_rank ?? Infinity))
    .slice(0, k);
}

/** Scale keyword ranks (ts_rank_cd, unbounded) into [0, 1] against the best hit. */
export function normalizeKeywordScores(scores: Map<string, number>): Map<string, number> {
  const max = Math.max(0, ...scores.values());
  return new Map([...scores].map(([id, s]) => [id, max > 0 ? Math.round((s / max) * 1e4) / 1e4 : 0]));
}
