import { describe, expect, it } from 'vitest';
import { normalizeKeywordScores, rrfFuse } from './rrf';

describe('rrfFuse', () => {
  it('ranks chunks found by both lists above single-list hits', () => {
    const fused = rrfFuse(
      [{ chunk_id: 'a', rank: 1 }, { chunk_id: 'b', rank: 2 }, { chunk_id: 'c', rank: 3 }],
      [{ chunk_id: 'c', rank: 1 }, { chunk_id: 'd', rank: 2 }],
      10,
    );
    expect(fused[0].chunk_id).toBe('c');
    expect(fused.map((h) => h.chunk_id)).toEqual(['c', 'a', 'b', 'd']);
    expect(fused[0]).toMatchObject({ vector_rank: 3, keyword_rank: 1 });
  });

  it('truncates to k and keeps the best rank on duplicates', () => {
    const fused = rrfFuse([{ chunk_id: 'a', rank: 5 }, { chunk_id: 'a', rank: 2 }], [], 1);
    expect(fused).toHaveLength(1);
    expect(fused[0].vector_rank).toBe(2);
    expect(fused[0].rrf).toBeCloseTo(1 / 62);
  });

  it('breaks ties toward the better vector rank', () => {
    const fused = rrfFuse([{ chunk_id: 'v', rank: 1 }], [{ chunk_id: 'k', rank: 1 }], 2);
    expect(fused.map((h) => h.chunk_id)).toEqual(['v', 'k']);
  });
});

describe('normalizeKeywordScores', () => {
  it('scales to the best hit and handles empty / zero input', () => {
    expect([...normalizeKeywordScores(new Map([['a', 0.4], ['b', 0.1]]))]).toEqual([['a', 1], ['b', 0.25]]);
    expect([...normalizeKeywordScores(new Map([['a', 0]]))]).toEqual([['a', 0]]);
    expect(normalizeKeywordScores(new Map()).size).toBe(0);
  });
});
