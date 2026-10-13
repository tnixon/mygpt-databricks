// Runs the upload-library hybrid search SQL against in-process Postgres (PGlite + pgvector).

import { PGlite } from '@electric-sql/pglite';
import { vector } from '@electric-sql/pglite-pgvector';
import { beforeAll, describe, expect, it } from 'vitest';
import { withAdmin, withUser, type Db, type Identity } from '../db/context';
import { migrate } from '../db/migrate';
import { Store, type Library } from '../db/store';
import { parseEmbedding, searchUploadLibrary } from './lakebaseSearch';

const SCHEMA = 'mygpt_search';
const store = new Store(SCHEMA);
const owner: Identity = { email: 'owner@stjude.org', groups: [] };
const stranger: Identity = { email: 'stranger@stjude.org', groups: [] };

/** A unit vector along `axis` (plus a small shared component) — makes cosine order obvious. */
const vec = (axis: number) => Array.from({ length: 1024 }, (_, i) => (i === axis ? 1 : 0.01));

let db: Db;
let lib: Library;

beforeAll(async () => {
  const pg = new PGlite({ extensions: { vector } });
  await pg.exec(`CREATE EXTENSION IF NOT EXISTS vector;
    CREATE ROLE app_sp NOSUPERUSER NOBYPASSRLS; GRANT CREATE ON DATABASE postgres TO app_sp; SET ROLE app_sp;`);
  db = {
    connect: () => Promise.resolve({
      query: async <T,>(text: string, values?: unknown[]) => ({ rows: (await pg.query<T>(text, values as never[])).rows }),
      exec: (sql: string) => pg.exec(sql),
      release: () => undefined,
    }),
  };
  await migrate(db, SCHEMA);
  lib = await withUser(db, owner, (q) =>
    store.createUploadLibrary(q, owner.email, { name: 'papers', visibility: 'private', documents_language: 'english' }));
  const docId = '00000000-0000-0000-0000-0000000000aa';
  await withUser(db, owner, (q) => store.addDocument(q, {
    id: docId, library_id: lib.id, filename: 'p.pdf', volume_path: '/v/p.pdf', size_bytes: 1, content_type: 'application/pdf',
  }));
  const chunks: [string, string, string, number][] = [
    ['c-imatinib', 'Methods', 'Imatinib inhibits the ABL1 kinase by binding its inactive conformation.', 0],
    ['c-egfr', 'Results', 'EGFR mutations confer sensitivity to gefitinib in lung cancer.', 1],
    ['c-other', 'Results', 'Patient recruitment followed the standard protocol.', 2],
  ];
  await withAdmin(db, async (q) => {
    for (const [id, section, text, axis] of chunks) {
      await q.query(
        `INSERT INTO ${SCHEMA}.upload_chunks (chunk_id, library_id, document_id, source_name, section, page, chunk_text, embedding, ts_config)
         VALUES ($1, $2, $3, 'p.pdf', $4, 1, $5, $6::vector, 'english')`,
        [id, lib.id, docId, section, text, `[${vec(axis).join(',')}]`]);
    }
  });
}, 60_000);

describe('searchUploadLibrary', () => {
  it('fuses dense + keyword hits and maps scores onto the agent conventions', async () => {
    const hits = await withUser(db, owner, (q) =>
      searchUploadLibrary(q, SCHEMA, lib, 'imatinib kinase', vec(0), { documents: [] }, 3));
    expect(hits[0]).toMatchObject({ chunk_id: 'c-imatinib', corpus: `upload:${lib.id}`, section: 'Methods', bm25_score: 1 });
    expect(hits[0].similarity).toBeCloseTo(1, 1);
    expect(hits.map((h) => h.chunk_id)).toHaveLength(3);
    expect(hits.find((h) => h.chunk_id === 'c-other')?.bm25_score).toBe(0);
  });

  it('applies the section focus', async () => {
    const hits = await withUser(db, owner, (q) =>
      searchUploadLibrary(q, SCHEMA, lib, 'kinase', vec(0), { documents: [], section: 'Results' }, 5));
    expect(hits.map((h) => h.chunk_id).sort()).toEqual(['c-egfr', 'c-other']);
  });

  it('returns nothing for a caller who cannot read the library (RLS)', async () => {
    const hits = await withUser(db, stranger, (q) =>
      searchUploadLibrary(q, SCHEMA, lib, 'imatinib', vec(0), { documents: [] }, 3));
    expect(hits).toEqual([]);
  });
});

describe('parseEmbedding', () => {
  it('accepts the gateway envelope and a bare vector list', () => {
    expect(parseEmbedding({ data: [{ embedding: [0.1, 0.2] }] })).toEqual([0.1, 0.2]);
    expect(parseEmbedding([[0.3]])).toEqual([0.3]);
    expect(() => parseEmbedding({})).toThrow();
  });
});
