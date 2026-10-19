// RLS suite for the Phase 4 schema, run against in-process Postgres (PGlite + pgvector).
//
// PGlite connects as a superuser, which bypasses RLS, so the suite first creates a plain
// NOSUPERUSER / NOBYPASSRLS role and runs everything as it. Like the app's service principal,
// that role owns the tables, and FORCE ROW LEVEL SECURITY is what makes the policies bind it.

import { PGlite } from '@electric-sql/pglite';
import { vector } from '@electric-sql/pglite-pgvector';
import { beforeAll, describe, expect, it } from 'vitest';
import { withAdmin, withUser, type Db, type DbClient, type Identity } from './context';
import { migrate } from './migrate';
import { Store, tsConfigFor } from './store';

const SCHEMA = 'mygpt_test';
const store = new Store(SCHEMA);

const alice: Identity = { email: 'alice@stjude.org', groups: ['kinase-lab'] };
const bob: Identity = { email: 'bob@stjude.org', groups: [] };
const carol: Identity = { email: 'carol@stjude.org', groups: ['kinase-lab'] };
const anon: Identity = { email: '', groups: [] };

let pg: PGlite;
let db: Db;

// PGlite has a single connection; serialize "pool" checkouts so transactions can't interleave.
function pgliteDb(p: PGlite): Db {
  let chain: Promise<void> = Promise.resolve();
  return {
    async connect(): Promise<DbClient> {
      let release!: () => void;
      const prev = chain;
      chain = new Promise<void>((r) => (release = r));
      await prev;
      return {
        query: async <T,>(text: string, values?: unknown[]) => {
          const r = await p.query<T>(text, values as never[]);
          return { rows: r.rows, rowCount: r.affectedRows ?? null };
        },
        exec: (sql: string) => p.exec(sql),
        release,
      };
    },
  };
}

const zeroVec = `[${Array.from({ length: 1024 }, () => 0.01).join(',')}]`;

beforeAll(async () => {
  pg = new PGlite({ extensions: { vector } });
  await pg.exec(`
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE ROLE app_sp NOSUPERUSER NOBYPASSRLS;
    GRANT CREATE ON DATABASE postgres TO app_sp;
    SET ROLE app_sp;
  `);
  db = pgliteDb(pg);
  await migrate(db, SCHEMA);
}, 60_000);

describe('migrations', () => {
  it('are idempotent', async () => {
    expect(await migrate(db, SCHEMA)).toEqual([]);
  });

  it('seed the governed corpora as public libraries', async () => {
    const libs = await withUser(db, anon, (q) => store.listLibraries(q));
    expect(libs.map((l) => l.corpus)).toEqual(
      expect.arrayContaining([null, 'biomed', 'health_policies_fr', 'health_policies_es', 'health_policies_hi',
        'kinase_literature']),
    );
    expect(libs.every((l) => l.kind === 'corpus' && l.visibility === 'public')).toBe(true);
    expect(libs[0].is_default).toBe(true);
  });
});

describe('library visibility', () => {
  let privateLib: string;
  let groupLib: string;
  let publicLib: string;

  beforeAll(async () => {
    privateLib = (await withUser(db, alice, (q) =>
      store.createUploadLibrary(q, alice.email, { name: 'alice private', visibility: 'private' }))).id;
    groupLib = (await withUser(db, alice, (q) =>
      store.createUploadLibrary(q, alice.email, {
        name: 'kinase lab', visibility: 'group', group_name: 'kinase-lab', share_history: true,
      }))).id;
    publicLib = (await withUser(db, alice, (q) =>
      store.createUploadLibrary(q, alice.email, { name: 'alice public', visibility: 'public' }))).id;
  });

  const visibleTo = async (who: Identity) =>
    (await withUser(db, who, (q) => store.listLibraries(q))).map((l) => l.id);

  it('private: owner only', async () => {
    expect(await visibleTo(alice)).toContain(privateLib);
    expect(await visibleTo(bob)).not.toContain(privateLib);
    expect(await visibleTo(carol)).not.toContain(privateLib);
  });

  it('group: owner + group members only', async () => {
    expect(await visibleTo(carol)).toContain(groupLib);
    expect(await visibleTo(bob)).not.toContain(groupLib);
  });

  it('upload libraries are scored with their language’s config', async () => {
    const fr = await withUser(db, alice, (q) =>
      store.createUploadLibrary(q, alice.email, { name: 'fr', visibility: 'private', documents_language: 'french' }));
    expect(fr.config_dataset).toBe('health_policies_fr');
    const de = await withUser(db, alice, (q) =>
      store.createUploadLibrary(q, alice.email, { name: 'de', visibility: 'private', documents_language: 'german' }));
    expect(de.config_dataset).toBe('default_german');
    expect((await withUser(db, alice, (q) => store.getLibrary(q, privateLib)))?.config_dataset).toBe('mygpt');
    await withUser(db, alice, (q) => store.deleteLibrary(q, fr.id));
    await withUser(db, alice, (q) => store.deleteLibrary(q, de.id));
  });

  it("Postgres has a text-search config for every library language (else 'simple')", async () => {
    const r = await withAdmin(db, (q) => q.query<{ cfgname: string }>('SELECT cfgname FROM pg_ts_config'));
    const have = new Set(r.rows.map((x) => x.cfgname));
    for (const lang of ['english', 'french', 'spanish', 'portuguese', 'german', 'italian', 'dutch', 'arabic',
      'hindi', 'hungarian']) {
      expect(have.has(tsConfigFor(lang)), lang).toBe(true);
    }
    expect(tsConfigFor('unknown')).toBe('simple');
  });

  it('public: everyone, including no context', async () => {
    expect(await visibleTo(bob)).toContain(publicLib);
    expect(await visibleTo(anon)).toContain(publicLib);
    expect(await visibleTo(anon)).not.toContain(privateLib);
  });

  it("others can't change or delete my library", async () => {
    expect(await withUser(db, bob, (q) => store.updateLibrary(q, publicLib, { name: 'pwned' }))).toBeNull();
    expect(await withUser(db, carol, (q) => store.deleteLibrary(q, groupLib))).toBe(false);
    expect((await withUser(db, alice, (q) => store.getLibrary(q, publicLib)))?.name).toBe('alice public');
  });

  it("users can't create libraries for someone else, or corpus libraries", async () => {
    await expect(
      withUser(db, bob, (q) => store.createUploadLibrary(q, alice.email, { name: 'spoof', visibility: 'private' })),
    ).rejects.toThrow(/row-level security/);
    await expect(
      withUser(db, bob, (q) =>
        q.query(`INSERT INTO ${SCHEMA}.libraries (name, owner_email, kind) VALUES ('x', $1, 'corpus')`, [bob.email])),
    ).rejects.toThrow(/row-level security/);
  });

  it("users can't edit or delete the seeded corpus libraries", async () => {
    const corpus = (await withUser(db, bob, (q) => store.getDefaultLibrary(q)))!;
    expect(await withUser(db, bob, (q) => store.deleteLibrary(q, corpus.id))).toBe(false);
  });

  it('documents + chunks follow library visibility; only the owner adds them', async () => {
    const docId = '00000000-0000-0000-0000-000000000001';
    await withUser(db, alice, (q) =>
      store.addDocument(q, {
        id: docId, library_id: privateLib, filename: 'a.pdf', volume_path: '/Volumes/x/a.pdf',
        size_bytes: 10, content_type: 'application/pdf',
      }));
    await withAdmin(db, (q) =>
      q.query(
        `INSERT INTO ${SCHEMA}.upload_chunks (chunk_id, library_id, document_id, source_name, chunk_text, embedding)
         VALUES ('c1', $1, $2, 'a.pdf', 'kinase inhibitors bind the ATP pocket', $3::vector)`,
        [privateLib, docId, zeroVec]));

    expect(await withUser(db, alice, (q) => store.listDocuments(q, privateLib))).toHaveLength(1);
    expect(await withUser(db, bob, (q) => store.listDocuments(q, privateLib))).toHaveLength(0);
    const bobChunks = await withUser(db, bob, (q) => q.query(`SELECT chunk_id FROM ${SCHEMA}.upload_chunks`));
    expect(bobChunks.rows).toHaveLength(0);
    await expect(
      withUser(db, bob, (q) =>
        store.addDocument(q, {
          id: '00000000-0000-0000-0000-000000000002', library_id: publicLib, filename: 'b.pdf',
          volume_path: '/Volumes/x/b.pdf', size_bytes: 1, content_type: 'application/pdf',
        })),
    ).rejects.toThrow(/row-level security/);
  });

  describe('conversations, history and feedback', () => {
    let privateConv: string;
    let sharedConv: string;
    let sharedAnswer: string;

    beforeAll(async () => {
      const turn = (conversation_id: string) => ({
        conversation_id, question_text: 'q?', model: 'sonnet', focus: {}, qrs: 50, answer_text: 'a',
        translated_question_text: 'q (fr)?', translated_answer_text: 'a (fr)',
        ars: 80, hi_eq: 5, hi_rf: null, model_endpoint: 'x', trace_id: 'tr-1', custom_outputs: {},
        sources: [{ rank: 1, source_name: 'a.pdf', page: 1, snippet: 's' }],
      });
      privateConv = await withUser(db, alice, (q) => store.createConversation(q, alice.email, [privateLib], 'mine'));
      await withUser(db, alice, (q) => store.saveTurn(q, turn(privateConv)));
      sharedConv = await withUser(db, alice, (q) => store.createConversation(q, alice.email, [groupLib], 'lab'));
      sharedAnswer = (await withUser(db, alice, (q) => store.saveTurn(q, turn(sharedConv)))).answer_id;
    });

    it('my conversations are private by default', async () => {
      const ids = async (who: Identity) =>
        (await withUser(db, who, (q) => store.listConversations(q, who.email))).map((c) => c.id);
      expect(await ids(alice)).toEqual(expect.arrayContaining([privateConv, sharedConv]));
      expect(await ids(bob)).toEqual([]);
    });

    it('share_history exposes a group library’s conversations to members, read-only', async () => {
      const carolView = await withUser(db, carol, (q) => store.getConversation(q, carol.email, sharedConv));
      expect(carolView?.conversation.shared).toBe(true);
      expect(carolView?.turns[0].sources).toHaveLength(1);
      expect([carolView?.turns[0].translated_question_text, carolView?.turns[0].translated_answer_text])
        .toEqual(['q (fr)?', 'a (fr)']);
      expect(await withUser(db, carol, (q) => store.getConversation(q, carol.email, privateConv))).toBeNull();
      expect(await withUser(db, carol, (q) => store.deleteConversation(q, sharedConv))).toBe(false);
      await expect(
        withUser(db, carol, (q) => store.saveTurn(q, {
          conversation_id: sharedConv, question_text: 'x', model: 'm', focus: {}, qrs: 0, answer_text: 'x',
          ars: 0, hi_eq: 0, hi_rf: null, model_endpoint: null, trace_id: null, custom_outputs: {}, sources: [],
        })),
      ).rejects.toThrow(/row-level security/);
    });

    it('a multi-library conversation stays owner-only, even when one library shares history', async () => {
      const multi = await withUser(db, alice, (q) => store.createConversation(q, alice.email, [groupLib, privateLib], 'both'));
      await withUser(db, alice, (q) => store.saveTurn(q, {
        conversation_id: multi, question_text: 'q', model: 'sonnet', focus: {}, qrs: null, answer_text: 'merged',
        ars: null, hi_eq: null, hi_rf: null, model_endpoint: null, trace_id: null,
        custom_outputs: { scores_scope: 'per_library' },
        sources: [{ rank: 1, library_id: groupLib, source_name: 'g.pdf' }, { rank: 2, library_id: privateLib, source_name: 'p.pdf' }],
      }));
      const mine = await withUser(db, alice, (q) => store.getConversation(q, alice.email, multi));
      expect(mine?.conversation.library_id).toBeNull();
      expect(mine?.conversation.library_ids).toEqual([groupLib, privateLib]);
      expect(mine?.turns[0].sources.map((s) => s.library_id)).toEqual([groupLib, privateLib]);
      expect(mine?.turns[0].custom_outputs).toEqual({ scores_scope: 'per_library' });
      expect(await withUser(db, alice, (q) => store.recentTurns(q, multi, 3))).toEqual([{ question: 'q', answer: 'merged' }]);
      expect(await withUser(db, carol, (q) => store.getConversation(q, carol.email, multi))).toBeNull();
    });

    it("users can't start a conversation as someone else or on a library they can't see", async () => {
      await expect(withUser(db, bob, (q) => store.createConversation(q, alice.email, [], 'x')))
        .rejects.toThrow(/row-level security/);
      await expect(withUser(db, bob, (q) => store.createConversation(q, bob.email, [privateLib], 'x')))
        .rejects.toThrow(/row-level security/);
    });

    it('feedback is per user, on visible answers only', async () => {
      expect(await withUser(db, carol, (q) => store.saveFeedback(q, carol.email, sharedAnswer, 1, 'good')))
        .toEqual({ trace_id: 'tr-1' });
      await withUser(db, alice, (q) => store.saveFeedback(q, alice.email, sharedAnswer, -1, 'wrong'));
      const carolTurn = (await withUser(db, carol, (q) => store.getConversation(q, carol.email, sharedConv)))!.turns[0];
      const aliceTurn = (await withUser(db, alice, (q) => store.getConversation(q, alice.email, sharedConv)))!.turns[0];
      expect([carolTurn.my_rating, aliceTurn.my_rating]).toEqual([1, -1]);
      expect(await withUser(db, bob, (q) => store.saveFeedback(q, bob.email, sharedAnswer, 1, ''))).toBeUndefined();
    });

    it('deleting a library cascades its documents and chunks; its conversations survive', async () => {
      expect(await withUser(db, alice, (q) => store.deleteLibrary(q, privateLib))).toBe(true);
      const left = await withAdmin(db, (q) =>
        q.query<{ n: number }>(`SELECT count(*)::int AS n FROM ${SCHEMA}.upload_chunks WHERE library_id = $1`, [privateLib]));
      expect(left.rows[0].n).toBe(0);
      const conv = await withUser(db, alice, (q) => store.getConversation(q, alice.email, privateConv));
      expect(conv?.conversation.library_id).toBeNull();
    });
  });
});
