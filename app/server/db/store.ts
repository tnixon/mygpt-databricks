// Typed data access over the Phase 4 Lakebase schema. Every method takes a Queryable that is
// already inside withUser()/withAdmin(), so RLS decides what each statement can see or change;
// these helpers never filter by user themselves (except to stamp ownership on inserts).

import type { Queryable } from './context';

export type Visibility = 'private' | 'group' | 'public';

export interface Library {
  id: string;
  name: string;
  description: string;
  owner_email: string;
  visibility: Visibility;
  group_name: string | null;
  share_history: boolean;
  kind: 'corpus' | 'upload';
  corpus: string | null;
  config_dataset: string;
  documents_language: string;
  status: 'empty' | 'ingesting' | 'ready' | 'failed';
  is_default: boolean;
  created_at: string;
  updated_at: string;
  n_documents?: number;
}

export interface LibraryDocument {
  id: string;
  library_id: string;
  filename: string;
  volume_path: string;
  size_bytes: number | null;
  content_type: string | null;
  status: 'uploaded' | 'parsing' | 'ready' | 'failed';
  error: string | null;
  n_chunks: number;
  uploaded_at: string;
}

export interface IngestRun {
  id: string;
  library_id: string;
  job_run_id: number | null;
  status: 'pending' | 'running' | 'succeeded' | 'failed';
  message: string | null;
  started_at: string;
  finished_at: string | null;
}

export interface SourceRow {
  rank: number;
  source_name?: string | null;
  corpus?: string | null;
  section?: string | null;
  page?: number | null;
  snippet?: string | null;
  vector_score?: number | null;
  bm25_score?: number | null;
  rerank_score?: number | null;
  nli?: string | null;
}

export interface ConversationSummary {
  id: string;
  owner_email: string;
  library_id: string | null;
  library_name: string | null;
  title: string;
  created_at: string;
  last_at: string;
  n_questions: number;
  shared: boolean;
}

export interface QaTurn {
  question_id: string;
  question_text: string;
  model: string | null;
  focus: unknown;
  qrs: number | null;
  asked_at: string;
  answer_id: string | null;
  answer_text: string | null;
  ars: number | null;
  hi_eq: number | null;
  hi_rf: number | null;
  model_endpoint: string | null;
  trace_id: string | null;
  sources: SourceRow[];
  my_rating: number | null;
  my_comment: string | null;
}

export interface LibraryInput {
  name: string;
  description?: string;
  visibility: Visibility;
  group_name?: string | null;
  share_history?: boolean;
  documents_language?: string;
}

/** Postgres text-search configs for MyGPT's document languages (others fall back to simple). */
const TS_CONFIGS: Record<string, string> = {
  english: 'english',
  french: 'french',
  spanish: 'spanish',
  portuguese: 'portuguese',
  german: 'german',
  italian: 'italian',
  dutch: 'dutch',
  arabic: 'arabic',
  hungarian: 'hungarian',
  hindi: 'hindi',
};

export function tsConfigFor(language: string): string {
  return TS_CONFIGS[language.toLowerCase()] ?? 'simple';
}

const LIB_COLS = `l.id, l.name, l.description, l.owner_email, l.visibility, l.group_name, l.share_history,
  l.kind, l.corpus, l.config_dataset, l.documents_language, l.status, l.is_default, l.created_at, l.updated_at`;

export class Store {
  constructor(readonly schema: string) {}

  // ------------------------------------------------------------------ libraries

  async listLibraries(q: Queryable): Promise<Library[]> {
    const s = this.schema;
    const r = await q.query<Library>(
      `SELECT ${LIB_COLS}, (SELECT count(*)::int FROM ${s}.library_documents d WHERE d.library_id = l.id) AS n_documents
       FROM ${s}.libraries l
       ORDER BY l.is_default DESC, (l.kind = 'corpus') DESC, lower(l.name)`,
    );
    return r.rows;
  }

  async getLibrary(q: Queryable, id: string): Promise<Library | null> {
    const r = await q.query<Library>(`SELECT ${LIB_COLS} FROM ${this.schema}.libraries l WHERE l.id = $1`, [id]);
    return r.rows[0] ?? null;
  }

  async getDefaultLibrary(q: Queryable): Promise<Library | null> {
    const r = await q.query<Library>(
      `SELECT ${LIB_COLS} FROM ${this.schema}.libraries l WHERE l.is_default ORDER BY l.created_at LIMIT 1`,
    );
    return r.rows[0] ?? null;
  }

  async createUploadLibrary(q: Queryable, owner: string, input: LibraryInput): Promise<Library> {
    const r = await q.query<Library>(
      `INSERT INTO ${this.schema}.libraries AS l
         (name, description, owner_email, visibility, group_name, share_history, kind, documents_language)
       VALUES ($1, $2, $3, $4, $5, $6, 'upload', $7)
       RETURNING ${LIB_COLS}`,
      [
        input.name,
        input.description ?? '',
        owner,
        input.visibility,
        input.visibility === 'group' ? (input.group_name ?? null) : null,
        input.share_history ?? false,
        input.documents_language ?? 'english',
      ],
    );
    return r.rows[0];
  }

  /** Returns null when the row isn't visible or isn't the caller's to change (RLS). */
  async updateLibrary(q: Queryable, id: string, patch: Partial<LibraryInput>): Promise<Library | null> {
    const r = await q.query<Library>(
      `UPDATE ${this.schema}.libraries AS l SET
         name          = coalesce($2, l.name),
         description   = coalesce($3, l.description),
         visibility    = coalesce($4, l.visibility),
         group_name    = CASE WHEN coalesce($4, l.visibility) = 'group' THEN coalesce($5, l.group_name) ELSE NULL END,
         share_history = coalesce($6, l.share_history),
         updated_at    = now()
       WHERE l.id = $1
       RETURNING ${LIB_COLS}`,
      [id, patch.name ?? null, patch.description ?? null, patch.visibility ?? null, patch.group_name ?? null,
       patch.share_history ?? null],
    );
    return r.rows[0] ?? null;
  }

  async setLibraryStatus(q: Queryable, id: string, status: Library['status']): Promise<void> {
    await q.query(`UPDATE ${this.schema}.libraries SET status = $2, updated_at = now() WHERE id = $1`, [id, status]);
  }

  async deleteLibrary(q: Queryable, id: string): Promise<boolean> {
    const r = await q.query(`DELETE FROM ${this.schema}.libraries WHERE id = $1 RETURNING id`, [id]);
    return r.rows.length > 0;
  }

  // ------------------------------------------------------------------ documents + ingest

  async listDocuments(q: Queryable, libraryId: string): Promise<LibraryDocument[]> {
    const r = await q.query<LibraryDocument>(
      `SELECT id, library_id, filename, volume_path, size_bytes, content_type, status, error, n_chunks, uploaded_at
       FROM ${this.schema}.library_documents WHERE library_id = $1 ORDER BY lower(filename)`,
      [libraryId],
    );
    return r.rows;
  }

  async addDocument(
    q: Queryable,
    doc: { id: string; library_id: string; filename: string; volume_path: string; size_bytes: number; content_type: string },
  ): Promise<LibraryDocument> {
    const r = await q.query<LibraryDocument>(
      `INSERT INTO ${this.schema}.library_documents (id, library_id, filename, volume_path, size_bytes, content_type)
       VALUES ($1, $2, $3, $4, $5, $6)
       RETURNING id, library_id, filename, volume_path, size_bytes, content_type, status, error, n_chunks, uploaded_at`,
      [doc.id, doc.library_id, doc.filename, doc.volume_path, doc.size_bytes, doc.content_type],
    );
    return r.rows[0];
  }

  async deleteDocument(q: Queryable, libraryId: string, docId: string): Promise<LibraryDocument | null> {
    const r = await q.query<LibraryDocument>(
      `DELETE FROM ${this.schema}.library_documents WHERE library_id = $1 AND id = $2
       RETURNING id, library_id, filename, volume_path, size_bytes, content_type, status, error, n_chunks, uploaded_at`,
      [libraryId, docId],
    );
    return r.rows[0] ?? null;
  }

  async countPendingDocuments(q: Queryable, libraryId: string): Promise<number> {
    const r = await q.query<{ n: number }>(
      `SELECT count(*)::int AS n FROM ${this.schema}.library_documents WHERE library_id = $1 AND status = 'uploaded'`,
      [libraryId],
    );
    return r.rows[0]?.n ?? 0;
  }

  async activeIngestRun(q: Queryable, libraryId: string): Promise<IngestRun | null> {
    const r = await q.query<IngestRun>(
      `SELECT id, library_id, job_run_id, status, message, started_at, finished_at
       FROM ${this.schema}.ingest_runs
       WHERE library_id = $1 AND status IN ('pending', 'running')
       ORDER BY started_at DESC LIMIT 1`,
      [libraryId],
    );
    return r.rows[0] ?? null;
  }

  async latestIngestRun(q: Queryable, libraryId: string): Promise<IngestRun | null> {
    const r = await q.query<IngestRun>(
      `SELECT id, library_id, job_run_id, status, message, started_at, finished_at
       FROM ${this.schema}.ingest_runs WHERE library_id = $1 ORDER BY started_at DESC LIMIT 1`,
      [libraryId],
    );
    return r.rows[0] ?? null;
  }

  async createIngestRun(q: Queryable, libraryId: string, createdBy: string): Promise<IngestRun> {
    const r = await q.query<IngestRun>(
      `INSERT INTO ${this.schema}.ingest_runs (library_id, created_by) VALUES ($1, $2)
       RETURNING id, library_id, job_run_id, status, message, started_at, finished_at`,
      [libraryId, createdBy],
    );
    return r.rows[0];
  }

  async updateIngestRun(
    q: Queryable,
    id: string,
    patch: { job_run_id?: number; status?: IngestRun['status']; message?: string; finished?: boolean },
  ): Promise<void> {
    await q.query(
      `UPDATE ${this.schema}.ingest_runs SET
         job_run_id  = coalesce($2, job_run_id),
         status      = coalesce($3, status),
         message     = coalesce($4, message),
         finished_at = CASE WHEN $5 THEN now() ELSE finished_at END
       WHERE id = $1`,
      [id, patch.job_run_id ?? null, patch.status ?? null, patch.message ?? null, patch.finished ?? false],
    );
  }

  // ------------------------------------------------------------------ sections (upload libraries)

  async uploadSections(q: Queryable, libraryId: string): Promise<{ section_title: string; section_count: number }[]> {
    const r = await q.query<{ section_title: string; section_count: number }>(
      `SELECT section AS section_title, count(*)::int AS section_count
       FROM ${this.schema}.upload_chunks WHERE library_id = $1
       GROUP BY section ORDER BY section_count DESC LIMIT 200`,
      [libraryId],
    );
    return r.rows;
  }

  // ------------------------------------------------------------------ conversations

  async createConversation(q: Queryable, owner: string, libraryId: string | null, title: string): Promise<string> {
    const r = await q.query<{ id: string }>(
      `INSERT INTO ${this.schema}.conversations (owner_email, library_id, title) VALUES ($1, $2, $3) RETURNING id`,
      [owner, libraryId, title.slice(0, 200)],
    );
    return r.rows[0].id;
  }

  /** True if the conversation exists and belongs to `owner` (only owners append turns). */
  async ownsConversation(q: Queryable, id: string, owner: string): Promise<boolean> {
    const r = await q.query(`SELECT 1 FROM ${this.schema}.conversations WHERE id = $1 AND owner_email = $2`, [id, owner]);
    return r.rows.length > 0;
  }

  /** Persist one grounded Q&A turn (question + answer + sources) and bump last_at. */
  async saveTurn(
    q: Queryable,
    turn: {
      conversation_id: string;
      question_text: string;
      model: string;
      focus: unknown;
      qrs: number | null;
      answer_text: string;
      ars: number | null;
      hi_eq: number | null;
      hi_rf: number | null;
      model_endpoint: string | null;
      trace_id: string | null;
      custom_outputs: unknown;
      sources: SourceRow[];
    },
  ): Promise<{ question_id: string; answer_id: string }> {
    const s = this.schema;
    const qr = await q.query<{ id: string }>(
      `INSERT INTO ${s}.questions (conversation_id, question_text, model, focus, qrs)
       VALUES ($1, $2, $3, $4, $5) RETURNING id`,
      [turn.conversation_id, turn.question_text, turn.model, JSON.stringify(turn.focus ?? null), turn.qrs],
    );
    const questionId = qr.rows[0].id;
    const ar = await q.query<{ id: string }>(
      `INSERT INTO ${s}.answers (question_id, answer_text, ars, hi_eq, hi_rf, model_endpoint, trace_id, custom_outputs)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id`,
      [questionId, turn.answer_text, turn.ars, turn.hi_eq, turn.hi_rf, turn.model_endpoint, turn.trace_id,
       JSON.stringify(turn.custom_outputs ?? null)],
    );
    const answerId = ar.rows[0].id;
    if (turn.sources.length > 0) {
      // one multi-row insert: 11 columns per source
      const values: unknown[] = [];
      const tuples = turn.sources.map((src, i) => {
        values.push(answerId, src.rank ?? i + 1, src.source_name ?? null, src.corpus ?? null, src.section ?? null,
          src.page ?? null, src.snippet ?? null, src.vector_score ?? null, src.bm25_score ?? null,
          src.rerank_score ?? null, src.nli ?? null);
        const b = i * 11;
        return `(${Array.from({ length: 11 }, (_, j) => `$${b + j + 1}`).join(', ')})`;
      });
      await q.query(
        `INSERT INTO ${s}.sources (answer_id, rank, source_name, corpus, section, page, snippet,
           vector_score, bm25_score, rerank_score, nli) VALUES ${tuples.join(', ')}`,
        values,
      );
    }
    await q.query(`UPDATE ${s}.conversations SET last_at = now() WHERE id = $1`, [turn.conversation_id]);
    return { question_id: questionId, answer_id: answerId };
  }

  async listConversations(q: Queryable, me: string, libraryId?: string): Promise<ConversationSummary[]> {
    const s = this.schema;
    const r = await q.query<ConversationSummary>(
      `SELECT c.id, c.owner_email, c.library_id, l.name AS library_name, c.title, c.created_at, c.last_at,
              (SELECT count(*)::int FROM ${s}.questions qq WHERE qq.conversation_id = c.id) AS n_questions,
              (c.owner_email <> $1) AS shared
       FROM ${s}.conversations c
       LEFT JOIN ${s}.libraries l ON l.id = c.library_id
       WHERE ($2::uuid IS NULL OR c.library_id = $2::uuid)
       ORDER BY c.last_at DESC
       LIMIT 200`,
      [me, libraryId ?? null],
    );
    return r.rows;
  }

  async getConversation(
    q: Queryable,
    me: string,
    id: string,
  ): Promise<{ conversation: ConversationSummary; turns: QaTurn[] } | null> {
    const s = this.schema;
    const cr = await q.query<ConversationSummary>(
      `SELECT c.id, c.owner_email, c.library_id, l.name AS library_name, c.title, c.created_at, c.last_at,
              (SELECT count(*)::int FROM ${s}.questions qq WHERE qq.conversation_id = c.id) AS n_questions,
              (c.owner_email <> $1) AS shared
       FROM ${s}.conversations c LEFT JOIN ${s}.libraries l ON l.id = c.library_id
       WHERE c.id = $2`,
      [me, id],
    );
    const conversation = cr.rows[0];
    if (!conversation) return null;
    const tr = await q.query<QaTurn>(
      `SELECT qq.id AS question_id, qq.question_text, qq.model, qq.focus, qq.qrs, qq.created_at AS asked_at,
              a.id AS answer_id, a.answer_text, a.ars, a.hi_eq, a.hi_rf, a.model_endpoint, a.trace_id,
              coalesce((SELECT json_agg(json_build_object(
                         'rank', so.rank, 'source_name', so.source_name, 'corpus', so.corpus, 'section', so.section,
                         'page', so.page, 'snippet', so.snippet, 'vector_score', so.vector_score,
                         'bm25_score', so.bm25_score, 'rerank_score', so.rerank_score, 'nli', so.nli)
                         ORDER BY so.rank)
                        FROM ${s}.sources so WHERE so.answer_id = a.id), '[]'::json) AS sources,
              f.rating AS my_rating, f.comment AS my_comment
       FROM ${s}.questions qq
       LEFT JOIN ${s}.answers a ON a.question_id = qq.id
       LEFT JOIN ${s}.feedback f ON f.answer_id = a.id AND f.user_email = $1
       WHERE qq.conversation_id = $2
       ORDER BY qq.created_at`,
      [me, id],
    );
    return { conversation, turns: tr.rows };
  }

  async deleteConversation(q: Queryable, id: string): Promise<boolean> {
    const r = await q.query(`DELETE FROM ${this.schema}.conversations WHERE id = $1 RETURNING id`, [id]);
    return r.rows.length > 0;
  }

  // ------------------------------------------------------------------ feedback

  /** Upsert my rating; returns the answer's trace id, or undefined if the answer isn't visible. */
  async saveFeedback(
    q: Queryable,
    me: string,
    answerId: string,
    rating: -1 | 0 | 1,
    comment: string,
  ): Promise<{ trace_id: string | null } | undefined> {
    const s = this.schema;
    const a = await q.query<{ trace_id: string | null }>(`SELECT trace_id FROM ${s}.answers WHERE id = $1`, [answerId]);
    if (a.rows.length === 0) return undefined;
    await q.query(
      `INSERT INTO ${s}.feedback (answer_id, user_email, rating, comment) VALUES ($1, $2, $3, $4)
       ON CONFLICT (answer_id, user_email) DO UPDATE SET rating = EXCLUDED.rating, comment = EXCLUDED.comment,
         updated_at = now()`,
      [answerId, me, rating, comment],
    );
    return a.rows[0];
  }
}
