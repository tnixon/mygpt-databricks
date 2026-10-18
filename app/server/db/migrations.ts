// Phase 4 Lakebase schema: libraries, the per-user upload index, conversation history and
// feedback, with row-level security (RLS) on every table.
//
// Migrations are inlined as strings (not .sql files) so tsdown bundles them with the server.
// `{{schema}}` is replaced at apply time (default `mygpt`; the RLS tests use a scratch schema).
//
// RLS model (session context + policies; ports MyGPT's resolve_dataset_for_user / get_datasets):
//   * The app connects as its service principal (SP), which OWNS these tables. FORCE ROW LEVEL
//     SECURITY makes the policies apply to the owner too, so every query is filtered.
//   * Per transaction the app sets `mygpt.user` (the signed-in email from the Apps proxy's
//     x-forwarded-email header), `mygpt.groups` (a JSON array of the user's workspace groups)
//     and `mygpt.role` (`admin` only for seeding / the upload-ingest job). See db/context.ts.
//   * TRUST BOUNDARY: users never get SQL access; only the app and the ingest job connect, and
//     they set the context. With no context set, only public libraries are visible (fail closed).
//   * Child tables inherit visibility through `IN (SELECT id FROM <parent>)` subqueries, which
//     are themselves RLS-filtered, so the library rule is written once.

export interface Migration {
  id: string;
  sql: string;
}

export const MIGRATIONS: Migration[] = [
  {
    id: '001_init',
    sql: `
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- session context
CREATE OR REPLACE FUNCTION {{schema}}.me() RETURNS text LANGUAGE sql STABLE AS $$
  SELECT coalesce(current_setting('mygpt.user', true), '')
$$;

CREATE OR REPLACE FUNCTION {{schema}}.my_groups() RETURNS text[] LANGUAGE sql STABLE AS $$
  SELECT coalesce(
    ARRAY(SELECT jsonb_array_elements_text(nullif(current_setting('mygpt.groups', true), '')::jsonb)),
    '{}'::text[])
$$;

CREATE OR REPLACE FUNCTION {{schema}}.is_admin() RETURNS boolean LANGUAGE sql STABLE AS $$
  SELECT coalesce(current_setting('mygpt.role', true), '') = 'admin'
$$;

-- ---------------------------------------------------------------- libraries
-- One row per library (MyGPT Dataset). kind=corpus points at a governed AI Search corpus
-- (seeded, admin-managed); kind=upload is a user library indexed in upload_chunks.
CREATE TABLE {{schema}}.libraries (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name               text NOT NULL CHECK (length(name) BETWEEN 1 AND 120),
  description        text NOT NULL DEFAULT '',
  owner_email        text NOT NULL CHECK (owner_email <> ''),
  visibility         text NOT NULL DEFAULT 'private' CHECK (visibility IN ('private', 'group', 'public')),
  group_name         text,
  -- shared chat history: members of a non-private library can read its conversations
  share_history      boolean NOT NULL DEFAULT false,
  kind               text NOT NULL CHECK (kind IN ('corpus', 'upload')),
  corpus             text,                                -- AI Search corpus (NULL = all corpora)
  config_dataset     text NOT NULL DEFAULT 'mygpt',       -- dataset_config row the agent scores with
  documents_language text NOT NULL DEFAULT 'english',
  status             text NOT NULL DEFAULT 'empty' CHECK (status IN ('empty', 'ingesting', 'ready', 'failed')),
  is_default         boolean NOT NULL DEFAULT false,
  created_at         timestamptz NOT NULL DEFAULT now(),
  updated_at         timestamptz NOT NULL DEFAULT now(),
  CHECK ((visibility = 'group') = (group_name IS NOT NULL)),
  UNIQUE (owner_email, name)
);

CREATE TABLE {{schema}}.library_documents (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  library_id   uuid NOT NULL REFERENCES {{schema}}.libraries ON DELETE CASCADE,
  filename     text NOT NULL,
  volume_path  text NOT NULL,
  size_bytes   bigint,
  content_type text,
  status       text NOT NULL DEFAULT 'uploaded' CHECK (status IN ('uploaded', 'parsing', 'ready', 'failed')),
  error        text,
  n_chunks     integer NOT NULL DEFAULT 0,
  uploaded_at  timestamptz NOT NULL DEFAULT now(),
  updated_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON {{schema}}.library_documents (library_id);

-- Per-user upload index (gap 7): bge-m3 dense vectors (cosine HNSW) + full-text (GIN),
-- fused with RRF at query time (server/retrieval/lakebaseSearch.ts).
CREATE TABLE {{schema}}.upload_chunks (
  chunk_id    text PRIMARY KEY,
  library_id  uuid NOT NULL REFERENCES {{schema}}.libraries ON DELETE CASCADE,
  document_id uuid NOT NULL REFERENCES {{schema}}.library_documents ON DELETE CASCADE,
  source_name text NOT NULL,
  section     text NOT NULL DEFAULT '(no section)',
  page        integer,
  chunk_text  text NOT NULL,
  embedding   vector(1024) NOT NULL,
  ts_config   regconfig NOT NULL DEFAULT 'simple',
  tsv         tsvector GENERATED ALWAYS AS (to_tsvector(ts_config, chunk_text)) STORED
);
CREATE INDEX ON {{schema}}.upload_chunks (library_id);
CREATE INDEX ON {{schema}}.upload_chunks (document_id);
CREATE INDEX ON {{schema}}.upload_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON {{schema}}.upload_chunks USING gin (tsv);

CREATE TABLE {{schema}}.ingest_runs (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  library_id  uuid NOT NULL REFERENCES {{schema}}.libraries ON DELETE CASCADE,
  job_run_id  bigint,
  status      text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'succeeded', 'failed')),
  message     text,
  created_by  text NOT NULL,
  started_at  timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz
);
CREATE INDEX ON {{schema}}.ingest_runs (library_id, started_at DESC);

-- ---------------------------------------------------------------- history + feedback (gap 9)
CREATE TABLE {{schema}}.conversations (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_email text NOT NULL CHECK (owner_email <> ''),
  library_id  uuid REFERENCES {{schema}}.libraries ON DELETE SET NULL,
  title       text NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  last_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON {{schema}}.conversations (owner_email, last_at DESC);
CREATE INDEX ON {{schema}}.conversations (library_id, last_at DESC);

CREATE TABLE {{schema}}.questions (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id uuid NOT NULL REFERENCES {{schema}}.conversations ON DELETE CASCADE,
  question_text   text NOT NULL,
  model           text,
  focus           jsonb,
  qrs             double precision,
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON {{schema}}.questions (conversation_id, created_at);

CREATE TABLE {{schema}}.answers (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  question_id    uuid NOT NULL UNIQUE REFERENCES {{schema}}.questions ON DELETE CASCADE,
  answer_text    text NOT NULL,
  ars            double precision,
  hi_eq          double precision,
  hi_rf          double precision,
  model_endpoint text,
  trace_id       text,
  custom_outputs jsonb,
  created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE {{schema}}.sources (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  answer_id    uuid NOT NULL REFERENCES {{schema}}.answers ON DELETE CASCADE,
  rank         integer NOT NULL,
  source_name  text,
  corpus       text,
  section      text,
  page         integer,
  snippet      text,
  vector_score double precision,
  bm25_score   double precision,
  rerank_score double precision,
  nli          text
);
CREATE INDEX ON {{schema}}.sources (answer_id, rank);

-- MyGPT rating_types: 1 correct, 0 n/a, -1 incorrect. One row per (answer, user), so members
-- of a shared library can each rate its answers.
CREATE TABLE {{schema}}.feedback (
  answer_id  uuid NOT NULL REFERENCES {{schema}}.answers ON DELETE CASCADE,
  user_email text NOT NULL CHECK (user_email <> ''),
  rating     smallint NOT NULL CHECK (rating IN (-1, 0, 1)),
  comment    text NOT NULL DEFAULT '',
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (answer_id, user_email)
);

-- ---------------------------------------------------------------- row-level security
ALTER TABLE {{schema}}.libraries         ENABLE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.library_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.upload_chunks     ENABLE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.ingest_runs       ENABLE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.conversations     ENABLE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.questions         ENABLE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.answers           ENABLE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.sources           ENABLE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.feedback          ENABLE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.libraries         FORCE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.library_documents FORCE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.upload_chunks     FORCE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.ingest_runs       FORCE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.conversations     FORCE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.questions         FORCE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.answers           FORCE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.sources           FORCE ROW LEVEL SECURITY;
ALTER TABLE {{schema}}.feedback          FORCE ROW LEVEL SECURITY;

-- Libraries: readable if public, mine, or shared with one of my groups. Users create/edit/
-- delete only their own UPLOAD libraries; corpus libraries are admin-seeded.
CREATE POLICY lib_read ON {{schema}}.libraries FOR SELECT USING (
  visibility = 'public'
  OR owner_email = {{schema}}.me()
  OR (visibility = 'group' AND group_name = ANY ({{schema}}.my_groups()))
  OR {{schema}}.is_admin());
CREATE POLICY lib_write ON {{schema}}.libraries FOR ALL
  USING ((owner_email = {{schema}}.me() AND kind = 'upload') OR {{schema}}.is_admin())
  WITH CHECK ((owner_email = {{schema}}.me() AND kind = 'upload') OR {{schema}}.is_admin());

-- Library contents: readable with the library, writable by its owner.
CREATE POLICY doc_read ON {{schema}}.library_documents FOR SELECT
  USING (library_id IN (SELECT id FROM {{schema}}.libraries));
CREATE POLICY doc_write ON {{schema}}.library_documents FOR ALL
  USING (EXISTS (SELECT 1 FROM {{schema}}.libraries l WHERE l.id = library_id
                 AND (l.owner_email = {{schema}}.me() OR {{schema}}.is_admin())))
  WITH CHECK (EXISTS (SELECT 1 FROM {{schema}}.libraries l WHERE l.id = library_id
                 AND (l.owner_email = {{schema}}.me() OR {{schema}}.is_admin())));

CREATE POLICY chunk_read ON {{schema}}.upload_chunks FOR SELECT
  USING (library_id IN (SELECT id FROM {{schema}}.libraries));
CREATE POLICY chunk_write ON {{schema}}.upload_chunks FOR ALL
  USING (EXISTS (SELECT 1 FROM {{schema}}.libraries l WHERE l.id = library_id
                 AND (l.owner_email = {{schema}}.me() OR {{schema}}.is_admin())))
  WITH CHECK (EXISTS (SELECT 1 FROM {{schema}}.libraries l WHERE l.id = library_id
                 AND (l.owner_email = {{schema}}.me() OR {{schema}}.is_admin())));

CREATE POLICY run_read ON {{schema}}.ingest_runs FOR SELECT
  USING (library_id IN (SELECT id FROM {{schema}}.libraries));
CREATE POLICY run_write ON {{schema}}.ingest_runs FOR ALL
  USING (EXISTS (SELECT 1 FROM {{schema}}.libraries l WHERE l.id = library_id
                 AND (l.owner_email = {{schema}}.me() OR {{schema}}.is_admin())))
  WITH CHECK (EXISTS (SELECT 1 FROM {{schema}}.libraries l WHERE l.id = library_id
                 AND (l.owner_email = {{schema}}.me() OR {{schema}}.is_admin())));

-- Conversations: mine, or (read-only) on a non-private library whose owner shares history.
CREATE POLICY conv_read ON {{schema}}.conversations FOR SELECT USING (
  owner_email = {{schema}}.me()
  OR {{schema}}.is_admin()
  OR (library_id IS NOT NULL AND EXISTS (
        SELECT 1 FROM {{schema}}.libraries l
        WHERE l.id = library_id AND l.share_history AND l.visibility <> 'private')));
CREATE POLICY conv_write ON {{schema}}.conversations FOR ALL
  USING (owner_email = {{schema}}.me() OR {{schema}}.is_admin())
  WITH CHECK ((owner_email = {{schema}}.me()
               AND (library_id IS NULL OR library_id IN (SELECT id FROM {{schema}}.libraries)))
              OR {{schema}}.is_admin());

-- Q&A rows follow their conversation; only its owner writes them.
CREATE POLICY q_read ON {{schema}}.questions FOR SELECT
  USING (conversation_id IN (SELECT id FROM {{schema}}.conversations));
CREATE POLICY q_write ON {{schema}}.questions FOR ALL
  USING (EXISTS (SELECT 1 FROM {{schema}}.conversations c WHERE c.id = conversation_id
                 AND (c.owner_email = {{schema}}.me() OR {{schema}}.is_admin())))
  WITH CHECK (EXISTS (SELECT 1 FROM {{schema}}.conversations c WHERE c.id = conversation_id
                 AND (c.owner_email = {{schema}}.me() OR {{schema}}.is_admin())));

CREATE POLICY a_read ON {{schema}}.answers FOR SELECT
  USING (question_id IN (SELECT id FROM {{schema}}.questions));
CREATE POLICY a_write ON {{schema}}.answers FOR ALL
  USING (EXISTS (SELECT 1 FROM {{schema}}.questions q JOIN {{schema}}.conversations c ON c.id = q.conversation_id
                 WHERE q.id = question_id AND (c.owner_email = {{schema}}.me() OR {{schema}}.is_admin())))
  WITH CHECK (EXISTS (SELECT 1 FROM {{schema}}.questions q JOIN {{schema}}.conversations c ON c.id = q.conversation_id
                 WHERE q.id = question_id AND (c.owner_email = {{schema}}.me() OR {{schema}}.is_admin())));

CREATE POLICY s_read ON {{schema}}.sources FOR SELECT
  USING (answer_id IN (SELECT id FROM {{schema}}.answers));
CREATE POLICY s_write ON {{schema}}.sources FOR ALL
  USING (EXISTS (SELECT 1 FROM {{schema}}.answers a
                 JOIN {{schema}}.questions q ON q.id = a.question_id
                 JOIN {{schema}}.conversations c ON c.id = q.conversation_id
                 WHERE a.id = answer_id AND (c.owner_email = {{schema}}.me() OR {{schema}}.is_admin())))
  WITH CHECK (EXISTS (SELECT 1 FROM {{schema}}.answers a
                 JOIN {{schema}}.questions q ON q.id = a.question_id
                 JOIN {{schema}}.conversations c ON c.id = q.conversation_id
                 WHERE a.id = answer_id AND (c.owner_email = {{schema}}.me() OR {{schema}}.is_admin())));

-- Feedback: visible with its answer; each user writes only their own rating.
CREATE POLICY fb_read ON {{schema}}.feedback FOR SELECT
  USING (answer_id IN (SELECT id FROM {{schema}}.answers));
CREATE POLICY fb_write ON {{schema}}.feedback FOR ALL
  USING (user_email = {{schema}}.me() OR {{schema}}.is_admin())
  WITH CHECK ((user_email = {{schema}}.me() AND answer_id IN (SELECT id FROM {{schema}}.answers))
              OR {{schema}}.is_admin());
`,
  },
  {
    // The governed AI Search corpora (Phase 2 ETL) as public, admin-owned libraries.
    // config_dataset / documents_language mirror etl/sql/03_sections_and_config.sql.
    id: '002_seed_corpus_libraries',
    sql: `
SELECT set_config('mygpt.role', 'admin', true);
INSERT INTO {{schema}}.libraries
  (name, description, owner_email, visibility, kind, corpus, config_dataset, documents_language, status, is_default)
VALUES
  ('All governed documents', 'Every governed corpus in the AI Search index.',
   'system', 'public', 'corpus', NULL, 'mygpt', 'english', 'ready', true),
  ('Biomedical seed (BioASQ)', 'BioASQ biomedical seed corpus.',
   'system', 'public', 'corpus', 'biomed', 'biomed', 'english', 'ready', false),
  ('SJ Global health policies (FR)', 'SJ Global French health-policy PDFs.',
   'system', 'public', 'corpus', 'health_policies_fr', 'health_policies_fr', 'french', 'ready', false),
  ('SJ Global health policies (ES)', 'SJ Global Spanish health-policy PDFs.',
   'system', 'public', 'corpus', 'health_policies_es', 'health_policies_es', 'spanish', 'ready', false),
  ('BlueSky kinase literature', 'BlueSky kinase papers.',
   'system', 'public', 'corpus', 'kinase_literature', 'kinase_literature', 'english', 'ready', false)
ON CONFLICT (owner_email, name) DO NOTHING;
`,
  },
  {
    // Upload libraries created before configDatasetFor() were all scored with the English
    // `mygpt` config (English-only reranker on French / Spanish text).
    id: '003_upload_config_by_language',
    sql: `
SELECT set_config('mygpt.role', 'admin', true);
UPDATE {{schema}}.libraries
SET config_dataset = CASE documents_language
      WHEN 'french' THEN 'health_policies_fr' WHEN 'spanish' THEN 'health_policies_es' ELSE 'mygpt' END,
    updated_at = now()
WHERE kind = 'upload' AND config_dataset = 'mygpt';
`,
  },
  {
    // Phase 6 (multilingual): MyGPT's translation fields; the Hindi governed corpus; and
    // per-language scoring configs (dataset_config default_<language>) for upload
    // libraries that 003 left on the English `mygpt` config.
    id: '004_multilingual',
    sql: `
SELECT set_config('mygpt.role', 'admin', true);
ALTER TABLE {{schema}}.questions ADD COLUMN IF NOT EXISTS translated_question_text text;
ALTER TABLE {{schema}}.answers ADD COLUMN IF NOT EXISTS translated_answer_text text;
INSERT INTO {{schema}}.libraries
  (name, description, owner_email, visibility, kind, corpus, config_dataset, documents_language, status, is_default)
VALUES
  ('SJ Global health policies (HI)', 'Public Hindi health-policy PDFs.',
   'system', 'public', 'corpus', 'health_policies_hi', 'health_policies_hi', 'hindi', 'ready', false)
ON CONFLICT (owner_email, name) DO NOTHING;
UPDATE {{schema}}.libraries
SET config_dataset = 'default_' || documents_language, updated_at = now()
WHERE kind = 'upload' AND config_dataset = 'mygpt' AND documents_language <> 'english';
`,
  },
];
