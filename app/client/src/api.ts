// Shapes + fetch helpers for the app's /api routes (server/routes/*).

export const MODELS = [
  { alias: 'sonnet', label: 'Claude Sonnet 5' },
  { alias: 'llama', label: 'Llama 4 Maverick' },
  { alias: 'gptoss', label: 'GPT-OSS 120B' },
] as const;

export type ModelAlias = (typeof MODELS)[number]['alias'];

export function modelLabel(alias?: string | null): string | undefined {
  return MODELS.find((m) => m.alias === alias)?.label ?? alias ?? undefined;
}

export const LANGUAGES = [
  'english', 'french', 'spanish', 'portuguese', 'german', 'italian', 'dutch', 'arabic', 'hindi', 'hungarian',
  'unknown',
] as const;

export interface Source {
  source_name?: string | null;
  section?: string | null;
  page?: number | null;
  snippet?: string | null;
  vector_score?: number | null;
  bm25_score?: number | null;
  rerank_score?: number | null;
  nli?: string | null;
}

export interface Whoami {
  user: string | null;
  groups: string[];
  persistence: boolean;
}

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
  documents_language: string;
  status: 'empty' | 'ingesting' | 'ready' | 'failed';
  is_default: boolean;
  n_documents?: number;
  can_edit: boolean;
}

export interface LibraryDocument {
  id: string;
  filename: string;
  size_bytes: number | null;
  status: 'uploaded' | 'parsing' | 'ready' | 'failed';
  error: string | null;
  n_chunks: number;
  uploaded_at: string;
}

export interface IngestRun {
  id: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed';
  message: string | null;
  started_at: string;
  finished_at: string | null;
}

export interface SectionOption {
  section_title: string;
  section_count: number;
}

export interface AskResponse {
  answer: string;
  sources: Source[];
  qrs: number;
  ars: number | null;
  hi_eq: number | null;
  hi_rf: number | null;
  model: string;
  library_id: string;
  /** Phase 6: the question as searched / the answer as scored, in the documents' language */
  documents_language?: string;
  translated_query?: string | null;
  translated_answer?: string | null;
  persisted: boolean;
  conversation_id?: string;
  question_id?: string;
  answer_id?: string;
}

export interface ConversationSummary {
  id: string;
  owner_email: string;
  library_id: string | null;
  library_name: string | null;
  title: string;
  last_at: string;
  n_questions: number;
  shared: boolean;
}

export interface QaTurn {
  question_id: string;
  question_text: string;
  model: string | null;
  qrs: number | null;
  translated_question_text: string | null;
  asked_at: string;
  answer_id: string | null;
  answer_text: string | null;
  translated_answer_text: string | null;
  ars: number | null;
  hi_eq: number | null;
  hi_rf: number | null;
  sources: Source[];
  my_rating: number | null;
  my_comment: string | null;
}

export type Rating = -1 | 0 | 1;

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  const body = (await r.json().catch(() => ({}))) as T & { error?: string };
  if (!r.ok) throw new Error(body.error ?? `${path} failed (${r.status})`);
  return body;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const api = {
  whoami: () => call<Whoami>('/api/whoami'),
  libraries: () => call<{ libraries: Library[]; persistence: boolean }>('/api/libraries'),
  createLibrary: (input: {
    name: string; description?: string; visibility: Visibility; group_name?: string;
    share_history?: boolean; documents_language: string;
  }) => call<{ library: Library }>('/api/libraries', json('POST', input)),
  updateLibrary: (id: string, patch: Partial<Pick<Library, 'name' | 'visibility' | 'group_name' | 'share_history'>>) =>
    call<{ library: Library }>(`/api/libraries/${id}`, json('PATCH', patch)),
  deleteLibrary: (id: string) => call<{ deleted: boolean }>(`/api/libraries/${id}`, { method: 'DELETE' }),
  documents: (id: string) => call<{ documents: LibraryDocument[] }>(`/api/libraries/${id}/documents`),
  uploadDocuments: (id: string, files: File[]) => {
    const form = new FormData();
    for (const f of files) form.append('files', f);
    return call<{ documents: LibraryDocument[]; ingest_error: string | null }>(
      `/api/libraries/${id}/documents`, { method: 'POST', body: form });
  },
  deleteDocument: (id: string, docId: string) =>
    call<{ deleted: boolean }>(`/api/libraries/${id}/documents/${docId}`, { method: 'DELETE' }),
  retryFailed: (id: string) =>
    call<{ requeued: number; ingest_run: IngestRun | null }>(`/api/libraries/${id}/retry`, { method: 'POST' }),
  status: (id: string) =>
    call<{ library: Library | null; documents: LibraryDocument[]; ingest_run: IngestRun | null }>(
      `/api/libraries/${id}/status`),
  sections: (libraryId?: string) =>
    call<{ sections: SectionOption[] }>(`/api/sections${libraryId ? `?library_id=${libraryId}` : ''}`),
  ask: (body: {
    question: string; model: ModelAlias; library_id?: string; conversation_id?: string;
    focus?: { section?: string };
  }) => call<AskResponse>('/api/ask', json('POST', body)),
  conversations: () => call<{ conversations: ConversationSummary[] }>('/api/conversations'),
  conversation: (id: string) =>
    call<{ conversation: ConversationSummary; turns: QaTurn[] }>(`/api/conversations/${id}`),
  deleteConversation: (id: string) => call<{ deleted: boolean }>(`/api/conversations/${id}`, { method: 'DELETE' }),
  feedback: (answerId: string, rating: Rating, comment: string) =>
    call<{ saved: boolean; traced: boolean }>(`/api/answers/${answerId}/feedback`, json('POST', { rating, comment })),
};
