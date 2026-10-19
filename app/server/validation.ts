// Request-body validation for the Phase 4 routes (pure; see validation.test.ts).

import type { LibraryInput, Visibility } from './db/store';

export const MODEL_ALIASES = ['sonnet', 'llama', 'gptoss'] as const;
export type ModelAlias = (typeof MODEL_ALIASES)[number];

// Supervisor routes: auto (it decides) or a forced documents / Genie / tools route.
export const ROUTES = ['auto', 'docs', 'genie', 'tools'] as const;
export type Route = (typeof ROUTES)[number];
/** Libraries one question may search (= the Supervisor's MAX_LIBRARIES). */
export const MAX_ASK_LIBRARIES = 8;

// MyGPT's document_languages (models.py). 'unknown' = mixed / unlisted languages: multilingual
// reranker, no translation, 'simple' text search.
export const LANGUAGES = [
  'english', 'french', 'spanish', 'portuguese', 'german', 'italian', 'dutch', 'arabic', 'hindi', 'hungarian',
  'unknown',
] as const;

// ai_parse_document formats; spreadsheets / TXT / video arrive with Phase 3.
export const UPLOAD_EXTENSIONS: Record<string, string> = {
  '.pdf': 'application/pdf',
  '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
};

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
export const isUuid = (v: unknown): v is string => typeof v === 'string' && UUID_RE.test(v);

export type Result<T> = { ok: true; value: T } | { ok: false; error: string };

const VISIBILITIES: Visibility[] = ['private', 'group', 'public'];

/** Validate a create (`partial = false`) or update (`partial = true`) library body. */
export function parseLibraryInput(body: unknown, myGroups: string[], partial: boolean): Result<Partial<LibraryInput>> {
  const b = (body ?? {}) as Record<string, unknown>;
  const out: Partial<LibraryInput> = {};
  if (b.name !== undefined || !partial) {
    const name = typeof b.name === 'string' ? b.name.trim().replace(/\s+/g, ' ') : '';
    if (name.length < 1 || name.length > 120) return { ok: false, error: 'name must be 1-120 characters' };
    out.name = name;
  }
  if (b.description !== undefined) {
    if (typeof b.description !== 'string' || b.description.length > 1000) {
      return { ok: false, error: 'description must be a string of at most 1000 characters' };
    }
    out.description = b.description.trim();
  }
  if (b.visibility !== undefined || !partial) {
    const visibility = (b.visibility ?? 'private') as Visibility;
    if (!VISIBILITIES.includes(visibility)) return { ok: false, error: 'visibility must be private, group or public' };
    out.visibility = visibility;
    if (visibility === 'group') {
      const group = typeof b.group_name === 'string' ? b.group_name : '';
      if (!myGroups.includes(group)) return { ok: false, error: 'group_name must be one of your workspace groups' };
      out.group_name = group;
    }
  }
  if (b.share_history !== undefined) {
    if (typeof b.share_history !== 'boolean') return { ok: false, error: 'share_history must be a boolean' };
    out.share_history = b.share_history;
  }
  if (b.documents_language !== undefined || !partial) {
    const language = typeof b.documents_language === 'string' ? b.documents_language.toLowerCase() : 'english';
    if (!(LANGUAGES as readonly string[]).includes(language)) return { ok: false, error: 'unsupported documents_language' };
    if (partial) return { ok: false, error: 'documents_language cannot be changed after creation' };
    out.documents_language = language;
  }
  return { ok: true, value: out };
}

export interface AskInput {
  question: string;
  model: ModelAlias;
  k: number;
  focus: { documents: string[]; section?: string };
  route: Route;
  /** the libraries the Supervisor may search (empty = the default library) */
  library_ids: string[];
  conversation_id?: string;
}

export function parseAskInput(body: unknown): Result<AskInput> {
  const b = (body ?? {}) as Record<string, unknown>;
  const question = typeof b.question === 'string' ? b.question.trim() : '';
  if (!question) return { ok: false, error: 'question is required' };
  if (question.length > 4000) return { ok: false, error: 'question is too long' };
  const model: ModelAlias = MODEL_ALIASES.includes(b.model as ModelAlias) ? (b.model as ModelAlias) : 'sonnet';
  const k = typeof b.k === 'number' && b.k > 0 && b.k <= 10 ? Math.floor(b.k) : 5;
  const f = (b.focus ?? {}) as { documents?: unknown; section?: unknown };
  const focus = {
    documents: Array.isArray(f.documents) ? f.documents.map(String).slice(0, 50) : [],
    section: typeof f.section === 'string' && f.section !== 'all' ? f.section : undefined,
  };
  const route: Route = ROUTES.includes(b.route as Route) ? (b.route as Route) : 'auto';
  const rawIds = b.library_ids ?? [];
  if (!Array.isArray(rawIds) || !rawIds.every(isUuid)) return { ok: false, error: 'library_ids must be library ids' };
  const library_ids = [...new Set(rawIds)];
  if (library_ids.length > MAX_ASK_LIBRARIES) {
    return { ok: false, error: `at most ${MAX_ASK_LIBRARIES} libraries per question` };
  }
  if (b.conversation_id !== undefined && b.conversation_id !== null && !isUuid(b.conversation_id)) {
    return { ok: false, error: 'invalid conversation_id' };
  }
  return {
    ok: true,
    value: {
      question, model, k, focus, route, library_ids,
      conversation_id: (b.conversation_id as string | undefined) ?? undefined,
    },
  };
}

export function parseFeedback(body: unknown): Result<{ rating: -1 | 0 | 1; comment: string }> {
  const b = (body ?? {}) as Record<string, unknown>;
  if (b.rating !== -1 && b.rating !== 0 && b.rating !== 1) return { ok: false, error: 'rating must be -1, 0 or 1' };
  const comment = typeof b.comment === 'string' ? b.comment.trim() : '';
  if (comment.length > 4000) return { ok: false, error: 'comment is too long' };
  return { ok: true, value: { rating: b.rating, comment } };
}

/** Lower-cased extension of an allowed upload, or null. */
export function uploadExtension(filename: string): string | null {
  const m = /\.[A-Za-z0-9]+$/.exec(filename);
  const ext = m ? m[0].toLowerCase() : '';
  return ext in UPLOAD_EXTENSIONS ? ext : null;
}
