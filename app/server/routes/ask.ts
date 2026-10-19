// Grounded questions (via the MyGPT Supervisor) + section catalogue, per library.
//
// /api/ask resolves the selected libraries under RLS (404 if any isn't visible), then hands
// them to the Supervisor as custom_inputs.libraries; it picks the ones to search per question:
//   * corpus library → the RAG agent retrieves from AI Search (corpus / dataset);
//   * upload library → the app runs the Lakebase hybrid search up front and hands the hits
//     over as that library's `sources` (the agents never touch Lakebase); the RAG agent still
//     reranks, scores (QRS/ARS/HI) and answers.
// The Supervisor may instead (or also) route to Genie or the UC-function tools. Recent turns of
// the conversation go along as input messages, so follow-ups resolve. The agent is called AS
// THE SIGNED-IN USER (OBO). The turn is then persisted to the user's conversation; a
// persistence failure is logged but never drops the answer.

import type { Application } from 'express';
import { createWorkspaceClient } from '@databricks/appkit';
import { askAgent, type AgentLibraryInput, type AgentMessage, type ProvidedSource } from '../agent';
import { config } from '../config';
import { withUser, type Identity } from '../db/context';
import type { Library } from '../db/store';
import { getIdentity, userToken } from '../identity';
import { embedQuery, searchUploadLibrary } from '../retrieval/lakebaseSearch';
import { isUuid, parseAskInput } from '../validation';
import { handle, type Deps } from './deps';
import { STATELESS_DEFAULT } from './libraries';

/** Earlier Q&A turns sent with a question (the Supervisor keeps its last 6 messages). */
const HISTORY_TURNS = 3;

async function resolveLibrary(deps: Deps, who: Identity, id: string | undefined): Promise<Library | null> {
  if (!deps.db) return STATELESS_DEFAULT;
  return withUser(deps.db, who, (q) => (id ? deps.store.getLibrary(q, id) : deps.store.getDefaultLibrary(q)));
}

/** The selected libraries (none = the default), or null if any isn't visible to the user. */
async function resolveLibraries(deps: Deps, who: Identity, ids: string[]): Promise<Library[] | null> {
  if (!deps.db) return [STATELESS_DEFAULT];
  return withUser(deps.db, who, async (q) => {
    if (ids.length === 0) {
      const lib = await deps.store.getDefaultLibrary(q);
      return lib ? [lib] : null;
    }
    const libs: Library[] = [];
    for (const id of ids) {
      const lib = await deps.store.getLibrary(q, id);
      if (!lib) return null;
      libs.push(lib);
    }
    return libs;
  });
}

/** Governed section catalogue (paper_sections, built by the ETL) for a corpus library.
 *  Read as the app service principal (SELECT on paper_sections via app resources): it's a
 *  non-sensitive catalogue, and the user token has no SQL scope. */
async function corpusSections(datasetName: string) {
  if (!config.warehouseId) return [];
  const wc = createWorkspaceClient();
  const out: unknown = await wc.apiClient.request({
    path: '/api/2.0/sql/statements',
    method: 'POST',
    headers: new Headers({ 'Content-Type': 'application/json' }),
    raw: false,
    payload: {
      warehouse_id: config.warehouseId,
      statement: `SELECT section_title, section_count FROM ${config.ucSchema}.paper_sections
                  WHERE dataset_name = :ds ORDER BY section_count DESC LIMIT 200`,
      parameters: [{ name: 'ds', value: datasetName }],
      wait_timeout: '30s',
    },
  });
  return ((out as { result?: { data_array?: (string | null)[][] } }).result?.data_array ?? []).map((r) => ({
    section_title: r[0] ?? '',
    section_count: Number(r[1] ?? 0),
  }));
}

export function registerAskRoutes(app: Application, deps: Deps) {
  const { store } = deps;

  // Section catalogue for the document/section focus selector (gap 13), per library.
  app.get('/api/sections', handle('/api/sections', async (req, res) => {
    const libraryId = typeof req.query.library_id === 'string' && req.query.library_id ? req.query.library_id : undefined;
    if (libraryId !== undefined && !isUuid(libraryId)) {
      res.status(400).json({ error: 'invalid library_id' });
      return;
    }
    const who = await getIdentity(req);
    const lib = await resolveLibrary(deps, who, libraryId);
    if (!lib) {
      res.status(404).json({ error: 'library not found' });
      return;
    }
    try {
      const sections = lib.kind === 'upload' && deps.db
        ? await withUser(deps.db, who, (q) => store.uploadSections(q, lib.id))
        : await corpusSections(lib.corpus ?? 'mygpt');
      res.json({ sections });
    } catch (err) {
      console.error('/api/sections failed', err);
      res.json({ sections: [] });
    }
  }));

  app.post('/api/ask', handle('/api/ask', async (req, res) => {
    const parsed = parseAskInput(req.body);
    if (!parsed.ok) {
      res.status(400).json({ error: parsed.error });
      return;
    }
    const input = parsed.value;
    const who = await getIdentity(req);
    const libs = await resolveLibraries(deps, who, input.library_ids);
    if (!libs) {
      res.status(404).json({ error: 'library not found' });
      return;
    }
    const libraryIds = libs.map((l) => l.id);

    // Upload libraries: search Lakebase up front (one query embedding, one search per library);
    // a library without hits is left out of what the Supervisor may search.
    const hits = new Map<string, ProvidedSource[]>();
    const uploads = libs.filter((l) => l.kind === 'upload');
    if (uploads.length > 0) {
      const db = deps.db;
      if (!db) throw new Error('upload library without Lakebase');
      const vector = await embedQuery(input.question);
      await Promise.all(uploads.map(async (lib) => {
        hits.set(lib.id, await withUser(db, who, (q) =>
          searchUploadLibrary(q, store.schema, lib, input.question, vector, input.focus, input.k)));
      }));
    }
    const libraries: AgentLibraryInput[] = libs
      .filter((l) => l.kind !== 'upload' || (hits.get(l.id)?.length ?? 0) > 0)
      .map((l) => ({
        id: l.id, name: l.name, kind: l.kind, description: l.description,
        documents_language: l.documents_language, dataset: l.config_dataset,
        ...(l.kind === 'upload' ? { sources: hits.get(l.id) } : l.corpus ? { corpus: l.corpus } : {}),
      }));
    if (libraries.length === 0 && input.route === 'docs') {
      res.json({
        answer: "I couldn't find anything relevant in these libraries' documents.",
        sources: [], qrs: null, ars: null, hi_eq: null, hi_rf: null, model: input.model, library_ids: libraryIds,
        route: [], scores_scope: null, libraries_searched: [], genie: null, tool_calls: [], persisted: false,
      });
      return;
    }

    // Multi-turn context: recent turns of the user's own conversation (RLS), oldest first.
    let history: AgentMessage[] = [];
    if (deps.db && who.email && input.conversation_id) {
      const conversationId = input.conversation_id;
      const turns = await withUser(deps.db, who, async (q) =>
        (await store.ownsConversation(q, conversationId, who.email)) ? store.recentTurns(q, conversationId, HISTORY_TURNS) : []);
      history = turns.flatMap((t): AgentMessage[] => [
        { role: 'user', content: t.question },
        { role: 'assistant', content: t.answer },
      ]);
    }

    const { answer, outputs } = await askAgent(
      [...history, { role: 'user', content: input.question }],
      { route: input.route, model: input.model, k: input.k, focus: input.focus, libraries },
      userToken(req),
    );
    if (outputs.error) console.warn('[ask] supervisor reported', outputs.error);
    const result = {
      answer,
      sources: outputs.sources ?? [],
      qrs: outputs.qrs ?? null,
      ars: outputs.ars ?? null,
      hi_eq: outputs.hi_eq ?? null,
      hi_rf: outputs.hi_rf ?? null,
      model: outputs.model ?? input.model,
      library_ids: libraryIds,
      documents_language: outputs.documents_language ?? (libs.length === 1 ? libs[0].documents_language : undefined),
      translated_query: outputs.translated_query ?? null,
      translated_answer: outputs.translated_answer ?? null,
      route: outputs.route ?? [],
      scores_scope: outputs.scores_scope ?? null,
      libraries_searched: outputs.libraries_searched ?? [],
      genie: outputs.genie ?? null,
      tool_calls: (outputs.tool_calls ?? []).map(({ name, ok, latency_ms }) => ({ name, ok, latency_ms })),
    };

    // Persist the turn into the user's conversation (Lakebase mode, signed-in users only).
    let ids: { conversation_id: string; question_id: string; answer_id: string } | null = null;
    if (deps.db && who.email) {
      try {
        ids = await withUser(deps.db, who, async (q) => {
          let conversationId = input.conversation_id;
          if (!conversationId || !(await store.ownsConversation(q, conversationId, who.email))) {
            conversationId = await store.createConversation(q, who.email, libraryIds, input.question);
          }
          const saved = await store.saveTurn(q, {
            conversation_id: conversationId,
            question_text: input.question,
            model: input.model,
            focus: input.focus,
            qrs: result.qrs,
            translated_question_text: result.translated_query,
            answer_text: answer,
            translated_answer_text: result.translated_answer,
            ars: result.ars,
            hi_eq: result.hi_eq,
            hi_rf: result.hi_rf,
            model_endpoint: outputs.model ?? null,
            trace_id: outputs.trace_id ?? null,
            custom_outputs: outputs,
            sources: result.sources.map((s, i) => ({ ...s, rank: i + 1 })),
          });
          return { conversation_id: conversationId, ...saved };
        });
      } catch (err) {
        console.error('[ask] could not persist the turn', err);
      }
    }
    res.json({ ...result, ...(ids ?? {}), persisted: ids !== null });
  }));
}
