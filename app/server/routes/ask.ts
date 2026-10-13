// Grounded RAG + section catalogue, per library.
//
// /api/ask resolves the library under RLS (404 if not visible), then:
//   * corpus library → the agent retrieves from AI Search (custom_inputs.corpus/dataset);
//   * upload library → the app runs the Lakebase hybrid search and hands the hits to the
//     agent as custom_inputs.sources; the agent still reranks, scores (QRS/ARS/HI) and answers.
// The agent is called AS THE SIGNED-IN USER (OBO). The turn is then persisted to the user's
// conversation; a persistence failure is logged but never drops the answer.

import type { Application } from 'express';
import { createWorkspaceClient } from '@databricks/appkit';
import { askAgent, type AgentCustomInputs } from '../agent';
import { config } from '../config';
import { withUser, type Identity } from '../db/context';
import type { Library } from '../db/store';
import { getIdentity, userToken } from '../identity';
import { embedQuery, searchUploadLibrary } from '../retrieval/lakebaseSearch';
import { isUuid, parseAskInput } from '../validation';
import { handle, type Deps } from './deps';
import { STATELESS_DEFAULT } from './libraries';

async function resolveLibrary(deps: Deps, who: Identity, id: string | undefined): Promise<Library | null> {
  if (!deps.db) return STATELESS_DEFAULT;
  return withUser(deps.db, who, (q) => (id ? deps.store.getLibrary(q, id) : deps.store.getDefaultLibrary(q)));
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
    const lib = await resolveLibrary(deps, who, input.library_id);
    if (!lib) {
      res.status(404).json({ error: 'library not found' });
      return;
    }

    const customInputs: AgentCustomInputs = {
      dataset: lib.config_dataset,
      model: input.model,
      k: input.k,
      focus: input.focus,
    };
    if (lib.kind === 'upload') {
      if (!deps.db) throw new Error('upload library without Lakebase');
      const vector = await embedQuery(input.question);
      customInputs.sources = await withUser(deps.db, who, (q) =>
        searchUploadLibrary(q, store.schema, lib, input.question, vector, input.focus, input.k));
      if (customInputs.sources.length === 0) {
        res.json({
          answer: "I couldn't find anything relevant in this library's documents.",
          sources: [], qrs: 0, ars: null, hi_eq: null, hi_rf: null, model: input.model,
          library_id: lib.id, persisted: false,
        });
        return;
      }
    } else if (lib.corpus) {
      customInputs.corpus = lib.corpus;
    }

    const { answer, outputs } = await askAgent(input.question, customInputs, userToken(req));
    const result = {
      answer,
      sources: outputs.sources ?? [],
      qrs: outputs.qrs ?? 0,
      ars: outputs.ars ?? null,
      hi_eq: outputs.hi_eq ?? null,
      hi_rf: outputs.hi_rf ?? null,
      model: outputs.model ?? input.model,
      library_id: lib.id,
    };

    // Persist the turn into the user's conversation (Lakebase mode, signed-in users only).
    let ids: { conversation_id: string; question_id: string; answer_id: string } | null = null;
    if (deps.db && who.email) {
      try {
        ids = await withUser(deps.db, who, async (q) => {
          let conversationId = input.conversation_id;
          if (!conversationId || !(await store.ownsConversation(q, conversationId, who.email))) {
            conversationId = await store.createConversation(q, who.email, lib.id, input.question);
          }
          const saved = await store.saveTurn(q, {
            conversation_id: conversationId,
            question_text: input.question,
            model: input.model,
            focus: input.focus,
            qrs: result.qrs,
            answer_text: answer,
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
