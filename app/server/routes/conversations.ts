// Conversation history + answer feedback (gap 9). Ports MyGPT's get_conversations_by_dataset,
// get_question_details and feedback_for_answers onto RLS-scoped Lakebase. Feedback is also
// logged as a human assessment on the agent's MLflow trace, so eval/monitoring see it.

import type { Application } from 'express';
import { createWorkspaceClient } from '@databricks/appkit';
import { withUser } from '../db/context';
import { getIdentity } from '../identity';
import { isUuid, parseFeedback } from '../validation';
import { handle, requireDb, type Deps } from './deps';

/** Best effort: user_rating (true = correct, false = incorrect) on the trace; n/a is skipped. */
async function logTraceFeedback(traceId: string, email: string, rating: -1 | 0 | 1, comment: string) {
  if (rating === 0) return;
  const wc = createWorkspaceClient();
  await wc.apiClient.request({
    path: `/api/3.0/mlflow/traces/${encodeURIComponent(traceId)}/assessments`,
    method: 'POST',
    headers: new Headers({ 'Content-Type': 'application/json' }),
    raw: false,
    payload: {
      assessment: {
        assessment_name: 'user_rating',
        trace_id: traceId,
        source: { source_type: 'HUMAN', source_id: email },
        feedback: { value: rating === 1 },
        ...(comment ? { rationale: comment } : {}),
      },
    },
  });
}

export function registerConversationRoutes(app: Application, deps: Deps) {
  const { store } = deps;

  app.get('/api/conversations', handle('/api/conversations', async (req, res) => {
    if (!deps.db) {
      res.json({ conversations: [] });
      return;
    }
    const libraryId = typeof req.query.library_id === 'string' ? req.query.library_id : undefined;
    if (libraryId !== undefined && !isUuid(libraryId)) {
      res.status(400).json({ error: 'invalid library_id' });
      return;
    }
    const who = await getIdentity(req);
    const conversations = await withUser(deps.db, who, (q) => store.listConversations(q, who.email, libraryId));
    res.json({ conversations });
  }));

  app.get('/api/conversations/:id', handle('/api/conversations/:id', async (req, res) => {
    const db = requireDb(deps, res);
    if (!db) return;
    const id = String(req.params.id);
    if (!isUuid(id)) {
      res.status(400).json({ error: 'invalid id' });
      return;
    }
    const who = await getIdentity(req);
    const conv = await withUser(db, who, (q) => store.getConversation(q, who.email, id));
    if (!conv) {
      res.status(404).json({ error: 'conversation not found' });
      return;
    }
    res.json(conv);
  }));

  app.delete('/api/conversations/:id', handle('DELETE /api/conversations', async (req, res) => {
    const db = requireDb(deps, res);
    if (!db) return;
    const id = String(req.params.id);
    if (!isUuid(id)) {
      res.status(400).json({ error: 'invalid id' });
      return;
    }
    const who = await getIdentity(req);
    const deleted = await withUser(db, who, (q) => store.deleteConversation(q, id));
    if (!deleted) {
      res.status(404).json({ error: 'conversation not found or not yours' });
      return;
    }
    res.json({ deleted: true });
  }));

  app.post('/api/answers/:id/feedback', handle('/api/answers/feedback', async (req, res) => {
    const db = requireDb(deps, res);
    if (!db) return;
    const id = String(req.params.id);
    const parsed = parseFeedback(req.body);
    if (!isUuid(id) || !parsed.ok) {
      res.status(400).json({ error: parsed.ok ? 'invalid id' : parsed.error });
      return;
    }
    const who = await getIdentity(req);
    if (!who.email) {
      res.status(401).json({ error: 'sign-in required' });
      return;
    }
    const { rating, comment } = parsed.value;
    const saved = await withUser(db, who, (q) => store.saveFeedback(q, who.email, id, rating, comment));
    if (!saved) {
      res.status(404).json({ error: 'answer not found' });
      return;
    }
    let traced = false;
    if (saved.trace_id) {
      try {
        await logTraceFeedback(saved.trace_id, who.email, rating, comment);
        traced = rating !== 0;
      } catch (err) {
        console.warn(`[feedback] could not log assessment on trace ${saved.trace_id}`, err);
      }
    }
    res.json({ saved: true, traced });
  }));
}
