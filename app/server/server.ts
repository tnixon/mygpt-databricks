import { createApp, server, serving, createWorkspaceClient } from '@databricks/appkit';

// The MyGPT retrieval+scoring ResponsesAgent (its own Model Serving endpoint).
const AGENT_ENDPOINT = process.env.MYGPT_AGENT_ENDPOINT ?? 'mygpt-rag-agent';
// Optional SQL warehouse for the section catalogue (focus selector). If unset,
// /api/sections returns an empty list and the UI simply omits the control.
const WAREHOUSE_ID = process.env.MYGPT_WAREHOUSE_ID ?? '';
const SCHEMA = process.env.MYGPT_SCHEMA ?? 'structural_biology.mygpt';

const MODEL_ALIASES = ['sonnet', 'llama', 'gptoss'] as const;
type ModelAlias = (typeof MODEL_ALIASES)[number];

interface AgentSource {
  source_name?: string;
  section?: string;
  page?: number | null;
  snippet?: string;
  vector_score?: number | null;
  bm25_score?: number | null;
  rerank_score?: number | null;
  nli?: string | null;
}
interface AgentResponse {
  output?: Array<{ content?: unknown; text?: string }>;
  custom_outputs?: {
    model?: string;
    qrs?: number;
    ars?: number;
    hi_eq?: number;
    hi_rf?: number | null;
    sources?: AgentSource[];
  };
}

// Pull the assistant text out of a ResponsesAgent output array (content may be a
// plain string or responses-style content parts).
function extractAnswer(resp: AgentResponse): string {
  for (const item of resp.output ?? []) {
    if (typeof item.text === 'string' && item.text) return item.text;
    const content = item.content;
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
      const text = content
        .map((p) => (typeof p === 'object' && p && 'text' in p ? String((p as { text: unknown }).text ?? '') : ''))
        .join('');
      if (text) return text;
    }
  }
  return '';
}

createApp({
  plugins: [
    server(),
    // FMAPI foundation models for DIRECT (non-grounded) chat streaming.
    serving({
      endpoints: {
        sonnet: { env: 'DATABRICKS_SERVING_ENDPOINT_NAME' },
        llama: { env: 'DATABRICKS_SERVING_ENDPOINT_LLAMA' },
        gptoss: { env: 'DATABRICKS_SERVING_ENDPOINT_GPTOSS' },
      },
      timeout: 120000,
    }),
  ],
  async onPluginsReady(appkit) {
    appkit.server.extend((app) => {
      // Identity for the AI-trust badge.
      app.get('/api/whoami', (req, res) => {
        const user = req.header('x-forwarded-email') ?? req.header('x-forwarded-user') ?? null;
        res.json({ user });
      });

      // Section catalogue for the document/section focus selector (gap 13).
      app.get('/api/sections', async (req, res) => {
        if (!WAREHOUSE_ID) {
          res.json({ sections: [] });
          return;
        }
        try {
          const userToken = req.header('x-forwarded-access-token');
          const wc = userToken ? createWorkspaceClient({ token: userToken }) : createWorkspaceClient();
          const out: unknown = await wc.apiClient.request({
            path: '/api/2.0/sql/statements',
            method: 'POST',
            headers: new Headers({ 'Content-Type': 'application/json' }),
            raw: false,
            payload: {
              warehouse_id: WAREHOUSE_ID,
              statement: `SELECT section_title, section_count FROM ${SCHEMA}.paper_sections WHERE dataset_name = 'mygpt' ORDER BY section_count DESC LIMIT 200`,
              wait_timeout: '30s',
            },
          });
          const rows = ((out as { result?: { data_array?: unknown[][] } }).result?.data_array ?? []).map((r) => ({
            section_title: String(r[0] ?? ''),
            section_count: Number(r[1] ?? 0),
          }));
          res.json({ sections: rows });
        } catch (err) {
          console.error('/api/sections failed', err);
          res.json({ sections: [] });
        }
      });

      // Grounded RAG: delegate to the ResponsesAgent, which owns retrieval, rerank,
      // QRS/ARS/HI scoring, and generation. Retrieve AS THE SIGNED-IN USER (OBO) by
      // forwarding their token so Unity Catalog + the agent enforce their access.
      app.post('/api/ask', async (req, res) => {
        const body = (req.body ?? {}) as {
          question?: unknown;
          model?: unknown;
          k?: unknown;
          focus?: { documents?: unknown; section?: unknown };
        };
        const question = typeof body.question === 'string' ? body.question.trim() : '';
        if (!question) {
          res.status(400).json({ error: 'question is required' });
          return;
        }
        const model: ModelAlias = MODEL_ALIASES.includes(body.model as ModelAlias)
          ? (body.model as ModelAlias)
          : 'sonnet';
        const k = typeof body.k === 'number' && body.k > 0 && body.k <= 10 ? Math.floor(body.k) : 5;
        const focus = {
          documents: Array.isArray(body.focus?.documents) ? body.focus?.documents.map(String) : [],
          section: typeof body.focus?.section === 'string' ? body.focus?.section : undefined,
        };

        try {
          const userToken = req.header('x-forwarded-access-token');
          const wc = userToken ? createWorkspaceClient({ token: userToken }) : createWorkspaceClient();

          const raw: unknown = await wc.apiClient.request({
            path: `/serving-endpoints/${AGENT_ENDPOINT}/invocations`,
            method: 'POST',
            headers: new Headers({ 'Content-Type': 'application/json' }),
            raw: false,
            payload: {
              input: [{ role: 'user', content: question }],
              custom_inputs: { dataset: 'mygpt', model, k, focus },
            },
          });
          const resp = raw as AgentResponse;
          const co = resp.custom_outputs ?? {};
          res.json({
            answer: extractAnswer(resp),
            sources: co.sources ?? [],
            qrs: co.qrs ?? 0,
            ars: co.ars ?? null,
            hi_eq: co.hi_eq ?? null,
            hi_rf: co.hi_rf ?? null,
            model: co.model ?? model,
          });
        } catch (err) {
          console.error('/api/ask failed', err);
          res.status(500).json({ error: 'ask failed' });
        }
      });
    });
  },
}).catch(console.error);
