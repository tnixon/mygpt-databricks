import { createApp, server, serving, createWorkspaceClient } from '@databricks/appkit';

// Databricks AI Search (Vector Search) index over the governed doc chunks.
const VS_INDEX = 'structural_biology.mygpt.doc_chunks_index';

interface VsQueryResponse {
  manifest?: { columns?: { name: string }[] };
  result?: { data_array?: unknown[][] };
}
// Programmatic serving.invoke() wraps the completion in an ExecutionResult.
interface ChatResponse {
  ok?: boolean;
  data?: { choices?: { message?: { content?: string } }[] };
}

const MODEL_ALIASES = ['sonnet', 'llama', 'gptoss'] as const;
type ModelAlias = (typeof MODEL_ALIASES)[number];

createApp({
  plugins: [
    server(),
    // FMAPI foundation models via the serving plugin (runs OBO by default).
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
    const wc = createWorkspaceClient();

    appkit.server.extend((app) => {
      // Identity for the AI-trust badge.
      app.get('/api/whoami', (req, res) => {
        const user = req.header('x-forwarded-email') ?? req.header('x-forwarded-user') ?? null;
        res.json({ user });
      });

      // Grounded RAG: retrieve from AI Search -> grounded prompt -> FMAPI answer.
      // Returns the answer plus the cited sources and a retrieval-relevance (QRS) score.
      app.post('/api/ask', async (req, res) => {
        const body = (req.body ?? {}) as { question?: unknown; model?: unknown; k?: unknown };
        const question = typeof body.question === 'string' ? body.question.trim() : '';
        if (!question) {
          res.status(400).json({ error: 'question is required' });
          return;
        }
        const model: ModelAlias = MODEL_ALIASES.includes(body.model as ModelAlias)
          ? (body.model as ModelAlias)
          : 'sonnet';
        const k = typeof body.k === 'number' && body.k > 0 && body.k <= 10 ? Math.floor(body.k) : 5;

        try {
          // 1) Retrieve top-k chunks (hybrid ANN + BM25).
          const vsRaw: unknown = await wc.apiClient.request({
            path: `/api/2.0/vector-search/indexes/${VS_INDEX}/query`,
            method: 'POST',
            headers: new Headers({ 'Content-Type': 'application/json' }),
            raw: false,
            payload: {
              columns: ['source_name', 'page', 'chunk_text'],
              query_text: question,
              num_results: k,
              query_type: 'HYBRID',
            },
          });
          const vs = vsRaw as VsQueryResponse;
          const cols = (vs.manifest?.columns ?? []).map((c) => c.name);
          const rows = (vs.result?.data_array ?? []).map((r) => {
            const o: Record<string, unknown> = {};
            cols.forEach((c, i) => {
              o[c] = r[i];
            });
            return o;
          });
          const sources = rows.map((r) => ({
            source_name: String(r.source_name ?? ''),
            page: r.page == null ? null : Number(r.page),
            score: Number(r.score ?? 0),
            snippet: String(r.chunk_text ?? '').slice(0, 240),
          }));

          if (sources.length === 0) {
            res.json({
              answer: "I couldn't find anything relevant in the indexed documents.",
              sources: [],
              qrs: 0,
              model,
            });
            return;
          }

          // 2) Grounded prompt (single user turn — the serving request type only
          //    allows user/assistant roles, so instructions go inline).
          const context = rows
            .map((r, i) => `[${i + 1}] ${r.source_name} (p${r.page ?? '?'}):\n${r.chunk_text}`)
            .join('\n\n');
          const prompt =
            'You are MyGPT, a governed research assistant for St. Jude. Answer the question ' +
            'using ONLY the numbered sources below, and cite them inline as [n]. If the sources ' +
            "do not contain the answer, say you couldn't find it in the documents.\n\n" +
            `Sources:\n${context}\n\nQuestion: ${question}`;

          // 3) Generate. When deployed, run the model call on behalf of the
          //    signed-in user (OBO) — Databricks Apps forwards their token in
          //    x-forwarded-access-token — so usage is attributed per-user via the
          //    Unity AI Gateway. In local dev that header is absent, so fall back
          //    to the app's service principal (resolved from the dev profile).
          //    Retrieval (above) stays on the service principal against the
          //    shared, governed AI Search index.
          const endpoint = appkit.serving(model);
          const asUser = req.header('x-forwarded-access-token') != null;
          const runner = asUser ? endpoint.asUser(req) : endpoint;
          const genRaw: unknown = await runner.invoke({
            messages: [{ role: 'user' as const, content: prompt }],
          });
          const gen = genRaw as ChatResponse;
          const answer = gen.data?.choices?.[0]?.message?.content ?? '';

          // 4) QRS = mean retrieval relevance across the cited sources.
          const qrs = sources.reduce((a, s) => a + s.score, 0) / sources.length;

          res.json({ answer, sources, qrs, model });
        } catch (err) {
          console.error('/api/ask failed', err);
          res.status(500).json({ error: 'ask failed' });
        }
      });
    });
  },
}).catch(console.error);
