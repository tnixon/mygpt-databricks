// Client for the MyGPT retrieval+scoring ResponsesAgent endpoint.

import { createWorkspaceClient } from '@databricks/appkit';
import { config } from './config';

export interface AgentSource {
  source_name?: string;
  corpus?: string;
  section?: string;
  page?: number | null;
  snippet?: string;
  vector_score?: number | null;
  bm25_score?: number | null;
  rerank_score?: number | null;
  nli?: string | null;
}

export interface AgentCustomOutputs {
  model?: string;
  corpus?: string | null;
  qrs?: number;
  ars?: number;
  hi_eq?: number;
  hi_rf?: number | null;
  sources?: AgentSource[];
  trace_id?: string | null;
  retrieve_error?: string | null;
}

interface AgentResponse {
  output?: Array<{ content?: unknown; text?: string }>;
  custom_outputs?: AgentCustomOutputs;
}

/** A pre-retrieved chunk handed to the agent (upload libraries), in its `_retrieve` shape. */
export interface ProvidedSource {
  chunk_id: string;
  source_name: string;
  corpus: string;
  section: string;
  page: number | null;
  chunk_text: string;
  similarity: number;
  bm25_score: number;
}

export interface AgentCustomInputs {
  dataset: string;
  model: string;
  k: number;
  focus: { documents: string[]; section?: string };
  corpus?: string;
  sources?: ProvidedSource[];
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
        .map((p) => {
          const text = typeof p === 'object' && p && 'text' in p ? (p as { text: unknown }).text : '';
          return typeof text === 'string' ? text : '';
        })
        .join('');
      if (text) return text;
    }
  }
  return '';
}

/** Ask the agent AS THE SIGNED-IN USER (OBO) when their token is available. */
export async function askAgent(
  question: string,
  customInputs: AgentCustomInputs,
  token: string | undefined,
): Promise<{ answer: string; outputs: AgentCustomOutputs }> {
  const wc = token ? createWorkspaceClient({ token }) : createWorkspaceClient();
  const raw: unknown = await wc.apiClient.request({
    path: `/serving-endpoints/${config.agentEndpoint}/invocations`,
    method: 'POST',
    headers: new Headers({ 'Content-Type': 'application/json' }),
    raw: false,
    payload: { input: [{ role: 'user', content: question }], custom_inputs: customInputs },
  });
  const resp = raw as AgentResponse;
  return { answer: extractAnswer(resp), outputs: resp.custom_outputs ?? {} };
}
