// Client for the MyGPT Supervisor ResponsesAgent endpoint (agents/supervisor/supervisor_agent.py).
// It routes each question to the libraries' documents (one mygpt-rag-agent call per library it
// picks), Genie or the UC-function tools, and passes the RAG scores through.

import { createWorkspaceClient } from '@databricks/appkit';
import { config } from './config';

export interface AgentSource {
  library_id?: string | null;
  library_name?: string | null;
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

/** One searched library: its own RAG scores (custom_outputs.libraries_searched). */
export interface LibrarySearched {
  library_id: string | null;
  name: string | null;
  kind?: string;
  query?: string;
  /** its sources are custom_outputs.sources[first_source - 1 … + n_sources] (global [n]) */
  first_source: number;
  n_sources: number;
  qrs?: number | null;
  ars?: number | null;
  hi_eq?: number | null;
  hi_rf?: number | null;
  trace_id?: string | null;
  retrieve_error?: string | null;
  documents_language?: string | null;
  translated_query?: string | null;
  translated_answer?: string | null;
}

export interface GenieResult {
  status?: string;
  text?: string;
  sql?: string | null;
  description?: string | null;
  columns?: string[];
  rows?: unknown[][];
  row_count?: number;
}

export interface ToolCallRecord {
  name: string;
  ok: boolean;
  latency_ms?: number;
}

/** answer = scores describe the answer (one library, verbatim); per_library = several
 *  libraries, see libraries_searched; documents_part = the answer also used other tools. */
export type ScoresScope = 'answer' | 'per_library' | 'documents_part' | null;

export interface AgentCustomOutputs {
  route?: string[];
  scores_scope?: ScoresScope;
  libraries_searched?: LibrarySearched[];
  genie?: GenieResult | null;
  tool_calls?: ToolCallRecord[];
  error?: string | null;
  model?: string;
  corpus?: string | null;
  qrs?: number | null;
  ars?: number | null;
  hi_eq?: number | null;
  hi_rf?: number | null;
  sources?: AgentSource[];
  trace_id?: string | null;
  retrieve_error?: string | null;
  // Phase 6: non-English corpora — the question as searched / the answer as scored, in the
  // documents' language (null when no translation applied).
  documents_language?: string;
  translated_query?: string | null;
  translated_answer?: string | null;
  translate_error?: string | null;
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

/** A library the Supervisor may search; upload libraries carry their pre-retrieved hits. */
export interface AgentLibraryInput {
  id: string;
  name: string;
  kind: 'corpus' | 'upload';
  description: string;
  documents_language: string;
  /** the dataset_config row it is scored with */
  dataset: string;
  corpus?: string;
  sources?: ProvidedSource[];
}

export interface AgentCustomInputs {
  route: 'auto' | 'docs' | 'genie' | 'tools';
  model: string;
  k: number;
  focus: { documents: string[]; section?: string };
  libraries: AgentLibraryInput[];
}

export interface AgentMessage {
  role: 'user' | 'assistant';
  content: string;
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

/** Ask the agent AS THE SIGNED-IN USER (OBO) when their token is available. `messages` ends
 *  with the question; earlier turns let the Supervisor resolve follow-ups. */
export async function askAgent(
  messages: AgentMessage[],
  customInputs: AgentCustomInputs,
  token: string | undefined,
): Promise<{ answer: string; outputs: AgentCustomOutputs }> {
  const wc = token ? createWorkspaceClient({ token }) : createWorkspaceClient();
  const raw: unknown = await wc.apiClient.request({
    path: `/serving-endpoints/${config.agentEndpoint}/invocations`,
    method: 'POST',
    headers: new Headers({ 'Content-Type': 'application/json' }),
    raw: false,
    payload: { input: messages, custom_inputs: customInputs },
  });
  const resp = raw as AgentResponse;
  return { answer: extractAnswer(resp), outputs: resp.custom_outputs ?? {} };
}
