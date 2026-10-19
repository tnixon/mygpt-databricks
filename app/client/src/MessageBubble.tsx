import { useState } from 'react';
import { Badge, Button, Input, Spinner } from '@databricks/appkit-ui/react';
import { ThumbsDown, ThumbsUp } from 'lucide-react';
import { api, modelLabel, type AgentTrace, type GenieResult, type LibrarySearched, type Rating, type Source } from './api';
import { Markdown } from './Markdown';

/** What the Supervisor did (route, per-library scores, Genie, tools) rides along as AgentTrace. */
export interface ChatMessage extends AgentTrace {
  role: 'user' | 'assistant';
  content: string;
  model?: string | null;
  sources?: Source[];
  qrs?: number | null;
  ars?: number | null;
  hi_eq?: number | null;
  hi_rf?: number | null;
  /** Phase 6: non-English documents — the question as searched, the answer as scored */
  documents_language?: string | null;
  translated_query?: string | null;
  translated_answer?: string | null;
  /** set once the turn is persisted (Lakebase mode) — enables feedback */
  answer_id?: string | null;
  my_rating?: number | null;
  my_comment?: string | null;
}

export function MessageBubble({ msg, streaming }: { msg: ChatMessage; streaming?: boolean }) {
  const isUser = msg.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] rounded-lg px-4 py-3 text-sm ${
          isUser ? 'bg-primary text-primary-foreground' : 'bg-muted text-foreground'
        }`}
      >
        {!isUser && (
          <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span className="font-medium">MyGPT</span>
            {msg.model && (
              <Badge variant="outline" className="text-[10px]">
                {modelLabel(msg.model)}
              </Badge>
            )}
            {(msg.route ?? []).map((r) => (
              <Badge key={r} variant="outline" className="text-[10px]" title="how MyGPT answered">
                {ROUTE_LABELS[r] ?? r}
              </Badge>
            ))}
            <Scores qrs={msg.qrs} ars={msg.ars} hi_eq={msg.hi_eq} hi_rf={msg.hi_rf} />
            {streaming && <Spinner className="h-3 w-3" />}
          </div>
        )}

        {isUser ? (
          <div className="whitespace-pre-wrap break-words">{msg.content}</div>
        ) : msg.content ? (
          <Markdown>{msg.content}</Markdown>
        ) : (
          streaming ? <span className="text-muted-foreground">…</span> : null
        )}

        {!isUser && (msg.translated_query || msg.translated_answer) && (
          <Translations msg={msg} />
        )}

        {!isUser && msg.genie && <GeniePanel genie={msg.genie} />}

        {!isUser && msg.sources && msg.sources.length > 0 && (
          <div className="mt-3 border-t border-border pt-2">
            {msg.scores_scope === 'documents_part' && (
              <div className="mb-1 text-[11px] text-muted-foreground">
                Trust scores cover the documents part of this answer only.
              </div>
            )}
            {sourceGroups(msg).map((g, gi) => (
              <div key={gi} className={gi > 0 ? 'mt-2' : ''}>
                <div className="mb-1 flex flex-wrap items-center gap-2 text-xs font-medium text-muted-foreground">
                  <span>{g.library ? `Sources · ${g.library.name ?? 'library'}` : 'Sources'}</span>
                  {g.library && <Scores {...g.library} />}
                </div>
                <SourceList sources={g.sources} first={g.first} />
              </div>
            ))}
          </div>
        )}

        {!isUser && msg.tool_calls && msg.tool_calls.length > 0 && (
          <details className="mt-2 text-xs text-muted-foreground">
            <summary className="cursor-pointer">Tools used ({msg.tool_calls.length})</summary>
            <ul className="mt-1 space-y-0.5">
              {msg.tool_calls.map((c, i) => (
                <li key={i}>
                  {c.name}
                  {c.ok ? '' : ' · failed'}
                  {typeof c.latency_ms === 'number' ? ` · ${(c.latency_ms / 1000).toFixed(1)} s` : ''}
                </li>
              ))}
            </ul>
          </details>
        )}

        {!isUser && msg.answer_id && (
          <Feedback answerId={msg.answer_id} rating={msg.my_rating ?? null} comment={msg.my_comment ?? ''} />
        )}
      </div>
    </div>
  );
}

const ROUTE_LABELS: Record<string, string> = {
  documents: 'Documents',
  genie: 'Structured data',
  tools: 'Tools',
  direct: 'Direct',
};

/** QRS / ARS / HI badges (each shown only when present). */
function Scores({ qrs, ars, hi_eq, hi_rf }: { qrs?: number | null; ars?: number | null; hi_eq?: number | null; hi_rf?: number | null }) {
  return (
    <>
      {typeof qrs === 'number' && (
        <Badge variant="secondary" className="text-[10px]">
          QRS {Math.round(qrs)}
        </Badge>
      )}
      {typeof ars === 'number' && (
        <Badge variant="secondary" className="text-[10px]">
          ARS {Math.round(ars)}
        </Badge>
      )}
      {typeof hi_eq === 'number' && (
        <Badge
          variant={hi_eq >= 50 ? 'destructive' : 'secondary'}
          className="text-[10px]"
          title={typeof hi_rf === 'number' ? `Random-forest HI: ${Math.round(hi_rf)}` : undefined}
        >
          HI {Math.round(hi_eq)}
          {typeof hi_rf === 'number' ? ` · RF ${Math.round(hi_rf)}` : ''}
        </Badge>
      )}
    </>
  );
}

/** Sources split per searched library when several were (citation numbers stay global). */
function sourceGroups(msg: ChatMessage): { library: LibrarySearched | null; sources: Source[]; first: number }[] {
  const sources = msg.sources ?? [];
  const searched = msg.libraries_searched ?? [];
  if (searched.length <= 1) return [{ library: null, sources, first: 1 }];
  return searched.map((lib) => ({
    library: lib,
    sources: sources.slice(lib.first_source - 1, lib.first_source - 1 + lib.n_sources),
    first: lib.first_source,
  }));
}

function SourceList({ sources, first }: { sources: Source[]; first: number }) {
  return (
            <ol className="space-y-1">
              {sources.map((s, i) => (
                <li key={i} className="flex items-start gap-2 text-xs">
                  <span className="text-muted-foreground">[{first + i}]</span>
                  <span className="flex-1 break-words">
                    <span className="font-medium">{s.source_name}</span>
                    {s.page != null ? ` · p${s.page}` : ''}
                    {s.section && s.section !== '(no section)' ? ` · ${s.section}` : ''}
                  </span>
                  {typeof s.rerank_score === 'number' && (
                    <Badge variant="outline" className="text-[10px]" title="cross-encoder rerank score">
                      ↕ {s.rerank_score.toFixed(2)}
                    </Badge>
                  )}
                  {s.nli && (
                    <Badge
                      variant={s.nli === 'contradiction' ? 'destructive' : 'outline'}
                      className="text-[10px]"
                      title="answer↔source NLI entailment"
                    >
                      {s.nli}
                    </Badge>
                  )}
                </li>
              ))}
            </ol>
  );
}

const cellText = (v: unknown): string =>
  v == null ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v as string | number | boolean);

/** Genie's answer: its text, the SQL it ran and (up to 20 of) the result rows. */
function GeniePanel({ genie }: { genie: GenieResult }) {
  const columns = genie.columns ?? [];
  const rows = (genie.rows ?? []).slice(0, 20);
  return (
    <details className="mt-3 border-t border-border pt-2 text-xs">
      <summary className="cursor-pointer font-medium text-muted-foreground">
        Structured data{typeof genie.row_count === 'number' ? ` · ${genie.row_count} row${genie.row_count === 1 ? '' : 's'}` : ''}
        {genie.status && genie.status !== 'COMPLETED' ? ` · ${genie.status.toLowerCase()}` : ''}
      </summary>
      {genie.sql && <pre className="mt-2 overflow-x-auto whitespace-pre-wrap rounded bg-background p-2 font-mono text-[11px]">{genie.sql}</pre>}
      {columns.length > 0 && rows.length > 0 && (
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-left text-[11px]">
            <thead>
              <tr>{columns.map((c) => <th key={c} className="border-b border-border px-1 py-0.5 font-medium">{c}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  {r.map((v, j) => <td key={j} className="border-b border-border/50 px-1 py-0.5">{cellText(v)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </details>
  );
}

/** MyGPT's translated question / answer (functional minimum; full UX in Phase 7). */
function Translations({ msg }: { msg: ChatMessage }) {
  const lang = msg.documents_language ? msg.documents_language[0].toUpperCase() + msg.documents_language.slice(1) : '';
  return (
    <div className="mt-2 space-y-1 text-xs text-muted-foreground">
      {msg.translated_query && (
        <div className="break-words">
          Searched{lang ? ` in ${lang}` : ''} as: <span className="italic">{msg.translated_query}</span>
        </div>
      )}
      {msg.translated_answer && (
        <details>
          <summary className="cursor-pointer">Answer as scored{lang ? ` (${lang})` : ''}</summary>
          <div className="mt-1 whitespace-pre-wrap break-words">{msg.translated_answer}</div>
        </details>
      )}
    </div>
  );
}

/** MyGPT answer feedback: correct / incorrect plus an optional comment (one per user). */
function Feedback({ answerId, rating, comment }: { answerId: string; rating: number | null; comment: string }) {
  const [current, setCurrent] = useState<number | null>(rating);
  const [text, setText] = useState(comment);
  const [state, setState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  const save = (r: Rating, c: string) => {
    setState('saving');
    api
      .feedback(answerId, r, c)
      .then(() => {
        setCurrent(r);
        setState('saved');
      })
      .catch(() => setState('error'));
  };
  const toggle = (r: Rating) => save(current === r ? 0 : r, text);

  return (
    <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border pt-2 text-xs">
      <Button
        variant={current === 1 ? 'default' : 'ghost'}
        size="sm"
        className="h-7 px-2"
        onClick={() => toggle(1)}
        title="Correct"
      >
        <ThumbsUp className="h-3.5 w-3.5" />
      </Button>
      <Button
        variant={current === -1 ? 'destructive' : 'ghost'}
        size="sm"
        className="h-7 px-2"
        onClick={() => toggle(-1)}
        title="Incorrect"
      >
        <ThumbsDown className="h-3.5 w-3.5" />
      </Button>
      {current !== null && current !== 0 && (
        <>
          <Input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Optional comment"
            className="h-7 flex-1 min-w-40 text-xs"
          />
          <Button variant="outline" size="sm" className="h-7" onClick={() => save(current as Rating, text)}>
            Save
          </Button>
        </>
      )}
      {state === 'saving' && <Spinner className="h-3 w-3" />}
      {state === 'saved' && <span className="text-muted-foreground">Thanks — saved</span>}
      {state === 'error' && <span className="text-destructive">Couldn’t save feedback</span>}
    </div>
  );
}
