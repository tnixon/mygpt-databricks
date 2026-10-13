import { useState } from 'react';
import { Badge, Button, Input, Spinner } from '@databricks/appkit-ui/react';
import { ThumbsDown, ThumbsUp } from 'lucide-react';
import { api, modelLabel, type Rating, type Source } from './api';
import { Markdown } from './Markdown';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  model?: string | null;
  sources?: Source[];
  qrs?: number | null;
  ars?: number | null;
  hi_eq?: number | null;
  hi_rf?: number | null;
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
            {typeof msg.qrs === 'number' && (
              <Badge variant="secondary" className="text-[10px]">
                QRS {Math.round(msg.qrs)}
              </Badge>
            )}
            {typeof msg.ars === 'number' && (
              <Badge variant="secondary" className="text-[10px]">
                ARS {Math.round(msg.ars)}
              </Badge>
            )}
            {typeof msg.hi_eq === 'number' && (
              <Badge
                variant={msg.hi_eq >= 50 ? 'destructive' : 'secondary'}
                className="text-[10px]"
                title={typeof msg.hi_rf === 'number' ? `Random-forest HI: ${Math.round(msg.hi_rf)}` : undefined}
              >
                HI {Math.round(msg.hi_eq)}
                {typeof msg.hi_rf === 'number' ? ` · RF ${Math.round(msg.hi_rf)}` : ''}
              </Badge>
            )}
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

        {!isUser && msg.sources && msg.sources.length > 0 && (
          <div className="mt-3 border-t border-border pt-2">
            <div className="mb-1 text-xs font-medium text-muted-foreground">Sources</div>
            <ol className="space-y-1">
              {msg.sources.map((s, i) => (
                <li key={i} className="flex items-start gap-2 text-xs">
                  <span className="text-muted-foreground">[{i + 1}]</span>
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
          </div>
        )}

        {!isUser && msg.answer_id && (
          <Feedback answerId={msg.answer_id} rating={msg.my_rating ?? null} comment={msg.my_comment ?? ''} />
        )}
      </div>
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
