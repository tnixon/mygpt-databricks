// Conversation history (functional minimum; search + CSV export are Phase 7): my conversations
// plus shared-history conversations on libraries shared with me, each re-openable read-only
// or continued in chat.

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import {
  Alert,
  AlertDescription,
  Badge,
  Button,
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  ScrollArea,
  Spinner,
} from '@databricks/appkit-ui/react';
import { Trash2 } from 'lucide-react';
import { api, type ConversationSummary, type QaTurn } from './api';
import { MessageBubble } from './MessageBubble';

export function HistoryPage() {
  const [conversations, setConversations] = useState<ConversationSummary[] | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reload = () => {
    api
      .conversations()
      .then((d) => setConversations(d.conversations))
      .catch((e: Error) => setError(e.message));
  };
  useEffect(reload, []);

  const remove = (c: ConversationSummary) => {
    if (!window.confirm(`Delete “${c.title}”?`)) return;
    api
      .deleteConversation(c.id)
      .then(() => {
        if (openId === c.id) setOpenId(null);
        reload();
      })
      .catch((e: Error) => setError(e.message));
  };

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-3">
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      {conversations === null && <Spinner className="h-5 w-5" />}
      {conversations?.length === 0 && (
        <p className="text-sm text-muted-foreground">No conversations yet — grounded questions are saved here.</p>
      )}
      {conversations?.map((c) => (
        <Card key={c.id}>
          <CardHeader className="py-3">
            <CardTitle className="flex flex-wrap items-center gap-2 text-base">
              <button className="text-left hover:underline" onClick={() => setOpenId(openId === c.id ? null : c.id)}>
                {c.title}
              </button>
              {c.shared && <Badge variant="outline">shared by {c.owner_email}</Badge>}
              {!c.shared && (
                <Button variant="ghost" size="sm" className="ml-auto h-7 px-2" onClick={() => remove(c)} title="Delete">
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              )}
            </CardTitle>
            <CardDescription>
              {c.library_name ?? ((c.library_ids?.length ?? 0) > 1 ? `${c.library_ids!.length} libraries` : 'deleted library')} · {c.n_questions} question{c.n_questions === 1 ? '' : 's'} ·{' '}
              {new Date(c.last_at).toLocaleString()}
            </CardDescription>
          </CardHeader>
          {openId === c.id && <ConversationView id={c.id} />}
        </Card>
      ))}
    </div>
  );
}

function ConversationView({ id }: { id: string }) {
  const navigate = useNavigate();
  const [turns, setTurns] = useState<QaTurn[] | null>(null);

  useEffect(() => {
    api.conversation(id).then((d) => setTurns(d.turns)).catch(() => setTurns([]));
  }, [id]);

  if (turns === null) return <Spinner className="m-4 h-4 w-4" />;
  return (
    <div className="flex flex-col gap-2 px-4 pb-4">
      <ScrollArea className="max-h-[60vh] rounded-md border bg-card">
        <div className="flex flex-col gap-4 p-4">
          {turns.map((t) => (
            <div key={t.question_id} className="flex flex-col gap-4">
              <MessageBubble msg={{ role: 'user', content: t.question_text }} />
              <MessageBubble
                msg={{
                  role: 'assistant', content: t.answer_text ?? '', model: t.model, sources: t.sources, qrs: t.qrs,
                  ars: t.ars, hi_eq: t.hi_eq, hi_rf: t.hi_rf, answer_id: t.answer_id, my_rating: t.my_rating,
                  my_comment: t.my_comment, translated_query: t.translated_question_text,
                  translated_answer: t.translated_answer_text, ...(t.custom_outputs ?? {}),
                }}
              />
            </div>
          ))}
        </div>
      </ScrollArea>
      <Button variant="outline" size="sm" className="self-start" onClick={() => void navigate(`/?conversation=${id}`)}>
        Continue in chat
      </Button>
    </div>
  );
}
