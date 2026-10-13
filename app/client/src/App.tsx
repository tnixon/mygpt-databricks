import { createBrowserRouter, RouterProvider, NavLink, Outlet, useSearchParams } from 'react-router';
import { useState, useEffect, useMemo, type KeyboardEvent } from 'react';
import {
  Button,
  Textarea,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Switch,
  Label,
  Badge,
  Alert,
  AlertTitle,
  AlertDescription,
  Spinner,
  ScrollArea,
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  useIsMobile,
  useServingStream,
} from '@databricks/appkit-ui/react';
import { Menu, Plus, Send } from 'lucide-react';
import { api, MODELS, type Library, type ModelAlias, type SectionOption } from './api';
import { HistoryPage } from './HistoryPage';
import { LibrariesPage } from './LibrariesPage';
import { MessageBubble, type ChatMessage } from './MessageBubble';

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
    isActive
      ? 'bg-primary text-primary-foreground'
      : 'text-sidebar-foreground/80 hover:bg-sidebar-accent hover:text-sidebar-foreground'
  }`;

const mobileNavLinkClass = ({ isActive }: { isActive: boolean }) =>
  `block px-3 py-2 rounded-md text-sm font-medium transition-colors ${
    isActive
      ? 'bg-primary text-primary-foreground'
      : 'text-muted-foreground hover:bg-muted hover:text-foreground'
  }`;

type NavLinkClassFn = (props: { isActive: boolean }) => string;

function NavLinks({ className, linkClass, onClick }: { className?: string; linkClass: NavLinkClassFn; onClick?: () => void }) {
  return (
    <nav className={className}>
      <NavLink to="/" end className={linkClass} onClick={onClick}>
        Chat
      </NavLink>
      <NavLink to="/libraries" className={linkClass} onClick={onClick}>
        Libraries
      </NavLink>
      <NavLink to="/history" className={linkClass} onClick={onClick}>
        History
      </NavLink>
    </nav>
  );
}

function Layout() {
  const isMobile = useIsMobile();
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  // The sheet only exists on mobile; leaving mobile layout closes it.
  const navOpen = isMobile && mobileNavOpen;

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <header className="bg-sidebar text-sidebar-foreground border-b border-sidebar-border px-4 md:px-6 py-3 flex items-center gap-4">
        <h1 className="font-edge text-xl font-bold tracking-tight text-sidebar-foreground">MyGPT</h1>
        <NavLinks className="hidden md:flex gap-1" linkClass={navLinkClass} />
        <div className="ml-auto md:hidden">
          <Sheet open={navOpen} onOpenChange={setMobileNavOpen}>
            <Button variant="ghost" size="icon" className="text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground" onClick={() => setMobileNavOpen(true)}>
              <Menu className="h-5 w-5" />
              <span className="sr-only">Open navigation</span>
            </Button>
            <SheetContent side="left">
              <SheetHeader>
                <SheetTitle>Navigation</SheetTitle>
              </SheetHeader>
              <NavLinks className="flex flex-col gap-1" linkClass={mobileNavLinkClass} onClick={() => setMobileNavOpen(false)} />
            </SheetContent>
          </Sheet>
        </div>
      </header>

      <main className="flex-1 min-h-0 p-4 md:p-6">
        <Outlet />
      </main>
    </div>
  );
}

const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: '/', element: <ChatPage /> },
      { path: '/libraries', element: <LibrariesPage /> },
      { path: '/history', element: <HistoryPage /> },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

// Pull the text delta out of an OpenAI-compatible streaming chunk.
function extractDelta(chunk: unknown): string {
  const c = chunk as { choices?: Array<{ delta?: { content?: string } }> };
  return c.choices?.[0]?.delta?.content ?? '';
}

function ChatPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [model, setModel] = useState<ModelAlias>(MODELS[0].alias);
  const [grounded, setGrounded] = useState(true);
  const [queued, setQueued] = useState(false); // fires the direct (non-grounded) stream
  const [asking, setAsking] = useState(false); // grounded request in flight
  const [user, setUser] = useState<string | null>(null);
  const [libraries, setLibraries] = useState<Library[]>([]);
  const [libraryId, setLibraryId] = useState<string>('');
  const [conversationId, setConversationId] = useState<string | undefined>(undefined);
  const [sections, setSections] = useState<SectionOption[]>([]);
  const [focusSection, setFocusSection] = useState<string>('all');

  useEffect(() => {
    api.whoami().then((d) => setUser(d.user)).catch(() => setUser(null));
    api
      .libraries()
      .then((d) => {
        // Only searchable libraries: governed corpora, and upload libraries with indexed docs.
        const usable = d.libraries.filter((l) => l.kind === 'corpus' || l.status === 'ready');
        setLibraries(usable);
        const requested = searchParams.get('library');
        const initial = usable.find((l) => l.id === requested) ?? usable.find((l) => l.is_default) ?? usable[0];
        setLibraryId((cur) => cur || initial?.id || '');
      })
      .catch(() => setLibraries([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // "Continue in chat" from History: /?conversation=<id> reloads that conversation.
  useEffect(() => {
    const id = searchParams.get('conversation');
    if (!id) return;
    api
      .conversation(id)
      .then(({ conversation, turns }) => {
        setConversationId(conversation.shared ? undefined : conversation.id);
        if (conversation.library_id) setLibraryId(conversation.library_id);
        setMessages(
          turns.flatMap((t): ChatMessage[] => [
            { role: 'user', content: t.question_text },
            {
              role: 'assistant', content: t.answer_text ?? '', model: t.model, sources: t.sources, qrs: t.qrs,
              ars: t.ars, hi_eq: t.hi_eq, hi_rf: t.hi_rf, answer_id: t.answer_id, my_rating: t.my_rating,
              my_comment: t.my_comment,
            },
          ]),
        );
      })
      .catch(() => undefined)
      .finally(() => setSearchParams({}, { replace: true }));
  }, [searchParams, setSearchParams]);

  useEffect(() => {
    if (!libraryId) return;
    setFocusSection('all');
    api
      .sections(libraryId)
      .then((d) => setSections(d.sections ?? []))
      .catch(() => setSections([]));
  }, [libraryId]);

  const newConversation = () => {
    setMessages([]);
    setConversationId(undefined);
  };

  const selectLibrary = (id: string) => {
    if (id === libraryId) return;
    setLibraryId(id);
    newConversation(); // a conversation belongs to one library
  };

  // Direct mode: stream straight from the model (no retrieval).
  const request = useMemo(
    () => ({ messages: messages.map((m) => ({ role: m.role, content: m.content })) }),
    [messages],
  );
  const { stream, chunks, streaming, error, reset } = useServingStream(request, {
    alias: model,
    onComplete: (finalChunks) => {
      const text = finalChunks.map(extractDelta).join('');
      if (text) setMessages((prev) => [...prev, { role: 'assistant', content: text, model }]);
      reset();
    },
  });
  useEffect(() => {
    if (queued) {
      stream();
      setQueued(false);
    }
  }, [queued, stream]);
  const streamingText = chunks.map(extractDelta).join('');

  // Grounded mode: retrieve + cite via /api/ask (persisted to the conversation).
  const askGrounded = (question: string) => {
    setAsking(true);
    api
      .ask({
        question,
        model,
        library_id: libraryId || undefined,
        conversation_id: conversationId,
        focus: focusSection !== 'all' ? { section: focusSection } : undefined,
      })
      .then((d) => {
        if (d.conversation_id) setConversationId(d.conversation_id);
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: d.answer,
            model: d.model,
            sources: d.sources,
            qrs: d.qrs,
            ars: d.ars,
            hi_eq: d.hi_eq,
            hi_rf: d.hi_rf,
            answer_id: d.answer_id,
          },
        ]);
      })
      .catch(() => {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: 'Something went wrong answering that. Please try again.', model },
        ]);
      })
      .finally(() => setAsking(false));
  };

  const busy = streaming || asking;

  const send = () => {
    const text = input.trim();
    if (!text || busy) return;
    setMessages((prev) => [...prev, { role: 'user', content: text }]);
    setInput('');
    if (grounded) askGrounded(text);
    else setQueued(true);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const library = libraries.find((l) => l.id === libraryId);

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col gap-4">
      {/* Toolbar: model · grounded toggle · library · section · identity · trust note */}
      <div className="flex flex-wrap items-center gap-3">
        <Select value={model} onValueChange={(v) => setModel(v as ModelAlias)}>
          <SelectTrigger className="w-52">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {MODELS.map((m) => (
              <SelectItem key={m.alias} value={m.alias}>
                {m.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div className="flex items-center gap-2">
          <Switch id="grounded" checked={grounded} onCheckedChange={setGrounded} />
          <Label htmlFor="grounded" className="text-sm">
            Grounded
          </Label>
        </div>
        {grounded && libraries.length > 0 && (
          <Select value={libraryId} onValueChange={selectLibrary}>
            <SelectTrigger className="w-64">
              <SelectValue placeholder="Library" />
            </SelectTrigger>
            <SelectContent>
              {libraries.map((l) => (
                <SelectItem key={l.id} value={l.id}>
                  {l.name}
                  {l.kind === 'upload' ? ` · ${l.visibility}` : ''}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        {grounded && sections.length > 0 && (
          <Select value={focusSection} onValueChange={setFocusSection}>
            <SelectTrigger className="w-52">
              <SelectValue placeholder="All sections" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All sections</SelectItem>
              {sections.map((s) => (
                <SelectItem key={s.section_title} value={s.section_title}>
                  {s.section_title} ({s.section_count})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        <Button variant="outline" size="sm" onClick={newConversation} disabled={busy || messages.length === 0}>
          <Plus className="h-4 w-4" />
          <span className="ml-1">New conversation</span>
        </Button>
        <Badge variant="secondary">{user ? `Docs as ${user}` : 'Service principal · dev'}</Badge>
        <span className="ml-auto text-xs text-muted-foreground">
          AI-generated — verify before use.
        </span>
      </div>

      {/* Conversation */}
      <ScrollArea className="flex-1 min-h-0 rounded-lg border bg-card">
        <div className="flex flex-col gap-4 p-4">
          {messages.length === 0 && !busy && (
            <div className="py-16 text-center">
              <h2 className="font-edge text-2xl font-bold text-foreground">Ask MyGPT</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                {grounded
                  ? `Grounded in ${library?.name ?? 'St. Jude research documents'} — answers cite their sources.`
                  : 'Direct model chat (no document retrieval).'}
              </p>
            </div>
          )}
          {messages.map((m, i) => (
            <MessageBubble key={i} msg={m} />
          ))}
          {streaming && <MessageBubble msg={{ role: 'assistant', content: streamingText, model }} streaming />}
          {asking && <MessageBubble msg={{ role: 'assistant', content: '', model }} streaming />}
        </div>
      </ScrollArea>

      {error && !grounded && (
        <Alert variant="destructive">
          <AlertTitle>Something went wrong</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Composer */}
      <div className="flex items-end gap-2">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={grounded ? 'Ask about the research documents…' : 'Ask anything…'}
          rows={2}
          className="flex-1 resize-none"
          disabled={busy}
        />
        <Button onClick={send} disabled={busy || !input.trim()} className="h-auto py-3">
          {busy ? <Spinner className="h-4 w-4" /> : <Send className="h-4 w-4" />}
          <span className="ml-2">Send</span>
        </Button>
      </div>
    </div>
  );
}
