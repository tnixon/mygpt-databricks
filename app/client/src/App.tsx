import { createBrowserRouter, RouterProvider, NavLink, Outlet } from 'react-router';
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
import { Menu, Send } from 'lucide-react';
import { Markdown } from './Markdown';

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
    </nav>
  );
}

function Layout() {
  const isMobile = useIsMobile();
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  useEffect(() => {
    if (!isMobile) setMobileNavOpen(false);
  }, [isMobile]);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <header className="bg-sidebar text-sidebar-foreground border-b border-sidebar-border px-4 md:px-6 py-3 flex items-center gap-4">
        <h1 className="font-edge text-xl font-bold tracking-tight text-sidebar-foreground">MyGPT</h1>
        <NavLinks className="hidden md:flex gap-1" linkClass={navLinkClass} />
        <div className="ml-auto md:hidden">
          <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
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
    children: [{ path: '/', element: <ChatPage /> }],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

const MODELS = [
  { alias: 'sonnet', label: 'Claude Sonnet 5' },
  { alias: 'llama', label: 'Llama 4 Maverick' },
  { alias: 'gptoss', label: 'GPT-OSS 120B' },
] as const;

type ModelAlias = (typeof MODELS)[number]['alias'];

interface Source {
  source_name: string;
  page: number | null;
  score: number;
  snippet: string;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  model?: string;
  sources?: Source[];
  qrs?: number;
}

interface AskResponse {
  answer: string;
  sources: Source[];
  qrs: number;
  model: string;
}

// Pull the text delta out of an OpenAI-compatible streaming chunk.
function extractDelta(chunk: unknown): string {
  const c = chunk as { choices?: Array<{ delta?: { content?: string } }> };
  return c.choices?.[0]?.delta?.content ?? '';
}

function modelLabel(alias?: string): string | undefined {
  return MODELS.find((m) => m.alias === alias)?.label ?? alias;
}

function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [model, setModel] = useState<ModelAlias>(MODELS[0].alias);
  const [grounded, setGrounded] = useState(true);
  const [queued, setQueued] = useState(false); // fires the direct (non-grounded) stream
  const [asking, setAsking] = useState(false); // grounded request in flight
  const [user, setUser] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/whoami')
      .then((r) => r.json())
      .then((d: { user: string | null }) => setUser(d.user))
      .catch(() => setUser(null));
  }, []);

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

  // Grounded mode: retrieve + cite via /api/ask.
  const askGrounded = (question: string) => {
    setAsking(true);
    fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, model }),
    })
      .then((r) => {
        if (!r.ok) throw new Error('ask failed');
        return r.json() as Promise<AskResponse>;
      })
      .then((d) => {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: d.answer, model: d.model, sources: d.sources, qrs: d.qrs },
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

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col gap-4">
      {/* Toolbar: model · grounded toggle · identity · trust note */}
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
        <Badge variant="secondary">{user ? `Running as ${user}` : 'Running as you (OBO)'}</Badge>
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
                  ? 'Grounded in St. Jude research documents — answers cite their sources.'
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

function MessageBubble({ msg, streaming }: { msg: ChatMessage; streaming?: boolean }) {
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
                QRS {msg.qrs.toFixed(2)}
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
                  </span>
                  <Badge variant="outline" className="text-[10px]">
                    {s.score.toFixed(2)}
                  </Badge>
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
    </div>
  );
}
