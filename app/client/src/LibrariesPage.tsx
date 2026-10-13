// Library self-service (functional minimum; the full UX is Phase 7): list the libraries I can
// see, create upload libraries, upload documents and watch them index, share, and delete.

import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router';
import {
  Alert,
  AlertDescription,
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Input,
  Label,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Spinner,
  Switch,
} from '@databricks/appkit-ui/react';
import { Trash2, Upload } from 'lucide-react';
import { api, LANGUAGES, type Library, type LibraryDocument, type Visibility, type Whoami } from './api';

const statusVariant = (s: string) => (s === 'failed' ? 'destructive' : s === 'ready' ? 'secondary' : 'outline');

export function LibrariesPage() {
  const [me, setMe] = useState<Whoami | null>(null);
  const [libraries, setLibraries] = useState<Library[]>([]);
  const [persistence, setPersistence] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const reload = useCallback(() => {
    api
      .libraries()
      .then((d) => {
        setLibraries(d.libraries);
        setPersistence(d.persistence);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    api.whoami().then(setMe).catch(() => setMe(null));
    reload();
  }, [reload]);

  const mine = libraries.filter((l) => l.can_edit);
  const others = libraries.filter((l) => !l.can_edit);

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6">
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      {!persistence && (
        <Alert>
          <AlertDescription>Libraries need Lakebase (LAKEBASE_ENDPOINT); only the governed corpus is available.</AlertDescription>
        </Alert>
      )}

      {persistence && me?.user && <CreateLibrary groups={me.groups} onCreated={(l) => { reload(); setSelected(l.id); }} />}

      <section className="flex flex-col gap-2">
        <h2 className="text-lg font-semibold">My libraries</h2>
        {mine.length === 0 && <p className="text-sm text-muted-foreground">None yet — create one above.</p>}
        {mine.map((l) => (
          <LibraryCard
            key={l.id}
            library={l}
            groups={me?.groups ?? []}
            open={selected === l.id}
            onToggle={() => setSelected(selected === l.id ? null : l.id)}
            onChanged={reload}
          />
        ))}
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="text-lg font-semibold">Shared with me &amp; governed corpora</h2>
        {others.map((l) => (
          <Card key={l.id}>
            <CardHeader className="py-3">
              <CardTitle className="flex flex-wrap items-center gap-2 text-base">
                {l.name}
                <Badge variant="outline">{l.kind === 'corpus' ? 'governed corpus' : l.visibility}</Badge>
                {l.kind === 'upload' && <Badge variant={statusVariant(l.status)}>{l.status}</Badge>}
                {l.share_history && <Badge variant="outline">shared history</Badge>}
                <Link to={`/?library=${l.id}`} className="ml-auto text-sm font-normal text-primary hover:underline">
                  Ask
                </Link>
              </CardTitle>
              <CardDescription>
                {l.description || (l.kind === 'upload' ? `${l.n_documents ?? 0} documents` : '')}
                {l.kind === 'upload' ? ` · owner ${l.owner_email}` : ''}
                {l.group_name ? ` · group ${l.group_name}` : ''}
              </CardDescription>
            </CardHeader>
          </Card>
        ))}
      </section>
    </div>
  );
}

function CreateLibrary({ groups, onCreated }: { groups: string[]; onCreated: (l: Library) => void }) {
  const [name, setName] = useState('');
  const [visibility, setVisibility] = useState<Visibility>('private');
  const [group, setGroup] = useState(groups[0] ?? '');
  const [language, setLanguage] = useState<string>('english');
  const [shareHistory, setShareHistory] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const create = () => {
    setBusy(true);
    setError(null);
    api
      .createLibrary({
        name, visibility, documents_language: language, share_history: shareHistory,
        group_name: visibility === 'group' ? group : undefined,
      })
      .then((d) => {
        setName('');
        onCreated(d.library);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">New library</CardTitle>
        <CardDescription>Upload your own PDF / DOCX / PPTX documents and ask questions over them.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <Label htmlFor="lib-name">Name</Label>
          <Input id="lib-name" value={name} onChange={(e) => setName(e.target.value)} className="w-56" />
        </div>
        <div className="flex flex-col gap-1">
          <Label>Visibility</Label>
          <VisibilitySelect value={visibility} onChange={setVisibility} hasGroups={groups.length > 0} />
        </div>
        {visibility === 'group' && (
          <div className="flex flex-col gap-1">
            <Label>Group</Label>
            <GroupSelect groups={groups} value={group} onChange={setGroup} />
          </div>
        )}
        <div className="flex flex-col gap-1">
          <Label>Language</Label>
          <Select value={language} onValueChange={setLanguage}>
            <SelectTrigger className="w-36">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {LANGUAGES.map((l) => (
                <SelectItem key={l} value={l}>
                  {l}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {visibility !== 'private' && (
          <div className="flex items-center gap-2 pb-2">
            <Switch id="share-history" checked={shareHistory} onCheckedChange={setShareHistory} />
            <Label htmlFor="share-history" className="text-sm">
              Share chat history
            </Label>
          </div>
        )}
        <Button onClick={create} disabled={busy || !name.trim() || (visibility === 'group' && !group)}>
          {busy ? <Spinner className="h-4 w-4" /> : 'Create'}
        </Button>
        {error && <p className="w-full text-sm text-destructive">{error}</p>}
      </CardContent>
    </Card>
  );
}

function VisibilitySelect({
  value, onChange, hasGroups,
}: { value: Visibility; onChange: (v: Visibility) => void; hasGroups: boolean }) {
  return (
    <Select value={value} onValueChange={(v) => onChange(v as Visibility)}>
      <SelectTrigger className="w-36">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="private">Private</SelectItem>
        {hasGroups && <SelectItem value="group">Group</SelectItem>}
        <SelectItem value="public">Public</SelectItem>
      </SelectContent>
    </Select>
  );
}

function GroupSelect({ groups, value, onChange }: { groups: string[]; value: string; onChange: (g: string) => void }) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="w-48">
        <SelectValue placeholder="Group" />
      </SelectTrigger>
      <SelectContent>
        {groups.map((g) => (
          <SelectItem key={g} value={g}>
            {g}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function LibraryCard({
  library, groups, open, onToggle, onChanged,
}: { library: Library; groups: string[]; open: boolean; onToggle: () => void; onChanged: () => void }) {
  const [lib, setLib] = useState(library);
  const [docs, setDocs] = useState<LibraryDocument[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => {
    api
      .status(library.id)
      .then((d) => {
        if (d.library) setLib(d.library);
        setDocs(d.documents);
      })
      .catch((e: Error) => setError(e.message));
  }, [library.id]);

  useEffect(() => {
    if (open) refresh();
  }, [open, refresh]);

  // Poll every 5 s while documents are waiting / indexing (status also drives follow-up runs).
  const pending = lib.status === 'ingesting' || docs.some((d) => d.status === 'uploaded' || d.status === 'parsing');
  useEffect(() => {
    if (!open || !pending) return;
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [open, pending, refresh]);

  const upload = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setBusy(true);
    setError(null);
    api
      .uploadDocuments(lib.id, Array.from(files))
      .then((d) => {
        if (d.ingest_error) setError(`Uploaded, but indexing could not start: ${d.ingest_error}`);
        refresh();
        onChanged();
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => {
        setBusy(false);
        if (fileInput.current) fileInput.current.value = '';
      });
  };

  const update = (patch: Parameters<typeof api.updateLibrary>[1]) => {
    api
      .updateLibrary(lib.id, patch)
      .then((d) => {
        setLib(d.library);
        onChanged();
      })
      .catch((e: Error) => setError(e.message));
  };

  const retry = () => {
    setError(null);
    api.retryFailed(lib.id).then(() => { refresh(); onChanged(); }).catch((e: Error) => setError(e.message));
  };

  const remove = () => {
    if (!window.confirm(`Delete library “${lib.name}” and all its documents?`)) return;
    api.deleteLibrary(lib.id).then(onChanged).catch((e: Error) => setError(e.message));
  };

  const removeDoc = (doc: LibraryDocument) => {
    if (!window.confirm(`Remove ${doc.filename}?`)) return;
    api.deleteDocument(lib.id, doc.id).then(() => { refresh(); onChanged(); }).catch((e: Error) => setError(e.message));
  };

  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="flex flex-wrap items-center gap-2 text-base">
          <button className="hover:underline" onClick={onToggle}>
            {lib.name}
          </button>
          <Badge variant="outline">{lib.visibility}{lib.group_name ? ` · ${lib.group_name}` : ''}</Badge>
          <Badge variant={statusVariant(lib.status)}>{lib.status}</Badge>
          {pending && <Spinner className="h-3 w-3" />}
          <span className="text-xs font-normal text-muted-foreground">{lib.documents_language}</span>
          {lib.status === 'ready' && (
            <Link to={`/?library=${lib.id}`} className="ml-auto text-sm font-normal text-primary hover:underline">
              Ask
            </Link>
          )}
        </CardTitle>
      </CardHeader>
      {open && (
        <CardContent className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <VisibilitySelect
              value={lib.visibility}
              hasGroups={groups.length > 0}
              onChange={(v) => update({ visibility: v, group_name: v === 'group' ? (lib.group_name ?? groups[0]) : null })}
            />
            {lib.visibility === 'group' && (
              <GroupSelect groups={groups} value={lib.group_name ?? ''} onChange={(g) => update({ group_name: g })} />
            )}
            {lib.visibility !== 'private' && (
              <div className="flex items-center gap-2">
                <Switch
                  id={`sh-${lib.id}`}
                  checked={lib.share_history}
                  onCheckedChange={(v) => update({ share_history: v })}
                />
                <Label htmlFor={`sh-${lib.id}`} className="text-sm">
                  Share chat history
                </Label>
              </div>
            )}
            <input
              ref={fileInput}
              type="file"
              multiple
              accept=".pdf,.docx,.pptx"
              className="hidden"
              onChange={(e) => upload(e.target.files)}
            />
            <Button variant="outline" size="sm" onClick={() => fileInput.current?.click()} disabled={busy}>
              {busy ? <Spinner className="h-4 w-4" /> : <Upload className="h-4 w-4" />}
              <span className="ml-1">Upload documents</span>
            </Button>
            {lib.status !== 'ingesting' && docs.some((d) => d.status === 'failed' || d.status === 'parsing') && (
              <Button variant="outline" size="sm" onClick={retry}>
                Retry failed
              </Button>
            )}
            <Button variant="ghost" size="sm" className="ml-auto text-destructive" onClick={remove}>
              <Trash2 className="h-4 w-4" />
              <span className="ml-1">Delete library</span>
            </Button>
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          {docs.length === 0 ? (
            <p className="text-sm text-muted-foreground">No documents yet.</p>
          ) : (
            <ul className="divide-y rounded-md border text-sm">
              {docs.map((d) => (
                <li key={d.id} className="flex items-center gap-2 px-3 py-2">
                  <span className="flex-1 break-all">{d.filename}</span>
                  {d.status === 'ready' && <span className="text-xs text-muted-foreground">{d.n_chunks} chunks</span>}
                  <Badge variant={statusVariant(d.status)} title={d.error ?? undefined}>
                    {d.status}
                  </Badge>
                  <Button variant="ghost" size="sm" className="h-7 px-2" onClick={() => removeDoc(d)} title="Remove">
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      )}
    </Card>
  );
}
