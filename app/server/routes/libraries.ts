// Library self-service (gap 14): list / create / update / delete libraries, upload documents
// into upload libraries, and poll their ingest status. Ports MyGPT's get_datasets,
// add_dataset_from_upload / upload_documents and delete_dataset onto RLS-scoped Lakebase.

import { randomUUID } from 'node:crypto';
import type { Application, Request } from 'express';
import multer from 'multer';
import { config } from '../config';
import { withUser, type Identity } from '../db/context';
import type { Library } from '../db/store';
import { getIdentity } from '../identity';
import { refreshIngest, startIngest } from '../ingest';
import { isUuid, parseLibraryInput, UPLOAD_EXTENSIONS, uploadExtension } from '../validation';
import { deleteDirectory, deleteFile, sanitizeFilename, uploadFile } from '../volumes';
import { handle, requireDb, type Deps } from './deps';

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: config.maxUploadBytes, files: 20 } });

/** The governed default when Lakebase isn't configured (stateless mode). */
export const STATELESS_DEFAULT: Library = {
  id: '00000000-0000-0000-0000-000000000000',
  name: 'All governed documents',
  description: 'Every governed corpus in the AI Search index.',
  owner_email: 'system',
  visibility: 'public',
  group_name: null,
  share_history: false,
  kind: 'corpus',
  corpus: null,
  config_dataset: 'mygpt',
  documents_language: 'english',
  status: 'ready',
  is_default: true,
  created_at: '',
  updated_at: '',
};

const canEdit = (lib: Library, who: Identity) => lib.kind === 'upload' && lib.owner_email === who.email;
const withFlags = (lib: Library, who: Identity) => ({ ...lib, can_edit: canEdit(lib, who) });

export function registerLibraryRoutes(app: Application, deps: Deps) {
  const { store } = deps;

  app.get('/api/libraries', handle('/api/libraries', async (req, res) => {
    const who = await getIdentity(req);
    if (!deps.db) {
      res.json({ libraries: [withFlags(STATELESS_DEFAULT, who)], persistence: false });
      return;
    }
    const libs = await withUser(deps.db, who, (q) => store.listLibraries(q));
    res.json({ libraries: libs.map((l) => withFlags(l, who)), persistence: true });
  }));

  app.post('/api/libraries', handle('POST /api/libraries', async (req, res) => {
    const db = requireDb(deps, res);
    if (!db) return;
    const who = await getIdentity(req);
    if (!who.email) {
      res.status(401).json({ error: 'sign-in required' });
      return;
    }
    const parsed = parseLibraryInput(req.body, who.groups, false);
    if (!parsed.ok) {
      res.status(400).json({ error: parsed.error });
      return;
    }
    try {
      const lib = await withUser(db, who, (q) =>
        store.createUploadLibrary(q, who.email, parsed.value as Required<typeof parsed.value>));
      res.status(201).json({ library: withFlags(lib, who) });
    } catch (err) {
      if (String(err).includes('duplicate key')) {
        res.status(409).json({ error: 'you already have a library with that name' });
        return;
      }
      throw err;
    }
  }));

  app.patch('/api/libraries/:id', handle('PATCH /api/libraries', async (req, res) => {
    const db = requireDb(deps, res);
    if (!db) return;
    const id = String(req.params.id);
    const who = await getIdentity(req);
    const parsed = parseLibraryInput(req.body, who.groups, true);
    if (!isUuid(id) || !parsed.ok) {
      res.status(400).json({ error: parsed.ok ? 'invalid id' : parsed.error });
      return;
    }
    const lib = await withUser(db, who, (q) => store.updateLibrary(q, id, parsed.value));
    if (!lib) {
      res.status(404).json({ error: 'library not found or not yours' });
      return;
    }
    res.json({ library: withFlags(lib, who) });
  }));

  app.delete('/api/libraries/:id', handle('DELETE /api/libraries', async (req, res) => {
    const db = requireDb(deps, res);
    if (!db) return;
    const id = String(req.params.id);
    if (!isUuid(id)) {
      res.status(400).json({ error: 'invalid id' });
      return;
    }
    const who = await getIdentity(req);
    // RLS: the DELETE only matches the caller's own upload libraries.
    const deleted = await withUser(db, who, (q) => store.deleteLibrary(q, id));
    if (!deleted) {
      res.status(404).json({ error: 'library not found or not yours' });
      return;
    }
    // Rows (documents, chunks, ingest runs) cascaded; now remove the uploaded files.
    await deleteDirectory(`${config.uploadVolume}/${id}`).catch((err: unknown) =>
      console.warn(`[libraries] could not remove ${config.uploadVolume}/${id}`, err));
    res.json({ deleted: true });
  }));

  app.get('/api/libraries/:id/documents', handle('/api/libraries/documents', async (req, res) => {
    const db = requireDb(deps, res);
    if (!db) return;
    const id = String(req.params.id);
    if (!isUuid(id)) {
      res.status(400).json({ error: 'invalid id' });
      return;
    }
    const who = await getIdentity(req);
    const docs = await withUser(db, who, (q) => store.listDocuments(q, id));
    res.json({ documents: docs });
  }));

  const loadOwnUploadLibrary = async (req: Request, who: Identity): Promise<Library | null> => {
    const id = String(req.params.id);
    if (!isUuid(id) || !deps.db) return null;
    const lib = await withUser(deps.db, who, (q) => store.getLibrary(q, id));
    return lib && canEdit(lib, who) ? lib : null;
  };

  app.post(
    '/api/libraries/:id/documents',
    upload.array('files'),
    handle('POST /api/libraries/documents', async (req, res) => {
      const db = requireDb(deps, res);
      if (!db) return;
      const who = await getIdentity(req);
      const lib = await loadOwnUploadLibrary(req, who);
      if (!lib) {
        res.status(404).json({ error: 'library not found or not yours' });
        return;
      }
      const files = (req.files as Express.Multer.File[] | undefined) ?? [];
      if (files.length === 0) {
        res.status(400).json({ error: 'no files' });
        return;
      }
      const rejected = files.filter((f) => !uploadExtension(f.originalname)).map((f) => f.originalname);
      if (rejected.length > 0) {
        res.status(400).json({
          error: `unsupported file type (allowed: ${Object.keys(UPLOAD_EXTENSIONS).join(', ')})`, rejected,
        });
        return;
      }
      const added = [];
      for (const f of files) {
        const docId = randomUUID();
        const ext = uploadExtension(f.originalname)!;
        const filename = sanitizeFilename(f.originalname);
        const volumePath = `${config.uploadVolume}/${lib.id}/${docId}__${filename}`;
        await uploadFile(volumePath, f.buffer);
        added.push(await withUser(db, who, (q) => store.addDocument(q, {
          id: docId, library_id: lib.id, filename, volume_path: volumePath, size_bytes: f.size,
          content_type: UPLOAD_EXTENSIONS[ext],
        })));
      }
      let run = null;
      let ingestError: string | null = null;
      try {
        run = await startIngest(db, store, who, lib.id);
      } catch (err) {
        ingestError = err instanceof Error ? err.message : String(err);
      }
      res.status(201).json({ documents: added, ingest_run: run, ingest_error: ingestError });
    }),
  );

  app.delete('/api/libraries/:id/documents/:docId', handle('DELETE /api/libraries/documents', async (req, res) => {
    const db = requireDb(deps, res);
    if (!db) return;
    const docId = String(req.params.docId);
    const who = await getIdentity(req);
    const lib = await loadOwnUploadLibrary(req, who);
    if (!lib || !isUuid(docId)) {
      res.status(404).json({ error: 'document not found or not yours' });
      return;
    }
    const doc = await withUser(db, who, (q) => store.deleteDocument(q, lib.id, docId));
    if (!doc) {
      res.status(404).json({ error: 'document not found or not yours' });
      return;
    }
    await deleteFile(doc.volume_path).catch((err: unknown) => console.warn(`[libraries] could not remove ${doc.volume_path}`, err));
    res.json({ deleted: true });
  }));

  app.get('/api/libraries/:id/status', handle('/api/libraries/status', async (req, res) => {
    const db = requireDb(deps, res);
    if (!db) return;
    const id = String(req.params.id);
    if (!isUuid(id)) {
      res.status(400).json({ error: 'invalid id' });
      return;
    }
    const who = await getIdentity(req);
    const lib = await withUser(db, who, (q) => store.getLibrary(q, id));
    if (!lib) {
      res.status(404).json({ error: 'library not found' });
      return;
    }
    if (lib.kind === 'upload') {
      await refreshIngest(db, store, who, id, lib.owner_email).catch((err: unknown) =>
        console.warn(`[ingest] refresh failed for ${id}`, err));
    }
    const [library, documents, run] = await withUser(db, who, async (q) => [
      await store.getLibrary(q, id),
      await store.listDocuments(q, id),
      await store.latestIngestRun(q, id),
    ] as const);
    res.json({ library: library && withFlags(library, who), documents, ingest_run: run });
  }));
}
