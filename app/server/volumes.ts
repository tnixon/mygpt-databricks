// UC volume file operations for upload libraries (app service principal, WRITE_VOLUME on the
// uploads volume). Raw Files API PUT, like AppKit's files connector, so Buffers stream as-is.

import { createWorkspaceClient } from '@databricks/appkit';

/** Keep a safe, readable file name: no path parts, only [A-Za-z0-9._-], max 120 chars. */
export function sanitizeFilename(name: string): string {
  const base = name.split(/[\\/]/).pop() ?? '';
  const cleaned = base
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^\w.-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^[._]+/, '');
  return (cleaned || 'document').slice(-120);
}

export async function uploadFile(path: string, contents: Buffer): Promise<void> {
  const wc = createWorkspaceClient();
  // getHost() resolves the SDK config first; `config.host` is unset until something has
  // (in Apps the host comes from the injected env, resolved lazily).
  const url = new URL(`/api/2.0/fs/files${encodeURI(path)}`, await wc.config.getHost());
  url.searchParams.set('overwrite', 'true');
  const headers = new Headers({
    'Content-Type': 'application/octet-stream',
    'Content-Length': String(contents.length),
  });
  await wc.config.authenticate(headers);
  const res = await fetch(url, { method: 'PUT', headers, body: new Uint8Array(contents) });
  if (!res.ok) throw new Error(`volume upload failed (${res.status}): ${(await res.text()).slice(0, 200)}`);
}

export async function deleteFile(path: string): Promise<void> {
  const wc = createWorkspaceClient();
  try {
    await wc.apiClient.request({
      path: `/api/2.0/fs/files${encodeURI(path)}`,
      method: 'DELETE',
      headers: new Headers(),
      raw: false,
    });
  } catch (err) {
    if (!String(err).includes('NOT_FOUND') && !String(err).includes('404')) throw err;
  }
}

/** Delete a directory and the files directly in it (upload folders are flat). */
export async function deleteDirectory(path: string): Promise<void> {
  const wc = createWorkspaceClient();
  let listing: { contents?: { path: string; is_directory?: boolean }[] };
  try {
    listing = (await wc.apiClient.request({
      path: `/api/2.0/fs/directories${encodeURI(path)}`,
      method: 'GET',
      headers: new Headers(),
      raw: false,
    })) as typeof listing;
  } catch (err) {
    if (String(err).includes('NOT_FOUND') || String(err).includes('404')) return;
    throw err;
  }
  for (const entry of listing.contents ?? []) {
    if (!entry.is_directory) await deleteFile(entry.path);
  }
  await wc.apiClient.request({
    path: `/api/2.0/fs/directories${encodeURI(path)}`,
    method: 'DELETE',
    headers: new Headers(),
    raw: false,
  });
}
