// Who is asking: email from the Databricks Apps proxy, workspace groups from SCIM /Me.
//
// The Apps proxy authenticates the user and sets x-forwarded-email / x-forwarded-access-token;
// the RLS session context (db/context.ts) is built from this. Groups are read with the
// user's own token (the default iam.current-user:read scope) and cached briefly. A failed
// lookup yields no groups, so group-shared libraries are hidden rather than exposed.

import type { Request } from 'express';
import { createWorkspaceClient } from '@databricks/appkit';
import type { Identity } from './db/context';

const GROUP_TTL_MS = 5 * 60 * 1000;
const groupCache = new Map<string, { groups: string[]; expires: number }>();

interface ScimMe {
  userName?: string;
  emails?: { value?: string; primary?: boolean }[];
  groups?: { display?: string }[];
}

async function fetchMe(token: string | undefined): Promise<ScimMe> {
  const wc = token ? createWorkspaceClient({ token }) : createWorkspaceClient();
  return (await wc.apiClient.request({
    path: '/api/2.0/preview/scim/v2/Me',
    method: 'GET',
    headers: new Headers({ Accept: 'application/scim+json' }),
    raw: false,
  })) as ScimMe;
}

export function userToken(req: Request): string | undefined {
  return req.header('x-forwarded-access-token') ?? undefined;
}

export async function getIdentity(req: Request): Promise<Identity> {
  const token = userToken(req);
  let email = (req.header('x-forwarded-email') ?? '').toLowerCase();
  const cached = email ? groupCache.get(email) : undefined;
  if (cached && cached.expires > Date.now()) return { email, groups: cached.groups };

  let groups: string[] = [];
  let ttl = GROUP_TTL_MS;
  // Locally (no proxy headers) fall back to the CLI identity in development only.
  if (token || process.env.NODE_ENV === 'development') {
    try {
      const me = await fetchMe(token);
      if (!email) email = (me.emails?.find((e) => e.primary)?.value ?? me.userName ?? '').toLowerCase();
      groups = (me.groups ?? []).map((g) => g.display ?? '').filter(Boolean).sort();
    } catch (err) {
      console.warn('[identity] SCIM /Me lookup failed; continuing without groups', err);
      ttl = 30 * 1000; // retry soon
    }
  }
  if (email) groupCache.set(email, { groups, expires: Date.now() + ttl });
  return { email, groups };
}
