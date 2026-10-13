// Lakebase access with the RLS session context (see migrations.ts for the policy model).
//
// Every user-facing query runs inside withUser(): one transaction on a pooled SP connection
// that first sets `mygpt.user` / `mygpt.groups` with set_config(..., is_local => true), so the
// context can never leak to the next request that reuses the connection.

export interface QueryResultLike<T> {
  rows: T[];
  rowCount?: number | null;
}

export interface Queryable {
  query<T = Record<string, unknown>>(text: string, values?: unknown[]): Promise<QueryResultLike<T>>;
}

export interface DbClient extends Queryable {
  release(): void;
  /** Multi-statement SQL without parameters (PGlite needs this; pg's query() handles it). */
  exec?(sql: string): Promise<unknown>;
}

export interface Db {
  connect(): Promise<DbClient>;
}

export interface Identity {
  email: string;
  groups: string[];
}

async function inTransaction<T>(
  db: Db,
  context: { user: string; groups: string[]; role: '' | 'admin' },
  fn: (q: Queryable) => Promise<T>,
): Promise<T> {
  const client = await db.connect();
  try {
    await client.query('BEGIN');
    await client.query(
      "SELECT set_config('mygpt.user', $1, true), set_config('mygpt.groups', $2, true), set_config('mygpt.role', $3, true)",
      [context.user, JSON.stringify(context.groups), context.role],
    );
    const result = await fn(client);
    await client.query('COMMIT');
    return result;
  } catch (err) {
    await client.query('ROLLBACK').catch(() => undefined);
    throw err;
  } finally {
    client.release();
  }
}

/** Run `fn` as the signed-in user: RLS filters every statement to what they may see. */
export function withUser<T>(db: Db, identity: Identity, fn: (q: Queryable) => Promise<T>): Promise<T> {
  return inTransaction(db, { user: identity.email, groups: identity.groups, role: '' }, fn);
}

/** Run `fn` with the admin context (seeding, server-side status bookkeeping). Never per request. */
export function withAdmin<T>(db: Db, fn: (q: Queryable) => Promise<T>): Promise<T> {
  return inTransaction(db, { user: '', groups: [], role: 'admin' }, fn);
}
