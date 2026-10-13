import type { Request, Response } from 'express';
import type { Db } from '../db/context';
import type { Store } from '../db/store';

export interface Deps {
  /** null = stateless mode (LAKEBASE_ENDPOINT unset): no libraries/history/uploads. */
  db: Db | null;
  store: Store;
}

/** Respond 503 when Lakebase isn't configured; returns the db otherwise. */
export function requireDb(deps: Deps, res: Response): Db | null {
  if (!deps.db) res.status(503).json({ error: 'persistence is not configured (LAKEBASE_ENDPOINT unset)' });
  return deps.db;
}

/** Wrap an async handler so a thrown error becomes a logged 500 instead of a hung request. */
export function handle(name: string, fn: (req: Request, res: Response) => Promise<void>) {
  return (req: Request, res: Response) => {
    fn(req, res).catch((err: unknown) => {
      console.error(`${name} failed`, err);
      if (!res.headersSent) res.status(500).json({ error: `${name} failed` });
    });
  };
}
