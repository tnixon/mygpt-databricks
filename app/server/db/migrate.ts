import type { Db } from './context';
import { MIGRATIONS, type Migration } from './migrations';

const SCHEMA_RE = /^[a-z_][a-z0-9_]{0,62}$/;

/**
 * Apply pending migrations, in order, each in its own transaction. Runs at app startup as
 * the service principal, which therefore owns the schema (the AppKit Lakebase docs: deploy
 * before any local `pnpm dev`, or your own identity ends up owning it).
 * An advisory lock serializes concurrent app instances.
 */
export async function migrate(db: Db, schema: string, migrations: Migration[] = MIGRATIONS): Promise<string[]> {
  if (!SCHEMA_RE.test(schema)) throw new Error(`invalid schema name: ${schema}`);
  const applied: string[] = [];
  const client = await db.connect();
  const exec = (sql: string) => (client.exec ? client.exec(sql) : client.query(sql));
  try {
    await exec(
      `CREATE SCHEMA IF NOT EXISTS ${schema};
       CREATE TABLE IF NOT EXISTS ${schema}.schema_migrations (
         id text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now());`,
    );
    for (const m of migrations) {
      await client.query('BEGIN');
      try {
        await client.query('SELECT pg_advisory_xact_lock(hashtext($1))', [`${schema}.migrations`]);
        const done = await client.query(`SELECT 1 FROM ${schema}.schema_migrations WHERE id = $1`, [m.id]);
        if (done.rows.length === 0) {
          await exec(m.sql.split('{{schema}}').join(schema));
          await client.query(`INSERT INTO ${schema}.schema_migrations (id) VALUES ($1)`, [m.id]);
          applied.push(m.id);
        }
        await client.query('COMMIT');
      } catch (err) {
        await client.query('ROLLBACK').catch(() => undefined);
        throw new Error(`migration ${m.id} failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  } finally {
    client.release();
  }
  return applied;
}
