import { createApp, createLakebasePool, server, serving } from '@databricks/appkit';
import { config } from './config';
import type { Db } from './db/context';
import { migrate } from './db/migrate';
import { Store } from './db/store';
import { getIdentity } from './identity';
import { registerAskRoutes } from './routes/ask';
import { registerConversationRoutes } from './routes/conversations';
import type { Deps } from './routes/deps';
import { registerLibraryRoutes } from './routes/libraries';

// Phase 4: Lakebase holds libraries, the per-user upload index, history and feedback (RLS
// via session context, see db/migrations.ts). Without LAKEBASE_ENDPOINT the app runs
// stateless: the default governed library only.
async function connectLakebase(): Promise<Db | null> {
  if (!config.lakebaseEndpoint) {
    console.warn('[lakebase] LAKEBASE_ENDPOINT unset — running without persistence');
    return null;
  }
  try {
    const db = createLakebasePool({ endpoint: config.lakebaseEndpoint, max: 10 }) as unknown as Db;
    const applied = await migrate(db, config.pgSchema);
    console.log(`[lakebase] schema ${config.pgSchema} ready${applied.length ? ` (applied ${applied.join(', ')})` : ''}`);
    return db;
  } catch (err) {
    // Keep answering from the governed corpus rather than crash-looping the app; /api/whoami
    // reports persistence: false, so this is visible in the UI as well as the logs.
    console.error('[lakebase] UNAVAILABLE — running without persistence', err);
    return null;
  }
}

const deps: Deps = { db: await connectLakebase(), store: new Store(config.pgSchema) };

createApp({
  plugins: [
    server(),
    // FMAPI foundation models for DIRECT (non-grounded) chat streaming.
    serving({
      endpoints: {
        sonnet: { env: 'DATABRICKS_SERVING_ENDPOINT_NAME' },
        llama: { env: 'DATABRICKS_SERVING_ENDPOINT_LLAMA' },
        gptoss: { env: 'DATABRICKS_SERVING_ENDPOINT_GPTOSS' },
      },
      timeout: 120000,
    }),
  ],
  onPluginsReady(appkit) {
    appkit.server.extend((app) => {
      // Identity for the AI-trust badge, the group picker, and whether persistence is on.
      app.get('/api/whoami', (req, res) => {
        getIdentity(req)
          .then((who) => res.json({ user: who.email || null, groups: who.groups, persistence: deps.db !== null }))
          .catch(() => res.json({ user: null, groups: [], persistence: deps.db !== null }));
      });
      registerLibraryRoutes(app, deps);
      registerAskRoutes(app, deps);
      registerConversationRoutes(app, deps);
    });
  },
}).catch(console.error);
