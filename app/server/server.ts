import { createApp, server, serving } from '@databricks/appkit';

createApp({
  plugins: [
    server(),
    // FMAPI foundation models, fronted by the serving plugin (runs OBO by default).
    // Aliases map to endpoint names injected via app.yaml / .env.
    serving({
      endpoints: {
        sonnet: { env: 'DATABRICKS_SERVING_ENDPOINT_NAME' },
        llama: { env: 'DATABRICKS_SERVING_ENDPOINT_LLAMA' },
        gptoss: { env: 'DATABRICKS_SERVING_ENDPOINT_GPTOSS' },
      },
      timeout: 120000,
    }),
  ],
  // Identity for the AI-trust "who is this running as" badge.
  async onPluginsReady(appkit) {
    appkit.server.extend((app) => {
      app.get('/api/whoami', (req, res) => {
        const user =
          req.header('x-forwarded-email') ??
          req.header('x-forwarded-user') ??
          null;
        res.json({ user });
      });
    });
  },
}).catch(console.error);
