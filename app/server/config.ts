// Runtime configuration (env vars set in app.yaml; see app/databricks.yml for the resources).

// The `uploads-volume` resource injects /Volumes/<catalog>/<schema>/<volume>; a dotted
// <catalog>.<schema>.<volume> is accepted too.
export function volumePath(volume: string): string {
  return volume.startsWith('/') ? volume : `/Volumes/${volume.split('.').join('/')}`;
}

// <catalog>.<schema> of a volume — the governed tables (paper_sections) live beside the uploads.
export function volumeSchema(volume: string): string {
  return volumePath(volume).split('/').slice(2, 4).join('.');
}

const uploadVolume = volumePath(process.env.MYGPT_UPLOAD_VOLUME || '/Volumes/structural_biology/mygpt/uploads');

export const config = {
  // The MyGPT Supervisor ResponsesAgent (routes to documents / Genie / tools; it calls the
  // mygpt-rag-agent endpoint itself, once per library it searches).
  agentEndpoint: process.env.MYGPT_AGENT_ENDPOINT ?? 'mygpt-supervisor',
  // Unity AI Gateway embeddings endpoint (bge-m3) — embeds queries for upload-library search.
  embedEndpoint: process.env.MYGPT_EMBED_ENDPOINT ?? 'mygpt-embeddings',
  // SQL warehouse for the governed section catalogue (paper_sections). Unset = no sections.
  warehouseId: process.env.MYGPT_WAREHOUSE_ID ?? '',
  ucSchema: process.env.MYGPT_SCHEMA || volumeSchema(uploadVolume),
  // Lakebase (Phase 4). Unset LAKEBASE_ENDPOINT = stateless mode: the default governed library
  // only, no history/feedback/uploads (local dev without a database).
  lakebaseEndpoint: process.env.LAKEBASE_ENDPOINT ?? '',
  pgSchema: process.env.MYGPT_PG_SCHEMA ?? 'mygpt',
  // Upload libraries: files land in this UC volume, then the ingest job indexes them.
  uploadVolume,
  uploadJobId: process.env.MYGPT_UPLOAD_JOB_ID ?? '',
  // Shared MLflow experiment: user feedback is logged as assessments on the agent's traces.
  experimentId: process.env.MYGPT_EXPERIMENT_ID ?? '',
  maxUploadBytes: 50 * 1024 * 1024,
};
