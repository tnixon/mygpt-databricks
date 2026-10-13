// Upload-library ingest orchestration: trigger the `mygpt_upload_ingest` job for a library
// and keep ingest_runs / libraries.status in step with the job run. The job itself
// (etl/upload_ingest.py) parses, chunks and embeds the library's `uploaded` documents.
//
// One active run per library: uploads that arrive while a run is in flight stay `uploaded`
// and are picked up by the follow-up run that refreshIngest() starts when the first one ends.

import { createWorkspaceClient } from '@databricks/appkit';
import { config } from './config';
import { withAdmin, withUser, type Db, type Identity } from './db/context';
import type { IngestRun, Store } from './db/store';

interface JobRunState {
  state?: { life_cycle_state?: string; result_state?: string; state_message?: string };
  status?: { state?: string; termination_details?: { code?: string; message?: string } };
}

function runOutcome(run: JobRunState): { done: boolean; ok: boolean; message: string } {
  const life = run.status?.state ?? run.state?.life_cycle_state ?? '';
  const done = ['TERMINATED', 'INTERNAL_ERROR', 'SKIPPED'].includes(life);
  const code = run.status?.termination_details?.code ?? run.state?.result_state ?? '';
  const ok = done && (code === 'SUCCESS' || run.state?.result_state === 'SUCCESS');
  const message = run.status?.termination_details?.message ?? run.state?.state_message ?? life;
  return { done, ok, message };
}

async function runNow(libraryId: string): Promise<number> {
  if (!config.uploadJobId) throw new Error('MYGPT_UPLOAD_JOB_ID is not configured');
  const wc = createWorkspaceClient();
  const res = (await wc.apiClient.request({
    path: '/api/2.1/jobs/run-now',
    method: 'POST',
    headers: new Headers({ 'Content-Type': 'application/json' }),
    raw: false,
    payload: { job_id: Number(config.uploadJobId), job_parameters: { library_id: libraryId } },
  })) as { run_id: number };
  return res.run_id;
}

async function getRun(runId: number): Promise<JobRunState> {
  const wc = createWorkspaceClient();
  return (await wc.apiClient.request({
    path: '/api/2.1/jobs/runs/get',
    method: 'GET',
    headers: new Headers(),
    raw: false,
    query: { run_id: String(runId) },
  })) as JobRunState;
}

/** Start an ingest run for the library unless one is already active. Caller must own it. */
export async function startIngest(db: Db, store: Store, who: Identity, libraryId: string): Promise<IngestRun | null> {
  const run = await withUser(db, who, async (q) => {
    if (await store.activeIngestRun(q, libraryId)) return null;
    if ((await store.countPendingDocuments(q, libraryId)) === 0) return null;
    await store.setLibraryStatus(q, libraryId, 'ingesting');
    return store.createIngestRun(q, libraryId, who.email);
  });
  if (!run) return null;
  try {
    const jobRunId = await runNow(libraryId);
    await withUser(db, who, (q) => store.updateIngestRun(q, run.id, { job_run_id: jobRunId, status: 'running' }));
    return { ...run, job_run_id: jobRunId, status: 'running' };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    await withUser(db, who, async (q) => {
      await store.updateIngestRun(q, run.id, { status: 'failed', message, finished: true });
      await store.setLibraryStatus(q, libraryId, 'failed');
    });
    throw err;
  }
}

/**
 * Reconcile the library's active run with the Jobs API (called from the status poll).
 * Bookkeeping runs as admin so a group member polling a shared library can't stall it;
 * a follow-up run starts only for the owner.
 */
export async function refreshIngest(db: Db, store: Store, who: Identity, libraryId: string, ownerEmail: string) {
  const active = await withUser(db, who, (q) => store.activeIngestRun(q, libraryId));
  if (active?.job_run_id) {
    const outcome = runOutcome(await getRun(Number(active.job_run_id)));
    if (outcome.done) {
      await withAdmin(db, async (q) => {
        await store.updateIngestRun(q, active.id, {
          status: outcome.ok ? 'succeeded' : 'failed', message: outcome.message, finished: true,
        });
        // The job sets per-document status and the library status; only fix a library left
        // `ingesting` by a run that died before it could.
        const lib = await store.getLibrary(q, libraryId);
        if (lib?.status === 'ingesting') await store.setLibraryStatus(q, libraryId, outcome.ok ? 'ready' : 'failed');
      });
    }
  }
  if (who.email === ownerEmail) await startIngest(db, store, who, libraryId);
}
