import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import { runPipeline, type PipelineMode } from "@/api/queries/integration";
import { fetchJobDetail } from "@/api/queries/jobs";
import { ConfirmDialog } from "@/components/ConfirmDialog";

const ACTIVE_STATUSES = new Set(["queued", "running"]);

/**
 * US19: trigger a whole-pipeline run (full reload or incremental refresh) and
 * watch its managed etl_pipeline job live. Full reloads require confirmation
 * because they wipe and reload every domain.
 */
export function PipelineRunner(): JSX.Element {
  const [mode, setMode] = useState<PipelineMode>("refresh");
  const [parallel, setParallel] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmFullReload, setConfirmFullReload] = useState(false);

  const jobQuery = useQuery({
    queryKey: ["integration", "pipeline-job", jobId],
    queryFn: () => fetchJobDetail(jobId as string),
    enabled: jobId != null,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status && ACTIVE_STATUSES.has(status) ? 2000 : false;
    },
  });

  const mutation = useMutation({
    mutationFn: runPipeline,
    onSuccess: (res) => {
      setJobId(res.job_id);
      setError(null);
    },
    onError: (err: unknown) =>
      setError(err instanceof Error ? err.message : "Failed to start pipeline"),
  });

  function runSelectedMode(): void {
    mutation.mutate({ mode, parallel });
  }

  function handleRun(): void {
    if (mode === "full") {
      setConfirmFullReload(true);
      return;
    }
    runSelectedMode();
  }

  const job = jobQuery.data;

  return (
    <div data-testid="pipeline-runner" className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          Mode
          <select
            aria-label="Pipeline mode"
            value={mode}
            onChange={(e) => setMode(e.target.value as PipelineMode)}
            className="rounded border px-2 py-1"
          >
            <option value="refresh">Incremental refresh</option>
            <option value="full">Full reload</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            aria-label="Parallel"
            checked={parallel}
            onChange={(e) => setParallel(e.target.checked)}
          />
          Parallel
        </label>
        <button
          type="button"
          onClick={handleRun}
          disabled={mutation.isPending}
          className="rounded border px-3 py-1 text-sm font-medium"
        >
          Run Pipeline
        </button>
      </div>

      {error && (
        <p role="alert" className="text-sm">
          {error}
        </p>
      )}

      {job && (
        <div data-testid="pipeline-status" className="text-sm">
          <span>Status: {job.status}</span>
          {" · "}
          <span>{job.progress_pct}%</span>
          {job.progress_msg ? (
            <>
              {" · "}
              <span>{job.progress_msg}</span>
            </>
          ) : null}
        </div>
      )}

      <ConfirmDialog
        open={confirmFullReload}
        onOpenChange={setConfirmFullReload}
        title="Run full pipeline reload?"
        description="A full reload clears and reloads every registered domain before refreshing downstream materialized views."
        tone="destructive"
        confirmLabel="Run full reload"
        pendingLabel="Starting..."
        isPending={mutation.isPending}
        details={[
          { label: "Mode", value: "Full reload" },
          { label: "Parallel", value: parallel ? "Enabled" : "Disabled" },
          { label: "Impact", value: "All domain tables are reloaded from source files." },
        ]}
        onConfirm={() => {
          setConfirmFullReload(false);
          runSelectedMode();
        }}
      />
    </div>
  );
}
