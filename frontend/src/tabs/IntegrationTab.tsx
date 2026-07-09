import { useEffect, useMemo, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  integrationKeys,
  listDomains,
  listJobs,
  purgeJobs,
  submitJob,
  type Job,
  type PurgeFilter,
  type SubmitJobRequest,
} from "@/api/queries/integration";
import type { ScanResult } from "@/api/queries/integration_chain";
import { DomainSelector } from "@/components/integration/DomainSelector";
import { PipelineRunner } from "@/components/integration/PipelineRunner";
import { ModeSelector } from "@/components/integration/ModeSelector";
import { JobHistoryTable } from "@/components/integration/JobHistoryTable";
import { ScanPanel } from "@/components/integration/ScanPanel";
import { ChainComposer } from "@/components/integration/ChainComposer";
import { ChainProgress } from "@/components/integration/ChainProgress";
import { CollapsibleSection } from "@/components/CollapsibleSection";
import { ConfirmDialog } from "@/components/ConfirmDialog";

type Mode = "onetime" | "delta" | "file";
type Feedback = { type: "success" | "error"; msg: string } | null;
type PendingSubmit = {
  request: SubmitJobRequest;
  target: string;
  targetList: string;
  hasCascade: boolean;
};
type PendingPurge = {
  label: string;
  filter: PurgeFilter;
};

const ACTIVE_STATUSES: ReadonlyArray<Job["status"]> = ["queued", "running"];

export default function IntegrationTab(): JSX.Element {
  const queryClient = useQueryClient();

  const [domain, setDomain] = useState<string>("");
  const [mode, setMode] = useState<Mode>("onetime");
  const [slice, setSlice] = useState<string>("");
  const [filterDomain, setFilterDomain] = useState<string>("");
  const [feedback, setFeedback] = useState<Feedback>(null);
  const [scan, setScan] = useState<ScanResult | null>(null);
  const [activeChainId, setActiveChainId] = useState<string | null>(null);
  // "all" = drop every terminal job; otherwise N = drop jobs older than N days
  const [purgeOlderDays, setPurgeOlderDays] = useState<"all" | number>("all");
  const [reindex, setReindex] = useState<boolean>(false);
  const [pendingSubmit, setPendingSubmit] = useState<PendingSubmit | null>(null);
  const [pendingPurge, setPendingPurge] = useState<PendingPurge | null>(null);

  const domainsQuery = useQuery({
    queryKey: integrationKeys.domains,
    queryFn: listDomains,
  });

  const jobsQuery = useQuery({
    queryKey: integrationKeys.jobs(),
    queryFn: () => listJobs(),
    refetchInterval: (query) => {
      const data = query.state.data as Job[] | undefined;
      const hasActive = data?.some((j) => ACTIVE_STATUSES.includes(j.status)) ?? false;
      return hasActive ? 2000 : false;
    },
  });

  const submitMutation = useMutation({
    mutationFn: submitJob,
    onSuccess: (data) => {
      setFeedback({ type: "success", msg: `Job ${data.job_id} submitted` });
      queryClient.invalidateQueries({
        predicate: (q) =>
          q.queryKey[0] === "integration" && q.queryKey[1] === "jobs",
      });
    },
    onError: (err: Error) =>
      setFeedback({ type: "error", msg: `Failed: ${err.message}` }),
  });

  const purgeMutation = useMutation({
    mutationFn: purgeJobs,
    onSuccess: (data) => {
      setFeedback({ type: "success", msg: `Cleared ${data.deleted} job(s)` });
      queryClient.invalidateQueries({
        predicate: (q) =>
          q.queryKey[0] === "integration" && q.queryKey[1] === "jobs",
      });
    },
    onError: (err: Error) =>
      setFeedback({ type: "error", msg: `Clear failed: ${err.message}` }),
  });

  const domains = domainsQuery.data ?? [];
  const selectedDomainInfo = useMemo(
    () => domains.find((d) => d.name === domain),
    [domains, domain],
  );

  const onetimeBlocked = selectedDomainInfo?.onetime_cascades === true;
  const cascadeTargets = selectedDomainInfo?.cascade_targets ?? [];
  // File mode is only meaningful for partitioned domains (per-slice reload).
  // For non-partitioned ones it has no useful operation, so block it in the UI.
  const fileBlocked = selectedDomainInfo !== undefined
    && selectedDomainInfo.partitioned === false;

  // Auto-flip to delta when the user lands on a mode that's blocked for the
  // currently-selected domain (onetime → cascade-risk, file → non-partitioned).
  useEffect(() => {
    if ((onetimeBlocked && mode === "onetime") || (fileBlocked && mode === "file")) {
      setMode("delta");
    }
  }, [onetimeBlocked, fileBlocked, mode]);

  const sliceRequired = mode === "file" && selectedDomainInfo?.partitioned === true;
  const submitDisabled =
    submitMutation.isPending ||
    !domain ||
    (sliceRequired && slice.trim() === "");

  const allJobs = jobsQuery.data ?? [];
  const activeJobs = useMemo(
    () => allJobs.filter((j) => ACTIVE_STATUSES.includes(j.status)),
    [allJobs],
  );
  const recentJobs = useMemo(() => {
    const recent = allJobs.filter((j) => !ACTIVE_STATUSES.includes(j.status));
    return filterDomain ? recent.filter((j) => j.domain === filterDomain) : recent;
  }, [allJobs, filterDomain]);

  const handleSubmit = (e: FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    if (submitDisabled) return;
    const request: SubmitJobRequest = {
      domain,
      mode,
      ...(sliceRequired ? { slice: slice.trim() } : {}),
      ...(reindex ? { reindex: true } : {}),
    };
    if (mode === "onetime") {
      const targetList = cascadeTargets.length > 0 ? cascadeTargets.join(", ") : "the target table";
      setPendingSubmit({
        request: {
          ...request,
          ...(cascadeTargets.length > 0 ? { confirm_destructive: true } : {}),
        },
        target: selectedDomainInfo?.name ?? domain,
        targetList,
        hasCascade: cascadeTargets.length > 0,
      });
      return;
    }
    setFeedback(null);
    submitMutation.mutate(request);
  };

  const handleConfirmedSubmit = (): void => {
    if (!pendingSubmit) return;
    setFeedback(null);
    submitMutation.mutate(pendingSubmit.request);
    setPendingSubmit(null);
  };

  const handlePurgeRequest = (): void => {
    const label =
      purgeOlderDays === "all"
        ? "all terminal jobs"
        : `terminal jobs older than ${purgeOlderDays} day${purgeOlderDays === 1 ? "" : "s"}`;
    setPendingPurge({
      label,
      filter:
        purgeOlderDays === "all"
          ? {}
          : { older_than_hours: purgeOlderDays * 24 },
    });
  };

  const handleConfirmedPurge = (): void => {
    if (!pendingPurge) return;
    setFeedback(null);
    purgeMutation.mutate(pendingPurge.filter);
    setPendingPurge(null);
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <header>
        <h1 className="text-xl font-bold text-foreground">Data Integration</h1>
        <p className="text-sm text-muted-foreground mt-0.5">
          Trigger and monitor ETL load jobs across all domains.
        </p>
      </header>

      {/* Smart change detection + chain composer (top-level so it's the
          first thing the user sees — one-click to scan + run masters→facts) */}
      <ScanPanel onScanned={setScan} />
      {scan && (
        <ChainComposer
          scan={scan}
          onSubmitted={(chainId) => {
            setActiveChainId(chainId);
            queryClient.invalidateQueries({
              predicate: (q) =>
                q.queryKey[0] === "integration" && q.queryKey[1] === "jobs",
            });
          }}
        />
      )}
      {activeChainId && (
        <ChainProgress
          chainId={activeChainId}
          onClose={() => setActiveChainId(null)}
        />
      )}

      {/* Run whole pipeline (full reload / incremental refresh) */}
      <CollapsibleSection title="Run Pipeline" storageKey="integration.run_pipeline">
        <PipelineRunner />
      </CollapsibleSection>

      {/* Submit Job */}
      <CollapsibleSection title="Submit Job" storageKey="integration.submit_job">
        {domainsQuery.isLoading ? (
          <p className="text-sm text-muted-foreground">Loading domains...</p>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <DomainSelector
                  value={domain}
                  onChange={setDomain}
                  domains={domains}
                  disabled={submitMutation.isPending}
                />
              </div>

              <div>
                <ModeSelector
                  value={mode}
                  onChange={setMode}
                  disabled={submitMutation.isPending}
                  disabledModes={(() => {
                    const blocks: Partial<Record<"onetime" | "delta" | "file", string>> = {};
                    if (onetimeBlocked) {
                      blocks.onetime =
                        `Cascades to ${cascadeTargets.join(", ")} — use Delta instead`;
                    }
                    if (fileBlocked) {
                      blocks.file =
                        "File mode requires a partitioned domain — use Delta instead";
                    }
                    return Object.keys(blocks).length > 0 ? blocks : undefined;
                  })()}
                  descriptionOverrides={
                    selectedDomainInfo?.partitioned
                      ? {
                          delta: "Upsert ALL partitions (use File for one slice)",
                          file: `Replace one slice (${selectedDomainInfo.partition_format ?? "YYYY-MM"})`,
                        }
                      : undefined
                  }
                />
              </div>

              {sliceRequired && (
                <div className="space-y-1.5">
                  <label
                    htmlFor="integration-slice"
                    className="text-xs font-medium text-foreground/80"
                  >
                    Slice ({selectedDomainInfo?.partition_format ?? "YYYY-MM"})
                  </label>
                  <input
                    id="integration-slice"
                    type="text"
                    placeholder="YYYY-MM"
                    value={slice}
                    onChange={(e) => setSlice(e.target.value)}
                    disabled={submitMutation.isPending}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50"
                  />
                </div>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <button
                type="submit"
                disabled={submitDisabled}
                className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {submitMutation.isPending ? "Submitting..." : "Submit Job"}
              </button>

              <label className="flex items-center gap-1.5 text-xs text-foreground/70 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={reindex}
                  onChange={(e) => setReindex(e.target.checked)}
                  disabled={submitMutation.isPending}
                  className="h-3.5 w-3.5 rounded border-input"
                />
                <span title="Run REINDEX TABLE after upsert — slow, only useful for very large bulk loads">
                  REINDEX after load
                </span>
              </label>

              {feedback && (
                <p
                  role="status"
                  className={
                    feedback.type === "success"
                      ? "text-sm text-emerald-600 dark:text-emerald-400"
                      : "text-sm text-destructive"
                  }
                >
                  {feedback.msg}
                </p>
              )}
            </div>
          </form>
        )}
      </CollapsibleSection>

      {/* Active Jobs */}
      <CollapsibleSection
        title="Active Jobs"
        storageKey="integration.active_jobs"
        headerRight={
          <span className="text-xs text-muted-foreground tabular-nums">
            {activeJobs.length}
          </span>
        }
      >
        <JobHistoryTable jobs={activeJobs} emptyMessage="No active jobs." />
      </CollapsibleSection>

      {/* Recent Jobs */}
      <CollapsibleSection
        title="Recent Jobs"
        storageKey="integration.recent_jobs"
        headerRight={
          <div className="flex items-center gap-2">
            <label
              htmlFor="integration-filter-domain"
              className="text-xs font-medium text-foreground/70"
            >
              Domain
            </label>
            <select
              id="integration-filter-domain"
              value={filterDomain}
              onChange={(e) => setFilterDomain(e.target.value)}
              className="rounded-md border border-input bg-background px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
            >
              <option value="">All domains</option>
              {domains.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name}
                </option>
              ))}
            </select>
            <label
              htmlFor="integration-clear-age"
              className="text-xs font-medium text-foreground/70"
            >
              Older than
            </label>
            <select
              id="integration-clear-age"
              value={String(purgeOlderDays)}
              onChange={(e) => {
                const v = e.target.value;
                setPurgeOlderDays(v === "all" ? "all" : Number(v));
              }}
              className="rounded-md border border-input bg-background px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
            >
              <option value="all">All terminal</option>
              <option value="1">1 day</option>
              <option value="7">7 days</option>
              <option value="30">30 days</option>
              <option value="90">90 days</option>
            </select>
            <button
              type="button"
              onClick={handlePurgeRequest}
              disabled={purgeMutation.isPending}
              className="rounded-md border border-border px-2 py-1 text-xs text-foreground/80 hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
              title="Delete terminal jobs (success / failed / skipped) matching the age filter"
            >
              {purgeMutation.isPending ? "Clearing…" : "Clear"}
            </button>
          </div>
        }
      >
        <JobHistoryTable jobs={recentJobs} emptyMessage="No recent jobs." />
      </CollapsibleSection>

      <ConfirmDialog
        open={pendingSubmit != null}
        onOpenChange={(open) => {
          if (!open) setPendingSubmit(null);
        }}
        title="Confirm one-time reload"
        description={
          pendingSubmit?.hasCascade
            ? "This reload truncates the selected domain and cascades into dependent fact tables."
            : "This reload replaces every row in the selected domain table."
        }
        tone="destructive"
        confirmLabel="Submit one-time reload"
        pendingLabel="Submitting..."
        isPending={submitMutation.isPending}
        details={[
          { label: "Domain", value: pendingSubmit?.target ?? "" },
          { label: "Affected tables", value: pendingSubmit?.targetList ?? "" },
          { label: "Running jobs", value: "Queued and running jobs are not cancelled automatically." },
        ]}
        onConfirm={handleConfirmedSubmit}
      />

      <ConfirmDialog
        open={pendingPurge != null}
        onOpenChange={(open) => {
          if (!open) setPendingPurge(null);
        }}
        title="Clear job history?"
        description="Only terminal integration jobs are deleted. Queued and running jobs are preserved server-side."
        tone="destructive"
        confirmLabel="Clear jobs"
        pendingLabel="Clearing..."
        isPending={purgeMutation.isPending}
        details={[
          { label: "Scope", value: pendingPurge?.label ?? "" },
          { label: "Preserved", value: "Queued and running jobs" },
        ]}
        onConfirm={handleConfirmedPurge}
      />
    </div>
  );
}
