import { useEffect, useMemo, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  integrationKeys,
  listDomains,
  listJobs,
  submitJob,
  type Job,
} from "@/api/queries/integration";
import { DomainSelector } from "@/components/integration/DomainSelector";
import { ModeSelector } from "@/components/integration/ModeSelector";
import { JobHistoryTable } from "@/components/integration/JobHistoryTable";

type Mode = "onetime" | "delta" | "file";
type Feedback = { type: "success" | "error"; msg: string } | null;

const ACTIVE_STATUSES: ReadonlyArray<Job["status"]> = ["queued", "running"];

export default function IntegrationTab(): JSX.Element {
  const queryClient = useQueryClient();

  const [domain, setDomain] = useState<string>("");
  const [mode, setMode] = useState<Mode>("onetime");
  const [slice, setSlice] = useState<string>("");
  const [filterDomain, setFilterDomain] = useState<string>("");
  const [feedback, setFeedback] = useState<Feedback>(null);

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
    let confirmDestructive = false;
    if (mode === "onetime") {
      const targetList = cascadeTargets.length > 0 ? cascadeTargets.join(", ") : "the target table";
      const ok = window.confirm(
        `One-time reload will TRUNCATE ${selectedDomainInfo?.name ?? domain}` +
          (cascadeTargets.length > 0
            ? ` and CASCADE-truncate: ${targetList}.\n\nAll rows in those fact tables will be lost. Continue?`
            : `.\n\nAll rows will be replaced. Continue?`),
      );
      if (!ok) return;
      confirmDestructive = cascadeTargets.length > 0;
    }
    setFeedback(null);
    submitMutation.mutate({
      domain,
      mode,
      ...(sliceRequired ? { slice: slice.trim() } : {}),
      ...(confirmDestructive ? { confirm_destructive: true } : {}),
    });
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

      {/* Submit Job */}
      <section className="rounded-lg border border-border bg-card text-card-foreground p-4 space-y-4">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-foreground/80">
          Submit Job
        </h2>

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

            <div className="flex items-center gap-3">
              <button
                type="submit"
                disabled={submitDisabled}
                className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {submitMutation.isPending ? "Submitting..." : "Submit Job"}
              </button>

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
      </section>

      {/* Active Jobs */}
      <section className="rounded-lg border border-border bg-card text-card-foreground p-4 space-y-3">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-foreground/80">
          Active Jobs
        </h2>
        <JobHistoryTable jobs={activeJobs} emptyMessage="No active jobs." />
      </section>

      {/* Recent Jobs */}
      <section className="rounded-lg border border-border bg-card text-card-foreground p-4 space-y-3">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-foreground/80">
            Recent Jobs
          </h2>
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
          </div>
        </div>
        <JobHistoryTable jobs={recentJobs} emptyMessage="No recent jobs." />
      </section>
    </div>
  );
}
