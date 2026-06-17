/**
 * AI Champion read-only panel + generate trigger (spec 02-27).
 * Mounted on the FVA tab — forward-only champion vs ai_champion comparison.
 */
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  aiChampionKeys,
  fetchAiChampionForecast,
  fetchAiChampionLatest,
  triggerAiChampionGenerate,
} from "@/api/queries/ai-champion";
import { fetchActiveJobs, fetchJobDetail } from "@/api/queries/jobs";

const JOB_TYPE = "generate_ai_champion";
const JOB_POLL_MS = 3_000;

function formatQty(value: number | null): string {
  if (value == null) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}

function formatPct(value: number | null): string {
  if (value == null) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}%`;
}

export function AiChampionPanel() {
  const queryClient = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  const [adjustedOnly, setAdjustedOnly] = useState(true);

  const { data: latest, isLoading: latestLoading } = useQuery({
    queryKey: aiChampionKeys.latest(),
    queryFn: fetchAiChampionLatest,
    staleTime: 60_000,
  });

  const { data: forecast, isLoading: forecastLoading } = useQuery({
    queryKey: aiChampionKeys.forecast({ adjusted_only: adjustedOnly, limit: 25 }),
    queryFn: () => fetchAiChampionForecast({ adjusted_only: adjustedOnly, limit: 25 }),
    enabled: latest?.run != null,
    staleTime: 60_000,
  });

  const { data: activeJobs } = useQuery({
    queryKey: ["active-jobs"],
    queryFn: fetchActiveJobs,
    staleTime: JOB_POLL_MS,
    refetchInterval: JOB_POLL_MS,
  });

  useEffect(() => {
    if (!activeJobs?.jobs || jobId) return;
    const running = activeJobs.jobs.find(
      (j) => j.job_type === JOB_TYPE && (j.status === "running" || j.status === "queued"),
    );
    if (running) setJobId(running.job_id);
  }, [activeJobs, jobId]);

  const generateMutation = useMutation({
    mutationFn: () => triggerAiChampionGenerate(),
    onSuccess: (data) => setJobId(data.job_id),
  });

  const { data: jobStatus } = useQuery({
    queryKey: ["jobs", jobId],
    queryFn: () => (jobId ? fetchJobDetail(jobId) : Promise.resolve(null)),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed" || status === "cancelled") return false;
      return JOB_POLL_MS;
    },
  });

  const jobDone = jobStatus?.status === "completed";
  const jobFailed = jobStatus?.status === "failed";
  const inFlight = generateMutation.isPending || (!!jobId && !jobDone && !jobFailed);

  useEffect(() => {
    if (jobDone) {
      queryClient.invalidateQueries({ queryKey: aiChampionKeys.latest() });
      queryClient.invalidateQueries({ queryKey: ["ai-champion", "forecast"] });
      const timer = setTimeout(() => setJobId(null), 3_000);
      return () => clearTimeout(timer);
    }
  }, [jobDone, queryClient]);

  const run = latest?.run;

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-amber-600 dark:text-amber-400" />
            <h3 className="text-sm font-medium text-foreground">AI Champion Forecast</h3>
          </div>
          <p className="mt-1 max-w-2xl text-xs text-muted-foreground">
            Forward-only LLM nudge on the promoted champion forecast. Not graded in the FVA ladder
            (no historical actuals for future months).
          </p>
        </div>
        <Button
          size="sm"
          variant="outline"
          disabled={inFlight}
          onClick={() => generateMutation.mutate()}
        >
          {inFlight ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Generating…
            </>
          ) : (
            "Generate"
          )}
        </Button>
      </div>

      {jobId && (
        <p
          className={`mt-3 rounded-md border px-3 py-2 text-xs ${
            jobDone
              ? "border-green-200 bg-green-50 text-green-800 dark:border-green-800 dark:bg-green-950/30 dark:text-green-300"
              : jobFailed
                ? "border-red-200 bg-red-50 text-red-800 dark:border-red-800 dark:bg-red-950/30 dark:text-red-300"
                : "border-blue-200 bg-blue-50 text-blue-800 dark:border-blue-800 dark:bg-blue-950/30 dark:text-blue-300"
          }`}
        >
          {jobDone && "Generation completed. Refreshing…"}
          {jobFailed && "Generation failed. Check Jobs tab for logs."}
          {!jobDone && !jobFailed && (
            <>
              Job {jobId.slice(0, 8)}…
              {jobStatus?.progress_pct != null ? ` (${jobStatus.progress_pct}%)` : ""}
              {jobStatus?.progress_msg ? ` — ${jobStatus.progress_msg}` : ""}
            </>
          )}
        </p>
      )}

      {latestLoading ? (
        <p className="mt-4 text-sm text-muted-foreground">Loading latest run…</p>
      ) : !run ? (
        <p className="mt-4 text-sm text-muted-foreground">
          No AI Champion run yet. Click Generate to create a forward forecast.
        </p>
      ) : (
        <>
          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-md border border-border/60 px-3 py-2">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Status</p>
              <p className="font-medium capitalize">{run.status}</p>
            </div>
            <div className="rounded-md border border-border/60 px-3 py-2">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Adjusted DFUs</p>
              <p className="font-medium">
                {run.n_adjusted.toLocaleString()} / {run.n_dfus.toLocaleString()}
              </p>
            </div>
            <div className="rounded-md border border-border/60 px-3 py-2">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Provider</p>
              <p className="font-medium truncate" title={run.ai_model}>
                {run.provider} · {run.ai_model}
              </p>
            </div>
            <div className="rounded-md border border-border/60 px-3 py-2">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Plan version</p>
              <p className="font-medium">{run.plan_version}</p>
            </div>
          </div>

          {latest.by_recommendation.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {latest.by_recommendation.map((r) => (
                <span
                  key={r.recommendation_code}
                  className="rounded-full border border-border bg-muted/40 px-2 py-0.5 text-xs"
                >
                  {r.recommendation_code}: {r.dfus.toLocaleString()}
                </span>
              ))}
            </div>
          )}

          <div className="mt-4 flex items-center justify-between">
            <h4 className="text-xs font-medium text-foreground">Champion vs AI (sample)</h4>
            <label className="flex items-center gap-2 text-xs text-muted-foreground">
              <input
                type="checkbox"
                checked={adjustedOnly}
                onChange={(e) => setAdjustedOnly(e.target.checked)}
                className="rounded border-border"
              />
              Adjusted only
            </label>
          </div>

          {forecastLoading ? (
            <p className="mt-2 text-sm text-muted-foreground">Loading forecast rows…</p>
          ) : (forecast?.rows.length ?? 0) === 0 ? (
            <p className="mt-2 text-sm text-muted-foreground">No rows match the current filter.</p>
          ) : (
            <div className="mt-2 overflow-x-auto">
              <table className="w-full min-w-[640px] text-left text-xs">
                <thead>
                  <tr className="border-b border-border text-muted-foreground">
                    <th className="py-2 pr-2 font-medium">DFU</th>
                    <th className="py-2 pr-2 font-medium">Month</th>
                    <th className="py-2 pr-2 font-medium">Champion</th>
                    <th className="py-2 pr-2 font-medium">AI</th>
                    <th className="py-2 pr-2 font-medium">Δ%</th>
                    <th className="py-2 font-medium">Code</th>
                  </tr>
                </thead>
                <tbody>
                  {forecast?.rows.map((row) => (
                    <tr key={`${row.item_id}-${row.loc}-${row.forecast_month}`} className="border-b border-border/40">
                      <td className="py-1.5 pr-2 font-mono">
                        {row.item_id}-{row.loc}
                      </td>
                      <td className="py-1.5 pr-2">{row.forecast_month?.slice(0, 7) ?? "—"}</td>
                      <td className="py-1.5 pr-2 font-mono">{formatQty(row.champion_qty)}</td>
                      <td className="py-1.5 pr-2 font-mono">{formatQty(row.ai_qty)}</td>
                      <td className="py-1.5 pr-2 font-mono">{formatPct(row.pct_change)}</td>
                      <td className="py-1.5">{row.recommendation_code}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {(forecast?.total ?? 0) > (forecast?.rows.length ?? 0) && (
                <p className="mt-2 text-xs text-muted-foreground">
                  Showing {forecast?.rows.length} of {forecast?.total.toLocaleString()} rows.
                </p>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
