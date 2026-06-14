/**
 * AI Planner FVA Backtest tab.
 *
 * Spec: docs/specs/PRD/PRD-ai-planner-fva-backtest.md (§7 UI Requirements)
 *
 * Shows: list of historical backtest runs, a "New Run" dialog, and per-run
 * detail panels (headline KPIs, FVA by recommendation, FVA by month,
 * top-DFU drill-down). Printable HTML report opens in a new tab.
 */
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { KpiCard } from "@/components/KpiCard";
import {
  type Provider,
  type StartRunRequest,
  aiFvaBacktestKeys,
  getFvaBacktestByMonth,
  getFvaBacktestByRecommendation,
  getFvaBacktestSummary,
  listFvaBacktestRuns,
  startFvaBacktestRun,
} from "@/api/queries";
import { DfuDrillPanel } from "@/tabs/ai-planner-fva/DfuDrillPanel";

const STALE_30S = 30_000;
const POLL_RUNNING_MS = 5_000;

export function fmtPct(n: number | null | undefined, places = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${n.toFixed(places)}%`;
}

export function fmtPp(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const sign = n > 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}pp`;
}

export function fmtNum(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString();
}

export function severityForLift(lift: number | null | undefined): "best" | "warning" | "neutral" {
  if (lift === null || lift === undefined) return "neutral";
  if (lift > 0.5) return "best";
  if (lift < -0.5) return "warning";
  return "neutral";
}

// ---------------------------------------------------------------------------
// New Run Dialog
// ---------------------------------------------------------------------------

function NewRunDialog() {
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState<StartRunRequest>({
    window_months: 10,
    horizon_months: 3,
    provider: "ollama",
    limit_dfus: 50,
    notes: "",
  });

  const startRun = useMutation({
    mutationFn: startFvaBacktestRun,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: aiFvaBacktestKeys.root });
      setOpen(false);
    },
  });

  const submit = () => {
    const body: StartRunRequest = {};
    if (form.window_months) body.window_months = form.window_months;
    if (form.horizon_months) body.horizon_months = form.horizon_months;
    if (form.provider) body.provider = form.provider;
    if (form.limit_dfus) body.limit_dfus = form.limit_dfus;
    if (form.notes) body.notes = form.notes;
    if (form.as_of_date) body.as_of_date = form.as_of_date;
    startRun.mutate(body);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>New Backtest Run</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Start AI FVA Backtest</DialogTitle>
          <DialogDescription>
            Configure a walk-forward backtest run. Defaults are safe for a quick
            Ollama smoke test — change provider and DFU limit for full runs.
          </DialogDescription>
        </DialogHeader>
        <div className="grid grid-cols-2 gap-3 py-2">
          <label className="text-sm flex flex-col gap-1">
            Window (months)
            <Input
              type="number"
              min={1}
              max={36}
              value={form.window_months ?? ""}
              onChange={(e) =>
                setForm((f) => ({ ...f, window_months: Number(e.target.value) || undefined }))
              }
            />
          </label>
          <label className="text-sm flex flex-col gap-1">
            Horizon (months)
            <Input
              type="number"
              min={1}
              max={12}
              value={form.horizon_months ?? ""}
              onChange={(e) =>
                setForm((f) => ({ ...f, horizon_months: Number(e.target.value) || undefined }))
              }
            />
          </label>
          <label className="text-sm flex flex-col gap-1">
            Provider
            <Select
              value={form.provider ?? "ollama"}
              onValueChange={(v) => setForm((f) => ({ ...f, provider: v as Provider }))}
            >
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="ollama">Ollama (local, free)</SelectItem>
                <SelectItem value="anthropic">Anthropic</SelectItem>
                <SelectItem value="openai">OpenAI</SelectItem>
                <SelectItem value="openai_compat">OpenAI-compatible</SelectItem>
              </SelectContent>
            </Select>
          </label>
          <label className="text-sm flex flex-col gap-1">
            DFU limit (smoke test)
            <Input
              type="number"
              min={1}
              max={50000}
              value={form.limit_dfus ?? ""}
              onChange={(e) =>
                setForm((f) => ({ ...f, limit_dfus: Number(e.target.value) || undefined }))
              }
              placeholder="50 = quick smoke"
            />
          </label>
          <label className="text-sm flex flex-col gap-1">
            As-of date (optional)
            <Input
              type="date"
              value={form.as_of_date ?? ""}
              onChange={(e) =>
                setForm((f) => ({ ...f, as_of_date: e.target.value || undefined }))
              }
            />
          </label>
          <label className="text-sm flex flex-col gap-1 col-span-2">
            Notes
            <Input
              value={form.notes ?? ""}
              onChange={(e) => setForm((f) => ({ ...f, notes: e.target.value }))}
              placeholder="Optional — appears in the runs list"
            />
          </label>
        </div>
        {startRun.isError && (
          <p className="text-sm text-destructive">
            {(startRun.error as Error)?.message ?? "Failed to start run."}
          </p>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
          <Button onClick={submit} disabled={startRun.isPending}>
            {startRun.isPending ? "Starting…" : "Start Backtest"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// Runs list (left column)
// ---------------------------------------------------------------------------

function RunsListPanel({
  selectedRunId,
  onSelect,
}: {
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
}) {
  const { data, isLoading, error } = useQuery({
    queryKey: aiFvaBacktestKeys.list(),
    queryFn: () => listFvaBacktestRuns(undefined, 50),
    refetchInterval: POLL_RUNNING_MS,
    staleTime: STALE_30S,
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Recent Runs</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <p className="text-sm text-muted-foreground">Loading…</p>}
        {error && (
          <p className="text-sm text-destructive">
            {(error as Error)?.message ?? "Failed to load runs"}
          </p>
        )}
        {data && data.runs.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No runs yet. Start one above.
          </p>
        )}
        {data && data.runs.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Status</TableHead>
                <TableHead>As-of</TableHead>
                <TableHead>Provider</TableHead>
                <TableHead>DFUs</TableHead>
                <TableHead>Recs</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.runs.map((r) => (
                <TableRow
                  key={r.run_id}
                  data-state={selectedRunId === r.run_id ? "selected" : undefined}
                  className="cursor-pointer hover:bg-muted/50"
                  onClick={() => onSelect(r.run_id)}
                >
                  <TableCell className="font-mono text-xs">
                    <span
                      className={
                        r.status === "succeeded"
                          ? "text-emerald-600 dark:text-emerald-400"
                          : r.status === "running"
                            ? "text-blue-600 dark:text-blue-400"
                            : r.status === "failed"
                              ? "text-rose-600 dark:text-rose-400"
                              : "text-muted-foreground"
                      }
                    >
                      {r.status}
                    </span>
                  </TableCell>
                  <TableCell className="text-xs">{r.as_of_date}</TableCell>
                  <TableCell className="text-xs">{r.provider}</TableCell>
                  <TableCell className="text-xs">{fmtNum(r.n_dfus_sampled)}</TableCell>
                  <TableCell className="text-xs">{fmtNum(r.n_recommendations)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Run detail (right column)
// ---------------------------------------------------------------------------

function SummaryKpis({ runId }: { runId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: aiFvaBacktestKeys.summary(runId),
    queryFn: () => getFvaBacktestSummary(runId),
    staleTime: STALE_30S,
  });

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading summary…</p>;
  }
  if (!data) return null;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <KpiCard
        label="Baseline WAPE"
        value={fmtPct(data.baseline_wape_pct)}
        size="md"
      />
      <KpiCard
        label="AI-Adjusted WAPE"
        value={fmtPct(data.ai_wape_pct)}
        size="md"
      />
      <KpiCard
        label="FVA Lift"
        value={fmtPp(data.lift_pct)}
        severity={severityForLift(data.lift_pct)}
        size="lg"
      />
      <KpiCard
        label="Win Rate"
        value={fmtPct(data.win_rate_pct, 1)}
        sublabel={`${fmtNum(data.n_winners)} W / ${fmtNum(data.n_losers)} L / ${fmtNum(data.n_ties)} T`}
        size="md"
      />
    </div>
  );
}

function ByRecommendationPanel({ runId }: { runId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: aiFvaBacktestKeys.byRecommendation(runId),
    queryFn: () => getFvaBacktestByRecommendation(runId),
    staleTime: STALE_30S,
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">FVA by Recommendation Type</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <p className="text-sm text-muted-foreground">Loading…</p>}
        {data && data.rows.length === 0 && (
          <p className="text-sm text-muted-foreground">No data yet.</p>
        )}
        {data && data.rows.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Code</TableHead>
                <TableHead>Baseline WAPE</TableHead>
                <TableHead>AI WAPE</TableHead>
                <TableHead>Lift</TableHead>
                <TableHead>Obs</TableHead>
                <TableHead>Avg Conf.</TableHead>
                <TableHead>Avg %Δ</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.rows.map((r) => (
                <TableRow key={r.recommendation_code}>
                  <TableCell className="font-mono text-xs">{r.recommendation_code}</TableCell>
                  <TableCell className="text-xs">{fmtPct(r.baseline_wape_pct)}</TableCell>
                  <TableCell className="text-xs">{fmtPct(r.ai_wape_pct)}</TableCell>
                  <TableCell
                    className={`text-xs font-semibold ${
                      (r.lift_pct ?? 0) > 0
                        ? "text-emerald-600 dark:text-emerald-400"
                        : (r.lift_pct ?? 0) < 0
                          ? "text-rose-600 dark:text-rose-400"
                          : ""
                    }`}
                  >
                    {fmtPp(r.lift_pct)}
                  </TableCell>
                  <TableCell className="text-xs">{fmtNum(r.n_obs)}</TableCell>
                  <TableCell className="text-xs">
                    {r.avg_confidence !== null ? r.avg_confidence.toFixed(3) : "—"}
                  </TableCell>
                  <TableCell className="text-xs">{fmtPct(r.avg_pct_change)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

function ByMonthPanel({ runId }: { runId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: aiFvaBacktestKeys.byMonth(runId),
    queryFn: () => getFvaBacktestByMonth(runId),
    staleTime: STALE_30S,
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">FVA by Month (Walk-Forward)</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <p className="text-sm text-muted-foreground">Loading…</p>}
        {data && data.rows.length === 0 && (
          <p className="text-sm text-muted-foreground">No data yet.</p>
        )}
        {data && data.rows.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Month T</TableHead>
                <TableHead>Baseline WAPE</TableHead>
                <TableHead>AI WAPE</TableHead>
                <TableHead>Lift</TableHead>
                <TableHead>DFUs</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.rows.map((r) => {
                // WAPE in this codebase is the ACCURACY form (100 - error%),
                // so higher = better. Lift = ai - baseline, NOT baseline - ai.
                // Matches mv_ai_fva_overall.lift_pct sign convention in sql/186.
                const lift =
                  r.baseline_wape_pct !== null && r.ai_wape_pct !== null
                    ? r.ai_wape_pct - r.baseline_wape_pct
                    : null;
                return (
                  <TableRow key={r.forecast_run_month}>
                    <TableCell className="text-xs">{r.forecast_run_month}</TableCell>
                    <TableCell className="text-xs">{fmtPct(r.baseline_wape_pct)}</TableCell>
                    <TableCell className="text-xs">{fmtPct(r.ai_wape_pct)}</TableCell>
                    <TableCell
                      className={`text-xs font-semibold ${
                        (lift ?? 0) > 0
                          ? "text-emerald-600 dark:text-emerald-400"
                          : (lift ?? 0) < 0
                            ? "text-rose-600 dark:text-rose-400"
                            : ""
                      }`}
                    >
                      {fmtPp(lift)}
                    </TableCell>
                    <TableCell className="text-xs">{fmtNum(r.n_dfus)}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

// DfuDrillPanel + DfuDetailDialog live in ./ai-planner-fva/DfuDrillPanel.tsx

// ---------------------------------------------------------------------------
// Main tab shell
// ---------------------------------------------------------------------------

export default function AiPlannerFvaTab() {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  const reportUrl = selectedRunId
    ? `/ai-planner/fva-backtest/runs/${selectedRunId}/report.html`
    : null;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">AI Planner — FVA Backtest</h2>
          <p className="text-sm text-muted-foreground">
            Walk-forward backtest measuring AI Planner forecast value-add vs. champion baseline.
            See <a className="underline" href="#" onClick={(e) => e.preventDefault()}>
              PRD 02-27
            </a>.
          </p>
        </div>
        <div className="flex gap-2">
          {reportUrl && (
            <a href={reportUrl} target="_blank" rel="noopener noreferrer">
              <Button variant="outline">Printable Report</Button>
            </a>
          )}
          <NewRunDialog />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-1">
          <RunsListPanel selectedRunId={selectedRunId} onSelect={setSelectedRunId} />
        </div>

        <div className="lg:col-span-2 space-y-4">
          {!selectedRunId && (
            <Card>
              <CardContent className="py-8 text-center text-muted-foreground">
                Select a run from the list, or start a new backtest.
              </CardContent>
            </Card>
          )}
          {selectedRunId && (
            <>
              <SummaryKpis runId={selectedRunId} />
              <ByMonthPanel runId={selectedRunId} />
              <ByRecommendationPanel runId={selectedRunId} />
              <DfuDrillPanel runId={selectedRunId} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
