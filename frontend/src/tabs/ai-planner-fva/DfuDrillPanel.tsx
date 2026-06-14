/**
 * Per-DFU drill panel + detail dialog for the AI FVA backtest.
 *
 * - Lists top-N DFUs by absolute error reduction (mv_ai_fva_by_dfu).
 * - A "Look up DFU" form lets a planner inspect any item_id + loc directly.
 * - Clicking a row or submitting the form opens DfuDetailDialog with the
 *   walk-forward baseline-vs-AI-vs-actual table and each month's recommendation
 *   rationale (GET /runs/:runId/dfu-detail).
 *
 * Split out of AiPlannerFvaTab.tsx to keep that file under the 600-LoC ceiling
 * (CLAUDE.md "Tab files MUST be < 600 lines").
 */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  aiFvaBacktestKeys,
  getFvaBacktestDfuDetail,
  getFvaBacktestDfus,
} from "@/api/queries";
import { fmtNum, fmtPct, fmtPp } from "@/tabs/AiPlannerFvaTab";

const STALE_30S = 30_000;

type SelectedDfu = { itemId: string; loc: string };

// ---------------------------------------------------------------------------
// DFU Drill Panel — top-N table + look-up form
// ---------------------------------------------------------------------------

export function DfuDrillPanel({ runId }: { runId: string }) {
  const [selected, setSelected] = useState<SelectedDfu | null>(null);
  const [lookupItem, setLookupItem] = useState("");
  const [lookupLoc, setLookupLoc] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: aiFvaBacktestKeys.dfus(runId, "error_reduction", 25),
    queryFn: () => getFvaBacktestDfus(runId, { limit: 25, sort: "error_reduction" }),
    staleTime: STALE_30S,
  });

  const submitLookup = () => {
    const item = lookupItem.trim();
    const loc = lookupLoc.trim();
    if (!item || !loc) return;
    setSelected({ itemId: item, loc });
  };

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Top DFUs by Absolute Error Reduction</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* Look-up form — pick any item_id / loc without searching the list. */}
          <div className="flex flex-wrap items-end gap-2 text-sm">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Item ID</span>
              <Input
                value={lookupItem}
                onChange={(e) => setLookupItem(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") submitLookup(); }}
                placeholder="e.g. 916045"
                className="w-40"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Location</span>
              <Input
                value={lookupLoc}
                onChange={(e) => setLookupLoc(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") submitLookup(); }}
                placeholder="e.g. 1401-BULK"
                className="w-40"
              />
            </label>
            <Button
              onClick={submitLookup}
              disabled={!lookupItem.trim() || !lookupLoc.trim()}
              size="sm"
            >
              Inspect DFU
            </Button>
          </div>

          {isLoading && <p className="text-sm text-muted-foreground">Loading…</p>}
          {data && data.rows.length === 0 && (
            <p className="text-sm text-muted-foreground">No DFU data yet.</p>
          )}
          {data && data.rows.length > 0 && (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Item</TableHead>
                  <TableHead>Loc</TableHead>
                  <TableHead>Σ |Baseline-A|</TableHead>
                  <TableHead>Σ |AI-A|</TableHead>
                  <TableHead>Reduction</TableHead>
                  <TableHead>n</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.rows.map((r) => (
                  <TableRow
                    key={`${r.item_id}-${r.loc}`}
                    className="cursor-pointer hover:bg-muted/50"
                    onClick={() => setSelected({ itemId: r.item_id, loc: r.loc })}
                  >
                    <TableCell className="font-mono text-xs">{r.item_id}</TableCell>
                    <TableCell className="font-mono text-xs">{r.loc}</TableCell>
                    <TableCell className="text-xs">{fmtNum(Math.round(r.sae_baseline))}</TableCell>
                    <TableCell className="text-xs">{fmtNum(Math.round(r.sae_ai))}</TableCell>
                    <TableCell
                      className={`text-xs font-semibold ${
                        r.abs_error_reduction > 0
                          ? "text-emerald-600 dark:text-emerald-400"
                          : r.abs_error_reduction < 0
                            ? "text-rose-600 dark:text-rose-400"
                            : ""
                      }`}
                    >
                      {fmtNum(Math.round(r.abs_error_reduction))}
                    </TableCell>
                    <TableCell className="text-xs">{r.n_obs}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {selected && (
        <DfuDetailDialog
          runId={runId}
          itemId={selected.itemId}
          loc={selected.loc}
          onClose={() => setSelected(null)}
        />
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// DFU Detail Dialog — walk-forward baseline vs AI vs actual + rationale
// ---------------------------------------------------------------------------

function DfuDetailDialog({
  runId, itemId, loc, onClose,
}: {
  runId: string;
  itemId: string;
  loc: string;
  onClose: () => void;
}) {
  const { data, isLoading, error } = useQuery({
    queryKey: aiFvaBacktestKeys.dfuDetail(runId, itemId, loc),
    queryFn: () => getFvaBacktestDfuDetail(runId, itemId, loc),
    staleTime: STALE_30S,
    retry: false,
  });

  return (
    <Dialog open onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="font-mono text-base">
            {itemId} @ {loc}
          </DialogTitle>
          <DialogDescription>
            Walk-forward detail: baseline vs AI vs actual per (run T, lag), plus
            the recommendation the AI emitted at each forecast-run month.
          </DialogDescription>
        </DialogHeader>

        {isLoading && <p className="text-sm text-muted-foreground">Loading detail…</p>}
        {error && (
          <p className="text-sm text-destructive">
            {(error as Error)?.message ?? "Failed to load DFU detail."}
          </p>
        )}

        {data && (
          <div className="space-y-4">
            {/* Summary KPIs */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
              <div className="border rounded p-2">
                <div className="text-xs text-muted-foreground">Observations</div>
                <div className="font-semibold">{data.summary.n_obs}</div>
              </div>
              <div className="border rounded p-2">
                <div className="text-xs text-muted-foreground">Baseline WAPE</div>
                <div className="font-semibold">{fmtPct(data.summary.baseline_wape_pct)}</div>
              </div>
              <div className="border rounded p-2">
                <div className="text-xs text-muted-foreground">AI WAPE</div>
                <div className="font-semibold">{fmtPct(data.summary.ai_wape_pct)}</div>
              </div>
              <div className="border rounded p-2">
                <div className="text-xs text-muted-foreground">Lift (AI − Baseline)</div>
                <div
                  className={`font-semibold ${
                    (data.summary.lift_pp ?? 0) > 0
                      ? "text-emerald-600 dark:text-emerald-400"
                      : (data.summary.lift_pp ?? 0) < 0
                        ? "text-rose-600 dark:text-rose-400"
                        : ""
                  }`}
                >
                  {fmtPp(data.summary.lift_pp)}
                </div>
              </div>
            </div>

            {/* Recommendations per run T */}
            <div>
              <h4 className="text-sm font-semibold mb-1">Recommendations by Forecast Run Month</h4>
              {data.recommendations.length === 0 ? (
                <p className="text-sm text-muted-foreground">No recommendations for this DFU in this run.</p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Run T</TableHead>
                      <TableHead>Code</TableHead>
                      <TableHead>%Δ</TableHead>
                      <TableHead>Conf.</TableHead>
                      <TableHead>Rationale</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.recommendations.map((r) => (
                      <TableRow key={r.forecast_run_month}>
                        <TableCell className="text-xs">{r.forecast_run_month}</TableCell>
                        <TableCell className="font-mono text-xs">{r.recommendation_code}</TableCell>
                        <TableCell className="text-xs">
                          {r.pct_change === null ? "—" : `${r.pct_change.toFixed(1)}%`}
                        </TableCell>
                        <TableCell className="text-xs">
                          {r.confidence === null ? "—" : r.confidence.toFixed(2)}
                        </TableCell>
                        <TableCell className="text-xs">{r.rationale ?? "—"}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </div>

            {/* Per-lag detail */}
            <div>
              <h4 className="text-sm font-semibold mb-1">Per-Month, Per-Lag Forecast vs Actual</h4>
              {data.lags.length === 0 ? (
                <p className="text-sm text-muted-foreground">No forecast rows for this DFU in this run.</p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Run T</TableHead>
                      <TableHead>Target</TableHead>
                      <TableHead>Lag</TableHead>
                      <TableHead className="text-right">Baseline</TableHead>
                      <TableHead className="text-right">AI</TableHead>
                      <TableHead className="text-right">Actual</TableHead>
                      <TableHead className="text-right">Base err</TableHead>
                      <TableHead className="text-right">AI err</TableHead>
                      <TableHead>Winner</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.lags.map((l) => {
                      const baseErr = l.baseline_qty !== null && l.actual_qty !== null
                        ? Math.abs(l.baseline_qty - l.actual_qty) : null;
                      const aiErr = l.ai_qty !== null && l.actual_qty !== null
                        ? Math.abs(l.ai_qty - l.actual_qty) : null;
                      const winner = baseErr !== null && aiErr !== null
                        ? (aiErr < baseErr ? "AI" : aiErr > baseErr ? "Baseline" : "tie")
                        : "—";
                      const winnerClass =
                        winner === "AI" ? "text-emerald-600 dark:text-emerald-400"
                        : winner === "Baseline" ? "text-rose-600 dark:text-rose-400" : "";
                      return (
                        <TableRow key={`${l.forecast_run_month}-${l.lag}`}>
                          <TableCell className="text-xs">{l.forecast_run_month}</TableCell>
                          <TableCell className="text-xs">{l.target_month}</TableCell>
                          <TableCell className="text-xs">{l.lag}</TableCell>
                          <TableCell className="text-right text-xs">
                            {l.baseline_qty === null ? "—" : l.baseline_qty.toFixed(1)}
                          </TableCell>
                          <TableCell className="text-right text-xs">
                            {l.ai_qty === null ? "—" : l.ai_qty.toFixed(1)}
                          </TableCell>
                          <TableCell className="text-right text-xs">
                            {l.actual_qty === null ? "—" : l.actual_qty.toFixed(1)}
                          </TableCell>
                          <TableCell className="text-right text-xs">
                            {baseErr === null ? "—" : baseErr.toFixed(1)}
                          </TableCell>
                          <TableCell className="text-right text-xs">
                            {aiErr === null ? "—" : aiErr.toFixed(1)}
                          </TableCell>
                          <TableCell className={`text-xs font-semibold ${winnerClass}`}>
                            {winner}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
