/**
 * Self-Heal panel — scan for fixable DQ issues and apply/reject fixes.
 */
import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import {
  Zap,
  Check,
  X,
  Loader2,
} from "lucide-react";

import {
  fetchDQFixPreview,
  applyDQFixes,
  dqKeys,
} from "@/api/queries";
import type { DQFixItem } from "@/api/queries/platform";

export function SelfHealPanel() {
  const [healOpen, setHealOpen] = useState(false);
  const [selectedFixes, setSelectedFixes] = useState<Set<number>>(new Set());
  const [appliedFixes, setAppliedFixes] = useState<Set<number>>(new Set());
  const [rejectedFixes, setRejectedFixes] = useState<Set<number>>(new Set());
  const [applyResult, setApplyResult] = useState<{ total_applied: number; total_rows_fixed: number } | null>(null);

  const { data: fixPreview, isFetching: fixLoading, refetch: refetchPreview } = useQuery({
    queryKey: dqKeys.fixPreview,
    queryFn: fetchDQFixPreview,
    enabled: healOpen,
    staleTime: 0,
  });

  const fixItems = fixPreview?.items ?? [];
  const pendingFixItems = fixItems.filter(
    (f: DQFixItem) => !appliedFixes.has(f.id) && !rejectedFixes.has(f.id)
  );

  const applyFixesMutation = useMutation({
    mutationFn: (ids: number[]) => applyDQFixes(ids),
    onSuccess: (data) => {
      setApplyResult({ total_applied: data.total_applied, total_rows_fixed: data.total_rows_fixed });
      const newApplied = new Set(appliedFixes);
      for (const a of data.applied) newApplied.add(a.id);
      setAppliedFixes(newApplied);
      setSelectedFixes(new Set());
    },
  });

  const toggleFix = (id: number) => {
    setSelectedFixes((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const selectAllFixes = () => {
    setSelectedFixes(new Set(pendingFixItems.map((f: DQFixItem) => f.id)));
  };

  const deselectAllFixes = () => setSelectedFixes(new Set());

  const rejectSelected = () => {
    setRejectedFixes((prev) => {
      const next = new Set(prev);
      for (const id of selectedFixes) next.add(id);
      return next;
    });
    setSelectedFixes(new Set());
  };

  const resetHeal = () => {
    setSelectedFixes(new Set());
    setAppliedFixes(new Set());
    setRejectedFixes(new Set());
    setApplyResult(null);
    refetchPreview();
  };

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap className="h-4 w-4 text-teal-500" />
          <h3 className="text-sm font-medium text-foreground">Self-Heal</h3>
          <span className="text-xs text-muted-foreground">
            Statistical auto-fix for detected issues
          </span>
        </div>
        <button
          onClick={() => { setHealOpen(!healOpen); if (!healOpen) resetHeal(); }}
          className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-xs font-medium text-foreground hover:bg-muted/50"
        >
          <Zap className="h-3 w-3" />
          {healOpen ? "Close" : "Scan for Fixes"}
        </button>
      </div>

      {healOpen && (
        <div className="space-y-3">
          {/* Success banner */}
          {applyResult && (
            <div className="rounded-md bg-green-50 px-4 py-2 text-sm text-green-800 dark:bg-green-900/20 dark:text-green-300">
              Applied {applyResult.total_applied} fix(es) affecting {applyResult.total_rows_fixed.toLocaleString()} rows.
            </div>
          )}

          {/* Loading */}
          {fixLoading && (
            <div className="flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" /> Scanning for fixable issues...
            </div>
          )}

          {/* Empty state */}
          {!fixLoading && fixItems.length === 0 && (
            <div className="py-6 text-center text-sm text-muted-foreground">
              No fixable issues found. All data quality checks are within acceptable bounds.
            </div>
          )}

          {/* Fix items */}
          {!fixLoading && fixItems.length > 0 && (
            <>
              {/* Toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-md bg-muted/30 px-3 py-2">
                <div className="flex items-center gap-2">
                  <button
                    onClick={selectedFixes.size === pendingFixItems.length && pendingFixItems.length > 0 ? deselectAllFixes : selectAllFixes}
                    className="rounded border border-border px-2 py-1 text-xs font-medium hover:bg-muted/50"
                  >
                    {selectedFixes.size === pendingFixItems.length && pendingFixItems.length > 0 ? "Deselect All" : "Select All"}
                  </button>
                  <span className="text-xs text-muted-foreground">
                    {selectedFixes.size} of {pendingFixItems.length} selected
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => applyFixesMutation.mutate(Array.from(selectedFixes))}
                    disabled={selectedFixes.size === 0 || applyFixesMutation.isPending}
                    className="inline-flex items-center gap-1 rounded-md bg-teal-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-teal-700 disabled:opacity-50"
                  >
                    {applyFixesMutation.isPending
                      ? <><Loader2 className="h-3 w-3 animate-spin" /> Applying...</>
                      : <><Check className="h-3 w-3" /> Accept Selected</>
                    }
                  </button>
                  <button
                    onClick={rejectSelected}
                    disabled={selectedFixes.size === 0}
                    className="inline-flex items-center gap-1 rounded-md border border-red-300 px-3 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50 disabled:opacity-50 dark:border-red-800 dark:text-red-400 dark:hover:bg-red-950/30"
                  >
                    <X className="h-3 w-3" /> Reject Selected
                  </button>
                  <button
                    onClick={resetHeal}
                    className="rounded px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
                  >
                    Reset
                  </button>
                </div>
              </div>

              {/* Fix list */}
              <div className="max-h-[500px] space-y-1.5 overflow-y-auto">
                {fixItems.map((f: DQFixItem) => {
                  const isApplied = appliedFixes.has(f.id);
                  const isRejected = rejectedFixes.has(f.id);
                  const isSelected = selectedFixes.has(f.id);
                  const isPending = !isApplied && !isRejected;

                  return (
                    <div
                      key={f.id}
                      className={`flex items-start gap-3 rounded-md border px-3 py-2.5 transition-all ${
                        isApplied
                          ? "border-green-200 bg-green-50/50 dark:border-green-900/40 dark:bg-green-950/20"
                          : isRejected
                            ? "border-border/30 bg-muted/20 opacity-50"
                            : isSelected
                              ? "border-teal-300 bg-teal-50/30 dark:border-teal-800 dark:bg-teal-950/20"
                              : "border-border bg-card hover:border-border/60"
                      }`}
                    >
                      {/* Checkbox */}
                      {isPending && (
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleFix(f.id)}
                          className="mt-0.5 h-4 w-4 rounded border-border accent-teal-600"
                          aria-label={`Select fix: ${f.description}`}
                        />
                      )}
                      {isApplied && <Check className="mt-0.5 h-4 w-4 flex-shrink-0 text-green-600" />}
                      {isRejected && <X className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />}

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${
                            f.fix_type === "outliers" || f.fix_type === "range"
                              ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400"
                              : f.fix_type === "completeness"
                                ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400"
                                : f.fix_type === "lead_time"
                                  ? "bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-400"
                                  : "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400"
                          }`}>
                            {f.fix_type}
                          </span>
                          <span className="text-xs font-medium text-foreground truncate">
                            {f.description}
                          </span>
                        </div>
                        <div className="mt-1 flex items-center gap-3 text-[11px] text-muted-foreground">
                          <span>{f.affected_rows.toLocaleString()} rows affected</span>
                          {f.recommendation && (
                            <span className="text-amber-600 dark:text-amber-400">
                              {f.recommendation}
                            </span>
                          )}
                          {isApplied && <span className="font-medium text-green-600">Applied</span>}
                          {isRejected && <span className="font-medium text-red-400">Rejected</span>}
                        </div>
                      </div>

                      {/* Single accept / reject buttons */}
                      {isPending && !isSelected && (
                        <div className="flex items-center gap-1 flex-shrink-0">
                          <button
                            onClick={() => applyFixesMutation.mutate([f.id])}
                            disabled={applyFixesMutation.isPending}
                            className="rounded p-1 text-teal-600 hover:bg-teal-50 dark:hover:bg-teal-950/30"
                            title="Accept this fix"
                          >
                            <Check className="h-3.5 w-3.5" />
                          </button>
                          <button
                            onClick={() => setRejectedFixes((prev) => new Set(prev).add(f.id))}
                            className="rounded p-1 text-red-400 hover:bg-red-50 dark:hover:bg-red-950/30"
                            title="Reject this fix"
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
