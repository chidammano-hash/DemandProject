import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  insightKeys,
  fetchConstrainedOpt,
  STALE_INSIGHTS,
  type ConstrainedOptItem,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatCurrency, formatPct, formatFixed, formatInt } from "@/lib/formatters";
import { Calculator } from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function ConstrainedOptPanel() {
  const [budgetInput, setBudgetInput] = useState("100000");
  const [activeBudget, setActiveBudget] = useState<number | null>(null);

  const { data, isLoading, error, isFetching } = useQuery({
    queryKey: insightKeys.constrainedOpt(activeBudget ?? 0),
    queryFn: () => fetchConstrainedOpt(activeBudget!),
    staleTime: STALE_INSIGHTS.FIVE_MIN,
    enabled: activeBudget != null && activeBudget > 0,
  });

  function handleOptimize() {
    const val = parseFloat(budgetInput);
    if (Number.isFinite(val) && val > 0) {
      setActiveBudget(val);
    }
  }

  if (error) {
    return (
      <div className="text-xs text-red-600 p-4">
        Failed to load optimization results: {(error as Error).message}
      </div>
    );
  }

  const items = data?.items ?? [];

  return (
    <div className="space-y-4">
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
        Budget-constrained safety stock optimization. Enter a budget and the optimizer allocates
        safety stock investment across items to maximize aggregate service level improvement per dollar spent.
      </div>

      {/* Budget input */}
      <div className="flex items-end gap-3">
        <div>
          <label className="text-xs text-muted-foreground font-medium block mb-1">
            Budget ($)
          </label>
          <input
            type="number"
            className="h-8 rounded border border-input bg-background px-3 text-xs w-40 font-mono"
            placeholder="100000"
            value={budgetInput}
            onChange={(e) => setBudgetInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleOptimize()}
            min={0}
            step={1000}
          />
        </div>
        <button
          className="h-8 px-4 text-xs rounded bg-foreground text-background hover:bg-foreground/90 transition-colors disabled:opacity-50"
          onClick={handleOptimize}
          disabled={isFetching || !budgetInput || parseFloat(budgetInput) <= 0}
        >
          {isFetching ? "Optimizing..." : "Optimize"}
        </button>
      </div>

      {/* Results */}
      {activeBudget == null ? (
        <div className="text-xs text-muted-foreground py-8 text-center">
          Enter a budget amount and click Optimize to see recommended safety stock allocations.
        </div>
      ) : isLoading ? (
        <p className="text-xs text-muted-foreground">Running optimization...</p>
      ) : !data ? (
        <EmptyState
          icon={Calculator}
          title="No optimization results"
          description="The constrained optimizer requires safety stock targets and cost data. Ensure safety stock has been computed."
          steps={[
            { label: "Compute safety stock", command: "make ss-compute" },
          ]}
        />
      ) : (
        <>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <KpiCard
              className={PANEL_KPI}
              label="Budget"
              value={formatCurrency(data.budget)}
            />
            <KpiCard
              className={PANEL_KPI}
              label="Allocated"
              value={formatCurrency(data.allocated)}
              colorClass={data.allocated > data.budget ? "text-red-600" : undefined}
            />
            <KpiCard
              className={PANEL_KPI}
              label="Items Improved"
              value={formatInt(data.items_improved)}
            />
            <KpiCard
              className={PANEL_KPI}
              label="CSL Before"
              value={formatPct(data.csl_before)}
            />
            <KpiCard
              className={PANEL_KPI}
              label="CSL After"
              value={formatPct(data.csl_after)}
              colorClass={
                data.csl_after != null && data.csl_before != null && data.csl_after > data.csl_before
                  ? "text-green-600"
                  : undefined
              }
            />
          </div>

          {items.length === 0 ? (
            <p className="text-xs text-muted-foreground text-center py-4">
              No items qualify for optimization at this budget level.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <p className="text-xs font-medium mb-2">
                Allocation Details ({items.length} items, sorted by CSL uplift per dollar)
              </p>
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b text-muted-foreground">
                    <th className="text-left py-1 pr-2">Item</th>
                    <th className="text-left py-1 pr-2">Location</th>
                    <th className="text-right py-1 pr-2">Current SS</th>
                    <th className="text-right py-1 pr-2">Recommended SS</th>
                    <th className="text-right py-1 pr-2">Investment</th>
                    <th className="text-right py-1 pr-2">CSL Before</th>
                    <th className="text-right py-1 pr-2">CSL After</th>
                    <th className="text-right py-1">Uplift</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item: ConstrainedOptItem, idx: number) => {
                    const uplift =
                      item.csl_after != null && item.csl_before != null
                        ? item.csl_after - item.csl_before
                        : null;
                    return (
                      <tr
                        key={`${item.item_no}-${item.loc}-${idx}`}
                        className="border-b last:border-0 hover:bg-muted/40"
                      >
                        <td className="py-1 pr-2 font-mono">{item.item_no}</td>
                        <td className="py-1 pr-2">{item.loc}</td>
                        <td className="py-1 pr-2 text-right font-mono tabular-nums">
                          {formatFixed(item.current_ss)}
                        </td>
                        <td className="py-1 pr-2 text-right font-mono tabular-nums">
                          {formatFixed(item.recommended_ss)}
                        </td>
                        <td className="py-1 pr-2 text-right font-mono tabular-nums">
                          {formatCurrency(item.investment)}
                        </td>
                        <td className="py-1 pr-2 text-right font-mono tabular-nums">
                          {formatPct(item.csl_before)}
                        </td>
                        <td className="py-1 pr-2 text-right font-mono tabular-nums text-green-600">
                          {formatPct(item.csl_after)}
                        </td>
                        <td className="py-1 text-right font-mono tabular-nums text-green-600 font-medium">
                          {uplift != null ? `+${formatPct(uplift)}` : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
