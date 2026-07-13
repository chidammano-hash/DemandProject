/**
 * F2.2 — Multi-Horizon Demand Plan Panel
 *
 * Displays P10/P50/P90 quantile forecasts with weekly disaggregation,
 * version selector, sigma uncertainty chips, and SS recommendation.
 */

import { useState, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchDemandPlan,
  fetchDemandPlanVersions,
  fetchDemandPlanWeekly,
  STALE,
} from "@/api/queries";
import { EmptyState } from "@/components/EmptyState";
import { ClipboardList } from "lucide-react";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";

function formatQty(v: number | null | undefined): string {
  if (v == null) return "—";
  return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function formatMonth(s: string | null): string {
  if (!s) return "—";
  const d = new Date(s + "T00:00:00");
  return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
}

const HORIZON_OPTIONS = [
  { label: "3M", value: 3 },
  { label: "6M", value: 6 },
  { label: "12M", value: 12 },
  { label: "18M", value: 18 },
];

export function DemandPlanPanel() {
  const { filters: globalFilters } = useGlobalFilterContext();
  const [itemNo, setItemNo] = useState("");
  const [loc, setLoc] = useState("");
  const [horizon, setHorizon] = useState(12);
  const [selectedVersion, setSelectedVersion] = useState<string>("");
  const [submitted, setSubmitted] = useState(false);

  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setItemNo(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setLoc(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  const versionsQ = useQuery({
    queryKey: ["demand-plan-versions"],
    queryFn: fetchDemandPlanVersions,
    staleTime: STALE.FIVE_MIN,
  });

  const planQ = useQuery({
    queryKey: ["demand-plan", { itemNo, loc, horizon, version: selectedVersion }],
    queryFn: () =>
      fetchDemandPlan({
        item_id: itemNo,
        loc,
        horizon,
        plan_version: selectedVersion || undefined,
      }),
    enabled: submitted && !!itemNo && !!loc,
    staleTime: STALE.TWO_MIN,
  });

  const weeklyQ = useQuery({
    queryKey: ["demand-plan-weekly", { itemNo, loc, version: selectedVersion }],
    queryFn: () =>
      fetchDemandPlanWeekly({
        item_id: itemNo,
        loc,
        plan_version: selectedVersion || undefined,
        weeks_ahead: 8,
      }),
    enabled: submitted && !!itemNo && !!loc,
    staleTime: STALE.TWO_MIN,
  });

  const versions = versionsQ.data?.versions ?? [];
  const plan = planQ.data;
  const weekly = weeklyQ.data;

  // Sigma chips from latest available month
  const latestRow = plan?.rows?.find((r) => r.sigma_combined != null);
  const sigmaF = latestRow?.sigma_forecast;
  const sigmaD = latestRow?.sigma_demand;
  const sigmaC = latestRow?.sigma_combined;

  // SS recommendation at 95% SL (Z=1.645), horizon=1 month LT placeholder
  const ssRecommended = sigmaC != null ? Math.round(1.645 * sigmaC * Math.sqrt(2)) : null;

  const handleSearch = () => {
    if (itemNo && loc) setSubmitted(true);
  };

  return (
    <div className="space-y-4">
      {/* Info banner */}
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
        <strong className="text-foreground">Demand Plan</strong> shows probabilistic multi-horizon forecasts (P10/P50/P90 quantiles) for a single item-location pair. P50 is the median forecast; P10 and P90 bound the likely demand range. Weekly disaggregation provides near-term visibility for replenishment scheduling.
      </div>

      {/* Header */}
      <div>
        <h3 className="font-semibold text-base">Demand Plan</h3>
        <p className="text-xs text-muted-foreground mt-0.5">
          Multi-horizon P10/P50/P90 quantile demand forecasts with weekly disaggregation.
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-2 items-end">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">Item No</label>
          <input
            className="rounded-md border bg-background px-2 py-1 text-xs w-32"
            placeholder="e.g. 100320"
            value={itemNo}
            onChange={(e) => setItemNo(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">Location</label>
          <input
            className="rounded-md border bg-background px-2 py-1 text-xs w-32"
            placeholder="e.g. 1401-BULK"
            value={loc}
            onChange={(e) => setLoc(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">Version</label>
          <select
            className="rounded-md border bg-background px-2 py-1 text-xs"
            value={selectedVersion}
            onChange={(e) => setSelectedVersion(e.target.value)}
          >
            <option value="">Latest active</option>
            {versions.map((v) => (
              <option key={v.plan_version} value={v.plan_version}>
                {v.plan_version} {v.status === "active" ? "✓" : ""}
              </option>
            ))}
          </select>
        </div>
        <div className="flex gap-1 items-end">
          {HORIZON_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setHorizon(opt.value)}
              className={`rounded px-2 py-1 text-xs font-medium border ${
                horizon === opt.value
                  ? "bg-primary text-primary-foreground border-primary"
                  : "bg-background hover:bg-muted"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <button
          onClick={handleSearch}
          disabled={!itemNo || !loc}
          className="rounded-md bg-primary px-3 py-1.5 text-xs text-primary-foreground font-medium hover:bg-primary/90 disabled:opacity-50"
        >
          View Plan
        </button>
      </div>

      {/* Sigma / SS recommendation chips */}
      {plan && latestRow && (
        <div className="rounded-lg border bg-muted/30 p-3 flex flex-wrap gap-4 text-xs">
          <div>
            <span className="text-muted-foreground">Forecast σ</span>
            <span className="ml-2 font-semibold">{sigmaF?.toFixed(1) ?? "—"}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Historical σ</span>
            <span className="ml-2 font-semibold">{sigmaD?.toFixed(1) ?? "—"}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Combined σ</span>
            <span className="ml-2 font-semibold">{sigmaC?.toFixed(1) ?? "—"}</span>
          </div>
          {ssRecommended != null && (
            <div className="rounded-full bg-amber-100 dark:bg-amber-900/30 px-3 py-0.5 text-amber-700 dark:text-amber-300 font-semibold">
              Recommended SS (95% SL): {ssRecommended} units
            </div>
          )}
        </div>
      )}

      {/* Monthly quantile table */}
      {!submitted ? (
        <EmptyState
          icon={ClipboardList}
          title="Enter an item and location to view the demand plan"
          description="The demand plan shows monthly forecast quantiles (P10/P50/P90) and weekly disaggregation for a single DFU. It combines the statistical baseline with any approved demand overrides."
          steps={[
            { label: "Enter Item No and Location in the fields above", command: "e.g. Item: 100320 | Loc: 1401-BULK" },
          ]}
        />
      ) : planQ.isLoading ? (
        <div className="text-xs text-muted-foreground py-4 text-center">Loading demand plan...</div>
      ) : planQ.isError ? (
        <EmptyState
          variant="error"
          icon={ClipboardList}
          title="Demand plan could not be loaded"
          description="Retry after checking the API connection. This panel reads existing demand-plan versions and does not generate forecasts."
          onAction={() => planQ.refetch()}
          actionLabel="Retry"
        />
      ) : !plan?.rows?.length ? (
        <EmptyState
          icon={ClipboardList}
          title="No saved demand plan for this item and location"
          description="This read-only panel displays previously loaded demand-plan versions. Use the Forecast view for the current production forecast and uncertainty bands."
        />
      ) : (
        <div className="rounded-lg border overflow-auto">
          <table className="w-full text-xs">
            <thead className="bg-muted/50">
              <tr>
                {["Month", "Horizon", "P10", "P50 (Median)", "P90", "σ_f", "σ_combined"].map(
                  (h) => (
                    <th
                      key={h}
                      className="px-3 py-2 text-left font-medium text-muted-foreground whitespace-nowrap"
                    >
                      {h}
                    </th>
                  )
                )}
              </tr>
            </thead>
            <tbody className="divide-y">
              {plan?.rows?.map(
                (r) => (
                  <tr key={r.plan_month} className="hover:bg-muted/30">
                    <td className="px-3 py-2 font-medium">{formatMonth(r.plan_month)}</td>
                    <td className="px-3 py-2 text-muted-foreground">{r.horizon_months}m</td>
                    <td className="px-3 py-2 text-blue-600 dark:text-blue-400">
                      {formatQty(r.p10)}
                    </td>
                    <td className="px-3 py-2 font-semibold">{formatQty(r.p50)}</td>
                    <td className="px-3 py-2 text-blue-600 dark:text-blue-400">
                      {formatQty(r.p90)}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">
                      {r.sigma_forecast?.toFixed(1) ?? "—"}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">
                      {r.sigma_combined?.toFixed(1) ?? "—"}
                    </td>
                  </tr>
                )
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Weekly disaggregation */}
      {submitted && (weekly?.weeks?.length ?? 0) > 0 && weekly && (
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
            Weekly View (next 8 weeks)
          </h4>
          <div className="grid grid-cols-4 md:grid-cols-8 gap-2">
            {weekly.weeks.map(
              (w) => (
                <div
                  key={w.plan_week}
                  className="rounded-lg border bg-card p-2 text-center"
                >
                  <p className="text-xs text-muted-foreground">W{w.iso_week}</p>
                  <p className="text-sm font-semibold mt-1">{formatQty(w.p50_weekly)}</p>
                  <p className="text-xs text-blue-500">
                    {formatQty(w.p10_weekly)}–{formatQty(w.p90_weekly)}
                  </p>
                </div>
              )
            )}
          </div>
        </div>
      )}
    </div>
  );
}
