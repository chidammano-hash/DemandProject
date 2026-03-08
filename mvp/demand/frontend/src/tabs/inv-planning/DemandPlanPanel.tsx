/**
 * F2.2 — Multi-Horizon Demand Plan Panel
 *
 * Displays P10/P50/P90 quantile forecasts with weekly disaggregation,
 * version selector, sigma uncertainty chips, and SS recommendation.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchDemandPlan,
  fetchDemandPlanVersions,
  fetchDemandPlanWeekly,
  STALE,
} from "@/api/queries";

function formatQty(v: number | null | undefined): string {
  if (v == null) return "—";
  return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function formatMonth(s: string | null): string {
  if (!s) return "—";
  const d = new Date(s + "T00:00:00");
  return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
}

function formatWeek(s: string | null): string {
  if (!s) return "—";
  const d = new Date(s + "T00:00:00");
  return `W${d.toLocaleDateString("en-US", { month: "numeric", day: "numeric" })}`;
}

const HORIZON_OPTIONS = [
  { label: "3M", value: 3 },
  { label: "6M", value: 6 },
  { label: "12M", value: 12 },
  { label: "18M", value: 18 },
];

export function DemandPlanPanel() {
  const [itemNo, setItemNo] = useState("");
  const [loc, setLoc] = useState("");
  const [horizon, setHorizon] = useState(12);
  const [selectedVersion, setSelectedVersion] = useState<string>("");
  const [submitted, setSubmitted] = useState(false);

  const versionsQ = useQuery({
    queryKey: ["demand-plan-versions"],
    queryFn: fetchDemandPlanVersions,
    staleTime: STALE.FIVE_MIN,
  });

  const planQ = useQuery({
    queryKey: ["demand-plan", { itemNo, loc, horizon, version: selectedVersion }],
    queryFn: () =>
      fetchDemandPlan({
        item_no: itemNo,
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
        item_no: itemNo,
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
  const latestRow = plan?.rows?.find((r: { sigma_combined?: number }) => r.sigma_combined != null);
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
            placeholder="100320"
            value={itemNo}
            onChange={(e) => setItemNo(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">Location</label>
          <input
            className="rounded-md border bg-background px-2 py-1 text-xs w-32"
            placeholder="1401-BULK"
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
            {versions.map((v: { plan_version: string; plan_label?: string; status?: string }) => (
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
        <div className="rounded-lg border bg-card p-8 text-center text-sm text-muted-foreground">
          Enter an Item No and Location, then click <strong>View Plan</strong>.
        </div>
      ) : planQ.isLoading ? (
        <div className="text-xs text-muted-foreground py-4 text-center">Loading demand plan...</div>
      ) : planQ.isError ? (
        <div className="text-xs text-red-600 py-4 text-center">
          No demand plan found for this item/location. Run{" "}
          <code className="font-mono">make quantile-train</code> to generate.
        </div>
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
                (r: {
                  plan_month: string;
                  horizon_months: number;
                  p10?: number;
                  p50?: number;
                  p90?: number;
                  sigma_forecast?: number;
                  sigma_combined?: number;
                }) => (
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
      {submitted && weekly?.weeks && weekly.weeks.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
            Weekly View (next 8 weeks)
          </h4>
          <div className="grid grid-cols-4 md:grid-cols-8 gap-2">
            {weekly.weeks.map(
              (w: {
                plan_week: string;
                iso_week: number;
                p10_weekly?: number;
                p50_weekly?: number;
                p90_weekly?: number;
              }) => (
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
