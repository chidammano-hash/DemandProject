/**
 * Error Decomposition Panel — the diagnostic view that localizes the forecast
 * accuracy gap. It pairs the headline VOLUME-WEIGHTED accuracy with the
 * UNWEIGHTED per-DFU mean/median (every SKU equal), and a Pareto of the DFUs
 * that own the most absolute error. Data: /forecast/accuracy/decomposition and
 * /forecast/accuracy/error-contributors (agg_accuracy_by_dfu, sql/193).
 */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { STALE } from "@/api/queries";
import {
  fetchAccuracyDecomposition,
  fetchErrorContributors,
  accuracyDecompositionKeys,
  errorContributorsKeys,
} from "@/api/queries/accuracy";
import type { DecompositionModelEntry } from "@/types";
import { formatPercent, formatNumber, titleCase } from "@/lib/formatters";

const GROUP_OPTIONS = [
  { value: "seasonality_profile", label: "Seasonality" },
  { value: "abc_vol", label: "ABC (volume)" },
  { value: "cluster_assignment", label: "Cluster" },
  { value: "ml_cluster", label: "ML cluster" },
  { value: "region", label: "Region" },
  { value: "dfu_execution_lag", label: "Execution lag" },
] as const;

const CONTRIBUTOR_LIMIT = 15;

interface ErrorDecompositionPanelProps {
  models: string;
  lag: number;
  monthFrom: string;
  clusterAssignment?: string;
  seasonalityProfile?: string;
  enabled: boolean;
}

function biasBadgeVariant(direction: string): "warning" | "info" | "outline" {
  if (direction === "over") return "warning"; // over-forecast → excess inventory risk
  if (direction === "under") return "info"; // under-forecast → stockout risk
  return "outline";
}

// Map a signed bias ratio to the same over/under/neutral direction language the
// Pareto table uses, so MASE (which is direction-blind) is never shown alone.
// A small dead-band keeps near-zero bias from reading as a directional signal.
const BIAS_DEADBAND = 0.02; // ±2% — below this, treat as balanced
function biasDirectionLabel(bias: number | null): string {
  if (bias === null || !Number.isFinite(bias)) return "—";
  if (bias > BIAS_DEADBAND) return "over";
  if (bias < -BIAS_DEADBAND) return "under";
  return "neutral";
}

// MASE band. Naive-relative: <1 beats the naive baseline, ≈1 on par, >1 worse.
// The on-par dead-band (0.95–1.05) keeps measurement noise from flipping a
// model between "beats naive" and "worse" on every refresh.
const MASE_ON_PAR_LOW = 0.95;
const MASE_ON_PAR_HIGH = 1.05;
function maseBandVariant(
  median: number | null,
): { variant: "success" | "outline" | "warning"; label: string } | null {
  if (median === null || !Number.isFinite(median)) return null;
  if (median < MASE_ON_PAR_LOW) return { variant: "success", label: "beats naive" };
  if (median > MASE_ON_PAR_HIGH) return { variant: "warning", label: "worse than naive — review" };
  return { variant: "outline", label: "on par" };
}

/**
 * MASE cell for the decomposition table. Leads with the per-DFU MEDIAN (the
 * mean is heavy-tailed under intermittency), tags the naive-relative band, and
 * pairs the volume-weighted bias direction so a direction-blind MASE is never
 * shown alone. The mean rides along as a tooltip for the analyst who wants it.
 */
function MaseCell({ entry }: { entry: DecompositionModelEntry }) {
  const band = maseBandVariant(entry.mase.median_mase);
  const biasDir = biasDirectionLabel(entry.volume_weighted.bias);

  if (band === null) {
    return (
      <span className="text-xs text-muted-foreground" title="No usable naive baseline for this segment.">
        — no baseline
      </span>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <span
        className="tabular-nums text-sm font-medium"
        title={`Mean MASE ${formatNumber(entry.mase.mean_mase)} (heavy-tailed; median shown)`}
      >
        {formatNumber(entry.mase.median_mase)}
      </span>
      <Badge variant={band.variant}>{band.label}</Badge>
      {biasDir !== "neutral" && biasDir !== "—" ? (
        <Badge variant={biasBadgeVariant(biasDir)} title="Forecast bias direction (MASE is direction-blind)">
          {biasDir}
        </Badge>
      ) : null}
    </div>
  );
}

export function ErrorDecompositionPanel({
  models,
  lag,
  monthFrom,
  clusterAssignment,
  seasonalityProfile,
  enabled,
}: ErrorDecompositionPanelProps) {
  const [groupBy, setGroupBy] = useState<string>("seasonality_profile");

  const decompParams = {
    group_by: groupBy,
    lag,
    models,
    month_from: monthFrom,
    cluster_assignment: clusterAssignment,
    seasonality_profile: seasonalityProfile,
  };
  const contribParams = {
    lag,
    limit: CONTRIBUTOR_LIMIT,
    models,
    month_from: monthFrom,
    cluster_assignment: clusterAssignment,
    seasonality_profile: seasonalityProfile,
  };

  const { data: decomp, isLoading: loadingDecomp } = useQuery({
    queryKey: accuracyDecompositionKeys.list(decompParams),
    queryFn: () => fetchAccuracyDecomposition(decompParams),
    staleTime: STALE.TWO_MIN,
    enabled,
  });
  const { data: contrib, isLoading: loadingContrib } = useQuery({
    queryKey: errorContributorsKeys.list(contribParams),
    queryFn: () => fetchErrorContributors(contribParams),
    staleTime: STALE.TWO_MIN,
    enabled,
  });

  // Flatten bucket × model into table rows.
  const decompRows: Array<{ bucket: string; model: string; entry: DecompositionModelEntry }> = [];
  for (const row of decomp?.rows ?? []) {
    for (const [model, entry] of Object.entries(row.by_model)) {
      decompRows.push({ bucket: row.bucket, model, entry });
    }
  }

  return (
    <div className="space-y-6">
      <p className="text-sm text-muted-foreground">
        <strong>Volume-weighted</strong> is the headline number (big SKUs dominate).{" "}
        <strong>Per-DFU mean/median</strong> weights every SKU equally and exposes the long tail.{" "}
        <strong>Error share</strong> is each segment&apos;s slice of total absolute error — the Pareto of where to fix first.
      </p>

      <p className="text-xs text-muted-foreground">
        <strong>MASE &lt;1 beats a naive baseline; &gt;1 worse</strong> — naive-relative, fair to the
        small-base long tail; shown alongside WAPE, not replacing it. The MASE median is
        per-DFU-equal / unweighted (like the per-DFU mean/median), not the volume-weighted WAPE headline.
        {decomp?.mase_seasonal_period_rule ? (
          <>
            {" "}
            Naive scale: {decomp.mase_seasonal_period_rule}.
          </>
        ) : null}
      </p>

      <div className="flex items-center gap-2">
        <span className="text-sm font-medium">Group by</span>
        <Select value={groupBy} onValueChange={setGroupBy}>
          <SelectTrigger className="w-44" aria-label="Group decomposition by">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {GROUP_OPTIONS.map((o) => (
              <SelectItem key={o.value} value={o.value}>
                {o.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Decomposition table */}
      {loadingDecomp ? (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading decomposition…
        </div>
      ) : decompRows.length === 0 ? (
        <p className="text-sm text-muted-foreground">No accuracy data for the current filters.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Segment</TableHead>
              <TableHead>Model</TableHead>
              <TableHead className="text-right">DFUs</TableHead>
              <TableHead className="text-right">Vol-weighted</TableHead>
              <TableHead className="text-right">Per-DFU mean</TableHead>
              <TableHead className="text-right">Per-DFU median</TableHead>
              <TableHead className="text-right">Undefined</TableHead>
              <TableHead>MASE (median)</TableHead>
              <TableHead>No naive baseline</TableHead>
              <TableHead>Error share</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {decompRows.map(({ bucket, model, entry }) => (
              <TableRow key={`${bucket}__${model}`}>
                <TableCell className="font-medium">{titleCase(bucket)}</TableCell>
                <TableCell>{model}</TableCell>
                <TableCell className="text-right">{formatNumber(entry.n_dfus)}</TableCell>
                <TableCell className="text-right">{formatPercent(entry.volume_weighted.accuracy_pct)}</TableCell>
                <TableCell className="text-right">{formatPercent(entry.unweighted.mean_accuracy_pct)}</TableCell>
                <TableCell className="text-right">{formatPercent(entry.unweighted.median_accuracy_pct)}</TableCell>
                <TableCell className="text-right">{formatNumber(entry.unweighted.n_undefined)}</TableCell>
                {/* MASE (median) — naive-relative band + paired bias direction so the
                    direction-blind MASE is never read in isolation. */}
                <TableCell>
                  <MaseCell entry={entry} />
                </TableCell>
                {/* Undefined MASE → a NAMED no-baseline state, never a bare 0. */}
                <TableCell>
                  <span
                    className="text-xs text-muted-foreground"
                    title="No in-sample naive baseline (cold-start / flat history) — excluded from the MASE median, not counted as a miss."
                  >
                    {formatNumber(entry.mase.n_undefined)} no baseline
                  </span>
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <div className="h-2 w-24 overflow-hidden rounded bg-muted">
                      <div
                        className="h-full rounded bg-primary"
                        style={{ width: `${Math.min(100, entry.error_contribution_pct ?? 0)}%` }}
                      />
                    </div>
                    <span className="tabular-nums text-xs">
                      {formatPercent(entry.error_contribution_pct)}
                    </span>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}

      {/* Top error contributors (Pareto) */}
      <div>
        <h4 className="mb-2 text-sm font-semibold">
          Top error contributors{contrib ? ` (of ${formatNumber(contrib.total_dfus)} DFUs)` : ""}
        </h4>
        {loadingContrib ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading contributors…
          </div>
        ) : (contrib?.contributors.length ?? 0) === 0 ? (
          <p className="text-sm text-muted-foreground">No contributors for the current filters.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Item</TableHead>
                <TableHead>Loc</TableHead>
                <TableHead className="text-right">Actual</TableHead>
                <TableHead className="text-right">Error share</TableHead>
                <TableHead className="text-right">Cumulative</TableHead>
                <TableHead className="text-right">Accuracy</TableHead>
                <TableHead>Bias</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {contrib?.contributors.map((c) => (
                <TableRow key={`${c.item_id}__${c.customer_group}__${c.loc}`}>
                  <TableCell className="font-medium">{c.item_id}</TableCell>
                  <TableCell>{c.loc}</TableCell>
                  <TableCell className="text-right">{formatNumber(c.sum_actual)}</TableCell>
                  <TableCell className="text-right">{formatPercent(c.error_contribution_pct)}</TableCell>
                  <TableCell className="text-right text-muted-foreground">
                    {formatPercent(c.cumulative_contribution_pct)}
                  </TableCell>
                  <TableCell className="text-right">{formatPercent(c.accuracy_pct)}</TableCell>
                  <TableCell>
                    <Badge variant={biasBadgeVariant(c.bias_direction)}>{c.bias_direction}</Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>
    </div>
  );
}
