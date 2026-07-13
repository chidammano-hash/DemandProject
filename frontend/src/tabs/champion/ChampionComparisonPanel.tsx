/**
 * ChampionComparisonPanel — Side-by-side comparison of two champion experiments.
 *
 * Shows overall metrics delta, per-lag table, per-month table,
 * model distribution comparison, and config diffs.
 */
import { useQuery } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, Minus } from "lucide-react";

import {
  championExperimentKeys,
  CHAMPION_EXP_STALE,
  compareChampionExperiments,
  type ChampionExperiment,
} from "@/api/queries";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface Props {
  baseline: ChampionExperiment;
  candidate: ChampionExperiment;
  execLag?: number;
}

export function ChampionComparisonPanel({ baseline, candidate, execLag }: Props) {
  const { data, isLoading, isError } = useQuery({
    queryKey: championExperimentKeys.compare(
      baseline.experiment_id,
      candidate.experiment_id,
      execLag,
    ),
    queryFn: () =>
      compareChampionExperiments(
        baseline.experiment_id,
        candidate.experiment_id,
        execLag,
      ),
    staleTime: CHAMPION_EXP_STALE.COMPARE,
    enabled: baseline.status === "completed" && candidate.status === "completed",
  });

  if (baseline.status !== "completed" || candidate.status !== "completed") {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          Both experiments must be completed to compare.
        </CardContent>
      </Card>
    );
  }

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          Loading comparison...
        </CardContent>
      </Card>
    );
  }

  if (isError || !data) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-red-500">
          Failed to load comparison.
        </CardContent>
      </Card>
    );
  }

  const overall = data.overall_comparison;
  const verdict = overall.verdict;

  return (
    <div className="space-y-4">
      {/* Verdict */}
      <Card>
        <CardHeader className="py-3">
          <CardTitle className="text-sm flex items-center gap-2">
            Overall Comparison
            <VerdictBadge verdict={verdict} />
          </CardTitle>
        </CardHeader>
        <CardContent className="pb-3">
          <div className="grid grid-cols-3 gap-4 text-xs">
            <MetricBlock
              label="Champion Accuracy"
              aVal={overall.experiment_a.champion_accuracy}
              bVal={overall.experiment_b.champion_accuracy}
              delta={overall.delta_champion_accuracy}
              higherIsBetter
            />
            <MetricBlock
              label="Ceiling Accuracy"
              aVal={overall.experiment_a.ceiling_accuracy}
              bVal={overall.experiment_b.ceiling_accuracy}
              delta={overall.delta_ceiling_accuracy}
              higherIsBetter
            />
            <MetricBlock
              label="Gap (bps)"
              aVal={overall.experiment_a.gap_bps}
              bVal={overall.experiment_b.gap_bps}
              delta={overall.delta_gap_bps}
              higherIsBetter={false}
            />
          </div>
        </CardContent>
      </Card>

      {/* Per-lag */}
      {data.per_lag_comparison.length > 0 && (
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="text-sm">Per-Lag Comparison</CardTitle>
          </CardHeader>
          <CardContent className="pb-3">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Lag</TableHead>
                  <TableHead className="text-xs text-right">A Acc%</TableHead>
                  <TableHead className="text-xs text-right">B Acc%</TableHead>
                  <TableHead className="text-xs text-right">Delta</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.per_lag_comparison.map((r) => (
                  <TableRow key={r.exec_lag}>
                    <TableCell className="text-xs">{r.exec_lag}</TableCell>
                    <TableCell className="text-xs text-right">
                      {r.a_champion_accuracy?.toFixed(2) ?? "--"}
                    </TableCell>
                    <TableCell className="text-xs text-right">
                      {r.b_champion_accuracy?.toFixed(2) ?? "--"}
                    </TableCell>
                    <TableCell className="text-xs text-right">
                      <DeltaText value={r.delta_accuracy} higherIsBetter />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Model distribution */}
      {data.model_dist_comparison.length > 0 && (
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="text-sm">Model Distribution</CardTitle>
          </CardHeader>
          <CardContent className="pb-3">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Model</TableHead>
                  <TableHead className="text-xs text-right">A %</TableHead>
                  <TableHead className="text-xs text-right">B %</TableHead>
                  <TableHead className="text-xs text-right">Delta</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.model_dist_comparison.map((r) => (
                  <TableRow key={r.model_id}>
                    <TableCell className="text-xs font-mono">
                      {r.model_id.replace("_cluster", "")}
                    </TableCell>
                    <TableCell className="text-xs text-right">
                      {r.a_pct.toFixed(1)}%
                    </TableCell>
                    <TableCell className="text-xs text-right">
                      {r.b_pct.toFixed(1)}%
                    </TableCell>
                    <TableCell className="text-xs text-right">
                      {r.delta_pct > 0 ? "+" : ""}
                      {r.delta_pct.toFixed(1)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Config diffs */}
      {data.config_diffs.length > 0 && (
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="text-sm">Config Differences</CardTitle>
          </CardHeader>
          <CardContent className="pb-3">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Param</TableHead>
                  <TableHead className="text-xs">A</TableHead>
                  <TableHead className="text-xs">B</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.config_diffs.map((d) => (
                  <TableRow key={d.key}>
                    <TableCell className="text-xs font-mono">{d.key}</TableCell>
                    <TableCell className="text-xs">
                      {JSON.stringify(d.a)}
                    </TableCell>
                    <TableCell className="text-xs">
                      {JSON.stringify(d.b)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type VerdictValue = "a_better" | "b_better" | "mixed";

const VERDICT_STYLES: Record<VerdictValue, string> = {
  a_better: "bg-blue-100 text-blue-800",
  b_better: "bg-emerald-100 text-emerald-800",
  mixed: "bg-gray-100 text-gray-800",
};

const VERDICT_LABELS: Record<VerdictValue, string> = {
  a_better: "A Better",
  b_better: "B Better",
  mixed: "Mixed",
};

function VerdictBadge({ verdict }: { verdict: string }) {
  const key = (verdict in VERDICT_STYLES ? verdict : "mixed") as VerdictValue;
  return (
    <Badge className={cn("text-[10px]", VERDICT_STYLES[key])}>
      {VERDICT_LABELS[key]}
    </Badge>
  );
}

function MetricBlock({
  label,
  aVal,
  bVal,
  delta,
  higherIsBetter,
}: {
  label: string;
  aVal: number | null;
  bVal: number | null;
  delta: number | null;
  higherIsBetter: boolean;
}) {
  return (
    <div className="text-center">
      <div className="text-muted-foreground mb-1">{label}</div>
      <div className="flex justify-center gap-4">
        <div>
          <div className="text-muted-foreground text-[10px]">A</div>
          <div className="font-medium">{aVal?.toFixed(2) ?? "--"}</div>
        </div>
        <div>
          <div className="text-muted-foreground text-[10px]">B</div>
          <div className="font-medium">{bVal?.toFixed(2) ?? "--"}</div>
        </div>
      </div>
      <DeltaText value={delta} higherIsBetter={higherIsBetter} className="mt-1" />
    </div>
  );
}

function DeltaText({
  value,
  higherIsBetter,
  className,
}: {
  value: number | null;
  higherIsBetter: boolean;
  className?: string;
}) {
  if (value === null || value === undefined) return <span className={className}>--</span>;
  const isPositive = value > 0;
  const isGood = higherIsBetter ? isPositive : !isPositive;
  const color = Math.abs(value) < 0.01 ? "text-muted-foreground" : isGood ? "text-emerald-600" : "text-red-600";
  const Icon = isPositive ? ArrowUp : value < 0 ? ArrowDown : Minus;

  return (
    <span className={cn("inline-flex items-center gap-0.5", color, className)}>
      <Icon className="h-3 w-3" />
      {value > 0 ? "+" : ""}
      {value.toFixed(2)}
    </span>
  );
}
