/**
 * SweepResultsPanel — leaderboard + per-segment winner map for a champion sweep.
 *
 * Two views: a global-ranked leaderboard (with gate-eligibility badges) and a
 * per-segment winner map with a "composite vs. best global" headline. Promote
 * the recommended winner via the existing Stage-1 promotion. See spec 30.
 */
import { useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Crown, Layers } from "lucide-react";

import {
  championSweepKeys,
  fetchChampionSweep,
  fetchSweepLeaderboard,
  fetchSweepSegments,
  promoteSweepWinner,
  type ChampionSweep,
} from "@/api/queries";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";

const fmtPct = (v: number | null | undefined) => (v == null ? "--" : `${Number(v).toFixed(2)}%`);
const fmtScore = (v: number | null | undefined) => (v == null ? "--" : Number(v).toFixed(2));

export function SweepResultsPanel({ sweepId }: { sweepId: number }) {
  const qc = useQueryClient();
  const running = (s?: ChampionSweep) => s?.status === "queued" || s?.status === "running";

  const { data: sweep } = useQuery({
    queryKey: championSweepKeys.detail(sweepId),
    queryFn: () => fetchChampionSweep(sweepId),
    refetchInterval: (q) => (running(q.state.data as ChampionSweep) ? 3000 : false),
  });
  const { data: lb } = useQuery({
    queryKey: championSweepKeys.leaderboard(sweepId),
    queryFn: () => fetchSweepLeaderboard(sweepId),
    enabled: sweep?.status === "completed",
  });
  const { data: segData } = useQuery({
    queryKey: championSweepKeys.segments(sweepId),
    queryFn: () => fetchSweepSegments(sweepId),
    enabled: sweep?.status === "completed" && sweep?.mode !== "global",
  });

  const promote = useMutation({
    mutationFn: () => promoteSweepWinner(sweepId),
    onSuccess: () => qc.invalidateQueries({ queryKey: championSweepKeys.detail(sweepId) }),
  });

  const members = lb?.members ?? [];
  const recommendedId = sweep?.recommended_experiment_id ?? null;
  const compositeMember = members.find((m) => m.is_composite) ?? null;
  const bestGlobal = useMemo(
    () => members.filter((m) => !m.is_composite).sort((a, b) => (a.global_rank ?? 999) - (b.global_rank ?? 999))[0] ?? null,
    [members],
  );

  if (!sweep) return <div className="py-8 text-center text-sm text-muted-foreground">Loading sweep…</div>;

  return (
    <div className="space-y-4">
      {/* Progress / recommendation header */}
      <Card>
        <CardContent className="flex items-center justify-between py-3">
          <div className="text-sm">
            <span className="font-medium">{sweep.label}</span>{" "}
            <Badge variant="outline" className="ml-1 text-xs">{sweep.status}</Badge>
            {running(sweep) ? (
              <span className="ml-2 text-xs text-muted-foreground">
                {sweep.completed_count}/{sweep.candidate_count ?? "?"} candidates
              </span>
            ) : null}
          </div>
          {sweep.status === "completed" && recommendedId != null ? (
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">
                Recommended #{recommendedId} · score {fmtScore(sweep.recommended_score)}
                {sweep.recommended_gate_eligible ? "" : " · gate ✗"}
              </span>
              <Button
                size="sm"
                disabled={!sweep.recommended_gate_eligible || promote.isPending}
                onClick={() => promote.mutate()}
                title={sweep.recommended_gate_eligible ? "Promote the recommended config" : "Recommendation did not pass the promote gate"}
              >
                <Crown className="mr-1 h-3.5 w-3.5" />
                {promote.isPending ? "Promoting…" : "Promote winner"}
              </Button>
            </div>
          ) : null}
        </CardContent>
      </Card>
      {promote.isError ? (
        <div className="text-xs text-red-500">{(promote.error as Error).message}</div>
      ) : null}

      {/* Composite vs best global headline */}
      {sweep.status === "completed" && compositeMember && bestGlobal ? (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Layers className="h-4 w-4" /> Per-segment composite vs. best global
            </CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-3 text-sm">
            <div className="rounded-md border p-3">
              <div className="text-xs text-muted-foreground">Best global config</div>
              <div className="font-medium">{bestGlobal.strategy}</div>
              <div>{fmtPct(bestGlobal.champion_accuracy)} acc · score {fmtScore(bestGlobal.global_score)}</div>
            </div>
            <div className="rounded-md border p-3">
              <div className="text-xs text-muted-foreground">Per-segment composite</div>
              <div className="font-medium">composite (per_segment)</div>
              <div>{fmtPct(compositeMember.champion_accuracy)} acc · score {fmtScore(compositeMember.global_score)}</div>
            </div>
          </CardContent>
        </Card>
      ) : null}

      {/* Global leaderboard */}
      {sweep.status === "completed" ? (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Global leaderboard</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-8 text-xs">#</TableHead>
                  <TableHead className="text-xs">Strategy</TableHead>
                  <TableHead className="text-xs">Accuracy</TableHead>
                  <TableHead className="text-xs">Ceiling</TableHead>
                  <TableHead className="text-xs">Gap (bps)</TableHead>
                  <TableHead className="text-xs">Score</TableHead>
                  <TableHead className="text-xs">Gate</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {members.map((m) => (
                  <TableRow
                    key={m.experiment_id}
                    className={m.experiment_id === recommendedId ? "bg-amber-50 dark:bg-amber-950/30" : undefined}
                  >
                    <TableCell className="text-xs">{m.global_rank ?? "--"}</TableCell>
                    <TableCell className="text-xs">
                      {m.strategy}
                      {m.is_composite ? <Badge variant="outline" className="ml-1 text-[10px]">composite</Badge> : null}
                      {m.skipped_duplicate ? <Badge variant="outline" className="ml-1 text-[10px]">reused</Badge> : null}
                    </TableCell>
                    <TableCell className="text-xs">{fmtPct(m.champion_accuracy)}</TableCell>
                    <TableCell className="text-xs">{fmtPct(m.ceiling_accuracy)}</TableCell>
                    <TableCell className="text-xs">{m.gap_bps == null ? "--" : Number(m.gap_bps).toFixed(0)}</TableCell>
                    <TableCell className="text-xs">{fmtScore(m.global_score)}</TableCell>
                    <TableCell className="text-xs">
                      {m.gate_eligible == null ? "--" : m.gate_eligible ? (
                        <Badge className="bg-emerald-100 text-emerald-700 text-[10px] dark:bg-emerald-900/40 dark:text-emerald-300">✓</Badge>
                      ) : (
                        <Badge variant="outline" className="text-[10px]">✗</Badge>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ) : null}

      {/* Per-segment winner map */}
      {segData && segData.segments.length > 0 ? (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Per-segment winners ({sweep.segment_axis})</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Segment</TableHead>
                  <TableHead className="text-xs">Winning strategy</TableHead>
                  <TableHead className="text-xs">Accuracy</TableHead>
                  <TableHead className="text-xs">DFUs</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {segData.segments.map((s) => (
                  <TableRow key={s.segment}>
                    <TableCell className="text-xs font-medium">{s.segment}</TableCell>
                    <TableCell className="text-xs">{s.winner?.strategy ?? "--"}</TableCell>
                    <TableCell className="text-xs">{fmtPct(s.winner?.accuracy)}</TableCell>
                    <TableCell className="text-xs">{s.winner?.n_dfus ?? "--"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
