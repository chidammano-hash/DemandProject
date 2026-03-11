/**
 * Data Quality & Pipeline Observability Tab (Spec 08-01).
 *
 * Shows: Domain health scores, check catalog, freshness status.
 */
import { useQuery } from "@tanstack/react-query";
import { fetchDQDashboard, fetchDQChecks, fetchDQFreshness, dqKeys, STALE_PLATFORM } from "@/api/queries";
import type { DQDomainScore, DQCheck } from "@/api/queries/platform";

const SCORE_COLORS: Record<string, string> = {
  high: "text-green-600 bg-green-50",
  medium: "text-amber-600 bg-amber-50",
  low: "text-red-600 bg-red-50",
};

function scoreLevel(score: number): string {
  if (score >= 90) return "high";
  if (score >= 70) return "medium";
  return "low";
}

export default function DataQualityTab() {
  const { data: dashboard } = useQuery({
    queryKey: dqKeys.dashboard,
    queryFn: fetchDQDashboard,
    staleTime: STALE_PLATFORM,
  });

  const { data: checks } = useQuery({
    queryKey: dqKeys.checks,
    queryFn: fetchDQChecks,
    staleTime: STALE_PLATFORM,
  });

  const { data: freshness } = useQuery({
    queryKey: dqKeys.freshness,
    queryFn: fetchDQFreshness,
    staleTime: STALE_PLATFORM,
  });

  const domains = dashboard?.domains ?? [];
  const checkList = checks?.checks ?? [];
  const tables = freshness?.tables ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-foreground">Data Quality & Observability</h2>
        <p className="text-sm text-muted-foreground">Monitor pipeline health and data quality across all domains</p>
      </div>

      {/* Domain Health Scores */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {domains.map((d: DQDomainScore) => {
          const level = scoreLevel(d.score);
          return (
            <div key={d.domain} className="rounded-lg border border-border bg-card p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium capitalize">{d.domain}</span>
                <span className={`rounded-full px-2 py-0.5 text-xs font-bold ${SCORE_COLORS[level]}`}>
                  {d.score}%
                </span>
              </div>
              <div className="mt-2 flex gap-3 text-xs text-muted-foreground">
                <span className="text-green-600">{d.passed} pass</span>
                <span className="text-red-600">{d.failed} fail</span>
                <span className="text-amber-600">{d.warnings} warn</span>
              </div>
            </div>
          );
        })}
        {domains.length === 0 && (
          <div className="col-span-full py-8 text-center text-sm text-muted-foreground">
            No data quality checks have been run yet. Use POST /data-quality/run to trigger a check.
          </div>
        )}
      </div>

      {/* Freshness Status */}
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 text-sm font-medium text-foreground">Pipeline Freshness</h3>
        <div className="space-y-2">
          {tables.map((t: { table: string; last_load: string | null }) => (
            <div key={t.table} className="flex items-center justify-between rounded-md border border-border/40 px-3 py-2 text-sm">
              <span className="font-mono text-xs">{t.table}</span>
              <span className={`text-xs ${t.last_load ? "text-muted-foreground" : "text-red-500"}`}>
                {t.last_load ? new Date(t.last_load).toLocaleString() : "Never loaded"}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Check Catalog */}
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 text-sm font-medium text-foreground">Check Catalog ({checkList.length})</h3>
        <div className="max-h-80 overflow-y-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border text-left text-muted-foreground">
                <th className="pb-2 pr-4">Check</th>
                <th className="pb-2 pr-4">Domain</th>
                <th className="pb-2 pr-4">Type</th>
                <th className="pb-2 pr-4">Severity</th>
                <th className="pb-2 pr-4">Status</th>
                <th className="pb-2">Last Run</th>
              </tr>
            </thead>
            <tbody>
              {checkList.map((c: DQCheck) => (
                <tr key={c.check_id} className="border-b border-border/30">
                  <td className="py-1.5 pr-4 font-medium">{c.check_name}</td>
                  <td className="py-1.5 pr-4 capitalize">{c.domain}</td>
                  <td className="py-1.5 pr-4">{c.check_type}</td>
                  <td className="py-1.5 pr-4">
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${c.severity === "critical" ? "bg-red-100 text-red-700" : "bg-amber-100 text-amber-700"}`}>
                      {c.severity}
                    </span>
                  </td>
                  <td className="py-1.5 pr-4">
                    <span className={`${c.last_status === "pass" ? "text-green-600" : c.last_status === "fail" ? "text-red-600" : "text-muted-foreground"}`}>
                      {c.last_status ?? "—"}
                    </span>
                  </td>
                  <td className="py-1.5 text-muted-foreground">{c.last_run?.slice(0, 16) ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
