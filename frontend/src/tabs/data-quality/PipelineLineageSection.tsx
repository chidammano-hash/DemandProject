import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { GitBranch } from "lucide-react";

import { EmptyState } from "@/components/EmptyState";
import {
  fetchBatches,
  lineageKeys,
  STALE_LINEAGE,
  type LoadBatch,
} from "@/api/queries";

const STATUS_OPTIONS = ["", "completed", "failed", "running", "skipped"] as const;

function statusDot(status: string): string {
  if (status === "completed") return "bg-emerald-500";
  if (status === "failed") return "bg-red-500";
  return "bg-amber-500";
}

/**
 * US20: unified load-batch lineage. Lists audit_load_batch rows (every domain,
 * including customer_demand now that it records batches), filterable by domain
 * and status, with sanitized error messages on failed batches.
 */
export function PipelineLineageSection(): JSX.Element {
  const [domain, setDomain] = useState("");
  const [status, setStatus] = useState("");

  const { data } = useQuery({
    queryKey: [...lineageKeys.batches, domain, status],
    queryFn: () => fetchBatches(domain || undefined, status || undefined, 50),
    staleTime: STALE_LINEAGE,
  });
  const batches: LoadBatch[] = data?.batches ?? [];

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-medium text-foreground">Pipeline Lineage</h3>
        <div className="flex items-center gap-2">
          <input
            aria-label="Filter lineage by domain"
            placeholder="domain…"
            value={domain}
            onChange={(e) => setDomain(e.target.value.trim())}
            className="w-32 rounded border px-2 py-1 text-xs"
          />
          <select
            aria-label="Filter lineage by status"
            value={status}
            onChange={(e) => setStatus(e.target.value)}
            className="rounded border px-2 py-1 text-xs"
          >
            {STATUS_OPTIONS.map((s) => (
              <option key={s || "all"} value={s}>
                {s || "all statuses"}
              </option>
            ))}
          </select>
        </div>
      </div>

      {batches.length === 0 ? (
        <EmptyState
          variant="no-data"
          icon={GitBranch}
          title="No pipeline batches yet"
          description="Batches appear after the ETL pipeline runs. Trigger an ingest to start the lineage log."
          steps={[
            { label: "Normalize source CSVs", command: "make normalize-all" },
            { label: "Load into Postgres", command: "make load-all" },
          ]}
        />
      ) : (
        <div className="space-y-2">
          {batches.map((b) => (
            <div
              key={b.batch_id}
              className="rounded-md border border-border/40 px-3 py-2 text-sm"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className={`inline-block h-2 w-2 rounded-full ${statusDot(b.status)}`} />
                  <span className="font-mono text-xs">{b.domain}</span>
                  <span className="text-xs text-muted-foreground">Batch #{b.batch_id}</span>
                </div>
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <span>{b.row_count_in?.toLocaleString() ?? "—"} in</span>
                  <span>{b.row_count_out?.toLocaleString() ?? "—"} out</span>
                  <span>{b.started_at ? new Date(b.started_at).toLocaleString() : "—"}</span>
                </div>
              </div>
              {b.status === "failed" && b.error_message ? (
                <p className="mt-1 text-xs text-red-500" role="alert">
                  {b.error_message}
                </p>
              ) : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
