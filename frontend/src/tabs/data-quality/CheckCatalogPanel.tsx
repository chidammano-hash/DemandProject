/**
 * Check Catalog Table — all DQ checks, filterable by status/severity, sortable.
 */
import { useMemo, useState } from "react";
import {
  MinusCircle,
  ArrowUpDown,
  Filter,
} from "lucide-react";

import type { DQCheck } from "@/api/queries/platform";
import {
  STATUS_ICON,
  SEVERITY_STYLE,
  relativeTime,
  type SortField,
  type SortDir,
} from "./dqShared";

interface CheckCatalogPanelProps {
  checkList: DQCheck[];
  domainFilter: string | null;
}

export function CheckCatalogPanel({ checkList, domainFilter }: CheckCatalogPanelProps) {
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [severityFilter, setSeverityFilter] = useState<string | null>(null);
  const [sortField, setSortField] = useState<SortField>("last_status");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("asc");
    }
  };

  const filteredChecks = useMemo(() => {
    let list = checkList;
    if (domainFilter) list = list.filter((c) => c.domain === domainFilter);
    if (statusFilter) list = list.filter((c) => (c.last_status ?? "none") === statusFilter);
    if (severityFilter) list = list.filter((c) => c.severity === severityFilter);

    const sorted = [...list].sort((a, b) => {
      const av = a[sortField] ?? "";
      const bv = b[sortField] ?? "";
      const cmp = String(av).localeCompare(String(bv));
      return sortDir === "asc" ? cmp : -cmp;
    });
    return sorted;
  }, [checkList, domainFilter, statusFilter, severityFilter, sortField, sortDir]);

  const uniqueStatuses = useMemo(() => {
    const set = new Set(checkList.map((c) => c.last_status ?? "none"));
    return Array.from(set).sort();
  }, [checkList]);

  const uniqueSeverities = useMemo(() => {
    const set = new Set(checkList.map((c) => c.severity));
    return Array.from(set).sort();
  }, [checkList]);

  const hasFilters = statusFilter || severityFilter;

  const clearFilters = () => {
    setStatusFilter(null);
    setSeverityFilter(null);
  };

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-medium text-foreground">
          Check Catalog ({filteredChecks.length}{filteredChecks.length !== checkList.length ? ` of ${checkList.length}` : ""})
        </h3>
        <div className="flex items-center gap-2">
          <Filter className="h-3.5 w-3.5 text-muted-foreground" />
          <select
            value={statusFilter ?? ""}
            onChange={(e) => setStatusFilter(e.target.value || null)}
            className="rounded border border-border bg-background px-2 py-1 text-xs"
            aria-label="Filter by status"
          >
            <option value="">All Statuses</option>
            {uniqueStatuses.map((s) => (
              <option key={s} value={s}>{s === "none" ? "Not Run" : s}</option>
            ))}
          </select>
          <select
            value={severityFilter ?? ""}
            onChange={(e) => setSeverityFilter(e.target.value || null)}
            className="rounded border border-border bg-background px-2 py-1 text-xs"
            aria-label="Filter by severity"
          >
            <option value="">All Severities</option>
            {uniqueSeverities.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          {hasFilters && (
            <button
              onClick={clearFilters}
              className="rounded px-2 py-1 text-xs text-primary hover:underline"
            >
              Clear Filters
            </button>
          )}
        </div>
      </div>

      <div className="max-h-96 overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-card">
            <tr className="border-b border-border text-left text-muted-foreground">
              <th className="pb-2 pr-3 w-10">Status</th>
              <th className="pb-2 pr-3 cursor-pointer select-none" onClick={() => toggleSort("severity")}>
                <span className="inline-flex items-center gap-1">
                  Severity <ArrowUpDown className="h-3 w-3" />
                </span>
              </th>
              <th className="pb-2 pr-3 cursor-pointer select-none" onClick={() => toggleSort("check_name")}>
                <span className="inline-flex items-center gap-1">
                  Check Name <ArrowUpDown className="h-3 w-3" />
                </span>
              </th>
              <th className="pb-2 pr-3">Type</th>
              <th className="pb-2 pr-3 cursor-pointer select-none" onClick={() => toggleSort("domain")}>
                <span className="inline-flex items-center gap-1">
                  Domain <ArrowUpDown className="h-3 w-3" />
                </span>
              </th>
              <th className="pb-2 pr-3">Table</th>
              <th className="pb-2 pr-3">Last Value</th>
              <th className="pb-2 cursor-pointer select-none" onClick={() => toggleSort("last_run")}>
                <span className="inline-flex items-center gap-1">
                  Last Run <ArrowUpDown className="h-3 w-3" />
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {filteredChecks.map((c: DQCheck) => {
              const st = STATUS_ICON[c.last_status ?? ""] ?? {
                icon: MinusCircle,
                color: "text-gray-400",
                label: c.last_status ?? "Not run",
              };
              const Icon = st.icon;
              return (
                <tr key={c.check_id} className="border-b border-border/30 hover:bg-muted/30">
                  <td className="py-1.5 pr-3">
                    <span className="inline-flex items-center gap-1" title={st.label}>
                      <Icon className={`h-3.5 w-3.5 ${st.color}`} />
                    </span>
                  </td>
                  <td className="py-1.5 pr-3">
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${SEVERITY_STYLE[c.severity] ?? SEVERITY_STYLE.low}`}>
                      {c.severity}
                    </span>
                  </td>
                  <td className="py-1.5 pr-3 font-medium">{c.check_name}</td>
                  <td className="py-1.5 pr-3 text-muted-foreground">{c.check_type}</td>
                  <td className="py-1.5 pr-3 capitalize text-muted-foreground">{c.domain}</td>
                  <td className="py-1.5 pr-3 font-mono text-muted-foreground">{c.table_name}</td>
                  <td className="py-1.5 pr-3 text-muted-foreground">
                    {c.last_value != null ? c.last_value.toFixed(2) : "\u2014"}
                  </td>
                  <td className="py-1.5 text-muted-foreground">{c.last_run ? relativeTime(c.last_run) : "\u2014"}</td>
                </tr>
              );
            })}
            {filteredChecks.length === 0 && (
              <tr>
                <td colSpan={8} className="py-6 text-center text-muted-foreground">
                  {checkList.length === 0 ? "No checks configured yet." : "No checks match the current filters."}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
