/** SQL Runner API queries */

import { fetchJson } from "./core";

export interface SqlResult {
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
  elapsed_ms: number;
}

export interface TableColumn {
  name: string;
  data_type: string;
  is_nullable: boolean;
}

export interface TableInfo {
  schema_name: string;
  table_name: string;
  table_type: string;
  columns: TableColumn[];
}

export interface QueryHistoryEntry {
  sql: string;
  executed_at: string;
  elapsed_ms: number;
  row_count: number;
  status: string;
}

export const sqlRunnerKeys = {
  schema: () => ["sql-runner-schema"] as const,
  history: () => ["sql-runner-history"] as const,
};

export async function fetchExecuteQuery(
  sql: string,
  maxRows?: number,
): Promise<SqlResult> {
  return fetchJson<SqlResult>("/sql-runner/execute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sql, max_rows: maxRows }),
  });
}

export async function fetchSchema(): Promise<{ tables: TableInfo[] }> {
  return fetchJson<{ tables: TableInfo[] }>("/sql-runner/schema");
}

export async function fetchQueryHistory(): Promise<{
  history: QueryHistoryEntry[];
}> {
  return fetchJson<{ history: QueryHistoryEntry[] }>("/sql-runner/history");
}
