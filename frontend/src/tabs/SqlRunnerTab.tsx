import { useCallback, useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Play,
  Trash2,
  Download,
  ChevronRight,
  ChevronDown,
  Clock,
  Database,
  AlertCircle,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/Skeleton";
import { cn } from "@/lib/utils";
import {
  fetchExecuteQuery,
  fetchSchema,
  fetchQueryHistory,
  sqlRunnerKeys,
} from "@/api/queries/sql-runner";
import type { SqlResult, TableInfo, QueryHistoryEntry } from "@/api/queries/sql-runner";

// ---------------------------------------------------------------------------
// Local storage history
// ---------------------------------------------------------------------------
const LS_KEY = "sql-runner-history";
const MAX_LOCAL_HISTORY = 50;

function loadLocalHistory(): QueryHistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveLocalHistory(entries: QueryHistoryEntry[]) {
  localStorage.setItem(LS_KEY, JSON.stringify(entries.slice(0, MAX_LOCAL_HISTORY)));
}

// ---------------------------------------------------------------------------
// Schema Browser
// ---------------------------------------------------------------------------
function SchemaBrowser({
  tables,
  isLoading,
  onInsertTable,
}: {
  tables: TableInfo[];
  isLoading: boolean;
  onInsertTable: (name: string) => void;
}) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const toggle = (key: string) =>
    setExpanded((prev) => ({ ...prev, [key]: !prev[key] }));

  if (isLoading) {
    return (
      <div className="space-y-2 p-2">
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} className="h-5 w-full" />
        ))}
      </div>
    );
  }

  return (
    <div className="overflow-y-auto text-xs">
      {tables.map((t) => {
        const key = `${t.schema_name}.${t.table_name}`;
        const isOpen = expanded[key];
        return (
          <div key={key}>
            <button
              className="flex w-full items-center gap-1 px-2 py-1 hover:bg-muted/50 text-left"
              onClick={() => toggle(key)}
              title={`${t.table_type} — click to expand`}
            >
              {isOpen ? <ChevronDown className="h-3 w-3 shrink-0" /> : <ChevronRight className="h-3 w-3 shrink-0" />}
              <Database className="h-3 w-3 shrink-0 text-muted-foreground" />
              <span
                className="truncate cursor-pointer hover:underline"
                onClick={(e) => {
                  e.stopPropagation();
                  onInsertTable(t.table_name);
                }}
              >
                {t.table_name}
              </span>
              <Badge variant="outline" className="ml-auto text-[9px] px-1 py-0">
                {t.table_type === "BASE TABLE" ? "T" : "V"}
              </Badge>
            </button>
            {isOpen && (
              <div className="pl-6 pb-1">
                {t.columns.map((c) => (
                  <div
                    key={c.name}
                    className="flex items-center gap-1 px-1 py-0.5 text-muted-foreground"
                  >
                    <span className="truncate">{c.name}</span>
                    <span className="ml-auto text-[9px] opacity-60">
                      {c.data_type}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Query History
// ---------------------------------------------------------------------------
function QueryHistory({
  entries,
  onRerun,
}: {
  entries: QueryHistoryEntry[];
  onRerun: (sql: string) => void;
}) {
  if (entries.length === 0) {
    return (
      <p className="text-xs text-muted-foreground p-2">No queries yet.</p>
    );
  }

  return (
    <div className="overflow-y-auto max-h-48 text-xs">
      {entries.map((e, i) => (
        <button
          key={i}
          className="flex w-full items-start gap-2 px-2 py-1.5 hover:bg-muted/50 text-left border-b border-border/30"
          onClick={() => onRerun(e.sql)}
        >
          {e.status === "ok" ? (
            <CheckCircle2 className="h-3 w-3 mt-0.5 shrink-0 text-emerald-500" />
          ) : (
            <XCircle className="h-3 w-3 mt-0.5 shrink-0 text-destructive" />
          )}
          <div className="flex-1 min-w-0">
            <p className="font-mono truncate">{e.sql}</p>
            <p className="text-muted-foreground">
              {e.row_count} rows &middot; {e.elapsed_ms}ms &middot;{" "}
              {e.executed_at}
            </p>
          </div>
        </button>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Results Table
// ---------------------------------------------------------------------------
function ResultsTable({ result }: { result: SqlResult }) {
  return (
    <div className="overflow-auto border rounded-md">
      <Table>
        <TableHeader>
          <TableRow>
            {result.columns.map((col) => (
              <TableHead key={col} className="whitespace-nowrap text-xs font-semibold">
                {col}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {result.rows.map((row, ri) => (
            <TableRow key={ri}>
              {row.map((cell, ci) => (
                <TableCell key={ci} className="whitespace-nowrap text-xs font-mono">
                  {cell === null ? (
                    <span className="text-muted-foreground italic">NULL</span>
                  ) : (
                    String(cell)
                  )}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CSV Export
// ---------------------------------------------------------------------------
function exportCsv(result: SqlResult) {
  const escape = (v: unknown) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return s.includes(",") || s.includes('"') || s.includes("\n")
      ? `"${s.replace(/"/g, '""')}"`
      : s;
  };
  const header = result.columns.map(escape).join(",");
  const rows = result.rows.map((r) => r.map(escape).join(","));
  const csv = [header, ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `query-results-${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
export function SqlRunnerTab() {
  const [sql, setSql] = useState("SELECT 1 AS test;");
  const [localHistory, setLocalHistory] = useState<QueryHistoryEntry[]>(loadLocalHistory);
  const [showSchema, setShowSchema] = useState(true);
  const [showHistory, setShowHistory] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Schema query
  const schemaQuery = useQuery({
    queryKey: sqlRunnerKeys.schema(),
    queryFn: fetchSchema,
    staleTime: 5 * 60 * 1000,
  });

  // Execute mutation
  const executeMutation = useMutation({
    mutationFn: (query: string) => fetchExecuteQuery(query),
    onSuccess: (_data, query) => {
      const entry: QueryHistoryEntry = {
        sql: query.slice(0, 500),
        executed_at: new Date().toISOString().slice(0, 19),
        elapsed_ms: _data.elapsed_ms,
        row_count: _data.row_count,
        status: "ok",
      };
      const updated = [entry, ...localHistory].slice(0, MAX_LOCAL_HISTORY);
      setLocalHistory(updated);
      saveLocalHistory(updated);
    },
    onError: (_err, query) => {
      const entry: QueryHistoryEntry = {
        sql: query.slice(0, 500),
        executed_at: new Date().toISOString().slice(0, 19),
        elapsed_ms: 0,
        row_count: 0,
        status: "error",
      };
      const updated = [entry, ...localHistory].slice(0, MAX_LOCAL_HISTORY);
      setLocalHistory(updated);
      saveLocalHistory(updated);
    },
  });

  const handleExecute = useCallback(() => {
    const trimmed = sql.trim();
    if (!trimmed) return;
    executeMutation.mutate(trimmed);
  }, [sql, executeMutation]);

  // Keyboard shortcut: Cmd/Ctrl+Enter
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        handleExecute();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleExecute]);

  const handleInsertTable = (name: string) => {
    setSql((prev) => {
      const ta = textareaRef.current;
      if (ta) {
        const start = ta.selectionStart;
        const end = ta.selectionEnd;
        return prev.slice(0, start) + name + prev.slice(end);
      }
      return prev + " " + name;
    });
    textareaRef.current?.focus();
  };

  const tables = schemaQuery.data?.tables ?? [];

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">SQL Runner</h2>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowSchema((v) => !v)}
          >
            <Database className="h-3.5 w-3.5 mr-1" />
            Schema
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowHistory((v) => !v)}
          >
            <Clock className="h-3.5 w-3.5 mr-1" />
            History
          </Button>
        </div>
      </div>

      <div className="flex gap-4 flex-1 min-h-0">
        {/* Schema sidebar */}
        {showSchema && (
          <Card className="w-64 shrink-0 overflow-hidden">
            <CardHeader className="py-2 px-3">
              <CardTitle className="text-xs font-medium">
                Tables ({tables.length})
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0 overflow-y-auto max-h-[calc(100vh-240px)]">
              <SchemaBrowser
                tables={tables}
                isLoading={schemaQuery.isLoading}
                onInsertTable={handleInsertTable}
              />
            </CardContent>
          </Card>
        )}

        {/* Main panel */}
        <div className="flex-1 flex flex-col gap-3 min-w-0">
          {/* Editor */}
          <Card>
            <CardContent className="p-3">
              <textarea
                ref={textareaRef}
                value={sql}
                onChange={(e) => setSql(e.target.value)}
                className={cn(
                  "w-full min-h-[120px] max-h-[300px] resize-y rounded-md border border-input bg-background px-3 py-2",
                  "font-mono text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                )}
                placeholder="Enter SQL query... (Cmd+Enter to run)"
                spellCheck={false}
                data-testid="sql-editor"
              />
              <div className="flex items-center gap-2 mt-2">
                <Button
                  size="sm"
                  onClick={handleExecute}
                  disabled={executeMutation.isPending || !sql.trim()}
                  data-testid="run-query-btn"
                >
                  <Play className="h-3.5 w-3.5 mr-1" />
                  {executeMutation.isPending ? "Running..." : "Run Query"}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSql("")}
                >
                  <Trash2 className="h-3.5 w-3.5 mr-1" />
                  Clear
                </Button>
                {executeMutation.data && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => exportCsv(executeMutation.data!)}
                  >
                    <Download className="h-3.5 w-3.5 mr-1" />
                    Export CSV
                  </Button>
                )}
                <span className="ml-auto text-xs text-muted-foreground">
                  {navigator.platform.includes("Mac") ? "Cmd" : "Ctrl"}+Enter
                  to run
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Error */}
          {executeMutation.isError && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
              <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
              <span>{(executeMutation.error as Error).message}</span>
            </div>
          )}

          {/* Results */}
          {executeMutation.data && (
            <Card className="flex-1 overflow-hidden">
              <CardHeader className="py-2 px-3">
                <div className="flex items-center gap-2">
                  <CardTitle className="text-xs font-medium">
                    Results
                  </CardTitle>
                  <Badge variant="outline" className="text-[10px]">
                    {executeMutation.data.row_count} rows
                  </Badge>
                  {executeMutation.data.truncated && (
                    <Badge variant="destructive" className="text-[10px]">
                      Truncated
                    </Badge>
                  )}
                  <span className="ml-auto text-[10px] text-muted-foreground">
                    {executeMutation.data.elapsed_ms}ms
                  </span>
                </div>
              </CardHeader>
              <CardContent className="p-0 overflow-auto max-h-[calc(100vh-480px)]">
                <ResultsTable result={executeMutation.data} />
              </CardContent>
            </Card>
          )}

          {/* Pending state */}
          {executeMutation.isPending && (
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <Skeleton className="h-4 w-4 rounded-full" />
                  <span className="text-sm text-muted-foreground">
                    Executing query...
                  </span>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* History drawer */}
      {showHistory && (
        <Card>
          <CardHeader className="py-2 px-3">
            <CardTitle className="text-xs font-medium">
              Query History ({localHistory.length})
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <QueryHistory
              entries={localHistory}
              onRerun={(s) => {
                setSql(s);
                textareaRef.current?.focus();
              }}
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default SqlRunnerTab;
