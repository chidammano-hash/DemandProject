import { useQuery } from "@tanstack/react-query";
import { buildQuerySuffix } from "./helpers";
import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ReferenceHistoryPoint {
  month: string;
  demand_qty: number;
  sales_qty: number;
}

export interface ReferenceCustomer {
  customer_no: string;
  customer_name: string;
  demand_qty: number;
  pct_share: number;
}

export interface ReferenceData {
  item_id: string;
  loc: string;
  item_description: string;
  location_name: string;
  history: ReferenceHistoryPoint[];
  top_customers: ReferenceCustomer[];
  trend_mom_pct: number;
  current_inventory: number | null;
  avg_lead_time: number | null;
  forecast_accuracy: number | null;
}

export interface DecompositionMonthly {
  month: string;
  customer_no: string;
  customer_name: string;
  demand_qty: number;
  pct_share: number;
}

export interface ParetoEntry {
  customer_no: string;
  customer_name: string;
  total_demand: number;
  pct_share: number;
  cumulative_pct: number;
}

export interface DecompositionData {
  item_id: string;
  loc: string;
  series: DecompositionMonthly[];
  pareto: ParetoEntry[];
}

export interface ComparisonMonthly {
  month: string;
  actual_qty: number;
  bottom_up_qty: number | null;
  top_down_qty: number | null;
  reconciled_qty: number | null;
}

export interface ComparisonData {
  item_id: string;
  loc: string;
  comparison: ComparisonMonthly[];
}

export interface WorkbenchSeriesMonth {
  month: string;
  demand_qty: number;
}

export interface WorkbenchSeries {
  key: string;
  label: string;
  total_demand: number;
  months: WorkbenchSeriesMonth[];
}

export interface WorkbenchChild {
  key: string;
  label: string;
  total_demand: number;
}

export interface WorkbenchData {
  grain: string;
  series: WorkbenchSeries[];
  hierarchy_children: string | null;
  total: number;
}

export interface MatrixData {
  rows: string[];
  cols: string[];
  cells: number[][];
  row_labels: Record<string, string>;
  col_labels: Record<string, string>;
}

export interface MatrixDrillData {
  item_id: string;
  loc: string;
  history: { month: string; demand_qty: number }[];
}

export type WorkbenchGrain = "item" | "item_loc" | "item_loc_customer";
export type MatrixDim = "item" | "location" | "customer";
export type MatrixMetric = "demand_qty" | "sales_qty" | "fill_rate";

// ---------------------------------------------------------------------------
// Query keys
// ---------------------------------------------------------------------------

export const demandHistoryKeys = {
  reference: (itemId: string, loc: string) =>
    ["demand-history-reference", itemId, loc] as const,
  decomposition: (itemId: string, loc: string, months?: number) =>
    ["demand-history-decomposition", itemId, loc, months] as const,
  comparison: (itemId: string, loc: string) =>
    ["demand-history-comparison", itemId, loc] as const,
  workbench: (grain: string, itemId?: string, loc?: string, customerNo?: string, months?: number, limit?: number, offset?: number) =>
    ["demand-history-workbench", grain, itemId, loc, customerNo, months, limit, offset] as const,
  matrix: (rowDim: string, colDim: string, metric?: string, months?: number, limit?: number) =>
    ["demand-history-matrix", rowDim, colDim, metric, months, limit] as const,
  matrixDrill: (itemId: string, loc: string, months?: number) =>
    ["demand-history-matrix-drill", itemId, loc, months] as const,
};

// ---------------------------------------------------------------------------
// Fetch functions
// ---------------------------------------------------------------------------

export function fetchReference(itemId: string, loc: string): Promise<ReferenceData> {
  const qs = buildQuerySuffix({ item_id: itemId, loc });
  return fetchJson(`/demand-history/reference${qs}`);
}

export function fetchDecomposition(
  itemId: string,
  loc: string,
  months?: number,
): Promise<DecompositionData> {
  const qs = buildQuerySuffix({ item_id: itemId, loc, months });
  return fetchJson(`/demand-history/decomposition${qs}`);
}

export function fetchComparison(itemId: string, loc: string): Promise<ComparisonData> {
  const qs = buildQuerySuffix({ item_id: itemId, loc });
  return fetchJson(`/demand-history/comparison${qs}`);
}

export function fetchWorkbench(
  grain: WorkbenchGrain,
  itemId?: string,
  loc?: string,
  customerNo?: string,
  months?: number,
  limit?: number,
  offset?: number,
): Promise<WorkbenchData> {
  const qs = buildQuerySuffix({
    grain, item_id: itemId, loc, customer_no: customerNo,
    months, limit, offset,
  });
  return fetchJson(`/demand-history/workbench${qs}`);
}

export function fetchMatrix(
  rowDim: MatrixDim,
  colDim: MatrixDim,
  metric?: MatrixMetric,
  months?: number,
  limit?: number,
): Promise<MatrixData> {
  const qs = buildQuerySuffix({ row_dim: rowDim, col_dim: colDim, metric, months, limit });
  return fetchJson(`/demand-history/matrix${qs}`);
}

export function fetchMatrixDrill(
  itemId: string,
  loc: string,
  months?: number,
): Promise<MatrixDrillData> {
  const qs = buildQuerySuffix({ item_id: itemId, loc, months });
  return fetchJson(`/demand-history/matrix/drill${qs}`);
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

export function useReference(itemId: string, loc: string) {
  return useQuery({
    queryKey: demandHistoryKeys.reference(itemId, loc),
    queryFn: () => fetchReference(itemId, loc),
    enabled: !!itemId && !!loc,
  });
}

export function useDecomposition(itemId: string, loc: string, months?: number) {
  return useQuery({
    queryKey: demandHistoryKeys.decomposition(itemId, loc, months),
    queryFn: () => fetchDecomposition(itemId, loc, months),
    enabled: !!itemId && !!loc,
  });
}

export function useComparison(itemId: string, loc: string) {
  return useQuery({
    queryKey: demandHistoryKeys.comparison(itemId, loc),
    queryFn: () => fetchComparison(itemId, loc),
    enabled: !!itemId && !!loc,
  });
}

export function useWorkbench(
  grain: WorkbenchGrain,
  itemId?: string,
  loc?: string,
  customerNo?: string,
  months?: number,
  limit?: number,
  offset?: number,
  enabled = true,
) {
  return useQuery({
    queryKey: demandHistoryKeys.workbench(grain, itemId, loc, customerNo, months, limit, offset),
    queryFn: () => fetchWorkbench(grain, itemId, loc, customerNo, months, limit, offset),
    enabled,
  });
}

export function useMatrix(
  rowDim: MatrixDim,
  colDim: MatrixDim,
  metric?: MatrixMetric,
  months?: number,
  limit?: number,
) {
  return useQuery({
    queryKey: demandHistoryKeys.matrix(rowDim, colDim, metric, months, limit),
    queryFn: () => fetchMatrix(rowDim, colDim, metric, months, limit),
  });
}

export function useMatrixDrill(itemId: string, loc: string, months?: number) {
  return useQuery({
    queryKey: demandHistoryKeys.matrixDrill(itemId, loc, months),
    queryFn: () => fetchMatrixDrill(itemId, loc, months),
    enabled: !!itemId && !!loc,
  });
}