import { useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ArrowDownWideNarrow, ArrowUpWideNarrow, ChartColumn, ChevronsUpDown, Globe, Loader2, MessageSquare, RefreshCcw, Send, Trophy } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";

type DomainMeta = {
  name: string;
  plural: string;
  default_sort: string;
  columns: string[];
  numeric_fields: string[];
  date_fields: string[];
  category_fields: string[];
};

type DomainPage = {
  total: number;
  total_approximate?: boolean;
  limit: number;
  offset: number;
  [key: string]: unknown;
};

type SuggestPayload = {
  values?: string[];
};

type SamplePairPayload = {
  item?: string | null;
  location?: string | null;
};

type ClusterInfo = {
  cluster_id: string;
  label: string;
  count: number;
  pct_of_total: number;
  avg_demand: number;
  cv_demand: number;
};

type DfuClustersPayload = {
  domain: string;
  total_assigned: number;
  clusters: ClusterInfo[];
};

type ClusterProfile = {
  cluster_id: number;
  label: string;
  mean_demand: number;
  cv_demand: number;
  seasonality_strength: number;
  trend_slope: number;
  growth_rate: number;
  zero_demand_pct: number;
};

type ClusterProfilesPayload = {
  profiles: ClusterProfile[];
  metadata: {
    optimal_k: number | null;
    silhouette_score: number | null;
    inertia: number | null;
  };
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sql?: string | null;
  data?: Record<string, unknown>[] | null;
  columns?: string[];
  row_count?: number | null;
  error?: string | null;
};

type AccuracyKpis = {
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
  sum_forecast: number;
  sum_actual: number;
  dfu_count: number;
};

type AccuracySliceRow = {
  bucket: string;
  n_rows: number;
  by_model: Record<string, AccuracyKpis>;
};

type AccuracySlicePayload = {
  group_by: string;
  rows: AccuracySliceRow[];
  common_dfu_count?: number;
  dfu_counts?: Record<string, number>;
};

type LagPoint = {
  lag: number;
  by_model: Record<string, AccuracyKpis>;
};

type LagCurvePayload = {
  by_lag: LagPoint[];
};

type MarketIntelSearchResult = {
  title: string;
  link: string;
  snippet: string;
};

type MarketIntelPayload = {
  item_no: string;
  location_id: string;
  item_desc: string | null;
  brand_name: string | null;
  category: string | null;
  state_id: string | null;
  site_desc: string | null;
  search_results: MarketIntelSearchResult[];
  narrative: string;
  generated_at: string;
};

type DfuAnalysisMode = "item_location" | "all_items_at_location" | "item_at_all_locations";

type DfuAnalysisKpis = {
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
  sum_forecast: number;
  sum_actual: number;
  months_covered: number;
};

type DfuModelMonthly = {
  month: string;
  forecast: number;
  actual: number;
};

type DfuAnalysisPayload = {
  mode: DfuAnalysisMode;
  item: string;
  location: string;
  points: number;
  models: string[];
  series: Record<string, number | string>[];
  model_monthly: Record<string, DfuModelMonthly[]>;
  dfu_attributes: Record<string, string | null>[];
};

function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState<T>(value);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // For objects (e.g. columnFilters record), compare by content instead of
  // reference so that re-renders producing an identical object don't reset the
  // debounce timer — which would prevent it from ever resolving.
  const serialized = typeof value === "object" ? JSON.stringify(value) : undefined;
  useEffect(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setDebounced(value), delay);
    return () => { if (timer.current) clearTimeout(timer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serialized ?? value, delay]);
  return debounced;
}

const FALLBACK_DOMAINS = ["item", "location", "customer", "time", "dfu", "sales", "forecast"];
const DIMENSION_DOMAINS = ["item", "location", "customer", "time", "dfu", "sales", "forecast"];
const titleCase = (value: string): string =>
  value
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

const TREND_COLORS = ["#4f46e5", "#0d9488", "#d97706", "#7c3aed", "#dc2626", "#0284c7"];

// Dedicated colors for DFU Analysis chart — sales measures + known forecast models
const DFU_SALES_COLORS: Record<string, string> = {
  tothist_dmd: "#e11d48",   // rose
  qty_shipped: "#2563eb",   // blue
  qty_ordered: "#059669",   // emerald
};
const DFU_MODEL_COLORS: Record<string, string> = {
  champion: "#f59e0b",        // amber
  ceiling: "#8b5cf6",         // violet
  external: "#06b6d4",        // cyan
  lgbm_global: "#84cc16",     // lime
  lgbm_cluster: "#14b8a6",    // teal
  lgbm_transfer: "#f97316",   // orange
  catboost_global: "#ec4899", // pink
  catboost_cluster: "#6366f1",// indigo
  catboost_transfer: "#a3e635",// lime-400
  xgboost_global: "#a855f7",  // purple
  xgboost_cluster: "#0ea5e9", // sky
  xgboost_transfer: "#fb923c",// orange-400
};
const DFU_MODEL_FALLBACK_COLORS = ["#64748b", "#78716c", "#0f766e", "#b45309", "#9333ea", "#e879f9"];
function dfuModelColor(model: string, idx: number): string {
  return DFU_MODEL_COLORS[model] ?? DFU_MODEL_FALLBACK_COLORS[idx % DFU_MODEL_FALLBACK_COLORS.length];
}

const ELEMENT_CONFIG: Record<string, { symbol: string; number: number; name: string; color: string; activeColor: string; glow: string }> = {
  explorer: { symbol: "Dx", number: 1, name: "Explorer", color: "bg-pink-50/90 text-pink-800 border-pink-200/60",        activeColor: "bg-pink-100 text-pink-950 border-pink-300", glow: "shadow-[0_0_12px_rgba(236,72,153,0.3)]" },
  item:     { symbol: "It", number: 26, name: "Item",     color: "bg-pink-50/90 text-pink-800 border-pink-200/60",        activeColor: "bg-pink-100 text-pink-950 border-pink-300", glow: "shadow-[0_0_12px_rgba(236,72,153,0.3)]" },
  location: { symbol: "Lo", number: 71, name: "Location", color: "bg-pink-50/90 text-pink-800 border-pink-200/60",        activeColor: "bg-pink-100 text-pink-950 border-pink-300", glow: "shadow-[0_0_12px_rgba(236,72,153,0.3)]" },
  customer: { symbol: "Cu", number: 29, name: "Customer", color: "bg-amber-50/90 text-amber-800 border-amber-200/60",     activeColor: "bg-amber-100 text-amber-950 border-amber-300", glow: "shadow-[0_0_12px_rgba(245,158,11,0.3)]" },
  time:     { symbol: "Ti", number: 22, name: "Time",     color: "bg-amber-50/90 text-amber-800 border-amber-200/60",     activeColor: "bg-amber-100 text-amber-950 border-amber-300", glow: "shadow-[0_0_12px_rgba(245,158,11,0.3)]" },
  dfu:      { symbol: "Df", number: 110, name: "DFU",      color: "bg-lime-50/90 text-lime-800 border-lime-200/60",        activeColor: "bg-lime-100 text-lime-950 border-lime-300", glow: "shadow-[0_0_12px_rgba(132,204,22,0.3)]" },
  clusters: { symbol: "Cl", number: 2, name: "Clusters", color: "bg-emerald-50/90 text-emerald-800 border-emerald-200/60", activeColor: "bg-emerald-100 text-emerald-950 border-emerald-300", glow: "shadow-[0_0_12px_rgba(16,185,129,0.3)]" },
  sales:    { symbol: "Sa", number: 3, name: "Sales",    color: "bg-sky-50/90 text-sky-800 border-sky-200/60",            activeColor: "bg-sky-100 text-sky-950 border-sky-300", glow: "shadow-[0_0_12px_rgba(14,165,233,0.3)]" },
  forecast: { symbol: "Fc", number: 4, name: "Forecast", color: "bg-indigo-50/90 text-indigo-800 border-indigo-200/60",    activeColor: "bg-indigo-100 text-indigo-950 border-indigo-300", glow: "shadow-[0_0_12px_rgba(99,102,241,0.3)]" },
  dfuAnalysis: { symbol: "Da", number: 7, name: "DFU Analysis", color: "bg-teal-50/90 text-teal-800 border-teal-200/60", activeColor: "bg-teal-100 text-teal-950 border-teal-300", glow: "shadow-[0_0_12px_rgba(20,184,166,0.3)]" },
  accuracy: { symbol: "Ac", number: 5, name: "Accuracy", color: "bg-purple-50/90 text-purple-800 border-purple-200/60",    activeColor: "bg-purple-100 text-purple-950 border-purple-300", glow: "shadow-[0_0_12px_rgba(168,85,247,0.3)]" },
  intel:    { symbol: "Mi", number: 6, name: "Intel",    color: "bg-cyan-50/90 text-cyan-800 border-cyan-200/60",          activeColor: "bg-cyan-100 text-cyan-950 border-cyan-300", glow: "shadow-[0_0_12px_rgba(6,182,212,0.3)]" },
};

const ACCURACY_KPI_OPTIONS = [
  { key: "accuracy_pct", label: "Accuracy %", format: "pct" },
  { key: "wape",         label: "WAPE %",     format: "pct" },
  { key: "bias",         label: "Bias",        format: "bias" },
  { key: "sum_forecast", label: "\u03A3 Forecast", format: "num" },
  { key: "sum_actual",   label: "\u03A3 Actual",   format: "num" },
  { key: "dfu_count",    label: "DFU Count",  format: "num" },
] as const;

const numberFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const compactNumberFmt = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

function formatCompactNumber(value: number | string): string {
  if (typeof value !== "number") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? compactNumberFmt.format(parsed) : String(value);
  }
  return compactNumberFmt.format(value);
}

function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "-";
  }
  return numberFmt.format(value);
}

function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "-";
  }
  return `${numberFmt.format(value)}%`;
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  if (typeof value === "number") {
    return formatNumber(value);
  }
  return String(value);
}

function getInitialDomain(): string {
  const queryDomain = new URLSearchParams(window.location.search).get("domain");
  return (queryDomain || "item").toLowerCase();
}

const ANALYTICS_TAB_DOMAINS = new Set(["sales", "forecast"]);

function getInitialTab(): string {
  const d = getInitialDomain();
  if (ANALYTICS_TAB_DOMAINS.has(d)) return d;
  return DIMENSION_DOMAINS.includes(d) ? "explorer" : d;
}

function updateDomainPath(domain: string) {
    const normalized = domain.toLowerCase();
  const url = new URL(window.location.href);
  url.searchParams.set("domain", normalized);
  window.history.replaceState(null, "", url);
}

export default function App() {
  const [domain, setDomain] = useState<string>(getInitialDomain);
  const [activeTab, setActiveTab] = useState<string>(getInitialTab);

  const [meta, setMeta] = useState<DomainMeta | null>(null);
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [total, setTotal] = useState(0);
  const [totalApproximate, setTotalApproximate] = useState(false);

  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(100);
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [columnFilters, setColumnFilters] = useState<Record<string, string>>({});
  const [visibleColumns, setVisibleColumns] = useState<Record<string, boolean>>({});
  const [showFieldPanel, setShowFieldPanel] = useState(false);

  const isFactDomain = domain === "sales" || domain === "forecast";
  const filterDebounceMs = isFactDomain ? 500 : 300;
  const debouncedSearch = useDebounce(search, filterDebounceMs);
  const debouncedColumnFilters = useDebounce(columnFilters, filterDebounceMs);

  const [itemFilter, setItemFilter] = useState("");
  const [locationFilter, setLocationFilter] = useState("");
  const debouncedItemFilter = useDebounce(itemFilter, filterDebounceMs);
  const debouncedLocationFilter = useDebounce(locationFilter, filterDebounceMs);
  const [selectedModel, setSelectedModel] = useState("");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedCluster, setSelectedCluster] = useState("");
  const [clusterSource, setClusterSource] = useState<"ml" | "source">("ml");
  const [clusterSummary, setClusterSummary] = useState<ClusterInfo[]>([]);
  const [clusterMeta, setClusterMeta] = useState<ClusterProfilesPayload["metadata"] | null>(null);
  const [showClusterViz, setShowClusterViz] = useState(false);
  const [autoSampledDomain, setAutoSampledDomain] = useState("");
  const [columnSuggestions, setColumnSuggestions] = useState<Record<string, string[]>>({});

  const [loadingTable, setLoadingTable] = useState(false);
  const [error, setError] = useState<string>("");

  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Accuracy comparison panel state (feature16)
  const [sliceGroupBy, setSliceGroupBy] = useState("cluster_assignment");
  const [sliceLag, setSliceLag] = useState(-1);
  const [sliceModels, setSliceModels] = useState("");
  const [sliceData, setSliceData] = useState<AccuracySliceRow[]>([]);
  const [lagCurveData, setLagCurveData] = useState<LagPoint[]>([]);
  const [loadingSlice, setLoadingSlice] = useState(false);
  const [sliceKpis, setSliceKpis] = useState<string[]>(["accuracy_pct", "wape", "bias"]);
  const [lagCurveMetric, setLagCurveMetric] = useState("accuracy_pct");
  const [sliceMonths, setSliceMonths] = useState(12); // 1-12 month rolling window
  const [commonDfus, setCommonDfus] = useState(false);
  const [commonDfuCount, setCommonDfuCount] = useState<number | null>(null);
  const [dfuCounts, setDfuCounts] = useState<Record<string, number> | null>(null);

  // Champion / model competition state (feature15)
  const [competitionConfig, setCompetitionConfig] = useState<{ metric: string; lag: string; min_dfu_rows: number; champion_model_id: string; models: string[] } | null>(null);
  const [championSummary, setChampionSummary] = useState<{ total_dfus: number; total_champion_rows: number; model_wins: Record<string, number>; overall_champion_wape: number | null; overall_champion_accuracy_pct: number | null; run_ts: string; total_ceiling_rows?: number; ceiling_model_wins?: Record<string, number>; overall_ceiling_wape?: number | null; overall_ceiling_accuracy_pct?: number | null } | null>(null);
  const [runningCompetition, setRunningCompetition] = useState(false);
  const [savingConfig, setSavingConfig] = useState(false);

  // Market Intelligence state (feature18 - market intel)
  const [miItemFilter, setMiItemFilter] = useState("");
  const [miLocationFilter, setMiLocationFilter] = useState("");
  const [miItemSuggestions, setMiItemSuggestions] = useState<string[]>([]);
  const [miLocationSuggestions, setMiLocationSuggestions] = useState<string[]>([]);
  const [miResult, setMiResult] = useState<MarketIntelPayload | null>(null);
  const [miLoading, setMiLoading] = useState(false);
  const [miError, setMiError] = useState("");

  // DFU Analysis tab state
  const [dfuMode, setDfuMode] = useState<DfuAnalysisMode>("item_location");
  const [dfuItem, setDfuItem] = useState("");
  const [dfuLocation, setDfuLocation] = useState("");
  const [dfuPoints, setDfuPoints] = useState(36);
  const [dfuKpiMonths, setDfuKpiMonths] = useState(12);
  const [dfuData, setDfuData] = useState<DfuAnalysisPayload | null>(null);
  const [dfuVisibleSeries, setDfuVisibleSeries] = useState<Set<string>>(new Set(["tothist_dmd", "qty_shipped", "qty_ordered"]));
  const [dfuTimeStart, setDfuTimeStart] = useState("");
  const [dfuTimeEnd, setDfuTimeEnd] = useState("");
  const [dfuDefaultStart, setDfuDefaultStart] = useState("");
  const [dfuLoading, setDfuLoading] = useState(false);
  const [dfuAutoSampled, setDfuAutoSampled] = useState(false);
  const [dfuItemSuggestions, setDfuItemSuggestions] = useState<string[]>([]);
  const [dfuLocationSuggestions, setDfuLocationSuggestions] = useState<string[]>([]);
  const debouncedDfuItem = useDebounce(dfuItem, 500);
  const debouncedDfuLocation = useDebounce(dfuLocation, 500);

  const visibleCols = useMemo(() => {
    if (!meta) {
      return [];
    }
    return meta.columns.filter((col) => visibleColumns[col] !== false);
  }, [meta, visibleColumns]);

  const itemField = useMemo(() => {
    if (!meta) {
      return "";
    }
    if (meta.columns.includes("dmdunit")) {
      return "dmdunit";
    }
    if (meta.columns.includes("item_no")) {
      return "item_no";
    }
    return "";
  }, [meta]);
  const locationField = useMemo(() => {
    if (!meta) {
      return "";
    }
    if (meta.columns.includes("loc")) {
      return "loc";
    }
    if (meta.columns.includes("location_id")) {
      return "location_id";
    }
    return "";
  }, [meta]);
  const showFactFilters = (domain === "sales" || domain === "forecast") && Boolean(itemField) && Boolean(locationField) && activeTab !== "explorer";
  const formatPairFilterValue = (value: string): string => {
    const trimmed = value.trim();
    if (!trimmed) {
      return "";
    }
    return `=${trimmed}`;
  };
  const effectiveFilters = useMemo(() => {
    const out = Object.fromEntries(Object.entries(debouncedColumnFilters).filter(([, value]) => value.trim() !== ""));
    if (showFactFilters && debouncedItemFilter.trim() && itemField) {
      out[itemField] = formatPairFilterValue(debouncedItemFilter);
    }
    if (showFactFilters && debouncedLocationFilter.trim() && locationField) {
      out[locationField] = formatPairFilterValue(debouncedLocationFilter);
    }
    if (domain === "forecast" && selectedModel.trim()) {
      out["model_id"] = `=${selectedModel.trim()}`;
    }
    if (domain === "dfu" && selectedCluster.trim()) {
      const filterCol = clusterSource === "ml" ? "ml_cluster" : "cluster_assignment";
      out[filterCol] = `=${selectedCluster.trim()}`;
    }
    return out;
  }, [debouncedColumnFilters, showFactFilters, debouncedItemFilter, debouncedLocationFilter, itemField, locationField, domain, selectedModel, selectedCluster, clusterSource]);

  useEffect(() => {
    let cancelled = false;

    async function loadDomains() {
      try {
        const res = await fetch("/domains");
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const payload = (await res.json()) as { domains?: string[] };
        const list = (payload.domains || []).map((d) => d.toLowerCase());
        if (!cancelled) {
          const nextDomains = list.length > 0 ? list : FALLBACK_DOMAINS;
          if (!nextDomains.includes(domain)) {
            setDomain(nextDomains[0]);
          }
        }
      } catch {
        if (!cancelled) {
          if (!FALLBACK_DOMAINS.includes(domain)) {
            setDomain("item");
          }
        }
      }
    }

    loadDomains();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadDomainData() {
      // loading meta
      setLoadingTable(true);
      setError("");

      // Fetch meta + initial page in parallel
      const metaPromise = fetch(`/domains/${encodeURIComponent(domain)}/meta`);
      const pagePromise = fetch(
        `/domains/${encodeURIComponent(domain)}/page?` +
          new URLSearchParams({ limit: String(limit), offset: "0", q: "", sort_by: "", sort_dir: "asc" }).toString()
      );

      try {
        const [metaRes, pageRes] = await Promise.all([metaPromise, pagePromise]);
        if (!metaRes.ok) throw new Error(`HTTP ${metaRes.status}`);
        const payload = (await metaRes.json()) as DomainMeta;
        if (cancelled) return;

        setMeta(payload);
        setOffset(0);
        setSearch("");
        setColumnFilters({});
        setSortBy(payload.default_sort);
        setSortDir("asc");
        setItemFilter("");
        setLocationFilter("");
        setSelectedModel("");
        setAvailableModels([]);
        setAutoSampledDomain("");
        setColumnSuggestions({});
        setVisibleColumns(Object.fromEntries(payload.columns.map((c) => [c, true])));
        updateDomainPath(domain);

        if (pageRes.ok) {
          const pagePl = (await pageRes.json()) as DomainPage;
          if (!cancelled) {
            setRows((pagePl[payload.plural] || []) as Record<string, unknown>[]);
            setTotal(Number(pagePl.total || 0));
            setTotalApproximate(Boolean(pagePl.total_approximate));
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load domain metadata");
          setMeta(null);
          setRows([]);
          setTotal(0);
        }
      } finally {
        if (!cancelled) {
          // done loading meta
          setLoadingTable(false);
        }
      }
    }

    loadDomainData();
    return () => {
      cancelled = true;
    };
  }, [domain]);

  useEffect(() => {
    if (!meta) {
      return;
    }

    // Skip page queries when data grid is not visible (only explorer tab shows the grid)
    if (activeTab !== "explorer") {
      return;
    }

    const plural = meta.plural;
    let cancelled = false;

    async function loadTable() {
      setLoadingTable(true);
      setError("");
      try {
        const params = new URLSearchParams({
          limit: String(limit),
          offset: String(offset),
          q: debouncedSearch,
          sort_by: sortBy,
          sort_dir: sortDir,
        });

        if (Object.keys(effectiveFilters).length > 0) {
          params.set("filters", JSON.stringify(effectiveFilters));
        }

        const res = await fetch(`/domains/${encodeURIComponent(domain)}/page?${params.toString()}`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const payload = (await res.json()) as DomainPage;
        const pageRows = (payload[plural] || []) as Record<string, unknown>[];

        if (!cancelled) {
          setRows(pageRows);
          setTotal(Number(payload.total || 0));
          setTotalApproximate(Boolean(payload.total_approximate));
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load records");
          setRows([]);
          setTotal(0);
          setTotalApproximate(false);
        }
      } finally {
        if (!cancelled) {
          setLoadingTable(false);
        }
      }
    }

    loadTable();
    return () => {
      cancelled = true;
    };
  }, [meta, domain, offset, limit, debouncedSearch, sortBy, sortDir, effectiveFilters, activeTab]);

  useEffect(() => {
    if (domain !== "forecast" || !meta) {
      setAvailableModels([]);
      return;
    }
    let cancelled = false;
    async function loadModels() {
      try {
        const res = await fetch("/domains/forecast/models");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = (await res.json()) as { models?: string[] };
        if (!cancelled) {
          setAvailableModels(payload.models || []);
        }
      } catch {
        if (!cancelled) setAvailableModels([]);
      }
    }
    loadModels();
    return () => { cancelled = true; };
  }, [domain, meta]);

  useEffect(() => {
    if (domain !== "dfu") {
      setClusterSummary([]);
      setSelectedCluster("");
      setClusterMeta(null);
      return;
    }
    let cancelled = false;
    async function loadClusters() {
      try {
        const res = await fetch(`/domains/dfu/clusters?source=${clusterSource}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = (await res.json()) as DfuClustersPayload;
        if (!cancelled && payload.clusters) {
          setClusterSummary(payload.clusters);
        }
      } catch {
        if (!cancelled) setClusterSummary([]);
      }
    }
    async function loadProfiles() {
      try {
        const res = await fetch("/domains/dfu/clusters/profiles");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = (await res.json()) as ClusterProfilesPayload;
        if (!cancelled) {
          setClusterMeta(payload.metadata || null);
        }
      } catch {
        if (!cancelled) {
          setClusterMeta(null);
        }
      }
    }
    loadClusters();
    if (clusterSource === "ml") loadProfiles();
    return () => { cancelled = true; };
  }, [domain, clusterSource]);

  useEffect(() => {
    if (!meta || !showFactFilters || autoSampledDomain === domain) {
      return;
    }
    if (itemFilter.trim() || locationFilter.trim()) {
      setAutoSampledDomain(domain);
      return;
    }
    let cancelled = false;
    async function loadSamplePair() {
      try {
        const res = await fetch(`/domains/${encodeURIComponent(domain)}/sample-pair`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const payload = (await res.json()) as SamplePairPayload;
        if (!cancelled) {
          if (payload.item) {
            setItemFilter(String(payload.item));
          }
          if (payload.location) {
            setLocationFilter(String(payload.location));
          }
          setOffset(0);
        }
      } catch {
        // Non-blocking; fall back to user-provided filters.
      } finally {
        if (!cancelled) {
          setAutoSampledDomain(domain);
        }
      }
    }
    loadSamplePair();
    return () => {
      cancelled = true;
    };
  }, [meta, showFactFilters, autoSampledDomain, domain, itemFilter, locationFilter]);

  // Column-level typeahead suggestions
  useEffect(() => {
    if (!meta) return;
    // Determine which columns are text (not numeric, not date)
    const textCols = new Set(
      meta.columns.filter(
        (c) => !meta.numeric_fields.includes(c) && !meta.date_fields.includes(c),
      ),
    );
    // Find columns that have a non-empty, non-exact filter typed
    const active = Object.entries(debouncedColumnFilters).filter(
      ([col, val]) => val.trim() !== "" && !val.startsWith("=") && textCols.has(col),
    );
    // Clear suggestions for columns no longer being filtered.
    // Return prev reference unchanged when nothing is stale to avoid re-renders.
    setColumnSuggestions((prev) => {
      const staleCols = Object.keys(prev).filter((col) => !debouncedColumnFilters[col]?.trim());
      if (staleCols.length === 0) return prev;
      const next = { ...prev };
      staleCols.forEach((col) => delete next[col]);
      return next;
    });
    if (active.length === 0) return;

    let cancelled = false;
    const timers: number[] = [];
    for (const [col, val] of active) {
      const tid = window.setTimeout(async () => {
        try {
          const params = new URLSearchParams({
            field: col,
            q: val.trim(),
            limit: "12",
          });
          // Pass other active column filters as scoped context
          const otherFilters: Record<string, string> = {};
          for (const [k, v] of Object.entries(debouncedColumnFilters)) {
            if (k !== col && v.trim()) otherFilters[k] = v.trim();
          }
          if (Object.keys(otherFilters).length > 0) {
            params.set("filters", JSON.stringify(otherFilters));
          }
          const res = await fetch(
            `/domains/${encodeURIComponent(domain)}/suggest?${params.toString()}`,
          );
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const payload = (await res.json()) as SuggestPayload;
          const values = Array.from(
            new Set((payload.values || []).filter(Boolean)),
          ).slice(0, 12);
          if (!cancelled) {
            setColumnSuggestions((prev) => ({ ...prev, [col]: values }));
          }
        } catch {
          if (!cancelled) {
            setColumnSuggestions((prev) => ({ ...prev, [col]: [] }));
          }
        }
      }, 180);
      timers.push(tid);
    }
    return () => {
      cancelled = true;
      timers.forEach((t) => window.clearTimeout(t));
    };
  }, [debouncedColumnFilters, domain, meta]);

  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(offset + limit, total);

  function toggleSort(column: string) {
    if (sortBy === column) {
      setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
      return;
    }
    setSortBy(column);
    setSortDir("asc");
  }

  function toggleColumn(column: string, checked: boolean) {
    setVisibleColumns((prev) => ({ ...prev, [column]: checked }));
  }

  async function sendChat() {
    const q = chatInput.trim();
    if (!q || chatLoading) return;
    const userMsg: ChatMessage = { role: "user", content: q };
    setChatMessages((prev) => [...prev, userMsg]);
    setChatInput("");
    setChatLoading(true);
    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, domain }),
      });
      if (!res.ok) {
        const detail = await res.text();
        setChatMessages((prev) => [...prev, { role: "assistant", content: `Error: ${detail}` }]);
        return;
      }
      const payload = await res.json();
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: payload.answer || "No answer returned.",
        sql: payload.sql || null,
        data: payload.data || null,
        columns: payload.columns || [],
        row_count: payload.row_count ?? null,
        error: payload.error || null,
      };
      setChatMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setChatMessages((prev) => [...prev, { role: "assistant", content: `Network error: ${err instanceof Error ? err.message : "unknown"}` }]);
    } finally {
      setChatLoading(false);
    }
  }

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  // Fetch accuracy slice + lag curve data when on the Accuracy tab (feature16)
  useEffect(() => {
    if (activeTab !== "accuracy") return;
    let cancelled = false;
    setLoadingSlice(true);
    // Compute trailing month window (skip when slicing by month — each row is already one month)
    let monthFrom = "";
    if (sliceGroupBy !== "month_start") {
      const now = new Date();
      const from = new Date(now.getFullYear(), now.getMonth() - sliceMonths, 1);
      monthFrom = from.toISOString().slice(0, 10);
    }
    const needDfuCount = sliceKpis.includes("dfu_count");
    const sliceParams = new URLSearchParams({ group_by: sliceGroupBy, lag: String(sliceLag) });
    if (sliceModels.trim()) sliceParams.set("models", sliceModels.trim());
    if (monthFrom) sliceParams.set("month_from", monthFrom);
    if (commonDfus) sliceParams.set("common_dfus", "true");
    if (needDfuCount) sliceParams.set("include_dfu_count", "true");
    const lagParams = new URLSearchParams();
    if (sliceModels.trim()) lagParams.set("models", sliceModels.trim());
    if (monthFrom) lagParams.set("month_from", monthFrom);
    if (commonDfus) lagParams.set("common_dfus", "true");
    if (needDfuCount) lagParams.set("include_dfu_count", "true");
    Promise.all([
      fetch(`/forecast/accuracy/slice?${sliceParams}`).then((r) => r.json() as Promise<AccuracySlicePayload>),
      fetch(`/forecast/accuracy/lag-curve?${lagParams}`).then((r) => r.json() as Promise<LagCurvePayload>),
    ])
      .then(([slicePayload, lagPayload]) => {
        if (cancelled) return;
        setSliceData(slicePayload.rows || []);
        setLagCurveData(lagPayload.by_lag || []);
        setCommonDfuCount(slicePayload.common_dfu_count ?? null);
        setDfuCounts(slicePayload.dfu_counts ?? null);
      })
      .catch(() => {
        if (!cancelled) { setSliceData([]); setLagCurveData([]); setCommonDfuCount(null); setDfuCounts(null); }
      })
      .finally(() => { if (!cancelled) setLoadingSlice(false); });
    return () => { cancelled = true; };
  }, [activeTab, sliceGroupBy, sliceLag, sliceModels, sliceMonths, commonDfus, sliceKpis]);

  // Fetch competition config + summary when on the accuracy tab (feature15)
  useEffect(() => {
    if (activeTab !== "accuracy") return;
    Promise.all([
      fetch("/competition/config").then((r) => r.ok ? r.json() : null),
      fetch("/competition/summary").then((r) => r.ok ? r.json() : null),
    ]).then(([cfgPayload, sumPayload]) => {
      if (cfgPayload?.config) {
        setCompetitionConfig(cfgPayload.config);
        setAvailableModels(cfgPayload.available_models || []);
      }
      if (sumPayload?.summary) setChampionSummary(sumPayload.summary);
    }).catch(() => {/* ignore */});
  }, [activeTab]);

  // Market Intelligence — item suggestions
  useEffect(() => {
    if (activeTab !== "intel") return;
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({ field: "item_no", q: miItemFilter.trim(), limit: "12" });
        const res = await fetch(`/domains/item/suggest?${params.toString()}`);
        if (!res.ok) throw new Error();
        const payload = (await res.json()) as SuggestPayload;
        if (!cancelled) setMiItemSuggestions((payload.values || []).filter(Boolean).slice(0, 12));
      } catch { if (!cancelled) setMiItemSuggestions([]); }
    }, 180);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [activeTab, miItemFilter]);

  // Market Intelligence — location suggestions
  useEffect(() => {
    if (activeTab !== "intel") return;
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({ field: "location_id", q: miLocationFilter.trim(), limit: "12" });
        const res = await fetch(`/domains/location/suggest?${params.toString()}`);
        if (!res.ok) throw new Error();
        const payload = (await res.json()) as SuggestPayload;
        if (!cancelled) setMiLocationSuggestions((payload.values || []).filter(Boolean).slice(0, 12));
      } catch { if (!cancelled) setMiLocationSuggestions([]); }
    }, 180);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [activeTab, miLocationFilter]);

  // Market Intelligence — generate briefing
  async function generateMarketIntel() {
    const item = miItemFilter.trim();
    const loc = miLocationFilter.trim();
    if (!item || !loc || miLoading) return;
    setMiLoading(true);
    setMiError("");
    setMiResult(null);
    try {
      const res = await fetch("/market-intelligence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ item_no: item, location_id: loc }),
      });
      if (!res.ok) {
        const detail = await res.text();
        setMiError(`Error: ${detail}`);
        return;
      }
      const payload = (await res.json()) as MarketIntelPayload;
      setMiResult(payload);
    } catch (err) {
      setMiError(err instanceof Error ? err.message : "Network error");
    } finally {
      setMiLoading(false);
    }
  }

  // DFU Analysis: auto-sample item+location on first visit
  useEffect(() => {
    if (activeTab !== "dfuAnalysis" || dfuAutoSampled) return;
    if (dfuItem.trim() || dfuLocation.trim()) {
      setDfuAutoSampled(true);
      return;
    }
    let cancelled = false;
    async function loadSample() {
      try {
        const res = await fetch("/domains/sales/sample-pair");
        if (!res.ok) throw new Error("HTTP error");
        const payload = (await res.json()) as SamplePairPayload;
        if (!cancelled) {
          if (payload.item) setDfuItem(String(payload.item));
          if (payload.location) setDfuLocation(String(payload.location));
        }
      } catch { /* non-blocking */ }
      finally { if (!cancelled) setDfuAutoSampled(true); }
    }
    loadSample();
    return () => { cancelled = true; };
  }, [activeTab, dfuAutoSampled, dfuItem, dfuLocation]);

  // DFU Analysis: fetch data from /dfu/analysis
  useEffect(() => {
    if (activeTab !== "dfuAnalysis") return;
    const needsItem = dfuMode !== "all_items_at_location";
    const needsLoc = dfuMode !== "item_at_all_locations";
    if (needsItem && !debouncedDfuItem.trim()) return;
    if (needsLoc && !debouncedDfuLocation.trim()) return;

    let cancelled = false;
    async function loadAnalysis() {
      setDfuLoading(true);
      try {
        const params = new URLSearchParams({
          mode: dfuMode,
          item: debouncedDfuItem.trim(),
          location: debouncedDfuLocation.trim(),
          points: String(dfuPoints),
        });
        const res = await fetch(`/dfu/analysis?${params}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = (await res.json()) as DfuAnalysisPayload;
        if (!cancelled) {
          setDfuData(payload);
          const allKeys = new Set(["tothist_dmd", "qty_shipped", "qty_ordered", ...payload.models.map((m) => `forecast_${m}`)]);
          setDfuVisibleSeries(allKeys);
          // Default "from" to the first month where all measures have data
          const measureKeys = new Set<string>();
          for (const pt of payload.series) {
            for (const k of Object.keys(pt)) {
              if (k !== "month") measureKeys.add(k);
            }
          }
          let smartStart = "";
          if (measureKeys.size > 0 && payload.series.length > 0) {
            const keys = Array.from(measureKeys);
            for (const pt of payload.series) {
              if (keys.every((k) => k in pt)) {
                smartStart = String(pt.month);
                break;
              }
            }
          }
          setDfuDefaultStart(smartStart);
          setDfuTimeStart(smartStart);
          setDfuTimeEnd("");
        }
      } catch {
        if (!cancelled) setDfuData(null);
      } finally {
        if (!cancelled) setDfuLoading(false);
      }
    }
    loadAnalysis();
    return () => { cancelled = true; };
  }, [activeTab, dfuMode, debouncedDfuItem, debouncedDfuLocation, dfuPoints]);

  // DFU Analysis: compute KPIs client-side from model_monthly data
  const dfuKpis = useMemo<Record<string, DfuAnalysisKpis>>(() => {
    if (!dfuData?.model_monthly) return {};
    const result: Record<string, DfuAnalysisKpis> = {};
    for (const [modelId, rows] of Object.entries(dfuData.model_monthly)) {
      // rows are sorted month desc; take the most recent kpi_months entries
      const window = rows.slice(0, dfuKpiMonths);
      if (window.length === 0) continue;
      let sumForecast = 0, sumActual = 0, sumAbsErr = 0;
      for (const r of window) {
        sumForecast += r.forecast;
        sumActual += r.actual;
        sumAbsErr += Math.abs(r.forecast - r.actual);
      }
      const absActual = Math.abs(sumActual);
      const wape = absActual > 0 ? (100 * sumAbsErr) / absActual : null;
      const accuracy = wape !== null ? 100 - wape : null;
      const bias = absActual > 0 ? (sumForecast / sumActual) - 1 : null;
      result[modelId] = {
        accuracy_pct: accuracy !== null ? Math.round(accuracy * 10000) / 10000 : null,
        wape: wape !== null ? Math.round(wape * 10000) / 10000 : null,
        bias: bias !== null ? Math.round(bias * 10000) / 10000 : null,
        sum_forecast: sumForecast,
        sum_actual: sumActual,
        months_covered: window.length,
      };
    }
    return result;
  }, [dfuData, dfuKpiMonths]);

  // DFU Analysis: available months and time-range-filtered series
  const dfuMonths = useMemo(() => {
    if (!dfuData?.series.length) return [] as string[];
    return dfuData.series.map((p) => String(p.month));
  }, [dfuData]);

  const dfuFilteredSeries = useMemo(() => {
    if (!dfuData?.series.length) return [];
    const start = dfuTimeStart || dfuMonths[0];
    const end = dfuTimeEnd || dfuMonths[dfuMonths.length - 1];
    return dfuData.series.filter((p) => {
      const m = String(p.month);
      return m >= start && m <= end;
    });
  }, [dfuData, dfuMonths, dfuTimeStart, dfuTimeEnd]);

  // DFU Analysis: item typeahead suggestions
  useEffect(() => {
    if (activeTab !== "dfuAnalysis" || !dfuItem.trim()) {
      setDfuItemSuggestions([]);
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({ field: "dmdunit", q: dfuItem.trim(), limit: "12" });
        if (debouncedDfuLocation.trim()) {
          params.set("filters", JSON.stringify({ loc: `=${debouncedDfuLocation.trim()}` }));
        }
        const res = await fetch(`/domains/sales/suggest?${params}`);
        if (!res.ok) throw new Error("HTTP error");
        const payload = (await res.json()) as SuggestPayload;
        if (!cancelled) setDfuItemSuggestions(payload.values || []);
      } catch { if (!cancelled) setDfuItemSuggestions([]); }
    }, 180);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [activeTab, dfuItem, debouncedDfuLocation]);

  // DFU Analysis: location typeahead suggestions
  useEffect(() => {
    if (activeTab !== "dfuAnalysis" || !dfuLocation.trim()) {
      setDfuLocationSuggestions([]);
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({ field: "loc", q: dfuLocation.trim(), limit: "12" });
        if (debouncedDfuItem.trim()) {
          params.set("filters", JSON.stringify({ dmdunit: `=${debouncedDfuItem.trim()}` }));
        }
        const res = await fetch(`/domains/sales/suggest?${params}`);
        if (!res.ok) throw new Error("HTTP error");
        const payload = (await res.json()) as SuggestPayload;
        if (!cancelled) setDfuLocationSuggestions(payload.values || []);
      } catch { if (!cancelled) setDfuLocationSuggestions([]); }
    }, 180);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [activeTab, dfuLocation, debouncedDfuItem]);

  return (
    <main className="mx-auto w-full max-w-[1800px] min-w-0 overflow-x-hidden p-4 md:p-6">
      <section className="animate-fade-in relative overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 p-5 text-white shadow-2xl">
        {/* Decorative orbs */}
        <div className="pointer-events-none absolute -left-20 -top-20 h-64 w-64 rounded-full bg-indigo-500/10 blur-3xl" />
        <div className="pointer-events-none absolute -right-16 -bottom-16 h-48 w-48 rounded-full bg-purple-500/10 blur-3xl" />
        <div className="pointer-events-none absolute left-1/2 top-0 h-px w-2/3 -translate-x-1/2 bg-gradient-to-r from-transparent via-indigo-400/30 to-transparent" />
        <div className="relative flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-500/20 ring-1 ring-indigo-400/30">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-indigo-300">
                  <circle cx="12" cy="12" r="3" />
                  <ellipse cx="12" cy="12" rx="10" ry="4" />
                  <ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(60 12 12)" />
                  <ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(120 12 12)" />
                </svg>
              </div>
              <h1 className="text-2xl font-bold tracking-tight md:text-3xl bg-gradient-to-r from-white via-indigo-100 to-indigo-200 bg-clip-text text-transparent">Planthium</h1>
            </div>
            <p className="mt-1 ml-[46px] text-sm text-indigo-200/70 md:text-base">Periodic Analytics for Demand Forecasting</p>
          </div>
          <div className="flex flex-wrap gap-2.5">
            {/* Data Explorer tab — consolidates item, location, customer, time */}
            {(() => {
              const el = ELEMENT_CONFIG["explorer"];
              const isActive = activeTab === "explorer";
              return (
                <button
                  key="explorer"
                  className={cn(
                    "group relative flex flex-col items-center justify-center rounded-xl border px-3.5 py-2 min-w-[68px] transition-all duration-200 backdrop-blur-sm",
                    isActive
                      ? el.activeColor + " " + el.glow + " scale-105 border-opacity-100"
                      : el.color + " hover:scale-105 hover:border-opacity-80 border-opacity-40"
                  )}
                  onClick={() => {
                    setActiveTab("explorer");
                    if (ANALYTICS_TAB_DOMAINS.has(domain) || !DIMENSION_DOMAINS.includes(domain)) {
                      setDomain("item");
                    }
                  }}
                >
                  {isActive && <span className="absolute -bottom-1 left-1/2 h-1 w-6 -translate-x-1/2 rounded-full bg-current opacity-60" />}
                  <span className="text-[9px] leading-none self-end font-mono opacity-50">{el.number}</span>
                  <span className="text-xl font-black leading-tight font-mono tracking-tight">{el.symbol}</span>
                  <span className="text-[9px] font-medium leading-none tracking-wide uppercase opacity-70">{el.name}</span>
                </button>
              );
            })()}
            {/* Clusters tab — DFU clustering info only */}
            {(() => {
              const el = ELEMENT_CONFIG["clusters"];
              const isActive = activeTab === "clusters";
              return (
                <button
                  key="clusters"
                  className={cn(
                    "group relative flex flex-col items-center justify-center rounded-xl border px-3.5 py-2 min-w-[68px] transition-all duration-200 backdrop-blur-sm",
                    isActive
                      ? el.activeColor + " " + el.glow + " scale-105 border-opacity-100"
                      : el.color + " hover:scale-105 hover:border-opacity-80 border-opacity-40"
                  )}
                  onClick={() => {
                    setActiveTab("clusters");
                    if (domain !== "dfu") setDomain("dfu");
                  }}
                >
                  {isActive && <span className="absolute -bottom-1 left-1/2 h-1 w-6 -translate-x-1/2 rounded-full bg-current opacity-60" />}
                  <span className="text-[9px] leading-none self-end font-mono opacity-50">{el.number}</span>
                  <span className="text-xl font-black leading-tight font-mono tracking-tight">{el.symbol}</span>
                  <span className="text-[9px] font-medium leading-none tracking-wide uppercase opacity-70">{el.name}</span>
                </button>
              );
            })()}
            {/* DFU Analysis tab */}
            {(() => {
              const el = ELEMENT_CONFIG["dfuAnalysis"];
              const isActive = activeTab === "dfuAnalysis";
              return (
                <button
                  key="dfuAnalysis"
                  className={cn(
                    "flex flex-col items-center justify-center rounded-lg border-2 px-3 py-1.5 min-w-[64px] transition-all",
                    isActive
                      ? el.activeColor + " shadow-md ring-2 ring-white/40"
                      : el.color + " opacity-80 hover:opacity-100 hover:shadow-sm"
                  )}
                  onClick={() => setActiveTab("dfuAnalysis")}
                >
                  <span className="text-[10px] leading-none self-start font-mono opacity-70">{el.number}</span>
                  <span className="text-lg font-bold leading-tight font-mono">{el.symbol}</span>
                  <span className="text-[10px] leading-none">{el.name}</span>
                </button>
              );
            })()}
            {/* Accuracy tab */}
            {(() => {
              const el = ELEMENT_CONFIG["accuracy"];
              const isActive = activeTab === "accuracy";
              return (
                <button
                  className={cn(
                    "group relative flex flex-col items-center justify-center rounded-xl border px-3.5 py-2 min-w-[68px] transition-all duration-200 backdrop-blur-sm",
                    isActive
                      ? el.activeColor + " " + el.glow + " scale-105 border-opacity-100"
                      : el.color + " hover:scale-105 hover:border-opacity-80 border-opacity-40"
                  )}
                  onClick={() => setActiveTab("accuracy")}
                >
                  {isActive && <span className="absolute -bottom-1 left-1/2 h-1 w-6 -translate-x-1/2 rounded-full bg-current opacity-60" />}
                  <span className="text-[9px] leading-none self-end font-mono opacity-50">{el.number}</span>
                  <span className="text-xl font-black leading-tight font-mono tracking-tight">{el.symbol}</span>
                  <span className="text-[9px] font-medium leading-none tracking-wide uppercase opacity-70">{el.name}</span>
                </button>
              );
            })()}
            {/* Market Intelligence tab */}
            {(() => {
              const el = ELEMENT_CONFIG["intel"];
              const isActive = activeTab === "intel";
              return (
                <button
                  className={cn(
                    "flex flex-col items-center justify-center rounded-lg border-2 px-3 py-1.5 min-w-[64px] transition-all",
                    isActive
                      ? el.activeColor + " shadow-md ring-2 ring-white/40"
                      : el.color + " opacity-80 hover:opacity-100 hover:shadow-sm"
                  )}
                  onClick={() => setActiveTab("intel")}
                >
                  <span className="text-[10px] leading-none self-start font-mono opacity-70">{el.number}</span>
                  <span className="text-lg font-bold leading-tight font-mono">{el.symbol}</span>
                  <span className="text-[10px] leading-none">{el.name}</span>
                </button>
              );
            })()}
          </div>
        </div>
      </section>

      {error ? (
        <Card className="mt-4 border-red-200 bg-red-50">
          <CardContent className="pt-4 text-sm text-red-700">{error}</CardContent>
        </Card>
      ) : null}

      {/* Explorer TABLE dropdown merged into the data grid card below */}

      {activeTab === "clusters" ? (
        <Card className="mt-4 animate-fade-in">
          <CardHeader>
            <CardTitle className="text-base">DFU Clustering</CardTitle>
            <CardDescription>
              Filter by demand-pattern cluster.{" "}
              {clusterSource === "ml"
                ? "ML pipeline clusters from KMeans."
                : "Source clusters from dfu.txt."}
            </CardDescription>
            <div className="flex flex-wrap items-end gap-3">
              <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Source
                <select
                  className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  value={clusterSource}
                  onChange={(e) => {
                    setClusterSource(e.target.value as "ml" | "source");
                    setSelectedCluster("");
                    setOffset(0);
                  }}
                >
                  <option value="ml">ML Pipeline</option>
                  <option value="source">Source (dfu.txt)</option>
                </select>
              </label>
              {clusterSummary.length > 0 ? (
                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Cluster
                  <select
                    className="h-9 w-full min-w-[200px] rounded-md border border-input bg-background px-3 text-sm"
                    value={selectedCluster}
                    onChange={(e) => {
                      setOffset(0);
                      setSelectedCluster(e.target.value);
                    }}
                  >
                    <option value="">All Clusters</option>
                    {clusterSummary.map((c) => (
                      <option key={c.label} value={c.label}>
                        {c.label} ({formatCompactNumber(c.count)})
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {clusterSummary.length > 0 ? (
              <>
                <p className="text-xs uppercase tracking-wide text-muted-foreground">Cluster summary &mdash; {clusterSummary.length} clusters, {formatCompactNumber(clusterSummary.reduce((s, c) => s + c.count, 0))} DFUs assigned</p>
                <div className="max-h-[320px] overflow-y-auto rounded-md border border-input">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-muted bg-muted/30">
                        <TableHead className="text-xs">Cluster</TableHead>
                        <TableHead className="text-xs text-right">DFUs</TableHead>
                        <TableHead className="text-xs text-right">%</TableHead>
                        <TableHead className="text-xs text-right">Avg demand</TableHead>
                        <TableHead className="text-xs text-right">CV</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {clusterSummary.map((c) => (
                        <TableRow
                          key={c.label}
                          className={cn(
                            "cursor-pointer transition-colors hover:bg-muted/40",
                            selectedCluster === c.label && "bg-primary/10 font-semibold"
                          )}
                          onClick={() => {
                            setOffset(0);
                            setSelectedCluster(selectedCluster === c.label ? "" : c.label);
                          }}
                        >
                          <TableCell className="font-medium text-sm">{c.label}</TableCell>
                          <TableCell className="text-right text-sm tabular-nums">{formatNumber(c.count)}</TableCell>
                          <TableCell className="text-right text-sm tabular-nums">{formatNumber(c.pct_of_total)}%</TableCell>
                          <TableCell className="text-right text-sm tabular-nums">{formatNumber(c.avg_demand)}</TableCell>
                          <TableCell className="text-right text-sm tabular-nums">{formatNumber(c.cv_demand)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
                <p className="text-xs text-muted-foreground">Click a row or use the dropdown above to filter the table below.</p>

                {/* Model metadata (ML source only) */}
                {clusterSource === "ml" && clusterMeta?.optimal_k ? (
                  <div className="mt-3 flex flex-wrap gap-3 text-xs">
                    <span className="rounded bg-muted px-2 py-1">K = {clusterMeta.optimal_k}</span>
                    <span className="rounded bg-muted px-2 py-1">Silhouette = {clusterMeta.silhouette_score?.toFixed(4)}</span>
                    <span className="rounded bg-muted px-2 py-1">Inertia = {formatCompactNumber(clusterMeta.inertia ?? 0)}</span>
                  </div>
                ) : null}

                {/* Visualization toggle (ML source only) */}
                {clusterSource === "ml" ? (
                  <>
                    <button
                      className="mt-2 text-xs text-primary underline underline-offset-2 hover:text-primary/80"
                      onClick={() => setShowClusterViz(!showClusterViz)}
                    >
                      {showClusterViz ? "Hide visualizations" : "Show cluster visualizations"}
                    </button>
                    {showClusterViz ? (
                      <div className="mt-2 grid gap-4 md:grid-cols-2">
                        <div>
                          <p className="mb-1 text-xs font-semibold text-muted-foreground">K Selection (Elbow / Silhouette / Gap)</p>
                          <img
                            src="/domains/dfu/clusters/visualization/k_selection_plots.png"
                            alt="K Selection Plots"
                            className="w-full rounded-md border"
                            onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                          />
                        </div>
                        <div>
                          <p className="mb-1 text-xs font-semibold text-muted-foreground">Cluster Visualization (2D PCA)</p>
                          <img
                            src="/domains/dfu/clusters/visualization/cluster_visualization.png"
                            alt="Cluster PCA Visualization"
                            className="w-full rounded-md border"
                            onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                          />
                        </div>
                      </div>
                    ) : null}
                  </>
                ) : null}
              </>
            ) : (
              <p className="text-sm text-muted-foreground">
                No cluster assignments yet. Run the clustering pipeline: <code className="rounded bg-muted px-1">make cluster-features</code>, then <code className="rounded bg-muted px-1">make cluster-train</code>, <code className="rounded bg-muted px-1">make cluster-label</code>, and <code className="rounded bg-muted px-1">make cluster-update</code> to see clusters here.
              </p>
            )}
          </CardContent>
        </Card>
      ) : null}

      {activeTab === "accuracy" ? (
        <section className="mt-4">
          <Card className="animate-fade-in">
            <CardHeader>
              <div className="flex items-center gap-2">
                <ChartColumn className="h-5 w-5" />
                <CardTitle className="text-base">Accuracy Comparison</CardTitle>
              </div>
              <CardDescription>Compare forecast accuracy across models by DFU attribute. Uses pre-aggregated views for fast results.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="flex flex-wrap items-end gap-3">
                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Slice by
                  <select
                    className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                    value={sliceGroupBy}
                    onChange={(e) => setSliceGroupBy(e.target.value)}
                    disabled={loadingSlice}
                  >
                    <option value="cluster_assignment">Cluster (Business Label)</option>
                    <option value="ml_cluster">Cluster (ML)</option>
                    <option value="supplier_desc">Supplier</option>
                    <option value="abc_vol">ABC Volume</option>
                    <option value="region">Region</option>
                    <option value="brand_desc">Brand</option>
                    <option value="dfu_execution_lag">Execution Lag</option>
                    <option value="month_start">Month</option>
                  </select>
                </label>
                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Lag Filter
                  <select
                    className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                    value={sliceLag}
                    onChange={(e) => setSliceLag(Number(e.target.value))}
                    disabled={loadingSlice}
                  >
                    <option value={-1}>Execution Lag (per DFU)</option>
                    <option value={0}>Lag 0 (same month)</option>
                    <option value={1}>Lag 1</option>
                    <option value={2}>Lag 2</option>
                    <option value={3}>Lag 3</option>
                    <option value={4}>Lag 4</option>
                  </select>
                </label>
                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Models (comma-separated, blank = all)
                  <input
                    className="h-9 w-52 rounded-md border border-input bg-background px-3 text-sm"
                    placeholder="e.g. lgbm_global,external"
                    value={sliceModels}
                    onChange={(e) => setSliceModels(e.target.value)}
                    disabled={loadingSlice}
                  />
                </label>
                {sliceGroupBy !== "month_start" ? (
                  <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    KPI Window
                    <select
                      className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                      value={sliceMonths}
                      onChange={(e) => setSliceMonths(Number(e.target.value))}
                      disabled={loadingSlice}
                    >
                      {Array.from({ length: 12 }, (_, idx) => idx + 1).map((m) => (
                        <option key={m} value={m}>{m} month{m > 1 ? "s" : ""}</option>
                      ))}
                    </select>
                  </label>
                ) : null}
                <label className="flex items-center gap-1.5 self-end pb-1.5 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    className="h-3.5 w-3.5 rounded border-input accent-indigo-700"
                    checked={commonDfus}
                    onChange={() => setCommonDfus((v) => !v)}
                    disabled={loadingSlice}
                  />
                  <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground whitespace-nowrap">Common DFUs Only</span>
                </label>
                {commonDfus && commonDfuCount != null && dfuCounts ? (
                  <div className="flex items-center gap-2 self-end pb-1.5 text-xs text-muted-foreground tabular-nums">
                    <Badge variant="secondary" className="font-mono text-[10px]">{commonDfuCount.toLocaleString()} common</Badge>
                    {Object.entries(dfuCounts).map(([m, cnt]) => (
                      <span key={m} className="font-mono">{m}: {cnt.toLocaleString()}</span>
                    ))}
                  </div>
                ) : null}
                {loadingSlice ? (
                  <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                ) : null}
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">KPIs</span>
                {ACCURACY_KPI_OPTIONS.map((opt) => {
                  const checked = sliceKpis.includes(opt.key);
                  const isLast = sliceKpis.length === 1 && checked;
                  return (
                    <label key={opt.key} className="flex items-center gap-1.5 text-sm cursor-pointer select-none">
                      <input
                        type="checkbox"
                        className="h-3.5 w-3.5 rounded border-input accent-indigo-700"
                        checked={checked}
                        disabled={isLast}
                        onChange={() => {
                          setSliceKpis((prev) =>
                            prev.includes(opt.key)
                              ? prev.filter((k) => k !== opt.key)
                              : [...prev, opt.key]
                          );
                        }}
                      />
                      {opt.label}
                    </label>
                  );
                })}
              </div>

              {sliceData.length > 0 ? (() => {
                const allModels = Array.from(
                  new Set(sliceData.flatMap((r) => Object.keys(r.by_model)))
                ).sort();
                return (
                  <div className="space-y-2">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">
                      Model Comparison — {sliceData.length} {sliceGroupBy.replace(/_/g, " ")} bucket(s)
                    </p>
                    <div className="max-h-[400px] overflow-auto rounded-md border border-input">
                      <Table>
                        <TableHeader>
                          <TableRow className="border-muted bg-muted/30">
                            <TableHead className="text-xs sticky left-0 bg-muted/30">
                              {titleCase(sliceGroupBy)}
                            </TableHead>
                            {allModels.flatMap((m) =>
                              ACCURACY_KPI_OPTIONS
                                .filter((k) => sliceKpis.includes(k.key))
                                .map((k) => (
                                  <TableHead key={`${m}-${k.key}`} className="text-xs text-right">{m} {k.label}</TableHead>
                                ))
                            )}
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {sliceData.map((row) => {
                            const accValues = allModels
                              .map((m) => row.by_model[m]?.accuracy_pct)
                              .filter((v): v is number => v !== null && v !== undefined);
                            const bestAcc = accValues.length > 0 ? Math.max(...accValues) : null;
                            return (
                              <TableRow key={row.bucket} className="hover:bg-muted/30">
                                <TableCell className="sticky left-0 bg-background font-medium text-sm">{row.bucket}</TableCell>
                                {allModels.flatMap((m) => {
                                  const kpi = row.by_model[m];
                                  return ACCURACY_KPI_OPTIONS
                                    .filter((k) => sliceKpis.includes(k.key))
                                    .map((k) => {
                                      const val = kpi?.[k.key as keyof AccuracyKpis] as number | null | undefined;
                                      const isBestAcc = k.key === "accuracy_pct" && val !== null && val !== undefined && val === bestAcc;
                                      const isBadBias = k.key === "bias" && val !== null && val !== undefined && Math.abs(val) > 0.15;
                                      let display: string;
                                      if (val === null || val === undefined) {
                                        display = "-";
                                      } else if (k.format === "pct") {
                                        display = formatPercent(val);
                                      } else if (k.format === "bias") {
                                        display = `${(val * 100).toFixed(1)}%`;
                                      } else {
                                        display = Number(val).toLocaleString(undefined, { maximumFractionDigits: 0 });
                                      }
                                      return (
                                        <TableCell
                                          key={`${m}-${k.key}`}
                                          className={cn(
                                            "text-right text-sm tabular-nums",
                                            isBestAcc ? "font-bold text-indigo-700" : "",
                                            isBadBias ? "text-red-600" : "",
                                          )}
                                        >
                                          {display}
                                        </TableCell>
                                      );
                                    });
                                })}
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                    </div>
                    <p className="text-xs text-muted-foreground">Bold = best accuracy for that row. Red bias = |bias| &gt; 15%.</p>
                  </div>
                );
              })() : (
                !loadingSlice ? (
                  <p className="text-sm text-muted-foreground">
                    No data. Run <code className="rounded bg-muted px-1">make backtest-load</code> to populate the accuracy views.
                  </p>
                ) : null
              )}

              {lagCurveData.length > 0 ? (() => {
                const lagModels = Array.from(
                  new Set(lagCurveData.flatMap((p) => Object.keys(p.by_model)))
                ).sort();
                const activeLagMetric = sliceKpis.includes(lagCurveMetric) ? lagCurveMetric : sliceKpis[0];
                const lagMetricOpt = ACCURACY_KPI_OPTIONS.find((k) => k.key === activeLagMetric);
                const chartData = lagCurveData.map((p) => {
                  const row: Record<string, number | string> = { lag: `Lag ${p.lag}` };
                  for (const m of lagModels) {
                    const val = p.by_model[m]?.[activeLagMetric as keyof AccuracyKpis];
                    if (val !== null && val !== undefined) row[m] = val as number;
                  }
                  return row;
                });
                const fmtIsPct = lagMetricOpt?.format === "pct";
                const fmtIsBias = lagMetricOpt?.format === "bias";
                const yFormatter = (v: number) =>
                  fmtIsPct ? `${Number(v).toFixed(0)}%`
                  : fmtIsBias ? `${(Number(v) * 100).toFixed(0)}%`
                  : Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
                const tooltipFormatter = (v: number) =>
                  fmtIsPct ? `${Number(v).toFixed(1)}%`
                  : fmtIsBias ? `${(Number(v) * 100).toFixed(1)}%`
                  : Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
                return (
                  <div className="space-y-2">
                    <div className="flex items-center gap-3">
                      <p className="text-xs uppercase tracking-wide text-muted-foreground">
                        {lagMetricOpt?.label ?? "KPI"} by Lag Horizon
                      </p>
                      <select
                        className="h-7 rounded-md border border-input bg-background px-2 text-xs"
                        value={activeLagMetric}
                        onChange={(e) => setLagCurveMetric(e.target.value)}
                      >
                        {ACCURACY_KPI_OPTIONS.filter((k) => sliceKpis.includes(k.key)).map((k) => (
                          <option key={k.key} value={k.key}>{k.label}</option>
                        ))}
                      </select>
                    </div>
                    <ResponsiveContainer width="100%" height={220}>
                      <LineChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis dataKey="lag" tick={{ fontSize: 11 }} />
                        <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11 }} tickFormatter={yFormatter} />
                        <Tooltip formatter={tooltipFormatter} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {lagModels.map((m, i) => (
                          <Line
                            key={m}
                            type="monotone"
                            dataKey={m}
                            stroke={TREND_COLORS[i % TREND_COLORS.length]}
                            strokeWidth={2}
                            dot={{ r: 4 }}
                            connectNulls
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                );
              })() : null}
            </CardContent>
          </Card>

          {/* Champion Selection panel (feature15) */}
          <Card className="animate-fade-in mt-4">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Trophy className="h-5 w-5" />
                <CardTitle className="text-base">Champion Selection</CardTitle>
              </div>
              <CardDescription>Pick the best model per DFU based on forecast accuracy. Configure which models compete, then run the selection.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              {competitionConfig ? (
                <>
                  <div className="space-y-3">
                    <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Competing Models</span>
                    <div className="flex flex-wrap gap-3">
                      {availableModels.filter((m) => m !== competitionConfig.champion_model_id && m !== "ceiling").map((m) => {
                        const checked = competitionConfig.models.includes(m);
                        const isLast = competitionConfig.models.length <= 2 && checked;
                        return (
                          <label key={m} className="flex items-center gap-1.5 text-sm cursor-pointer select-none">
                            <input
                              type="checkbox"
                              className="h-3.5 w-3.5 rounded border-input accent-indigo-700"
                              checked={checked}
                              disabled={isLast || runningCompetition}
                              onChange={() => {
                                setCompetitionConfig((prev) => {
                                  if (!prev) return prev;
                                  const next = checked
                                    ? prev.models.filter((x) => x !== m)
                                    : [...prev.models, m];
                                  return { ...prev, models: next };
                                });
                              }}
                            />
                            <span className="font-mono text-xs">{m}</span>
                          </label>
                        );
                      })}
                    </div>
                  </div>

                  <div className="flex flex-wrap items-end gap-3">
                    <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      Metric
                      <select
                        className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                        value={competitionConfig.metric}
                        onChange={(e) => setCompetitionConfig((prev) => prev ? { ...prev, metric: e.target.value } : prev)}
                        disabled={runningCompetition}
                      >
                        <option value="wape">WAPE (Lowest Wins)</option>
                        <option value="accuracy_pct">Accuracy % (Highest Wins)</option>
                      </select>
                    </label>
                    <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      Lag
                      <select
                        className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                        value={competitionConfig.lag}
                        onChange={(e) => setCompetitionConfig((prev) => prev ? { ...prev, lag: e.target.value } : prev)}
                        disabled={runningCompetition}
                      >
                        <option value="execution">Execution Lag (per DFU)</option>
                        <option value="0">Lag 0 (same month)</option>
                        <option value="1">Lag 1</option>
                        <option value="2">Lag 2</option>
                        <option value="3">Lag 3</option>
                        <option value="4">Lag 4</option>
                      </select>
                    </label>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={savingConfig || runningCompetition}
                      onClick={() => {
                        setSavingConfig(true);
                        fetch("/competition/config", {
                          method: "PUT",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify(competitionConfig),
                        })
                          .then((r) => { if (!r.ok) throw new Error("Save failed"); })
                          .finally(() => setSavingConfig(false));
                      }}
                    >
                      {savingConfig ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : null}
                      Save Config
                    </Button>
                    <Button
                      size="sm"
                      disabled={runningCompetition || competitionConfig.models.length < 2}
                      onClick={() => {
                        setRunningCompetition(true);
                        // Save config first, then run
                        fetch("/competition/config", {
                          method: "PUT",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify(competitionConfig),
                        })
                          .then(() => fetch("/competition/run", { method: "POST" }))
                          .then((r) => { if (!r.ok) throw new Error("Run failed"); return r.json(); })
                          .then((summary) => {
                            setChampionSummary(summary);
                            // Re-fetch accuracy slice data (champion model now in DB)
                            const sliceParams = new URLSearchParams({ group_by: sliceGroupBy, lag: String(sliceLag) });
                            if (sliceModels.trim()) sliceParams.set("models", sliceModels.trim());
                            fetch(`/forecast/accuracy/slice?${sliceParams}`)
                              .then((r) => r.json())
                              .then((p) => setSliceData(p.rows || []));
                          })
                          .catch(() => {/* ignore */})
                          .finally(() => setRunningCompetition(false));
                      }}
                    >
                      {runningCompetition ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Trophy className="mr-1 h-3 w-3" />}
                      Run Competition
                    </Button>
                  </div>
                </>
              ) : (
                <p className="text-sm text-muted-foreground">Loading competition config...</p>
              )}

              {/* Results summary */}
              {championSummary ? (
                <div className="space-y-3 rounded-lg border bg-muted/40 p-4">
                  <div className="flex flex-wrap items-center gap-4">
                    <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Results</span>
                    <span className="text-xs text-muted-foreground">
                      Last run: {new Date(championSummary.run_ts).toLocaleString()}
                    </span>
                  </div>
                  {/* Champion KPI cards */}
                  <div className="flex flex-wrap gap-4 text-sm">
                    <div className="rounded-md border bg-card px-3 py-2">
                      <p className="text-xs text-muted-foreground">DFUs Evaluated</p>
                      <p className="text-lg font-bold tabular-nums">{championSummary.total_dfus.toLocaleString()}</p>
                    </div>
                    <div className="rounded-md border bg-card px-3 py-2">
                      <p className="text-xs text-muted-foreground">Champion Accuracy</p>
                      <p className="text-lg font-bold tabular-nums text-indigo-700">
                        {championSummary.overall_champion_accuracy_pct != null
                          ? `${championSummary.overall_champion_accuracy_pct.toFixed(2)}%`
                          : "-"}
                      </p>
                    </div>
                    <div className="rounded-md border bg-card px-3 py-2">
                      <p className="text-xs text-muted-foreground">Champion WAPE</p>
                      <p className="text-lg font-bold tabular-nums">
                        {championSummary.overall_champion_wape != null
                          ? `${championSummary.overall_champion_wape.toFixed(2)}%`
                          : "-"}
                      </p>
                    </div>
                    <div className="rounded-md border bg-card px-3 py-2">
                      <p className="text-xs text-muted-foreground">Champion Rows</p>
                      <p className="text-lg font-bold tabular-nums">{championSummary.total_champion_rows.toLocaleString()}</p>
                    </div>
                  </div>

                  {/* Ceiling (Oracle) KPI cards */}
                  {championSummary.overall_ceiling_accuracy_pct != null && (
                    <div className="flex flex-wrap gap-4 text-sm">
                      <div className="rounded-md border bg-card px-3 py-2 border-emerald-200">
                        <p className="text-xs text-muted-foreground">Ceiling Accuracy <span className="text-[10px]">(oracle)</span></p>
                        <p className="text-lg font-bold tabular-nums text-emerald-700">
                          {championSummary.overall_ceiling_accuracy_pct.toFixed(2)}%
                        </p>
                      </div>
                      <div className="rounded-md border bg-card px-3 py-2 border-emerald-200">
                        <p className="text-xs text-muted-foreground">Ceiling WAPE <span className="text-[10px]">(oracle)</span></p>
                        <p className="text-lg font-bold tabular-nums text-emerald-700">
                          {championSummary.overall_ceiling_wape != null
                            ? `${championSummary.overall_ceiling_wape.toFixed(2)}%`
                            : "-"}
                        </p>
                      </div>
                      {championSummary.total_ceiling_rows != null && (
                        <div className="rounded-md border bg-card px-3 py-2 border-emerald-200">
                          <p className="text-xs text-muted-foreground">Ceiling Rows</p>
                          <p className="text-lg font-bold tabular-nums">{championSummary.total_ceiling_rows.toLocaleString()}</p>
                        </div>
                      )}
                      {/* Gap indicator: how far champion is from ceiling */}
                      {championSummary.overall_champion_accuracy_pct != null && (
                        <div className="rounded-md border bg-card px-3 py-2 border-amber-200">
                          <p className="text-xs text-muted-foreground">Gap to Ceiling</p>
                          <p className="text-lg font-bold tabular-nums text-amber-700">
                            {(championSummary.overall_ceiling_accuracy_pct - championSummary.overall_champion_accuracy_pct).toFixed(2)} pp
                          </p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Champion model wins bar chart */}
                  <div className="space-y-1.5">
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Champion Model Wins (best model per DFU overall)</p>
                    {Object.entries(championSummary.model_wins).map(([model, wins]) => {
                      const pct = championSummary.total_dfus > 0 ? (wins / championSummary.total_dfus) * 100 : 0;
                      return (
                        <div key={model} className="flex items-center gap-2 text-sm">
                          <span className="w-40 truncate font-mono text-xs text-right">{model}</span>
                          <div className="flex-1 h-5 rounded bg-muted overflow-hidden">
                            <div
                              className="h-full rounded bg-indigo-500 transition-all"
                              style={{ width: `${Math.max(pct, 1)}%` }}
                            />
                          </div>
                          <span className="w-24 text-xs tabular-nums text-muted-foreground">
                            {wins.toLocaleString()} ({pct.toFixed(1)}%)
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  {/* Ceiling model wins bar chart */}
                  {championSummary.ceiling_model_wins && Object.keys(championSummary.ceiling_model_wins).length > 0 && (
                    <div className="space-y-1.5">
                      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Ceiling Model Wins — Oracle (best model per DFU per month)</p>
                      {(() => {
                        const totalCeil = Object.values(championSummary.ceiling_model_wins!).reduce((a, b) => a + b, 0);
                        return Object.entries(championSummary.ceiling_model_wins!).map(([model, wins]) => {
                          const pct = totalCeil > 0 ? (wins / totalCeil) * 100 : 0;
                          return (
                            <div key={model} className="flex items-center gap-2 text-sm">
                              <span className="w-40 truncate font-mono text-xs text-right">{model}</span>
                              <div className="flex-1 h-5 rounded bg-muted overflow-hidden">
                                <div
                                  className="h-full rounded bg-emerald-500 transition-all"
                                  style={{ width: `${Math.max(pct, 1)}%` }}
                                />
                              </div>
                              <span className="w-24 text-xs tabular-nums text-muted-foreground">
                                {wins.toLocaleString()} ({pct.toFixed(1)}%)
                              </span>
                            </div>
                          );
                        });
                      })()}
                    </div>
                  )}
                </div>
              ) : null}
            </CardContent>
          </Card>
        </section>
      ) : null}

      {activeTab === "intel" ? (
        <Card className="mt-4 animate-fade-in">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Globe className="h-5 w-5" />
              <CardTitle className="text-base">Market Intelligence</CardTitle>
            </div>
            <CardDescription>
              Select a product and location to generate an AI-powered market briefing with web search results and demographic context.
            </CardDescription>
            <div className="grid gap-2 md:grid-cols-[1fr_1fr_auto]">
              <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Item (item_no)
                <Input
                  className="h-9"
                  placeholder="Search for item..."
                  list="mi-item-suggest"
                  value={miItemFilter}
                  onChange={(e) => setMiItemFilter(e.target.value)}
                />
                <datalist id="mi-item-suggest">
                  {miItemSuggestions.map((val) => (
                    <option key={val} value={val} />
                  ))}
                </datalist>
              </label>
              <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Location (location_id)
                <Input
                  className="h-9"
                  placeholder="Search for location..."
                  list="mi-location-suggest"
                  value={miLocationFilter}
                  onChange={(e) => setMiLocationFilter(e.target.value)}
                />
                <datalist id="mi-location-suggest">
                  {miLocationSuggestions.map((val) => (
                    <option key={val} value={val} />
                  ))}
                </datalist>
              </label>
              <div className="flex items-end">
                <Button
                  onClick={generateMarketIntel}
                  disabled={!miItemFilter.trim() || !miLocationFilter.trim() || miLoading}
                  className="h-9"
                >
                  {miLoading ? (
                    <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Generating...</>
                  ) : (
                    <><Send className="mr-2 h-4 w-4" /> Generate Briefing</>
                  )}
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {miError ? (
              <Card className="border-red-200 bg-red-50">
                <CardContent className="pt-4 text-sm text-red-700">{miError}</CardContent>
              </Card>
            ) : null}

            {miLoading ? (
              <div className="flex flex-col items-center justify-center py-12 gap-3">
                <div className="flex flex-col items-center justify-center rounded-lg border-2 border-sky-300 bg-sky-50 px-4 py-2 shadow-md animate-pulse-glow">
                  <span className="text-[9px] leading-none self-start font-mono text-sky-500 opacity-70">6</span>
                  <span className="text-lg font-bold leading-tight font-mono text-sky-900">Mi</span>
                  <span className="text-[9px] leading-none text-sky-600">Loading</span>
                </div>
                <span className="text-sm text-muted-foreground">Searching the web and generating market briefing...</span>
              </div>
            ) : null}

            {miResult && !miLoading ? (
              <>
                {/* Product + Location context badges */}
                <div className="flex flex-wrap gap-2">
                  {miResult.item_desc ? <Badge variant="secondary">{miResult.item_desc}</Badge> : null}
                  {miResult.brand_name ? <Badge variant="secondary">{miResult.brand_name}</Badge> : null}
                  {miResult.category ? <Badge variant="secondary">{miResult.category}</Badge> : null}
                  {miResult.state_id ? <Badge variant="outline">State: {miResult.state_id}</Badge> : null}
                  {miResult.site_desc ? <Badge variant="outline">{miResult.site_desc}</Badge> : null}
                </div>

                {/* Search Results Cards */}
                {miResult.search_results.length > 0 ? (
                  <div>
                    <p className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
                      Web Search Results ({miResult.search_results.length})
                    </p>
                    <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
                      {miResult.search_results.map((sr, idx) => (
                        <Card key={idx} className="border-muted bg-muted/10 shadow-none">
                          <CardContent className="pt-3 pb-3">
                            <a href={sr.link} target="_blank" rel="noopener noreferrer"
                               className="text-sm font-medium text-blue-700 hover:underline line-clamp-2">
                              {sr.title}
                            </a>
                            <p className="mt-1 text-xs text-muted-foreground line-clamp-3">{sr.snippet}</p>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>
                ) : null}

                {/* Narrative Story */}
                <Card className="border-sky-200 bg-sky-50/30">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Market Intelligence Briefing</CardTitle>
                    <CardDescription className="text-xs">
                      Generated {new Date(miResult.generated_at).toLocaleString()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="prose prose-sm max-w-none text-sm whitespace-pre-wrap">
                      {miResult.narrative}
                    </div>
                  </CardContent>
                </Card>
              </>
            ) : null}
          </CardContent>
        </Card>
      ) : null}

      {activeTab === "dfuAnalysis" ? (
        <section className="mt-4 grid gap-4 [&>*]:min-w-0 xl:grid-cols-1">
          <Card className="animate-fade-in">
            <CardHeader className="space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <CardTitle className="text-base">DFU Analysis</CardTitle>
                  <CardDescription>
                    {dfuMode === "item_location"
                      ? "Sales + multi-model forecasts for a specific DFU (item @ location)"
                      : dfuMode === "all_items_at_location"
                        ? "Aggregated sales + forecasts across all items at a location"
                        : "Aggregated sales + forecasts for an item across all locations"}
                  </CardDescription>
                </div>
                <Button variant="outline" size="sm" onClick={() => { setDfuData(null); setDfuAutoSampled(false); setDfuItem(""); setDfuLocation(""); }}>
                  <RefreshCcw className="mr-1 h-4 w-4" /> Reset
                </Button>
              </div>

              {/* Row 1: Analysis scope selector */}
              <div className="grid gap-2 md:grid-cols-3">
                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Analysis Scope
                  <select
                    className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                    value={dfuMode}
                    onChange={(e) => setDfuMode(e.target.value as DfuAnalysisMode)}
                  >
                    <option value="item_location">Item @ Location (single DFU)</option>
                    <option value="all_items_at_location">All Items @ Location</option>
                    <option value="item_at_all_locations">Item @ All Locations</option>
                  </select>
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Points
                    <select
                      className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                      value={dfuPoints}
                      onChange={(e) => setDfuPoints(Number(e.target.value))}
                    >
                      {[12, 24, 36, 48, 60].map((v) => (
                        <option key={v} value={v}>{v}</option>
                      ))}
                    </select>
                  </label>
                  <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    KPI Window
                    <select
                      className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                      value={dfuKpiMonths}
                      onChange={(e) => setDfuKpiMonths(Number(e.target.value))}
                    >
                      {Array.from({ length: 12 }, (_, i) => i + 1).map((m) => (
                        <option key={m} value={m}>{m} mo</option>
                      ))}
                    </select>
                  </label>
                </div>
              </div>

              {/* Row 2: Item + Location filters */}
              <div className="grid gap-2 md:grid-cols-2">
                {dfuMode !== "all_items_at_location" ? (
                  <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Item (dmdunit)
                    <Input
                      className="h-9"
                      placeholder="Type to search items..."
                      list="dfu-analysis-item-suggest"
                      value={dfuItem}
                      onChange={(e) => setDfuItem(e.target.value)}
                    />
                    <datalist id="dfu-analysis-item-suggest">
                      {dfuItemSuggestions.map((val) => (
                        <option key={val} value={val} />
                      ))}
                    </datalist>
                  </label>
                ) : (
                  <div className="flex items-end">
                    <p className="pb-2 text-xs text-muted-foreground italic">Item: All (aggregated at location level)</p>
                  </div>
                )}
                {dfuMode !== "item_at_all_locations" ? (
                  <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Location (loc)
                    <Input
                      className="h-9"
                      placeholder="Type to search locations..."
                      list="dfu-analysis-loc-suggest"
                      value={dfuLocation}
                      onChange={(e) => setDfuLocation(e.target.value)}
                    />
                    <datalist id="dfu-analysis-loc-suggest">
                      {dfuLocationSuggestions.map((val) => (
                        <option key={val} value={val} />
                      ))}
                    </datalist>
                  </label>
                ) : (
                  <div className="flex items-end">
                    <p className="pb-2 text-xs text-muted-foreground italic">Location: All (aggregated at item level)</p>
                  </div>
                )}
              </div>
            </CardHeader>

            <CardContent className="space-y-4">
              {/* DFU Attributes */}
              {dfuData && dfuData.dfu_attributes && dfuData.dfu_attributes.length > 0 && (
                <details className="group rounded-md border border-input bg-background">
                  <summary className="cursor-pointer select-none px-3 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground">
                    DFU Attributes ({dfuData.dfu_attributes.length} {dfuData.dfu_attributes.length === 1 ? "record" : "records"})
                    <span className="ml-1 text-[10px] text-muted-foreground group-open:hidden">+ expand</span>
                  </summary>
                  <div className="border-t border-input px-3 py-2 space-y-3">
                    {dfuData.dfu_attributes.map((attrs, dfuIdx) => (
                      <div key={dfuIdx}>
                        {dfuData.dfu_attributes.length > 1 && (
                          <p className="mb-1 text-xs font-medium text-foreground">
                            {attrs.dmdunit} / {attrs.dmdgroup} @ {attrs.loc}
                          </p>
                        )}
                        <div className="grid grid-cols-2 gap-x-6 gap-y-0.5 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
                          {Object.entries(attrs).map(([key, val]) => (
                            <div key={key} className="flex items-baseline gap-1 text-xs truncate">
                              <span className="font-medium text-muted-foreground shrink-0">{titleCase(key)}:</span>
                              <span className="text-foreground truncate" title={val ?? "—"}>{val ?? "—"}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </details>
              )}

              {/* Measure toggles */}
              {dfuData && dfuData.series.length > 0 ? (
                <>
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Visible Measures</span>
                      {(() => {
                        const allKeys = new Set(["tothist_dmd", "qty_shipped", "qty_ordered", ...dfuData.models.map((m) => `forecast_${m}`)]);
                        const allSelected = [...allKeys].every((k) => dfuVisibleSeries.has(k));
                        return (
                          <button
                            className="text-xs font-medium text-primary hover:underline"
                            onClick={() => setDfuVisibleSeries(allSelected ? new Set() : allKeys)}
                          >
                            {allSelected ? "Deselect All" : "Select All"}
                          </button>
                        );
                      })()}
                    </div>
                    <div className="flex flex-wrap gap-x-4 gap-y-1 rounded-md border border-input bg-background p-2">
                      {(
                        [
                          { key: "tothist_dmd", label: "Sale Qty (external)", color: DFU_SALES_COLORS.tothist_dmd },
                          { key: "qty_shipped", label: "Qty Shipped", color: DFU_SALES_COLORS.qty_shipped },
                          { key: "qty_ordered", label: "Qty Ordered", color: DFU_SALES_COLORS.qty_ordered },
                        ] as { key: string; label: string; color: string }[]
                      ).map(({ key, label, color }) => (
                        <label key={key} className="flex items-center gap-2 text-xs font-medium">
                          <Checkbox
                            checked={dfuVisibleSeries.has(key)}
                            onCheckedChange={(v) => {
                              setDfuVisibleSeries((prev) => {
                                const next = new Set(prev);
                                if (v) next.add(key); else next.delete(key);
                                return next;
                              });
                            }}
                          />
                          <span className="flex items-center gap-1">
                            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
                            {label}
                          </span>
                        </label>
                      ))}
                      {dfuData.models.map((model, idx) => {
                        const seriesKey = `forecast_${model}`;
                        return (
                          <label key={model} className="flex items-center gap-2 text-xs font-medium">
                            <Checkbox
                              checked={dfuVisibleSeries.has(seriesKey)}
                              onCheckedChange={(v) => {
                                setDfuVisibleSeries((prev) => {
                                  const next = new Set(prev);
                                  if (v) next.add(seriesKey); else next.delete(seriesKey);
                                  return next;
                                });
                              }}
                            />
                            <span className="flex items-center gap-1">
                              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: dfuModelColor(model, idx) }} />
                              {model}
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  </div>

                  {/* Time range */}
                  <div className="flex flex-wrap items-end gap-3">
                    <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      From
                      <select
                        className="h-8 w-36 rounded-md border border-input bg-background px-2 text-sm"
                        value={dfuTimeStart || dfuMonths[0] || ""}
                        onChange={(e) => setDfuTimeStart(e.target.value)}
                      >
                        {dfuMonths.map((m) => (
                          <option key={m} value={m}>{m}</option>
                        ))}
                      </select>
                    </label>
                    <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      To
                      <select
                        className="h-8 w-36 rounded-md border border-input bg-background px-2 text-sm"
                        value={dfuTimeEnd || dfuMonths[dfuMonths.length - 1] || ""}
                        onChange={(e) => setDfuTimeEnd(e.target.value)}
                      >
                        {dfuMonths.map((m) => (
                          <option key={m} value={m}>{m}</option>
                        ))}
                      </select>
                    </label>
                    <button
                      className="h-8 rounded-md border border-input bg-background px-3 text-xs font-medium text-muted-foreground hover:text-foreground"
                      onClick={() => { setDfuTimeStart(""); setDfuTimeEnd(""); }}
                    >
                      Show All
                    </button>
                    <button
                      className="h-8 rounded-md border border-input bg-background px-3 text-xs font-medium text-muted-foreground hover:text-foreground"
                      onClick={() => { setDfuTimeStart(dfuDefaultStart); setDfuTimeEnd(""); }}
                    >
                      Default
                    </button>
                  </div>

                  {/* Chart */}
                  <Card className="min-w-0 border-muted shadow-none">
                    <CardHeader className="pb-0">
                      <CardTitle className="flex items-center gap-2 text-sm">
                        <ChartColumn className="h-4 w-4" /> Sales vs Forecast Overlay
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="h-[380px] pt-2">
                      <div className="h-full overflow-x-scroll overflow-y-hidden pb-2 [scrollbar-gutter:stable]">
                        <div className="h-full" style={{ minWidth: `${Math.max(1200, dfuFilteredSeries.length * 100)}px` }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={dfuFilteredSeries} margin={{ top: 8, right: 16, left: 18, bottom: 8 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
                              <XAxis dataKey="month" />
                              <YAxis yAxisId="left" width={84} tickFormatter={formatCompactNumber} tickMargin={10} />
                              <Tooltip
                                formatter={(value: number, name: string) => [
                                  formatNumber(Number.isFinite(Number(value)) ? Number(value) : null),
                                  String(name),
                                ]}
                              />
                              <Legend />
                              {dfuVisibleSeries.has("tothist_dmd") ? (
                                <Line
                                  type="monotone"
                                  dataKey="tothist_dmd"
                                  yAxisId="left"
                                  name="Sale Qty (external)"
                                  stroke={DFU_SALES_COLORS.tothist_dmd}
                                  strokeWidth={2.5}
                                  dot={false}
                                  activeDot={{ r: 5 }}
                                />
                              ) : null}
                              {dfuVisibleSeries.has("qty_shipped") ? (
                                <Line
                                  type="monotone"
                                  dataKey="qty_shipped"
                                  yAxisId="left"
                                  name="Qty Shipped"
                                  stroke={DFU_SALES_COLORS.qty_shipped}
                                  strokeWidth={2}
                                  dot={false}
                                  activeDot={{ r: 4 }}
                                />
                              ) : null}
                              {dfuVisibleSeries.has("qty_ordered") ? (
                                <Line
                                  type="monotone"
                                  dataKey="qty_ordered"
                                  yAxisId="left"
                                  name="Qty Ordered"
                                  stroke={DFU_SALES_COLORS.qty_ordered}
                                  strokeWidth={2}
                                  dot={false}
                                  activeDot={{ r: 4 }}
                                />
                              ) : null}
                              {dfuData.models
                                .filter((m) => dfuVisibleSeries.has(`forecast_${m}`))
                                .map((model, idx) => (
                                  <Line
                                    key={model}
                                    type="monotone"
                                    dataKey={`forecast_${model}`}
                                    yAxisId="left"
                                    name={model}
                                    stroke={dfuModelColor(model, idx)}
                                    strokeWidth={model === "champion" ? 2.5 : 1.5}
                                    strokeDasharray={model === "champion" ? undefined : "5 3"}
                                    dot={false}
                                    activeDot={{ r: 4 }}
                                  />
                                ))}
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* KPI Cards per model */}
                  {Object.keys(dfuKpis).length > 0 ? (
                    <div className="space-y-2">
                      <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Model KPIs ({dfuKpiMonths}-month window)</span>
                      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                        {dfuData.models
                          .filter((m) => dfuVisibleSeries.has(`forecast_${m}`) && dfuKpis[m])
                          .map((model) => {
                            const kpi = dfuKpis[model];
                            const colorIdx = dfuData!.models.indexOf(model) + 1;
                            return (
                              <Card key={model} className="border-muted bg-muted/20 shadow-none">
                                <CardContent className="pt-4">
                                  <div className="flex items-center gap-2 mb-2">
                                    <span className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: TREND_COLORS[colorIdx % TREND_COLORS.length] }} />
                                    <p className="text-xs uppercase tracking-wide text-muted-foreground font-semibold">{model}</p>
                                    <span className="text-[10px] text-muted-foreground ml-auto">{kpi.months_covered} mo</span>
                                  </div>
                                  <div className="grid grid-cols-5 gap-2">
                                    <div>
                                      <p className="text-[10px] uppercase text-muted-foreground">Accuracy</p>
                                      <p className="text-sm font-semibold tabular-nums">{formatPercent(kpi.accuracy_pct)}</p>
                                    </div>
                                    <div>
                                      <p className="text-[10px] uppercase text-muted-foreground">WAPE</p>
                                      <p className="text-sm font-semibold tabular-nums">{formatPercent(kpi.wape)}</p>
                                    </div>
                                    <div>
                                      <p className="text-[10px] uppercase text-muted-foreground">Bias</p>
                                      <p className="text-sm font-semibold tabular-nums">{formatNumber(kpi.bias)}</p>
                                    </div>
                                    <div>
                                      <p className="text-[10px] uppercase text-muted-foreground">Fcst</p>
                                      <p className="text-sm font-semibold tabular-nums">{formatCompactNumber(kpi.sum_forecast)}</p>
                                    </div>
                                    <div>
                                      <p className="text-[10px] uppercase text-muted-foreground">Actual</p>
                                      <p className="text-sm font-semibold tabular-nums">{formatCompactNumber(kpi.sum_actual)}</p>
                                    </div>
                                  </div>
                                </CardContent>
                              </Card>
                            );
                          })}
                      </div>
                    </div>
                  ) : null}
                </>
              ) : dfuLoading ? (
                <div className="flex h-[320px] items-center justify-center">
                  <div className="flex flex-col items-center gap-2">
                    <div className="flex flex-col items-center justify-center rounded-lg border-2 border-cyan-300 bg-cyan-50 px-4 py-2 shadow-md animate-pulse-glow">
                      <span className="text-[9px] leading-none self-start font-mono text-cyan-500 opacity-70">6</span>
                      <span className="text-lg font-bold leading-tight font-mono text-cyan-900">Da</span>
                      <span className="text-[9px] leading-none text-cyan-600">Loading</span>
                    </div>
                    <span className="text-xs text-muted-foreground">Fetching DFU analysis...</span>
                  </div>
                </div>
              ) : (
                <div className="flex h-[320px] items-center justify-center text-sm text-muted-foreground">
                  {dfuMode === "item_location" && (!dfuItem.trim() || !dfuLocation.trim())
                    ? "Enter both item and location to view DFU analysis."
                    : dfuMode === "all_items_at_location" && !dfuLocation.trim()
                      ? "Enter a location to view aggregated analysis."
                      : dfuMode === "item_at_all_locations" && !dfuItem.trim()
                        ? "Enter an item to view aggregated analysis."
                        : "No data available for the selected filters."}
                </div>
              )}
            </CardContent>
          </Card>
        </section>
      ) : null}

      {activeTab === "explorer" ? <section className="mt-4 grid gap-4 [&>*]:min-w-0 xl:grid-cols-1">
        <Card className="animate-fade-in">
          <CardHeader className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <CardTitle className="text-base">Data Explorer</CardTitle>
                <CardDescription>Browse, search, filter, and sort.</CardDescription>
              </div>
              <div className="flex items-center gap-3">
                <select
                  className="h-9 w-[180px] rounded-md border border-input bg-background px-3 text-sm"
                  value={domain}
                  onChange={(e) => setDomain(e.target.value)}
                >
                  {DIMENSION_DOMAINS.map((d) => {
                    const el = ELEMENT_CONFIG[d];
                    return (
                      <option key={d} value={d}>{el?.name || titleCase(d)}</option>
                    );
                  })}
                </select>
                <Badge variant="secondary">
                  {meta ? `${titleCase(meta.name)} (${totalApproximate ? `${formatNumber(total - 1)}+` : formatNumber(total)})` : "Loading"}
                </Badge>
              </div>
            </div>

            <div className="grid gap-2 md:grid-cols-[2fr_120px_1fr]">
              <Input
                placeholder="Search across configured fields"
                value={search}
                onChange={(e) => {
                  setOffset(0);
                  setSearch(e.target.value);
                }}
                disabled={!meta}
              />
              <select
                className="h-9 rounded-md border border-input bg-background px-3 text-sm"
                value={limit}
                onChange={(e) => {
                  setOffset(0);
                  setLimit(Number(e.target.value));
                }}
              >
                {[50, 100, 250, 500].map((v) => (
                  <option key={v} value={v}>
                    {v}/page
                  </option>
                ))}
              </select>
              <Button variant="outline" onClick={() => setShowFieldPanel((v) => !v)}>
                <ChevronsUpDown className="mr-2 h-4 w-4" /> Fields
              </Button>
            </div>

            {showFieldPanel && meta ? (
              <div className="rounded-md border p-2">
                <div className="flex gap-2 mb-2">
                  <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => {
                    const all: Record<string, boolean> = {};
                    meta.columns.forEach((c) => { all[c] = true; });
                    setVisibleColumns(all);
                  }}>Select All</Button>
                  <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => {
                    const none: Record<string, boolean> = {};
                    meta.columns.forEach((c) => { none[c] = false; });
                    setVisibleColumns(none);
                  }}>Deselect All</Button>
                </div>
                <div className="grid max-h-40 grid-cols-2 gap-2 overflow-y-auto overflow-x-hidden lg:grid-cols-3">
                  {meta.columns.map((col) => (
                    <label key={col} className="flex items-center gap-2 text-sm">
                      <Checkbox
                        checked={visibleColumns[col] !== false}
                        onCheckedChange={(checked) => toggleColumn(col, checked === true)}
                      />
                      <span>{titleCase(col)}</span>
                    </label>
                  ))}
                </div>
              </div>
            ) : null}
          </CardHeader>

          <CardContent>
            <div className="relative">
              {/* Chemistry-themed loading overlay */}
              {loadingTable && (
                <div className="absolute inset-0 z-30 flex items-center justify-center rounded-md bg-background/70 backdrop-blur-[1px]">
                  <div className="flex flex-col items-center gap-2">
                    <div className="flex flex-col items-center justify-center rounded-lg border-2 border-indigo-300 bg-indigo-50 px-5 py-2.5 shadow-lg animate-pulse-glow">
                      <span className="text-[10px] leading-none self-start font-mono text-indigo-500 opacity-70">{ELEMENT_CONFIG[domain]?.number ?? 0}</span>
                      <span className="text-2xl font-bold leading-tight font-mono text-indigo-900">{ELEMENT_CONFIG[domain]?.symbol ?? "Ld"}</span>
                      <span className="text-[10px] leading-none text-indigo-600">Loading</span>
                    </div>
                    <span className="text-xs text-muted-foreground">Querying {titleCase(domain)}...</span>
                  </div>
                </div>
              )}
              <div className="max-h-[680px] overflow-x-scroll overflow-y-auto rounded-md border pb-2 [scrollbar-gutter:stable]">
                <Table style={{ minWidth: `${Math.max(visibleCols.length * 260, 1800)}px` }}>
                  <TableHeader className="sticky top-0 z-20 bg-muted/80 backdrop-blur">
                    <TableRow>
                      {visibleCols.map((col) => (
                        <TableHead key={col} className="min-w-[180px] bg-muted/70 align-top">
                          <Button
                            variant={sortBy === col ? "secondary" : "ghost"}
                            size="sm"
                            className="mb-1 h-7 w-full justify-between px-2"
                            onClick={() => toggleSort(col)}
                          >
                            <span>{titleCase(col)}</span>
                            {sortBy === col ? (
                              sortDir === "asc" ? <ArrowUpWideNarrow className="h-3.5 w-3.5" /> : <ArrowDownWideNarrow className="h-3.5 w-3.5" />
                            ) : (
                              <ChevronsUpDown className="h-3.5 w-3.5" />
                            )}
                          </Button>
                          <Input
                            className="h-7 text-xs"
                            placeholder="Filter (=exact)"
                            list={`col-suggest-${domain}-${col}`}
                            value={columnFilters[col] || ""}
                            onChange={(e) => {
                              setOffset(0);
                              setColumnFilters((prev) => ({ ...prev, [col]: e.target.value }));
                            }}
                          />
                          {(columnSuggestions[col]?.length ?? 0) > 0 && (
                            <datalist id={`col-suggest-${domain}-${col}`}>
                              {columnSuggestions[col].map((v) => (
                                <option key={v} value={v} />
                              ))}
                            </datalist>
                          )}
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {rows.length === 0 && !loadingTable ? (
                      <TableRow>
                        <TableCell colSpan={Math.max(visibleCols.length, 1)} className="h-24 text-center text-muted-foreground">
                          No records
                        </TableCell>
                      </TableRow>
                    ) : (
                      rows.map((row, idx) => (
                        <TableRow key={`row-${offset + idx}`}>
                          {visibleCols.map((col) => (
                            <TableCell key={`${offset + idx}-${col}`} className="whitespace-nowrap">
                              {formatCell(row[col])}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </div>
            </div>

            <div className="mt-3 flex items-center justify-between gap-2 text-sm">
              <span className="text-muted-foreground">
                Showing {start}-{end} of {totalApproximate ? `${formatNumber(total - 1)}+` : formatNumber(total)}
              </span>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={offset + limit >= total}
                  onClick={() => setOffset(offset + limit)}
                >
                  Next
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </section> : null}

      <section className="mt-4">
        <Card className="animate-fade-in">
          <CardHeader className="cursor-pointer" onClick={() => setChatOpen((v) => !v)}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5" />
                <CardTitle className="text-base">Chat with Planthium</CardTitle>
              </div>
              <Badge variant="outline">{chatOpen ? "Collapse" : "Expand"}</Badge>
            </div>
            <CardDescription>Ask questions about your data in plain English.</CardDescription>
          </CardHeader>
          {chatOpen ? (
            <CardContent className="space-y-3">
              <div className="max-h-[400px] min-h-[120px] overflow-y-auto rounded-md border bg-muted/10 p-3 space-y-3">
                {chatMessages.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No messages yet. Try asking: &quot;What are the top 10 items by total sales quantity?&quot;</p>
                ) : (
                  chatMessages.map((msg, idx) => (
                    <div key={idx} className={cn("rounded-lg px-3 py-2 text-sm", msg.role === "user" ? "ml-8 bg-indigo-100 text-indigo-900" : "mr-8 bg-white border shadow-sm")}>
                      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">{msg.role === "user" ? "You" : "Assistant"}</p>
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                      {msg.error ? <p className="mt-1 text-xs text-red-600">{msg.error}</p> : null}
                      {msg.sql ? (
                        <details className="mt-2">
                          <summary className="cursor-pointer text-xs font-medium text-muted-foreground">SQL Query</summary>
                          <pre className="mt-1 overflow-x-auto rounded bg-slate-100 p-2 text-xs">{msg.sql}</pre>
                        </details>
                      ) : null}
                      {msg.data && msg.data.length > 0 && msg.columns && msg.columns.length > 0 ? (
                        <div className="mt-2">
                          <div className="flex items-center gap-2 mb-1">
                            <Badge variant="secondary" className="text-xs">{msg.row_count ?? msg.data.length} row(s)</Badge>
                          </div>
                          <div className="max-h-[200px] overflow-auto rounded border">
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  {msg.columns.map((col) => (
                                    <TableHead key={col} className="text-xs whitespace-nowrap">{col}</TableHead>
                                  ))}
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {msg.data.slice(0, 10).map((row, rIdx) => (
                                  <TableRow key={rIdx}>
                                    {msg.columns!.map((col) => (
                                      <TableCell key={col} className="text-xs whitespace-nowrap">{formatCell(row[col])}</TableCell>
                                    ))}
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </div>
                          {(msg.row_count ?? 0) > 10 ? <p className="mt-1 text-xs text-muted-foreground">Showing 10 of {msg.row_count} rows.</p> : null}
                        </div>
                      ) : null}
                    </div>
                  ))
                )}
                {chatLoading ? (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" /> Thinking...
                  </div>
                ) : null}
                <div ref={chatEndRef} />
              </div>
              <div className="flex gap-2">
                <Input
                  placeholder="Ask a question about your data..."
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } }}
                  disabled={chatLoading}
                  className="flex-1"
                />
                <Button onClick={sendChat} disabled={chatLoading || !chatInput.trim()} size="sm">
                  <Send className="mr-1 h-4 w-4" /> Send
                </Button>
              </div>
            </CardContent>
          ) : null}
        </Card>
      </section>
    </main>
  );
}
