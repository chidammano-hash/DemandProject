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
import { ArrowDownWideNarrow, ArrowUpWideNarrow, ChartColumn, ChevronsUpDown, Loader2, MessageSquare, RefreshCcw, Send } from "lucide-react";

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
  limit: number;
  offset: number;
  [key: string]: unknown;
};

type AnalyticsSummary = {
  total_rows: number;
  metric_total: number;
  metric_avg: number | null;
  metric_min: number | null;
  metric_max: number | null;
  min_date: string | null;
  max_date: string | null;
};

type AnalyticsPoint = { x: string; y: number };

type DomainAnalytics = {
  config: {
    metric: string;
    trend_metrics?: string[];
    date_field: string;
    category_field: string;
    points: number;
    top_n: number;
    kpi_months?: number;
  };
  available: {
    metrics: string[];
    date_fields: string[];
    category_fields: string[];
  };
  summary: AnalyticsSummary;
  trend: AnalyticsPoint[];
  trend_multi?: Record<string, AnalyticsPoint[]>;
  top_categories: AnalyticsPoint[];
  kpis?: {
    months_window?: number | null;
    months_covered?: number | null;
    total_forecast?: number | null;
    total_actual?: number | null;
    abs_error?: number | null;
    bias?: number | null;
    wape_pct?: number | null;
    mape_pct?: number | null;
    accuracy_pct?: number | null;
  };
};

type SuggestPayload = {
  values?: string[];
};

type SamplePairPayload = {
  item?: string | null;
  location?: string | null;
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

function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState<T>(value);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setDebounced(value), delay);
    return () => { if (timer.current) clearTimeout(timer.current); };
  }, [value, delay]);
  return debounced;
}

const FALLBACK_DOMAINS = ["item", "location", "customer", "time", "dfu", "sales", "forecast"];
const ANALYTICS_ENABLED_DOMAINS = new Set(["sales", "forecast"]);
const EXCLUDED_TREND_FIELDS = new Set(["type", "lag", "execution_lag"]);
const FORECAST_ACCURACY_METRIC = "accuracy_pct";

const titleCase = (value: string): string =>
  value
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

const TREND_COLORS = ["#0f766e", "#f97316", "#1d4ed8", "#a21caf", "#b45309", "#0891b2"];

const numberFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const compactNumberFmt = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

function metricLabel(value: string): string {
  if (value === FORECAST_ACCURACY_METRIC) {
    return "Forecast Accuracy %";
  }
  return value === "__count__" ? "Count" : titleCase(value);
}

function trendMetricCandidates(fields: string[]): string[] {
  return fields.filter((field) => !EXCLUDED_TREND_FIELDS.has(field.toLowerCase()));
}

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

function updateDomainPath(domain: string) {
    const normalized = domain.toLowerCase();
  const url = new URL(window.location.href);
  url.searchParams.set("domain", normalized);
  window.history.replaceState(null, "", url);
}

export default function App() {
  const [domains, setDomains] = useState<string[]>([]);
  const [domain, setDomain] = useState<string>(getInitialDomain);

  const [meta, setMeta] = useState<DomainMeta | null>(null);
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [total, setTotal] = useState(0);

  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(100);
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [columnFilters, setColumnFilters] = useState<Record<string, string>>({});
  const [visibleColumns, setVisibleColumns] = useState<Record<string, boolean>>({});
  const [showFieldPanel, setShowFieldPanel] = useState(false);

  const debouncedSearch = useDebounce(search, 300);
  const debouncedColumnFilters = useDebounce(columnFilters, 300);

  const [analytics, setAnalytics] = useState<DomainAnalytics | null>(null);
  const [trendMetrics, setTrendMetrics] = useState<string[]>([]);
  const [trendSeries, setTrendSeries] = useState<Record<string, AnalyticsPoint[]>>({});
  const [points, setPoints] = useState(24);
  const [forecastKpiMonths, setForecastKpiMonths] = useState(12);
  const [itemFilter, setItemFilter] = useState("");
  const [locationFilter, setLocationFilter] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [autoSampledDomain, setAutoSampledDomain] = useState("");
  const [itemSuggestions, setItemSuggestions] = useState<string[]>([]);
  const [locationSuggestions, setLocationSuggestions] = useState<string[]>([]);

  const [loadingMeta, setLoadingMeta] = useState(false);
  const [loadingTable, setLoadingTable] = useState(false);
  const [loadingAnalytics, setLoadingAnalytics] = useState(false);
  const [error, setError] = useState<string>("");

  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const visibleCols = useMemo(() => {
    if (!meta) {
      return [];
    }
    return meta.columns.filter((col) => visibleColumns[col] !== false);
  }, [meta, visibleColumns]);

  const analyticsEnabled = ANALYTICS_ENABLED_DOMAINS.has(domain);
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
  const showFactFilters = (domain === "sales" || domain === "forecast") && Boolean(itemField) && Boolean(locationField);
  const trendMetricOptions = useMemo(() => {
    const base = trendMetricCandidates(meta?.numeric_fields || []);
    if (domain === "forecast") {
      return [...base, FORECAST_ACCURACY_METRIC];
    }
    return base;
  }, [meta, domain]);
  const activeDateField = useMemo(() => {
    if (!meta || meta.date_fields.length === 0) {
      return "";
    }
    const preferred = meta.date_fields.find((field) => field.toLowerCase() === "startdate");
    return preferred || meta.date_fields[0] || "";
  }, [meta]);
  const selectedTrendMetrics = useMemo(() => Array.from(new Set(trendMetrics.filter(Boolean))), [trendMetrics]);
  const primaryTrendMetric = selectedTrendMetrics[0] || trendMetricOptions[0] || "__count__";
  const itemListId = `item-suggest-${domain}`;
  const locationListId = `location-suggest-${domain}`;
  const formatPairFilterValue = (value: string): string => {
    const trimmed = value.trim();
    if (!trimmed) {
      return "";
    }
    return `=${trimmed}`;
  };
  const effectiveFilters = useMemo(() => {
    const out = Object.fromEntries(Object.entries(debouncedColumnFilters).filter(([, value]) => value.trim() !== ""));
    if (showFactFilters && itemFilter.trim() && itemField) {
      out[itemField] = formatPairFilterValue(itemFilter);
    }
    if (showFactFilters && locationFilter.trim() && locationField) {
      out[locationField] = formatPairFilterValue(locationFilter);
    }
    if (domain === "forecast" && selectedModel.trim()) {
      out["model_id"] = `=${selectedModel.trim()}`;
    }
    return out;
  }, [columnFilters, showFactFilters, itemFilter, locationFilter, itemField, locationField, domain, selectedModel]);

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
          setDomains(nextDomains);
          if (!nextDomains.includes(domain)) {
            setDomain(nextDomains[0]);
          }
        }
      } catch {
        if (!cancelled) {
          setDomains(FALLBACK_DOMAINS);
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
      setLoadingMeta(true);
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

        const filteredMetrics = trendMetricCandidates(payload.numeric_fields || []);
        const nextMetrics =
          domain === "forecast"
            ? filteredMetrics.length > 0
              ? [filteredMetrics[0], FORECAST_ACCURACY_METRIC]
              : [FORECAST_ACCURACY_METRIC]
            : filteredMetrics.length > 0
              ? [filteredMetrics[0]]
              : [];

        setMeta(payload);
        setOffset(0);
        setSearch("");
        setColumnFilters({});
        setSortBy(payload.default_sort);
        setSortDir("asc");
        setTrendMetrics(nextMetrics);
        setTrendSeries({});
        setItemFilter("");
        setLocationFilter("");
        setSelectedModel("");
        setAvailableModels([]);
        setForecastKpiMonths(12);
        setAutoSampledDomain("");
        setItemSuggestions([]);
        setLocationSuggestions([]);
        setVisibleColumns(Object.fromEntries(payload.columns.map((c) => [c, true])));
        updateDomainPath(domain);

        if (pageRes.ok) {
          const pagePl = (await pageRes.json()) as DomainPage;
          if (!cancelled) {
            setRows((pagePl[payload.plural] || []) as Record<string, unknown>[]);
            setTotal(Number(pagePl.total || 0));
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
          setLoadingMeta(false);
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
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load records");
          setRows([]);
          setTotal(0);
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
  }, [meta, domain, offset, limit, debouncedSearch, sortBy, sortDir, effectiveFilters]);

  useEffect(() => {
    if (!meta || !analyticsEnabled) {
      setAnalytics(null);
      setTrendSeries({});
      return;
    }

    let cancelled = false;

    async function loadAnalytics() {
      setLoadingAnalytics(true);
      try {
        const params = new URLSearchParams({
          q: debouncedSearch,
          metric: primaryTrendMetric,
          metrics: selectedTrendMetrics.join(","),
          date_field: activeDateField,
          category_field: "",
          points: String(points),
          top_n: "12",
        });
        if (domain === "forecast") {
          params.set("kpi_months", String(forecastKpiMonths));
        }

        if (Object.keys(effectiveFilters).length > 0) {
          params.set("filters", JSON.stringify(effectiveFilters));
        }

        const res = await fetch(`/domains/${encodeURIComponent(domain)}/analytics?${params.toString()}`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const payload = (await res.json()) as DomainAnalytics;

        if (!cancelled) {
          setAnalytics(payload);
          if (payload.trend_multi && Object.keys(payload.trend_multi).length > 0) {
            setTrendSeries(payload.trend_multi);
          } else if (payload.config.metric && payload.trend) {
            setTrendSeries({ [payload.config.metric]: payload.trend });
          } else {
            setTrendSeries({});
          }
        }
      } catch {
        if (!cancelled) {
          setAnalytics(null);
          setTrendSeries({});
        }
      } finally {
        if (!cancelled) {
          setLoadingAnalytics(false);
        }
      }
    }

    loadAnalytics();
    return () => {
      cancelled = true;
    };
  }, [meta, analyticsEnabled, domain, primaryTrendMetric, selectedTrendMetrics, activeDateField, points, forecastKpiMonths, debouncedSearch, effectiveFilters]);

  useEffect(() => {
    if (trendMetricOptions.length === 0) {
      setTrendMetrics([]);
      return;
    }
    setTrendMetrics((prev) => {
      const filtered = Array.from(new Set(prev.filter((metric) => trendMetricOptions.includes(metric))));
      if (filtered.length > 0) {
        return filtered;
      }
      return [trendMetricOptions[0]];
    });
  }, [trendMetricOptions]);

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

  useEffect(() => {
    if (!showFactFilters || !itemField) {
      setItemSuggestions([]);
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({
          field: itemField,
          q: itemFilter.trim(),
          limit: "12",
        });
        if (locationFilter.trim() && locationField) {
          params.set("filters", JSON.stringify({ [locationField]: formatPairFilterValue(locationFilter) }));
        }
        const res = await fetch(`/domains/${encodeURIComponent(domain)}/suggest?${params.toString()}`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const payload = (await res.json()) as SuggestPayload;
        const values = Array.from(new Set((payload.values || []).filter(Boolean))).slice(0, 12);
        if (!cancelled) {
          setItemSuggestions(values);
        }
      } catch {
        if (!cancelled) {
          setItemSuggestions([]);
        }
      }
    }, 180);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [showFactFilters, itemField, itemFilter, domain, locationField, locationFilter]);

  useEffect(() => {
    if (!showFactFilters || !locationField) {
      setLocationSuggestions([]);
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({
          field: locationField,
          q: locationFilter.trim(),
          limit: "12",
        });
        if (itemFilter.trim() && itemField) {
          params.set("filters", JSON.stringify({ [itemField]: formatPairFilterValue(itemFilter) }));
        }
        const res = await fetch(`/domains/${encodeURIComponent(domain)}/suggest?${params.toString()}`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const payload = (await res.json()) as SuggestPayload;
        const values = Array.from(new Set((payload.values || []).filter(Boolean))).slice(0, 12);
        if (!cancelled) {
          setLocationSuggestions(values);
        }
      } catch {
        if (!cancelled) {
          setLocationSuggestions([]);
        }
      }
    }, 180);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [showFactFilters, locationField, locationFilter, domain, itemField, itemFilter]);

  const trendChartData = useMemo(() => {
    const bucket = new Map<string, Record<string, number | string>>();
    for (const m of selectedTrendMetrics) {
      const pointsForMetric = trendSeries[m] || [];
      for (const p of pointsForMetric) {
        const key = String(p.x);
        const row = bucket.get(key) || { x: key };
        row[m] = p.y;
        bucket.set(key, row);
      }
    }
    return Array.from(bucket.values()).sort((a, b) => String(a.x).localeCompare(String(b.x)));
  }, [selectedTrendMetrics, trendSeries]);
  const fallbackTrendData = useMemo(() => {
    if (!analytics?.trend?.length || !primaryTrendMetric) {
      return [];
    }
    return analytics.trend.map((p) => ({ x: p.x, [primaryTrendMetric]: p.y }));
  }, [analytics, primaryTrendMetric]);
  const effectiveTrendData = trendChartData.length > 0 ? trendChartData : fallbackTrendData;
  const renderedTrendMetrics = useMemo(() => {
    const available = selectedTrendMetrics.filter((m) =>
      effectiveTrendData.some((row) => typeof row[m] === "number" && Number.isFinite(Number(row[m]))),
    );
    if (available.length > 0) {
      return available;
    }
    return primaryTrendMetric ? [primaryTrendMetric] : selectedTrendMetrics;
  }, [effectiveTrendData, selectedTrendMetrics, primaryTrendMetric]);
  const isTrendBusy = loadingAnalytics;
  const hasAccuracyTrend = renderedTrendMetrics.includes(FORECAST_ACCURACY_METRIC);
  const hasNonAccuracyTrend = renderedTrendMetrics.some((m) => m !== FORECAST_ACCURACY_METRIC);
  const forecastKpis = domain === "forecast" ? analytics?.kpis : undefined;

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

  return (
    <main className="mx-auto w-full max-w-[1800px] min-w-0 overflow-x-hidden p-4 md:p-6">
      <section className="animate-fade-in rounded-xl border border-white/20 bg-gradient-to-r from-slate-900/90 via-teal-900/80 to-cyan-800/80 p-4 text-white shadow-xl">
        <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight md:text-3xl">Demand Studio</h1>
            <p className="text-sm text-teal-50/90 md:text-base">Shadcn-powered analytics UI for demand datasets.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {domains.map((d) => (
              <Button
                key={d}
                variant={d === domain ? "secondary" : "outline"}
                className={d === domain ? "bg-orange-200 text-slate-900 hover:bg-orange-100" : "border-white/35 bg-transparent text-white hover:bg-white/10"}
                onClick={() => setDomain(d)}
              >
                {titleCase(d)}
              </Button>
            ))}
          </div>
        </div>
      </section>

      {error ? (
        <Card className="mt-4 border-red-200 bg-red-50">
          <CardContent className="pt-4 text-sm text-red-700">{error}</CardContent>
        </Card>
      ) : null}

      <section className={cn("mt-4 grid gap-4 [&>*]:min-w-0", analyticsEnabled ? "2xl:grid-cols-[1.15fr_1fr]" : "xl:grid-cols-1")}>
        {analyticsEnabled ? <Card className="animate-fade-in">
          <CardHeader className="space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <CardTitle className="text-base">Analytics</CardTitle>
                <CardDescription>
                  {meta ? `${titleCase(meta.name)} metrics` : "Loading domain metadata"}
                </CardDescription>
              </div>
              <Button variant="outline" size="sm" onClick={() => setColumnFilters((prev) => ({ ...prev }))}>
                <RefreshCcw className="mr-1 h-4 w-4" /> Refresh
              </Button>
            </div>
            <div className="grid gap-2 md:grid-cols-2">
              <div className="space-y-1">
                <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Trend Measures</span>
                <div className="max-h-[84px] overflow-y-auto rounded-md border border-input bg-background p-2">
                  <div className="grid grid-cols-1 gap-1">
                    {trendMetricOptions.map((m) => {
                      const checked = selectedTrendMetrics.includes(m);
                      return (
                        <label key={m} className="flex items-center gap-2 text-xs font-medium normal-case tracking-normal text-foreground">
                          <Checkbox
                            checked={checked}
                            onCheckedChange={(v) => {
                              const nextChecked = v === true;
                              if (nextChecked) {
                                if (!checked) {
                                  setTrendMetrics((prev) => [...prev, m]);
                                }
                                return;
                              }
                              const remaining = selectedTrendMetrics.filter((x) => x !== m);
                              if (remaining.length === 0) {
                                return;
                              }
                              setTrendMetrics(remaining);
                            }}
                          />
                          <span>{metricLabel(m)}</span>
                        </label>
                      );
                    })}
                  </div>
                </div>
              </div>
              <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Trend Points
                <select
                  className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  value={points}
                  onChange={(e) => setPoints(Number(e.target.value))}
                  disabled={loadingAnalytics}
                >
                  {[12, 24, 36, 60].map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {showFactFilters ? (
              <div className="grid gap-2 md:grid-cols-2">
                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Item ({itemField})
                  <Input
                    className="h-9"
                    placeholder="Filter item"
                    list={itemListId}
                    value={itemFilter}
                    onChange={(e) => {
                      setOffset(0);
                      setItemFilter(e.target.value);
                    }}
                  />
                  <datalist id={itemListId}>
                    {itemSuggestions.map((val) => (
                      <option key={val} value={val} />
                    ))}
                  </datalist>
                </label>
                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Location ({locationField})
                  <Input
                    className="h-9"
                    placeholder="Filter location"
                    list={locationListId}
                    value={locationFilter}
                    onChange={(e) => {
                      setOffset(0);
                      setLocationFilter(e.target.value);
                    }}
                  />
                  <datalist id={locationListId}>
                    {locationSuggestions.map((val) => (
                      <option key={val} value={val} />
                    ))}
                  </datalist>
                </label>
              </div>
            ) : null}
            {domain === "forecast" && availableModels.length > 0 ? (
              <div className="grid gap-2 md:grid-cols-2">
                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Forecast Model
                  <select
                    className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                    value={selectedModel}
                    onChange={(e) => {
                      setOffset(0);
                      setSelectedModel(e.target.value);
                    }}
                  >
                    <option value="">All Models</option>
                    {availableModels.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                </label>
              </div>
            ) : null}
          </CardHeader>

          <CardContent className="space-y-4">
            <Card className="border-muted bg-muted/20 shadow-none">
              <CardContent className="pt-4">
                <p className="text-xs uppercase tracking-wide text-muted-foreground">Rows</p>
                <p className="mt-1 text-lg font-semibold">{formatNumber(analytics?.summary.total_rows)}</p>
              </CardContent>
            </Card>
            {domain === "forecast" ? (
              <div className="space-y-3">
                <div className="flex flex-wrap items-end gap-2 rounded-md border border-input bg-muted/20 px-3 py-2">
                  <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Accuracy Window (Months)
                    <select
                      className="h-9 w-[140px] rounded-md border border-input bg-background px-3 text-sm"
                      value={forecastKpiMonths}
                      onChange={(e) => setForecastKpiMonths(Number(e.target.value))}
                      disabled={loadingAnalytics}
                    >
                      {Array.from({ length: 12 }, (_, idx) => idx + 1).map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  </label>
                  <p className="pb-1 text-xs text-muted-foreground">
                    Averaged across last {analytics?.kpis?.months_covered ?? 0} month(s) in current filter context.
                  </p>
                </div>
                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                <Card className="border-muted bg-muted/20 shadow-none">
                  <CardContent className="pt-4">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">Forecast Accuracy</p>
                    <p className="mt-1 text-lg font-semibold">{formatPercent(forecastKpis?.accuracy_pct)}</p>
                  </CardContent>
                </Card>
                <Card className="border-muted bg-muted/20 shadow-none">
                  <CardContent className="pt-4">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">Avg WAPE</p>
                    <p className="mt-1 text-lg font-semibold">{formatPercent(forecastKpis?.wape_pct)}</p>
                  </CardContent>
                </Card>
                <Card className="border-muted bg-muted/20 shadow-none">
                  <CardContent className="pt-4">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">Avg MAPE</p>
                    <p className="mt-1 text-lg font-semibold">{formatPercent(forecastKpis?.mape_pct)}</p>
                  </CardContent>
                </Card>
                <Card className="border-muted bg-muted/20 shadow-none">
                  <CardContent className="pt-4">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">Total Forecast</p>
                    <p className="mt-1 text-lg font-semibold">{formatNumber(forecastKpis?.total_forecast)}</p>
                  </CardContent>
                </Card>
                <Card className="border-muted bg-muted/20 shadow-none">
                  <CardContent className="pt-4">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">Total Actual</p>
                    <p className="mt-1 text-lg font-semibold">{formatNumber(forecastKpis?.total_actual)}</p>
                  </CardContent>
                </Card>
                <Card className="border-muted bg-muted/20 shadow-none">
                  <CardContent className="pt-4">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">Absolute Error</p>
                    <p className="mt-1 text-lg font-semibold">{formatNumber(forecastKpis?.abs_error)}</p>
                  </CardContent>
                </Card>
                <Card className="border-muted bg-muted/20 shadow-none">
                  <CardContent className="pt-4">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">Bias (Fcst/Hist)</p>
                    <p className="mt-1 text-lg font-semibold">{formatNumber(forecastKpis?.bias)}</p>
                  </CardContent>
                </Card>
              </div>
              </div>
            ) : null}
            <Card className="min-w-0 border-muted shadow-none">
              <CardHeader className="pb-0">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <ChartColumn className="h-4 w-4" /> Trend
                </CardTitle>
              </CardHeader>
              <CardContent className="h-[320px] pt-2">
                {isTrendBusy ? (
                  <div className="flex h-full items-center justify-center text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                  </div>
                ) : !activeDateField ? (
                  <div className="flex h-full items-center justify-center text-sm text-muted-foreground">No date field configured for this domain.</div>
                ) : effectiveTrendData.length ? (
                  <div className="h-full overflow-x-scroll overflow-y-hidden pb-2 [scrollbar-gutter:stable]">
                    <div className="h-full" style={{ minWidth: `${Math.max(1200, effectiveTrendData.length * 120)}px` }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={effectiveTrendData} margin={{ top: 8, right: 16, left: 18, bottom: 8 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
                          <XAxis dataKey="x" />
                          <YAxis yAxisId="left" hide={!hasNonAccuracyTrend} width={84} tickFormatter={formatCompactNumber} tickMargin={10} />
                          <YAxis yAxisId="right" hide={!hasAccuracyTrend} orientation="right" width={64} tickFormatter={(v) => `${formatNumber(Number(v))}%`} tickMargin={10} domain={[0, 100]} />
                          <Tooltip
                            formatter={(value, name) => {
                              const n = Number(value);
                              if (String(name).toLowerCase().includes("accuracy")) {
                                return [formatPercent(Number.isFinite(n) ? n : null), String(name)];
                              }
                              return [formatNumber(Number.isFinite(n) ? n : null), String(name)];
                            }}
                          />
                          <Legend />
                          {renderedTrendMetrics.map((m, idx) => (
                            <Line
                              key={m}
                              type="monotone"
                              dataKey={m}
                              yAxisId={m === FORECAST_ACCURACY_METRIC ? "right" : "left"}
                              name={metricLabel(m)}
                              stroke={TREND_COLORS[idx % TREND_COLORS.length]}
                              strokeWidth={2}
                              dot={effectiveTrendData.length <= 1 ? { r: 4 } : false}
                              activeDot={{ r: 5 }}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                    {(analytics?.summary.total_rows || 0) === 0 ? "No rows for current filters (item/location)." : "No trend points for selected measures."}
                  </div>
                )}
              </CardContent>
            </Card>
          </CardContent>
        </Card> : null}

        <Card className="animate-fade-in">
          <CardHeader className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <CardTitle className="text-base">Data Explorer</CardTitle>
                <CardDescription>Search, filter, sort, and select fields.</CardDescription>
              </div>
              <Badge variant="secondary">
                {meta ? `${titleCase(meta.name)} (${formatNumber(total)})` : "Loading"}
              </Badge>
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
              <div className="grid max-h-40 grid-cols-2 gap-2 overflow-y-auto overflow-x-hidden rounded-md border p-2 lg:grid-cols-3">
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
            ) : null}
          </CardHeader>

          <CardContent>
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
                          placeholder="Filter"
                          value={columnFilters[col] || ""}
                          onChange={(e) => {
                            setOffset(0);
                            setColumnFilters((prev) => ({ ...prev, [col]: e.target.value }));
                          }}
                        />
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {loadingTable ? (
                    <TableRow>
                      <TableCell colSpan={Math.max(visibleCols.length, 1)} className="h-24 text-center text-muted-foreground">
                        <Loader2 className="mx-auto h-4 w-4 animate-spin" />
                      </TableCell>
                    </TableRow>
                  ) : rows.length === 0 ? (
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

            <div className="mt-3 flex items-center justify-between gap-2 text-sm">
              <span className="text-muted-foreground">
                Showing {start}-{end} of {formatNumber(total)}
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
      </section>

      <section className="mt-4">
        <Card className="animate-fade-in">
          <CardHeader className="cursor-pointer" onClick={() => setChatOpen((v) => !v)}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5" />
                <CardTitle className="text-base">Chat with Demand Studio</CardTitle>
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
                    <div key={idx} className={cn("rounded-lg px-3 py-2 text-sm", msg.role === "user" ? "ml-8 bg-teal-100 text-teal-900" : "mr-8 bg-white border shadow-sm")}>
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
