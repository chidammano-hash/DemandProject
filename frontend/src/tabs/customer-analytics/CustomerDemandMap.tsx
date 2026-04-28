import { useEffect, useMemo, useRef, useCallback } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Tooltip } from "react-leaflet";
import type { Layer, PathOptions } from "leaflet";
import "leaflet/dist/leaflet.css";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsMap,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters, MapLocation } from "@/api/queries/customer-analytics";
import usStatesGeo from "@/assets/us-states.json";
import { useDashboardFilter } from "./DashboardFilterContext";
import { ExportButtons } from "./ExportButtons";
import { formatInt as fmtNum } from "@/lib/formatters";

const statesGeoJSON = usStatesGeo as unknown as GeoJSON.FeatureCollection;

const STATE_NAME_TO_CODE: Record<string, string> = {
  Alabama: "AL", Alaska: "AK", Arizona: "AZ", Arkansas: "AR", California: "CA",
  Colorado: "CO", Connecticut: "CT", Delaware: "DE", Florida: "FL", Georgia: "GA",
  Hawaii: "HI", Idaho: "ID", Illinois: "IL", Indiana: "IN", Iowa: "IA",
  Kansas: "KS", Kentucky: "KY", Louisiana: "LA", Maine: "ME", Maryland: "MD",
  Massachusetts: "MA", Michigan: "MI", Minnesota: "MN", Mississippi: "MS",
  Missouri: "MO", Montana: "MT", Nebraska: "NE", Nevada: "NV",
  "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
  "North Carolina": "NC", "North Dakota": "ND", Ohio: "OH", Oklahoma: "OK",
  Oregon: "OR", Pennsylvania: "PA", "Rhode Island": "RI", "South Carolina": "SC",
  "South Dakota": "SD", Tennessee: "TN", Texas: "TX", Utah: "UT", Vermont: "VT",
  Virginia: "VA", Washington: "WA", "West Virginia": "WV", Wisconsin: "WI",
  Wyoming: "WY", "District of Columbia": "DC",
};

const SCALE = [
  [237, 242, 247], [204, 251, 241], [99, 226, 210],
  [20, 184, 166], [15, 118, 110], [19, 78, 74],
];

function choroplethColor(val: number, maxVal: number): string {
  if (val === 0 || maxVal === 0) return `rgb(${SCALE[0].join(",")})`;
  const t = Math.log(val + 1) / Math.log(maxVal + 1);
  const idx = Math.min(Math.floor(t * (SCALE.length - 1)) + 1, SCALE.length - 1);
  return `rgb(${SCALE[idx].join(",")})`;
}

function fillRateColor(fr: number): string {
  if (fr >= 95) return "#22c55e";
  if (fr >= 90) return "#84cc16";
  if (fr >= 85) return "#eab308";
  if (fr >= 80) return "#f97316";
  return "#ef4444";
}

function bubbleRadius(val: number, maxVal: number): number {
  if (maxVal <= 0) return 5;
  return 5 + 20 * Math.sqrt(val / maxVal);
}

// Lightweight grid clustering — aggregates markers into ~1° lat/lon cells
// (≈ 70 mile bins). One bubble per non-empty cell, sized by total demand,
// with a click-to-expand affordance handled by the existing pan/zoom UX.
// Avoids adding a marker-cluster npm dep at the cost of less polished UX.
const GRID_DEG = 1;
interface ClusteredBubble {
  lat: number;
  lon: number;
  count: number;
  total_demand: number;
  total_oos: number;
  fill_rate: number;
}
function clusterByGrid(points: MapLocation[]): ClusteredBubble[] {
  const cells = new Map<string, ClusteredBubble & { _wlat: number; _wlon: number; _wsum: number }>();
  for (const p of points) {
    if (p.lat == null || p.lon == null) continue;
    const cellLat = Math.round(p.lat / GRID_DEG) * GRID_DEG;
    const cellLon = Math.round(p.lon / GRID_DEG) * GRID_DEG;
    const key = `${cellLat}:${cellLon}`;
    const w = p.demand_qty || 1;
    const cur = cells.get(key);
    if (!cur) {
      cells.set(key, {
        lat: cellLat, lon: cellLon, count: 1,
        total_demand: p.demand_qty, total_oos: p.oos_qty,
        fill_rate: p.fill_rate,
        _wlat: p.lat * w, _wlon: p.lon * w, _wsum: w,
      });
    } else {
      cur.count += 1;
      cur.total_demand += p.demand_qty;
      cur.total_oos += p.oos_qty;
      cur._wlat += p.lat * w;
      cur._wlon += p.lon * w;
      cur._wsum += w;
      // weighted-mean fill rate by demand
      cur.fill_rate = cur.total_demand > 0
        ? Math.round(((cur.total_demand - cur.total_oos) / cur.total_demand) * 1000) / 10
        : cur.fill_rate;
    }
  }
  // Replace cell-center coords with demand-weighted centroid for nicer placement
  return Array.from(cells.values()).map((c) => ({
    lat: c._wsum > 0 ? c._wlat / c._wsum : c.lat,
    lon: c._wsum > 0 ? c._wlon / c._wsum : c.lon,
    count: c.count, total_demand: c.total_demand,
    total_oos: c.total_oos, fill_rate: c.fill_rate,
  }));
}

// fmtNum aliased to canonical formatInt — replaces local thousands-separator helper.

// Single composite tileset: doubles as base + labels in one request, halving
// network/tile-decode work. Previously stacked light_nolabels + light_only_labels.
const TILE_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png";

type MapMetric = "customer_count" | "demand_qty" | "sales_qty" | "oos_qty" | "fill_rate";

const METRIC_LABELS: Record<MapMetric, string> = {
  customer_count: "Customers",
  demand_qty: "Demand (cases)",
  sales_qty: "Sales (cases)",
  oos_qty: "OOS (cases)",
  fill_rate: "Fill Rate %",
};

interface Props {
  filters: CustomerAnalyticsFilters;
  metric: MapMetric;
  groupBy: "state" | "city" | "zip";
  onMetricChange: (m: MapMetric) => void;
  onGroupByChange: (g: "state" | "city" | "zip") => void;
}

export function CustomerDemandMap({ filters, metric, groupBy, onMetricChange, onGroupByChange }: Props) {
  const { dispatch } = useDashboardFilter();
  const geoRef = useRef(0);

  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.map(metric, groupBy, filters),
    queryFn: () => fetchCustomerAnalyticsMap(metric, groupBy, filters),
    staleTime: 60 * 60_000, // monthly data; pin to 1h to suppress thundering-herd refetches
    placeholderData: keepPreviousData, // keep prior chart visible during filter-change refetch
  });

  const locations = data?.locations ?? [];
  const maxVal = useMemo(() => {
    if (!locations.length) return 0;
    return Math.max(...locations.map((l) => l[metric] ?? l.demand_qty));
  }, [locations, metric]);

  const stateLookup = useMemo(() => {
    const m = new Map<string, MapLocation>();
    for (const loc of locations) {
      const key = (loc.state || loc.label || "").toUpperCase();
      m.set(key, loc);
    }
    return m;
  }, [locations]);

  const onEachFeature = useCallback(
    (feature: GeoJSON.Feature, layer: Layer) => {
      const name = feature.properties?.name ?? "";
      const code = STATE_NAME_TO_CODE[name] || name;
      const loc = stateLookup.get(code.toUpperCase());
      if (loc) {
        layer.bindTooltip(
          `<b>${name}</b><br/>Customers: ${fmtNum(loc.customer_count)}<br/>Demand: ${fmtNum(loc.demand_qty)}<br/>Sales: ${fmtNum(loc.sales_qty)}<br/>OOS: ${fmtNum(loc.oos_qty)}<br/>Fill Rate: ${loc.fill_rate}%`,
        );
      }
      layer.on("click", () => {
        const stateCode = STATE_NAME_TO_CODE[name] || name;
        dispatch({ type: "SET_STATE", payload: stateCode.toUpperCase() });
      });
    },
    [stateLookup, dispatch],
  );

  const geoStyle = useCallback(
    (feature?: GeoJSON.Feature): PathOptions => {
      if (!feature) return {};
      const name = feature.properties?.name ?? "";
      const code = STATE_NAME_TO_CODE[name] || name;
      const loc = stateLookup.get(code.toUpperCase());
      const val = loc ? (loc[metric] ?? loc.demand_qty) : 0;
      return {
        fillColor: choroplethColor(val, maxVal),
        fillOpacity: 0.7,
        color: "#fff",
        weight: 1,
      };
    },
    [stateLookup, maxVal, metric],
  );

  // Force GeoJSON to remount only when the styling inputs actually change.
  // Previously we incremented on EVERY render, which destroyed Leaflet's
  // layer caching and rebuilt the whole choropleth on each parent re-render.
  useEffect(() => {
    geoRef.current += 1;
  }, [stateLookup, maxVal, metric]);

  const bubbles = useMemo(
    () => locations.filter((l) => l.lat != null && l.lon != null),
    [locations],
  );

  // Cluster bubbles into ~1° grid cells when there are too many to render
  // smoothly as individual markers. 100 markers is the empirical knee on a
  // mid-spec laptop; below that, render individuals so tooltips read raw
  // city/zip names instead of grid cells.
  const clustered = useMemo(
    () => (bubbles.length > 100 ? clusterByGrid(bubbles) : null),
    [bubbles],
  );
  const maxClusterDemand = useMemo(
    () => (clustered ? Math.max(...clustered.map((c) => c.total_demand), 1) : 1),
    [clustered],
  );

  return (
    <Card aria-label="Customer demand map">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Customer Demand Map</CardTitle>
          <ExportButtons panelId="demand-map" getData={() => locations} />
        </div>
        <div className="flex gap-2 flex-wrap mt-1">
          <div className="flex gap-1">
            {(["state", "city", "zip"] as const).map((g) => (
              <button
                key={g}
                onClick={() => onGroupByChange(g)}
                className={`px-2 py-0.5 text-xs rounded ${groupBy === g ? "bg-teal-600 text-white" : "bg-gray-100 text-gray-600"}`}
              >
                {g}
              </button>
            ))}
          </div>
          <div className="flex gap-1">
            {(Object.keys(METRIC_LABELS) as MapMetric[]).map((m) => (
              <button
                key={m}
                onClick={() => onMetricChange(m)}
                className={`px-2 py-0.5 text-xs rounded ${metric === m ? "bg-indigo-600 text-white" : "bg-gray-100 text-gray-600"}`}
              >
                {METRIC_LABELS[m]}
              </button>
            ))}
          </div>
        </div>
        {data && (
          <p className="text-xs text-muted-foreground mt-1">
            {fmtNum(data.total_customers)} customers | {fmtNum(data.total_demand)} cases total demand
          </p>
        )}
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[480px] flex items-center justify-center text-sm text-muted-foreground">Loading map...</div>
        ) : locations.length === 0 ? (
          <div className="h-[480px] flex flex-col items-center justify-center text-sm text-muted-foreground gap-1">
            <span className="font-medium">No data for the selected filters</span>
            <span className="text-xs">Try a different item or widen the date range</span>
          </div>
        ) : (
          <MapContainer
            center={[39.5, -98.35]}
            zoom={4}
            style={{ height: 480, width: "100%" }}
            scrollWheelZoom={false}
          >
            <TileLayer url={TILE_URL} />
            <GeoJSON
              key={geoRef.current}
              data={statesGeoJSON}
              style={geoStyle}
              onEachFeature={onEachFeature}
            />
            {groupBy !== "state" && clustered &&
              clustered.map((c, i) => (
                <CircleMarker
                  key={`cluster-${i}`}
                  center={[c.lat, c.lon]}
                  radius={bubbleRadius(c.total_demand, maxClusterDemand)}
                  pathOptions={{ color: fillRateColor(c.fill_rate), fillColor: fillRateColor(c.fill_rate), fillOpacity: 0.7, weight: 1 }}
                >
                  <Tooltip>
                    <b>{c.count} locations</b>
                    <br />
                    Demand: {fmtNum(c.total_demand)} | Fill Rate: {c.fill_rate}%
                  </Tooltip>
                </CircleMarker>
              ))}
            {groupBy !== "state" && !clustered &&
              bubbles.map((loc, i) => (
                <CircleMarker
                  key={`${loc.label}-${i}`}
                  center={[loc.lat!, loc.lon!]}
                  radius={bubbleRadius(loc[metric] ?? loc.demand_qty, maxVal)}
                  pathOptions={{ color: fillRateColor(loc.fill_rate), fillColor: fillRateColor(loc.fill_rate), fillOpacity: 0.7, weight: 1 }}
                >
                  <Tooltip>
                    <b>{loc.label}</b>
                    {loc.state && `, ${loc.state}`}
                    <br />
                    Customers: {fmtNum(loc.customer_count)}
                    <br />
                    Demand: {fmtNum(loc.demand_qty)} | Fill Rate: {loc.fill_rate}%
                  </Tooltip>
                </CircleMarker>
              ))}
          </MapContainer>
        )}
      </CardContent>
    </Card>
  );
}
