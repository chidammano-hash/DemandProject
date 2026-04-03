import { useMemo, useRef, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
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

function fmtNum(n: number): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

const TILE_URL = "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png";
const LABEL_URL = "https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png";

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
  const geoRef = useRef(0);

  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.map(metric, groupBy, filters),
    queryFn: () => fetchCustomerAnalyticsMap(metric, groupBy, filters),
    staleTime: 5 * 60_000,
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
          `<b>${name}</b><br/>Customers: ${fmtNum(loc.customer_count)}<br/>Demand: ${fmtNum(loc.demand_qty)}<br/>Fill Rate: ${loc.fill_rate}%`,
        );
      }
    },
    [stateLookup],
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

  geoRef.current += 1;

  const bubbles = locations.filter((l) => l.lat != null && l.lon != null);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Customer Demand Map</CardTitle>
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
            {groupBy !== "state" &&
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
            <TileLayer url={LABEL_URL} />
          </MapContainer>
        )}
      </CardContent>
    </Card>
  );
}
