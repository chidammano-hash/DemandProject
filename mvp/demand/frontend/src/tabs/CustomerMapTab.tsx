import { useState, useMemo, useEffect, useCallback, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Tooltip } from "react-leaflet";
import type { Layer, PathOptions } from "leaflet";
import "leaflet/dist/leaflet.css";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { queryKeys, fetchCustomerMap, STALE } from "@/api/queries";
import type { CustomerMapLocation } from "@/api/queries/core";
import usStatesGeo from "@/assets/us-states.json";

// ---------------------------------------------------------------------------
// US States GeoJSON (bundled static asset)
// ---------------------------------------------------------------------------

const statesGeoJSON = usStatesGeo as unknown as GeoJSON.FeatureCollection;

// Map full state names to 2-letter codes
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

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

/** 5-step sequential color scale: light gray → teal → blue → indigo → purple */
const CHOROPLETH_SCALE = [
  [237, 242, 247],  // gray-100 (no data)
  [204, 251, 241],  // teal-100
  [99, 226, 210],   // teal-300
  [20, 184, 166],   // teal-500
  [15, 118, 110],   // teal-700
  [19, 78, 74],     // teal-900
];

function choroplethColor(count: number, maxCount: number): string {
  if (count === 0 || maxCount === 0) return `rgb(${CHOROPLETH_SCALE[0].join(",")})`;
  // log scale for better visual spread
  const t = Math.log(count + 1) / Math.log(maxCount + 1);
  const idx = Math.min(Math.floor(t * (CHOROPLETH_SCALE.length - 1)) + 1, CHOROPLETH_SCALE.length - 1);
  const c = CHOROPLETH_SCALE[idx];
  return `rgb(${c.join(",")})`;
}

/** Circle marker colors for city/zip overlay */
function markerColor(count: number, maxCount: number): string {
  const t = maxCount > 0 ? Math.sqrt(count / maxCount) : 0;
  if (t < 0.25) return "#f97316";  // orange-500
  if (t < 0.5) return "#ef4444";   // red-500
  if (t < 0.75) return "#dc2626";  // red-600
  return "#991b1b";                 // red-800
}

function radius(count: number, maxCount: number): number {
  if (maxCount <= 0) return 4;
  return 4 + 18 * Math.sqrt(count / maxCount);
}

function fmtNum(n: number): string {
  return n.toLocaleString();
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

type GroupBy = "state" | "zip" | "city";

const TILE_URL = "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png";
const LABEL_URL = "https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png";
const TILE_ATTR =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>';

export function CustomerMapTab() {
  const [groupBy, setGroupBy] = useState<GroupBy>("state");
  const geoRef = useRef(0); // force GeoJSON re-render when data changes

  // Always fetch state data for the choropleth base
  const { data: stateData } = useQuery({
    queryKey: queryKeys.customerMap("state"),
    queryFn: () => fetchCustomerMap("state"),
    staleTime: STALE.FIVE_MIN,
  });

  const { data, isLoading } = useQuery({
    queryKey: queryKeys.customerMap(groupBy),
    queryFn: () => fetchCustomerMap(groupBy),
    staleTime: STALE.FIVE_MIN,
  });

  const locations = data?.locations ?? [];
  const total = data?.total ?? 0;
  const stateLocations = stateData?.locations ?? [];

  // Build state code → count lookup
  const stateCountMap = useMemo(() => {
    const m = new Map<string, number>();
    for (const loc of stateLocations) {
      m.set(loc.label.toUpperCase(), loc.customer_count);
    }
    return m;
  }, [stateLocations]);

  const stateMaxCount = useMemo(
    () => Math.max(...stateLocations.map((l) => l.customer_count), 1),
    [stateLocations],
  );

  // City/zip markers (only when not in state view)
  const markers = useMemo(() => {
    if (groupBy === "state") return [];
    return locations
      .filter(
        (l): l is CustomerMapLocation & { lat: number; lon: number } =>
          l.lat != null && l.lon != null,
      )
      .slice(0, 500); // limit markers for performance
  }, [locations, groupBy]);

  const markerMax = useMemo(
    () => Math.max(...markers.map((l) => l.customer_count), 1),
    [markers],
  );

  // GeoJSON style callback
  const geoStyle = useCallback(
    (feature?: GeoJSON.Feature): PathOptions => {
      const name = feature?.properties?.name ?? "";
      const code = STATE_NAME_TO_CODE[name] ?? "";
      const count = stateCountMap.get(code) ?? 0;
      return {
        fillColor: choroplethColor(count, stateMaxCount),
        fillOpacity: 0.85,
        color: "#ffffff",
        weight: 1.5,
        opacity: 1,
      };
    },
    [stateCountMap, stateMaxCount],
  );

  // GeoJSON tooltip callback
  const onEachFeature = useCallback(
    (feature: GeoJSON.Feature, layer: Layer) => {
      const name = feature.properties?.name ?? "";
      const code = STATE_NAME_TO_CODE[name] ?? "";
      const count = stateCountMap.get(code) ?? 0;
      layer.bindTooltip(
        `<strong>${name} (${code})</strong><br/>${fmtNum(count)} customers`,
        { sticky: true, className: "leaflet-tooltip-custom" },
      );
    },
    [stateCountMap],
  );

  // Force GeoJSON re-render when data changes (Leaflet caches styles)
  useEffect(() => { geoRef.current += 1; }, [stateCountMap]);

  const center: [number, number] = [39.5, -98.35];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold tracking-tight">Customer Map</h2>
          <p className="text-sm text-muted-foreground">
            Geographic distribution of {fmtNum(total)} customers across all locations.
          </p>
        </div>
        <div className="flex items-center gap-1 rounded-md border border-border bg-muted/40 p-0.5 text-xs">
          {(["state", "city", "zip"] as GroupBy[]).map((g) => (
            <button
              key={g}
              onClick={() => setGroupBy(g)}
              className={`rounded px-3 py-1 capitalize transition-colors ${
                groupBy === g
                  ? "bg-background font-medium text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {g}
            </button>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span>Customers:</span>
        <div className="flex h-3 w-40 overflow-hidden rounded-sm">
          {CHOROPLETH_SCALE.slice(1).map((c, i) => (
            <div key={i} className="flex-1" style={{ background: `rgb(${c.join(",")})` }} />
          ))}
        </div>
        <span>Low</span>
        <span className="ml-auto mr-0">High</span>
        {groupBy !== "state" && (
          <>
            <span className="ml-4 flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-orange-500" />
              {groupBy} overlay
            </span>
          </>
        )}
      </div>

      {/* Map + table */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">
            {groupBy === "state"
              ? "Customer density by state"
              : `Customer density by state + ${groupBy} overlay`}
          </CardTitle>
          <CardDescription>
            {isLoading
              ? "Loading..."
              : `${fmtNum(locations.length)} unique ${groupBy} values`}
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <div className="h-[540px] w-full overflow-hidden border-b border-border/40">
            <MapContainer
              center={center}
              zoom={4}
              scrollWheelZoom
              className="h-full w-full"
              style={{ background: "#f0f4f8" }}
            >
              {/* Base tiles (no labels — labels go on top of choropleth) */}
              <TileLayer attribution={TILE_ATTR} url={TILE_URL} />

              {/* Choropleth layer */}
              {statesGeoJSON && (
                <GeoJSON
                  key={geoRef.current}
                  data={statesGeoJSON}
                  style={geoStyle}
                  onEachFeature={onEachFeature}
                />
              )}

              {/* Labels on top of choropleth */}
              <TileLayer url={LABEL_URL} />

              {/* City/zip circle markers overlay */}
              {markers.map((loc) => (
                <CircleMarker
                  key={`${loc.label}-${loc.state ?? ""}`}
                  center={[loc.lat, loc.lon]}
                  radius={radius(loc.customer_count, markerMax)}
                  pathOptions={{
                    fillColor: markerColor(loc.customer_count, markerMax),
                    color: "#fff",
                    weight: 1,
                    fillOpacity: 0.75,
                  }}
                >
                  <Tooltip>
                    <strong>{loc.label}</strong>
                    {loc.state && <span> ({loc.state})</span>}
                    <br />
                    {fmtNum(loc.customer_count)} customers
                  </Tooltip>
                </CircleMarker>
              ))}
            </MapContainer>
          </div>

          {/* Table */}
          <div className="max-h-[300px] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-muted/80 backdrop-blur">
                <tr>
                  <th className="px-4 py-2 text-left font-medium capitalize">{groupBy}</th>
                  {groupBy !== "state" && (
                    <th className="px-4 py-2 text-left font-medium">State</th>
                  )}
                  <th className="px-4 py-2 text-right font-medium">Customers</th>
                  <th className="px-4 py-2 text-right font-medium">% of Total</th>
                </tr>
              </thead>
              <tbody>
                {locations.slice(0, 200).map((loc) => (
                  <tr
                    key={`${loc.label}-${loc.state ?? ""}`}
                    className="border-t border-border/40 hover:bg-muted/30"
                  >
                    <td className="px-4 py-1.5">
                      <span className="flex items-center gap-2">
                        <span
                          className="inline-block h-2.5 w-2.5 flex-shrink-0 rounded-sm"
                          style={{
                            background:
                              groupBy === "state"
                                ? choroplethColor(loc.customer_count, stateMaxCount)
                                : choroplethColor(
                                    stateCountMap.get((loc.state ?? "").toUpperCase()) ?? 0,
                                    stateMaxCount,
                                  ),
                          }}
                        />
                        {loc.label}
                      </span>
                    </td>
                    {groupBy !== "state" && (
                      <td className="px-4 py-1.5 text-muted-foreground">{loc.state ?? "—"}</td>
                    )}
                    <td className="px-4 py-1.5 text-right tabular-nums">
                      {fmtNum(loc.customer_count)}
                    </td>
                    <td className="px-4 py-1.5 text-right tabular-nums text-muted-foreground">
                      {total > 0 ? ((loc.customer_count / total) * 100).toFixed(1) : 0}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default CustomerMapTab;
