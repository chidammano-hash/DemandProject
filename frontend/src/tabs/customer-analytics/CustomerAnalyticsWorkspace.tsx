import { lazy, Suspense, useState } from "react";

import type {
  CustomerAnalyticsFilters,
  CustomerAnalyticsView,
} from "@/api/queries/customer-analytics";
import { CustomerDemandMap } from "./CustomerDemandMap";
import { CustomerTreemap } from "./CustomerTreemap";

const CustomerHeatmap = lazy(() =>
  import("./CustomerHeatmap").then((module) => ({ default: module.CustomerHeatmap })),
);
const ChannelSunburst = lazy(() =>
  import("./ChannelSunburst").then((module) => ({ default: module.ChannelSunburst })),
);
const SegmentSparklines = lazy(() =>
  import("./SegmentSparklines").then((module) => ({ default: module.SegmentSparklines })),
);
const CustomerRanking = lazy(() =>
  import("./CustomerRanking").then((module) => ({ default: module.CustomerRanking })),
);
const OosImpactBubble = lazy(() =>
  import("./OosImpactBubble").then((module) => ({ default: module.OosImpactBubble })),
);
const CustomerLifecycle = lazy(() =>
  import("./CustomerLifecycle").then((module) => ({ default: module.CustomerLifecycle })),
);
const DemandAtRisk = lazy(() =>
  import("./DemandAtRisk").then((module) => ({ default: module.DemandAtRisk })),
);
const CustomerItemAffinity = lazy(() =>
  import("./CustomerItemAffinity").then((module) => ({ default: module.CustomerItemAffinity })),
);
const OrderPatterns = lazy(() =>
  import("./OrderPatterns").then((module) => ({ default: module.OrderPatterns })),
);
const DemandFlowSankey = lazy(() =>
  import("./DemandFlowSankey").then((module) => ({ default: module.DemandFlowSankey })),
);

type MapMetric = "customer_count" | "demand_qty" | "sales_qty" | "oos_qty" | "fill_rate";
type GroupBy = "state" | "city" | "zip";
type SegmentBy = "rpt_channel_desc" | "store_type_desc" | "chain_type_desc" | "state";
type SortMode = "demand_desc" | "fill_rate_asc";
type Grain = "customer" | "state";

interface CustomerAnalyticsWorkspaceProps {
  activeView: CustomerAnalyticsView;
  filters: CustomerAnalyticsFilters;
}

function PanelFallback({ label, tall = false }: { label: string; tall?: boolean }) {
  return (
    <div
      className={`flex ${tall ? "min-h-[440px]" : "min-h-[320px]"} items-center justify-center rounded-xl border bg-card`}
      role="status"
    >
      <div className="flex items-center gap-3 text-sm text-muted-foreground">
        <span className="h-2 w-2 animate-pulse rounded-full bg-primary motion-reduce:animate-none" />
        Loading {label}…
      </div>
    </div>
  );
}

export function CustomerAnalyticsWorkspace({
  activeView,
  filters,
}: CustomerAnalyticsWorkspaceProps) {
  const [mapMetric, setMapMetric] = useState<MapMetric>("demand_qty");
  const [groupBy, setGroupBy] = useState<GroupBy>("state");
  const [segmentBy, setSegmentBy] = useState<SegmentBy>("rpt_channel_desc");
  const [rankSort, setRankSort] = useState<SortMode>("demand_desc");
  const [oosGrain, setOosGrain] = useState<Grain>("customer");

  return (
    <section
      id={`customer-analytics-panel-${activeView}`}
      role="tabpanel"
      aria-labelledby={`customer-analytics-tab-${activeView}`}
      className="space-y-4"
    >
      {activeView === "overview" && (
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.25fr)_minmax(380px,0.75fr)]">
          <CustomerDemandMap
            filters={filters}
            metric={mapMetric}
            groupBy={groupBy}
            onMetricChange={setMapMetric}
            onGroupByChange={setGroupBy}
          />
          <CustomerTreemap filters={filters} />
        </div>
      )}

      {activeView === "customers" && (
        <Suspense fallback={<PanelFallback label="customer portfolio" tall />}>
          <CustomerRanking
            filters={filters}
            sort={rankSort}
            topN={20}
            onSortChange={setRankSort}
          />
          <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
            <CustomerLifecycle filters={filters} />
            <DemandAtRisk filters={filters} />
          </div>
        </Suspense>
      )}

      {activeView === "segments" && (
        <Suspense fallback={<PanelFallback label="segment analysis" tall />}>
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <ChannelSunburst filters={filters} />
            <SegmentSparklines
              filters={filters}
              segmentBy={segmentBy}
              onSegmentByChange={setSegmentBy}
            />
          </div>
          <div className="mt-4">
            <CustomerHeatmap filters={filters} metric="demand_qty" topN={25} />
          </div>
        </Suspense>
      )}

      {activeView === "service" && (
        <Suspense fallback={<PanelFallback label="service-risk analysis" tall />}>
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <OosImpactBubble
              filters={filters}
              grain={oosGrain}
              onGrainChange={setOosGrain}
            />
            <DemandAtRisk filters={filters} />
          </div>
          <div className="mt-4">
            <CustomerHeatmap filters={filters} metric="fill_rate" topN={25} />
          </div>
        </Suspense>
      )}

      {activeView === "behavior" && (
        <Suspense fallback={<PanelFallback label="buying-behavior analysis" tall />}>
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <CustomerItemAffinity filters={filters} />
            <OrderPatterns filters={filters} />
          </div>
          <div className="mt-4">
            <DemandFlowSankey filters={filters} />
          </div>
        </Suspense>
      )}
    </section>
  );
}
