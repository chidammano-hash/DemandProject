/**
 * DemandIntelligencePanel — Unified demand panel with 3 internal tabs:
 *   - Production Forecast (from DemandForecastPanel)
 *   - Demand Plan (from DemandPlanPanel)
 *   - Blended Demand (from BlendedDemandPanel)
 *
 * Consolidates 3 formerly separate panels into a single tabbed view.
 */

import { useState, Suspense, lazy } from "react";
import { cn } from "@/lib/utils";
import { LoadingElement } from "@/components/LoadingElement";

const DemandForecastPanel = lazy(() =>
  import("./DemandForecastPanel").then((m) => ({ default: m.DemandForecastPanel })),
);
const DemandPlanPanel = lazy(() =>
  import("./DemandPlanPanel").then((m) => ({ default: m.DemandPlanPanel })),
);
const BlendedDemandPanel = lazy(() =>
  import("./BlendedDemandPanel").then((m) => ({ default: m.BlendedDemandPanel })),
);

const TABS = [
  { key: "forecast" as const, label: "Production Forecast" },
  { key: "plan" as const, label: "Demand Plan" },
  { key: "blended" as const, label: "Blended Demand" },
];

type TabKey = (typeof TABS)[number]["key"];

export function DemandIntelligencePanel() {
  const [activeTab, setActiveTab] = useState<TabKey>("forecast");

  return (
    <div className="space-y-4">
      {/* Tab selector */}
      <div className="flex gap-1 border-b">
        {TABS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={cn(
              "px-4 py-2 text-sm font-medium border-b-2 transition-colors",
              activeTab === key
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground",
            )}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <Suspense fallback={<LoadingElement message="Loading demand view..." />}>
        {activeTab === "forecast" && <DemandForecastPanel />}
        {activeTab === "plan" && <DemandPlanPanel />}
        {activeTab === "blended" && <BlendedDemandPanel />}
      </Suspense>
    </div>
  );
}
