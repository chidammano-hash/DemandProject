/**
 * GenerateForecastCard -- Step 3 of the ForecastPanel.
 *
 * Horizon input, confidence-interval toggle, the Generate Forecast action,
 * and a read-only "Latest Version" summary card. Pure presentation; state
 * and handlers are lifted in.
 */
import { Play, Loader2 } from "lucide-react";

import type { ProductionForecastVersion } from "@/api/queries/production-forecast";
import { modelLabel } from "@/lib/model-labels";
import { timeAgo } from "@/components/shared-tuning-utils";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";

import { ConfigRow } from "./ConfigRow";

interface GenerateForecastCardProps {
  selectedModel: string;
  effectiveHorizon: number;
  onHorizonChange: (horizon: number) => void;
  includeCI: boolean;
  onIncludeCIChange: (checked: boolean) => void;
  isSubmitting: boolean;
  isForecastRunning: boolean;
  onGenerateForecast: () => void;
  latestVersion: ProductionForecastVersion | null;
}

export function GenerateForecastCard({
  selectedModel,
  effectiveHorizon,
  onHorizonChange,
  includeCI,
  onIncludeCIChange,
  isSubmitting,
  isForecastRunning,
  onGenerateForecast,
  latestVersion,
}: GenerateForecastCardProps) {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold">Step 3: Generate Forecast</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Horizon input */}
          <div>
            <label htmlFor="forecast-horizon" className="text-xs text-muted-foreground block mb-1">
              Horizon (months)
            </label>
            <input
              id="forecast-horizon"
              type="number"
              min={1}
              max={60}
              value={effectiveHorizon}
              onChange={(e) => onHorizonChange(Math.max(1, parseInt(e.target.value, 10) || 1))}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
          </div>

          {/* Selected model badge */}
          <div className="text-xs text-muted-foreground">
            Model:{" "}
            <span className="font-medium text-foreground">
              {selectedModel === "champion" ? "Champion (meta-learner)" : modelLabel(selectedModel)}
            </span>
          </div>

          {/* Confidence intervals checkbox */}
          <div className="flex items-center gap-2">
            <Checkbox
              id="include-ci"
              checked={includeCI}
              onCheckedChange={(checked) => onIncludeCIChange(checked === true)}
            />
            <label
              htmlFor="include-ci"
              className="text-xs cursor-pointer select-none"
            >
              Include Confidence Intervals (P10/P90)
            </label>
          </div>

          {/* Generate button */}
          <Button
            className="w-full"
            size="lg"
            onClick={onGenerateForecast}
            disabled={isSubmitting || isForecastRunning}
          >
            {isSubmitting || isForecastRunning ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {isForecastRunning ? "Forecast Running..." : "Submitting..."}
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Generate Forecast
              </>
            )}
          </Button>

          {isForecastRunning && (
            <p className="text-xs text-amber-600 dark:text-amber-400">
              A forecast generation job is currently running. Wait for it to complete before submitting another.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Latest version info card */}
      {latestVersion && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold">Latest Version</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1 text-sm">
            <ConfigRow label="Version" value={latestVersion.plan_version} />
            <ConfigRow label="DFUs" value={(latestVersion.sku_count ?? latestVersion.dfu_count)?.toLocaleString() ?? "--"} />
            <ConfigRow label="Rows" value={latestVersion.total_rows?.toLocaleString() ?? "--"} />
            <ConfigRow
              label="Generated"
              value={latestVersion.generated_at ? timeAgo(latestVersion.generated_at) : "--"}
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
