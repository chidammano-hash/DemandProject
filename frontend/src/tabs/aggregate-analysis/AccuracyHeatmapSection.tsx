/**
 * Accuracy Heatmap section for Portfolio Analysis — accuracy (100 − WAPE) grid
 * with model + row/column grain selectors. Extracted verbatim from
 * AggregateAnalysisTab.
 */
import { Skeleton } from "@/components/Skeleton";
import { HeatmapGrid } from "@/components/HeatmapGrid";
import { CollapsibleSection } from "@/components/CollapsibleSection";

import { formatHeatmapAccuracy } from "./aggregateShared";

export type HmGrain = "category" | "brand" | "location" | "class" | "sub_class" | "date";

interface HeatmapRow {
  label: string;
  values: number[];
  counts?: number[];
}

interface AccuracyHeatmapSectionProps {
  isLoading: boolean;
  rows: HeatmapRow[];
  columnLabels: string[];
  colorScale: (value: number) => string;
  heatmapModel: string;
  heatmapModels: string[] | undefined;
  heatmapRowGrain: HmGrain;
  heatmapColGrain: HmGrain;
  onHeatmapModelChange: (model: string) => void;
  onRowGrainChange: (grain: HmGrain) => void;
  onColGrainChange: (grain: HmGrain) => void;
}

export function AccuracyHeatmapSection({
  isLoading,
  rows,
  columnLabels,
  colorScale,
  heatmapModel,
  heatmapModels,
  heatmapRowGrain,
  heatmapColGrain,
  onHeatmapModelChange,
  onRowGrainChange,
  onColGrainChange,
}: AccuracyHeatmapSectionProps) {
  return (
    <CollapsibleSection
      title="Accuracy Heatmap"
      headerRight={
        <div className="flex items-center gap-2 text-[10px]">
          <span className="text-muted-foreground font-medium">Model</span>
          <select
            className="h-6 rounded border border-input bg-background px-1.5 text-[10px]"
            value={heatmapModel}
            onChange={(e) => onHeatmapModelChange(e.target.value)}
          >
            {(heatmapModels ?? ["external"]).map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
          <span className="text-muted-foreground font-medium">Rows</span>
          <select
            className="h-6 rounded border border-input bg-background px-1.5 text-[10px]"
            value={heatmapRowGrain}
            onChange={(e) => {
              const v = e.target.value as HmGrain;
              onRowGrainChange(v);
              if (v === heatmapColGrain) onColGrainChange(v === "date" ? "category" : "date");
            }}
          >
            <option value="category">Category</option>
            <option value="brand">Brand</option>
            <option value="class">Class</option>
            <option value="sub_class">Sub-class</option>
            <option value="location">Location</option>
            <option value="date">Date</option>
          </select>
          <span className="text-muted-foreground font-medium">Columns</span>
          <select
            className="h-6 rounded border border-input bg-background px-1.5 text-[10px]"
            value={heatmapColGrain}
            onChange={(e) => {
              const v = e.target.value as HmGrain;
              onColGrainChange(v);
              if (v === heatmapRowGrain) onRowGrainChange(v === "date" ? "category" : "date");
            }}
          >
            <option value="category">Category</option>
            <option value="brand">Brand</option>
            <option value="class">Class</option>
            <option value="sub_class">Sub-class</option>
            <option value="location">Location</option>
            <option value="date">Date</option>
          </select>
        </div>
      }
    >
      {isLoading ? (
        <Skeleton className="h-[200px]" />
      ) : (
        <>
          <HeatmapGrid
            rows={rows}
            columnLabels={columnLabels}
            colorScale={colorScale}
            valueFormat={formatHeatmapAccuracy}
            showLegend
            minLabel="<0%"
            maxLabel="100%"
          />
          {/* F3.2 / U3.6 — explain the floored cells so a low-base artifact
              isn't mistaken for a broken model. */}
          <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
            Accuracy = 100 − WAPE. Cells marked <span className="font-medium">&lt;0%*</span> have
            actuals near zero on a tiny base (WAPE &gt; 100%) — review WAPE rather than reading
            the negative literally.
          </p>
        </>
      )}
    </CollapsibleSection>
  );
}
