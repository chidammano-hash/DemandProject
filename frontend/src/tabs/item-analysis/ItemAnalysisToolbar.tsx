import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";

const DEMAND_PANELS = [
  { key: "overlay", label: "Chart" },
  { key: "shap", label: "SHAP" },
  { key: "forecastKpis", label: "Forecast KPIs" },
  { key: "dqCorrections", label: "DQ Corrections" },
  { key: "aiChampion", label: "AI Champion" },
] as const;

interface ItemAnalysisToolbarProps {
  panels: Record<string, boolean>;
  allOn: boolean;
  onSetAll: (enabled: boolean) => void;
  onToggle: (key: string) => void;
  shapModels: string[];
  selectedModel: string | null;
  onSelectedModelChange: (model: string | null) => void;
}

export function ItemAnalysisToolbar({
  panels,
  allOn,
  onSetAll,
  onToggle,
  shapModels,
  selectedModel,
  onSelectedModelChange,
}: ItemAnalysisToolbarProps) {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t px-6 py-2 text-xs">
      <Button
        variant="ghost"
        size="sm"
        className="h-6 px-2 text-[10px] font-semibold uppercase tracking-wider"
        onClick={() => onSetAll(!allOn)}
      >
        {allOn ? "Deselect All" : "Select All"}
      </Button>
      <span className="hidden h-4 w-px bg-border sm:block" />

      {DEMAND_PANELS.map((panel) => (
        <label key={panel.key} className="flex cursor-pointer select-none items-center gap-1.5">
          <Checkbox
            checked={panels[panel.key]}
            onCheckedChange={() => onToggle(panel.key)}
            aria-label={`Toggle ${panel.label}`}
          />
          <span className={panels[panel.key] ? "text-foreground" : "text-muted-foreground"}>
            {panel.label}
          </span>
        </label>
      ))}

      {panels.shap && shapModels.length > 0 && (
        <>
          <span className="mx-1 hidden h-4 w-px bg-border sm:block" />
          <label className="flex select-none items-center gap-1.5">
            <span className="font-semibold uppercase tracking-wider text-muted-foreground">
              SHAP
            </span>
            <select
              className="h-7 rounded border border-input bg-background px-2 text-xs"
              value={selectedModel ?? ""}
              onChange={(event) => onSelectedModelChange(event.target.value || null)}
              aria-label="SHAP model"
            >
              <option value="">None</option>
              {shapModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </label>
        </>
      )}
    </div>
  );
}
