/**
 * Column visibility panel: per-column checkboxes plus Select/Deselect All.
 */
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import type { DomainMeta } from "@/types";
import { titleCase } from "@/lib/formatters";

export interface FieldVisibilityPanelProps {
  meta: DomainMeta;
  visibleColumns: Record<string, boolean>;
  onToggleColumn: (column: string, checked: boolean) => void;
  onSelectAll: () => void;
  onDeselectAll: () => void;
}

export function FieldVisibilityPanel({
  meta,
  visibleColumns,
  onToggleColumn,
  onSelectAll,
  onDeselectAll,
}: FieldVisibilityPanelProps) {
  return (
    <div className="rounded-md border p-2">
      <div className="flex gap-2 mb-2">
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs"
          onClick={onSelectAll}
        >
          Select All
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs"
          onClick={onDeselectAll}
        >
          Deselect All
        </Button>
      </div>
      <div className="grid max-h-40 grid-cols-2 gap-2 overflow-y-auto overflow-x-hidden lg:grid-cols-3">
        {meta.columns.map((col) => (
          <label key={col} className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={visibleColumns[col] !== false}
              onCheckedChange={(checked) => onToggleColumn(col, checked === true)}
            />
            <span>{titleCase(col)}</span>
          </label>
        ))}
      </div>
    </div>
  );
}
