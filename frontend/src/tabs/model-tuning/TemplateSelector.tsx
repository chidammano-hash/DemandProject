/**
 * TemplateSelector -- Radio-button grid for choosing an experiment template.
 * Extracted from ExperimentBuilder for reusability and readability.
 */
import { cn } from "@/lib/utils";
import type { TemplateOption } from "@/lib/model-params";

export interface TemplateSelectorProps {
  templates: TemplateOption[];
  selectedTemplate: string;
  onSelect: (templateId: string) => void;
}

export function TemplateSelector({
  templates,
  selectedTemplate,
  onSelect,
}: TemplateSelectorProps) {
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
        Template
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {templates.map((tmpl) => (
          <label
            key={tmpl.id}
            className={cn(
              "flex items-start gap-2 rounded-md border px-3 py-2 cursor-pointer transition-colors",
              selectedTemplate === tmpl.id
                ? "border-primary bg-primary/5 ring-1 ring-primary/30"
                : "border-border hover:bg-muted/30",
            )}
          >
            <input
              type="radio"
              name="template"
              value={tmpl.id}
              checked={selectedTemplate === tmpl.id}
              onChange={() => onSelect(tmpl.id)}
              className="mt-0.5"
            />
            <div className="min-w-0">
              <p className="text-xs font-medium text-foreground">
                {tmpl.label}
              </p>
              <p className="text-[10px] text-muted-foreground truncate">
                {tmpl.description}
              </p>
            </div>
          </label>
        ))}
      </div>
    </div>
  );
}
