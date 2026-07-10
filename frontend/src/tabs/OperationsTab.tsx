import { useState } from "react";
import { DatabaseZap, Library, SlidersHorizontal, Sparkles } from "lucide-react";

import { WorkflowScanPanel } from "@/components/workflows/WorkflowScanPanel";
import { cn } from "@/lib/utils";
import IntegrationTab from "./IntegrationTab";
import JobsTab from "./JobsTab";

type OperationsView = "guide" | "library" | "manual";

interface OperationsTabProps {
  onNavigateToScenario?: (jobId: string) => void;
}

const VIEWS = [
  {
    id: "guide" as const,
    label: "Plan & Run",
    description: "AI-guided next best workflow",
    icon: Sparkles,
  },
  {
    id: "library" as const,
    label: "Workflow Library",
    description: "Pipelines, schedules and monitoring",
    icon: Library,
  },
  {
    id: "manual" as const,
    label: "Manual Load",
    description: "Advanced domain controls",
    icon: SlidersHorizontal,
  },
];

export default function OperationsTab({ onNavigateToScenario }: OperationsTabProps): JSX.Element {
  const [view, setView] = useState<OperationsView>("guide");

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="mb-1 flex items-center gap-2 text-primary">
            <DatabaseZap className="h-5 w-5" aria-hidden="true" />
            <span className="text-xs font-semibold uppercase tracking-[0.16em]">Operations</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">
            Workflow Command Center
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            One safe path from changed input files through clustering, forecasting, inventory, and
            archival workflows.
          </p>
        </div>
        <span className="rounded-full border border-emerald-500/25 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-700 dark:text-emerald-300">
          Guarded execution
        </span>
      </header>

      <nav
        aria-label="Workflow command center views"
        className="grid gap-2 rounded-xl border border-border bg-card/60 p-2 md:grid-cols-3"
      >
        {VIEWS.map((item) => {
          const active = view === item.id;
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => setView(item.id)}
              aria-current={active ? "page" : undefined}
              className={cn(
                "flex min-h-14 items-center gap-3 rounded-lg px-3 py-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                active
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-foreground hover:bg-muted"
              )}
            >
              <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
              <span>
                <span className="block text-sm font-semibold">{item.label}</span>
                <span
                  className={cn(
                    "block text-xs",
                    active ? "text-primary-foreground/75" : "text-muted-foreground"
                  )}
                >
                  {item.description}
                </span>
              </span>
            </button>
          );
        })}
      </nav>

      {view === "guide" && (
        <div className="space-y-6">
          <WorkflowScanPanel />
          <IntegrationTab view="guided" embedded />
        </div>
      )}
      {view === "library" && <JobsTab embedded onNavigateToScenario={onNavigateToScenario} />}
      {view === "manual" && <IntegrationTab view="manual" embedded />}
    </div>
  );
}
