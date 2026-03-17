/**
 * CausalChainCard — visual [Forecast]→[Inventory]→[Policy]→[Financial] chain
 */
import { cn } from "@/lib/utils";
import { ChevronRight } from "lucide-react";
import type { AiInsight } from "@/types/ai-planner";
import { buildCausalChain, CAUSAL_LAYER_LABELS, CAUSAL_LAYER_ICONS } from "./aiPlannerShared";

export function CausalChainCard({ insight }: { insight: AiInsight }) {
  const chain = buildCausalChain(insight);
  if (chain.length < 2) return null;

  return (
    <div className="mt-3 rounded-lg border bg-muted/20 p-3">
      <p className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        Causal Chain
      </p>
      <div className="flex items-stretch gap-0 overflow-x-auto">
        {chain.map((link, idx) => {
          const LayerIcon = CAUSAL_LAYER_ICONS[link.layer];
          return (
            <div key={link.layer} className="flex items-center">
              <div
                className={cn(
                  "flex min-w-[90px] flex-col items-center gap-0.5 rounded-md px-2 py-2 text-center",
                  link.isAlert
                    ? "bg-red-50 dark:bg-red-950/30"
                    : "bg-muted/40",
                )}
              >
                <LayerIcon
                  className={cn(
                    "h-3.5 w-3.5",
                    link.isAlert ? "text-red-500" : "text-muted-foreground",
                  )}
                />
                <span className="text-[9px] font-semibold uppercase tracking-wide text-muted-foreground">
                  {CAUSAL_LAYER_LABELS[link.layer]}
                </span>
                <span
                  className={cn(
                    "text-[11px] font-bold leading-tight",
                    link.isAlert ? "text-red-700 dark:text-red-400" : "text-foreground",
                  )}
                >
                  {link.signal}
                </span>
                <span className="text-[9px] leading-tight text-muted-foreground">
                  {link.impact}
                </span>
              </div>
              {idx < chain.length - 1 && (
                <ChevronRight className="mx-0.5 h-3.5 w-3.5 flex-shrink-0 text-muted-foreground/40" />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
