/**
 * TabStrip — the single sanctioned tab-strip primitive.
 *
 * Modeled on the keyboard-correct tab strip in `tabs/CustomerAnalyticsTab.tsx`
 * (role="tablist"/"tab", roving tabindex, Arrow/Home/End navigation with
 * automatic activation) generalized into a controlled component with four
 * visual variants:
 *
 *   - "underline" (default) — active tab gets a 2px bottom border in primary.
 *   - "pills"     — rounded-full pill track, active = bg-primary.
 *   - "segmented" — a single joined, bordered group, active = bg-primary.
 *   - "cards"     — the CustomerAnalyticsTab look: icon + title + optional
 *                   description in a raised active card.
 *
 * Usage:
 *   <TabStrip
 *     aria-label="Customer Analytics views"
 *     value={activeView}
 *     onValueChange={setActiveView}
 *     items={[{ key: "overview", label: "Overview", icon: MapPinned, description: "..." }]}
 *     variant="cards"
 *   />
 */
import { useRef, type KeyboardEvent, type ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

export interface TabStripItem {
  key: string;
  label: string;
  icon?: LucideIcon;
  /** Shown as a subtitle line under the label — only rendered by the "cards" variant. */
  description?: string;
  /** Small count/status pill rendered after the label. */
  badge?: ReactNode;
}

export type TabStripVariant = "underline" | "pills" | "segmented" | "cards";
export type TabStripSize = "sm" | "md" | "lg";

export interface TabStripProps {
  value: string;
  onValueChange: (key: string) => void;
  items: TabStripItem[];
  variant?: TabStripVariant;
  size?: TabStripSize;
  className?: string;
  "aria-label": string;
}

const SIZE_CLASSES: Record<TabStripSize, { pad: string; text: string; icon: string; cardMinH: string }> = {
  sm: { pad: "px-2.5 py-1", text: "text-xs", icon: "h-3 w-3", cardMinH: "min-h-11" },
  md: { pad: "px-3 py-1.5", text: "text-sm", icon: "h-3.5 w-3.5", cardMinH: "min-h-14" },
  lg: { pad: "px-4 py-2", text: "text-sm", icon: "h-4 w-4", cardMinH: "min-h-16" },
};

const CONTAINER_VARIANT: Record<TabStripVariant, string> = {
  underline: "gap-4 border-b border-border",
  pills: "gap-1 rounded-full bg-muted p-1",
  segmented: "divide-x divide-border rounded-md border border-border bg-card",
  cards: "gap-1 rounded-xl border border-border bg-card p-1 shadow-card",
};

const TAB_BASE =
  "relative flex shrink-0 select-none items-center justify-center whitespace-nowrap font-normal text-muted-foreground outline-none transition-colors duration-200 ease-smooth motion-reduce:transition-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background";

function tabVariantClasses(variant: TabStripVariant, selected: boolean): string {
  switch (variant) {
    case "underline":
      return cn(
        "gap-1.5 border-b-2 -mb-px pb-2 pt-1",
        selected ? "border-primary font-medium text-foreground" : "border-transparent hover:border-border hover:text-foreground",
      );
    case "pills":
      return cn(
        "gap-1.5 rounded-full",
        selected ? "bg-primary text-primary-foreground shadow-sm" : "hover:bg-background/70 hover:text-foreground",
      );
    case "segmented":
      return cn(
        "flex-1 gap-1.5",
        selected ? "bg-primary text-primary-foreground" : "hover:bg-muted hover:text-foreground",
      );
    case "cards":
      return cn(
        "flex-1 basis-40 items-center justify-start gap-2 rounded-lg px-3 text-left",
        selected ? "bg-primary text-primary-foreground shadow-sm" : "hover:bg-muted hover:text-foreground",
      );
  }
}

function TabBadge({ children, selected }: { children: ReactNode; selected: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex min-w-[1.125rem] items-center justify-center rounded-full px-1 text-2xs font-semibold leading-none",
        selected ? "bg-primary-foreground/20 text-primary-foreground" : "bg-muted text-muted-foreground",
      )}
    >
      {children}
    </span>
  );
}

export function TabStrip({
  value,
  onValueChange,
  items,
  variant = "underline",
  size = "md",
  className,
  "aria-label": ariaLabel,
}: TabStripProps) {
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const sizeCfg = SIZE_CLASSES[size];
  // Roving tabindex needs exactly one tabbable tab even if `value` doesn't
  // match any item (e.g. before the parent has synced state) — fall back to
  // the first item so Tab still enters the strip.
  const activeIndex = Math.max(0, items.findIndex((item) => item.key === value));

  const selectIndex = (index: number) => {
    const next = items[index];
    if (!next) return;
    onValueChange(next.key);
    tabRefs.current[index]?.focus();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let nextIndex: number | null = null;
    if (event.key === "ArrowRight") nextIndex = (index + 1) % items.length;
    else if (event.key === "ArrowLeft") nextIndex = (index - 1 + items.length) % items.length;
    else if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = items.length - 1;
    if (nextIndex === null) return;
    event.preventDefault();
    selectIndex(nextIndex);
  };

  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={cn("flex items-center overflow-x-auto overscroll-x-none", CONTAINER_VARIANT[variant], className)}
    >
      {items.map((item, index) => {
        const selected = item.key === value;
        const Icon = item.icon;
        const tabIndex = index === activeIndex ? 0 : -1;

        return (
          <button
            key={item.key}
            ref={(el) => {
              tabRefs.current[index] = el;
            }}
            type="button"
            role="tab"
            aria-selected={selected}
            tabIndex={tabIndex}
            title={variant !== "cards" ? item.description : undefined}
            onClick={() => onValueChange(item.key)}
            onKeyDown={(event) => handleKeyDown(event, index)}
            className={cn(TAB_BASE, sizeCfg.pad, sizeCfg.text, variant === "cards" && sizeCfg.cardMinH, tabVariantClasses(variant, selected))}
          >
            {Icon && <Icon className={cn(sizeCfg.icon, "shrink-0")} aria-hidden="true" />}
            {variant === "cards" ? (
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-1 text-xs font-semibold">
                  {item.label}
                  {item.badge != null && <TabBadge selected={selected}>{item.badge}</TabBadge>}
                </span>
                {item.description && (
                  <span className={cn("block truncate text-2xs", selected ? "text-primary-foreground/75" : "text-muted-foreground")}>
                    {item.description}
                  </span>
                )}
              </span>
            ) : (
              <>
                <span>{item.label}</span>
                {item.badge != null && <TabBadge selected={selected}>{item.badge}</TabBadge>}
              </>
            )}
          </button>
        );
      })}
    </div>
  );
}
