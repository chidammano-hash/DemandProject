import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import { NAV_ITEMS } from "./AppSidebar";
import {
  Search,
  CornerDownLeft,
  Keyboard,
  MoonStar,
  PanelLeft,
  type LucideIcon,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  onNavigate: (tab: string) => void;
  onToggleDarkMode?: () => void;
  onToggleSidebar?: () => void;
  onShowKeyboardHelp?: () => void;
  activeTab: string;
}

interface PaletteItem {
  id: string;
  label: string;
  section: string;
  icon?: LucideIcon;
  shortcut?: string;
  keywords?: string;
  type: "navigate" | "action";
}

// ---------------------------------------------------------------------------
// Quick actions beyond navigation
// ---------------------------------------------------------------------------
const QUICK_ACTIONS: PaletteItem[] = [
  {
    id: "toggle-dark",
    label: "Toggle dark mode",
    section: "Actions",
    icon: MoonStar,
    keywords: "dark light theme mode",
    type: "action",
  },
  {
    id: "toggle-sidebar",
    label: "Toggle sidebar",
    section: "Actions",
    icon: PanelLeft,
    keywords: "sidebar collapse expand navigation nav",
    type: "action",
  },
  {
    id: "keyboard-help",
    label: "Keyboard shortcuts",
    section: "Actions",
    icon: Keyboard,
    keywords: "shortcuts keys help hotkeys",
    type: "action",
  },
];

const SECTION_ORDER: Record<string, number> = {
  Command: 0,
  Demand: 1,
  Supply: 2,
  Operations: 3,
  System: 4,
  Actions: 5,
};

const SECTION_FROM_NAV: Record<string, string> = {
  tower: "Command",
  command: "Command",
  demand: "Demand",
  supply: "Supply",
  operations: "Operations",
  plan: "Operations",
  system: "System",
  overview: "Command",
  intelligence: "Command",
};

const NAV_KEYWORDS: Record<string, string> = {
  commandCenter: "control tower executive overview priorities alerts readiness",
  aggregateAnalysis: "portfolio demand forecast accuracy aggregate dashboard",
  itemAnalysis: "sku item dfu forecast actual overlay detail",
  fva: "forecast value added roi accuracy contribution",
  lgbmTuning: "forecasting model tuning champion backtest period roll ml parameters",
  customerAnalytics: "customer geography market channel account map",
  demandHistory: "history sales demand time series actuals",
  skuFeatures: "features feature lab clustering attributes",
  skuChat: "chat assistant ai sku analysis",
  invPlanning: "inventory replenishment policy orders safety stock eoq rop supply",
  invBacktest: "inventory backtest simulation service level",
  sop: "sop s&op consensus planning override collaboration",
  integration:
    "workflows jobs runs active scheduler queue ai etl pipeline load normalize refresh clustering forecast inventory archive",
  dataQuality: "quality lineage validation exceptions rules checks",
  explorer: "database table records dimensions facts browse",
  sqlRunner: "sql query runner console database",
  settings: "settings preferences configuration theme",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function CommandPalette({
  open,
  onClose,
  onNavigate,
  onToggleDarkMode,
  onToggleSidebar,
  onShowKeyboardHelp,
  activeTab,
}: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Build full item list from NAV_ITEMS + quick actions
  const allItems = useMemo<PaletteItem[]>(() => {
    const navItems: PaletteItem[] = NAV_ITEMS.filter((n) => n.key !== "chat").map((n) => ({
      id: n.key,
      label: n.label,
      section: SECTION_FROM_NAV[n.section] ?? n.section,
      icon: n.icon,
      shortcut: n.shortcut,
      keywords: `${n.key} ${n.section} ${NAV_KEYWORDS[n.key] ?? ""}`,
      type: "navigate" as const,
    }));
    return [...navItems, ...QUICK_ACTIONS];
  }, []);

  // Filter items by query
  const filtered = useMemo(() => {
    if (!query.trim()) return allItems;
    const q = query.toLowerCase();
    return allItems.filter(
      (item) =>
        item.label.toLowerCase().includes(q) ||
        item.section.toLowerCase().includes(q) ||
        (item.keywords && item.keywords.toLowerCase().includes(q))
    );
  }, [allItems, query]);

  // Group filtered items by section
  const grouped = useMemo(() => {
    const groups: { section: string; items: PaletteItem[] }[] = [];
    const seen = new Set<string>();
    // Sort by section order
    const sorted = [...filtered].sort(
      (a, b) => (SECTION_ORDER[a.section] ?? 99) - (SECTION_ORDER[b.section] ?? 99)
    );
    for (const item of sorted) {
      if (!seen.has(item.section)) {
        seen.add(item.section);
        groups.push({ section: item.section, items: [] });
      }
      groups.find((g) => g.section === item.section)!.items.push(item);
    }
    return groups;
  }, [filtered]);

  // Flat list for keyboard navigation
  const flatItems = useMemo(() => grouped.flatMap((g) => g.items), [grouped]);
  const selectedItemId = flatItems[selectedIndex]?.id
    ? `command-palette-option-${flatItems[selectedIndex].id}`
    : undefined;

  // Reset on open/close
  useEffect(() => {
    if (open) {
      setQuery("");
      setSelectedIndex(0);
      // Focus the input after a tick to allow the DOM to render
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  // Clamp selected index when filtered list changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  // Scroll selected item into view
  useEffect(() => {
    if (!listRef.current) return;
    const el = listRef.current.querySelector("[data-selected='true']");
    if (el) el.scrollIntoView({ block: "nearest" });
  }, [selectedIndex]);

  // Execute the selected item
  const executeItem = useCallback(
    (item: PaletteItem) => {
      if (item.type === "navigate") {
        onNavigate(item.id);
      } else if (item.id === "toggle-dark") {
        onToggleDarkMode?.();
      } else if (item.id === "toggle-sidebar") {
        onToggleSidebar?.();
      } else if (item.id === "keyboard-help") {
        onShowKeyboardHelp?.();
      }
      onClose();
    },
    [onNavigate, onToggleDarkMode, onToggleSidebar, onShowKeyboardHelp, onClose]
  );

  // Keyboard handler
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((i) => (i + 1) % Math.max(flatItems.length, 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((i) => (i - 1 + flatItems.length) % Math.max(flatItems.length, 1));
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (flatItems[selectedIndex]) executeItem(flatItems[selectedIndex]);
      } else if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    },
    [flatItems, selectedIndex, executeItem, onClose]
  );

  if (!open) return null;

  let flatIdx = 0;

  return (
    <div
      className="fixed inset-0 z-[100] flex justify-center bg-black/50 backdrop-blur-sm"
      onClick={onClose}
      data-testid="command-palette-backdrop"
    >
      <div
        className="mt-[20vh] h-fit w-full max-w-lg animate-in fade-in zoom-in-95 rounded-xl border border-border bg-card shadow-2xl duration-150"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
      >
        {/* Search input */}
        <div className="flex items-center gap-2 border-b border-border px-4 py-3">
          <Search className="h-4 w-4 shrink-0 text-muted-foreground" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search commands..."
            className="flex-1 bg-transparent text-lg text-foreground outline-none placeholder:text-muted-foreground"
            aria-label="Search commands"
            aria-controls="command-palette-results"
            aria-activedescendant={selectedItemId}
          />
        </div>

        {/* Results list */}
        <div
          ref={listRef}
          id="command-palette-results"
          role="listbox"
          aria-label="Command results"
          className="max-h-[60vh] overflow-y-auto px-2 py-2"
        >
          {flatItems.length === 0 && (
            <p className="px-3 py-6 text-center text-sm text-muted-foreground">No results found.</p>
          )}
          {grouped.map((group) => (
            <div key={group.section}>
              <p className="px-3 py-1.5 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                {group.section}
              </p>
              {group.items.map((item) => {
                const idx = flatIdx++;
                const isSelected = idx === selectedIndex;
                const isActive = item.type === "navigate" && item.id === activeTab;
                const Icon = item.icon;
                return (
                  <button
                    key={item.id}
                    id={`command-palette-option-${item.id}`}
                    role="option"
                    aria-selected={isSelected}
                    data-selected={isSelected}
                    onClick={() => executeItem(item)}
                    className={cn(
                      "flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                      isSelected
                        ? "bg-primary/10 text-foreground"
                        : "text-foreground/80 hover:bg-muted"
                    )}
                  >
                    {Icon && (
                      <Icon
                        className="h-[18px] w-[18px] shrink-0 text-muted-foreground"
                        strokeWidth={1.5}
                      />
                    )}
                    <span className="flex-1 truncate text-left">
                      {item.label}
                      {isActive && (
                        <span className="ml-2 text-xs text-muted-foreground">(current)</span>
                      )}
                    </span>
                    {item.shortcut && (
                      <kbd className="rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                        {item.shortcut}
                      </kbd>
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </div>

        {/* Footer hints */}
        <div className="flex items-center gap-3 border-t border-border px-4 py-2 text-xs text-muted-foreground">
          <span>
            {flatItems.length} result{flatItems.length === 1 ? "" : "s"}
          </span>
          <span className="h-3 w-px bg-border" aria-hidden="true" />
          <span className="flex items-center gap-1">
            <CornerDownLeft className="h-3 w-3" /> to select
          </span>
          <span>esc to close</span>
        </div>
      </div>
    </div>
  );
}
