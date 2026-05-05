import { useEffect, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface CollapsibleSectionProps {
  title: string;
  /** Optional subtitle line under the title. */
  subtitle?: string;
  /** Start expanded (default true) when no persisted state exists. */
  defaultOpen?: boolean;
  /** Extra elements rendered in the header row (right side) */
  headerRight?: React.ReactNode;
  /**
   * If provided, collapse state is persisted to localStorage under this key
   * (prefixed). Lets user preferences survive reloads.
   */
  storageKey?: string;
  children: React.ReactNode;
  className?: string;
}

export function CollapsibleSection({
  title,
  subtitle,
  defaultOpen = true,
  headerRight,
  storageKey,
  children,
  className,
}: CollapsibleSectionProps) {
  const fullKey = storageKey ? `collapsible.${storageKey}` : null;
  const [open, setOpen] = useState<boolean>(() => {
    if (!fullKey) return defaultOpen;
    try {
      const v = window.localStorage.getItem(fullKey);
      return v === null ? defaultOpen : v === "true";
    } catch {
      // localStorage unavailable (jsdom without polyfill, private mode, etc.)
      return defaultOpen;
    }
  });

  useEffect(() => {
    if (!fullKey) return;
    try {
      window.localStorage.setItem(fullKey, String(open));
    } catch {
      // localStorage unavailable; collapse state stays in-memory only.
    }
  }, [fullKey, open]);

  return (
    <Card className={cn("animate-fade-in", className)}>
      <CardHeader
        className="cursor-pointer select-none flex flex-row items-center justify-between pb-2"
        onClick={() => setOpen((v) => !v)}
      >
        <div className="flex items-center gap-2">
          {open ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}
          <div className="flex flex-col">
            <CardTitle className="text-sm font-medium">{title}</CardTitle>
            {subtitle && (
              <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>
            )}
          </div>
        </div>
        {headerRight && (
          <div onClick={(e) => e.stopPropagation()}>{headerRight}</div>
        )}
      </CardHeader>
      {open && <CardContent>{children}</CardContent>}
    </Card>
  );
}
