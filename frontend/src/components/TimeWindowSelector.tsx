import { cn } from "@/lib/utils";

interface TimeWindowSelectorProps {
  windows: number[];
  selected: number;
  onChange: (months: number) => void;
  suffix?: string;
  className?: string;
}

export function TimeWindowSelector({ windows, selected, onChange, suffix = "mo", className }: TimeWindowSelectorProps) {
  return (
    <div className={cn("flex items-center gap-0.5 rounded-lg border border-border p-0.5", className)}>
      {windows.map((w) => (
        <button
          key={w}
          onClick={() => onChange(w)}
          className={cn(
            "rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
            w === selected
              ? "bg-primary/10 text-primary"
              : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
          )}
        >
          {w}{suffix}
        </button>
      ))}
    </div>
  );
}
