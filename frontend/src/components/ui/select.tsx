import * as React from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Minimal controlled Select with open/close toggle and outside-click dismiss.
// Matches the shadcn/ui Select API used in AIPlannerTab.
// ---------------------------------------------------------------------------

interface SelectContextValue {
  value?: string;
  onValueChange?: (value: string) => void;
  open: boolean;
  setOpen: (v: boolean) => void;
}

const SelectContext = React.createContext<SelectContextValue>({
  open: false,
  setOpen: () => {},
});

interface SelectProps {
  value?: string;
  onValueChange?: (value: string) => void;
  children?: React.ReactNode;
}

function Select({ value, onValueChange, children }: SelectProps) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef<HTMLDivElement>(null);

  // Close on outside click
  React.useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  return (
    <SelectContext.Provider value={{ value, onValueChange, open, setOpen }}>
      <div ref={ref} className={cn("relative inline-block", open && "z-[200]")}>
        {children}
      </div>
    </SelectContext.Provider>
  );
}

function SelectTrigger({ className, children }: { className?: string; children?: React.ReactNode }) {
  const { open, setOpen } = React.useContext(SelectContext);
  return (
    <button
      type="button"
      onClick={() => setOpen(!open)}
      className={cn(
        "flex items-center justify-between gap-1 rounded-md border border-input bg-background px-3 py-1.5 text-sm shadow-sm hover:bg-accent focus:outline-none",
        className,
      )}
    >
      {children}
      <ChevronDown className={cn("h-3.5 w-3.5 text-muted-foreground transition-transform", open && "rotate-180")} />
    </button>
  );
}

function SelectValue({ placeholder }: { placeholder?: string }) {
  const { value } = React.useContext(SelectContext);
  return <span className="truncate">{value ?? placeholder}</span>;
}

function SelectContent({ children }: { children?: React.ReactNode }) {
  const { open } = React.useContext(SelectContext);
  if (!open) return null;
  return (
    <div className="absolute left-0 top-full z-[200] mt-1 min-w-full rounded-md border bg-popover py-1 shadow-lg">
      {children}
    </div>
  );
}

function SelectItem({
  value,
  children,
  className,
}: {
  value: string;
  children?: React.ReactNode;
  className?: string;
}) {
  const { value: selected, onValueChange, setOpen } = React.useContext(SelectContext);
  return (
    <div
      role="option"
      aria-selected={selected === value}
      className={cn(
        "cursor-pointer px-3 py-1.5 text-sm hover:bg-accent",
        selected === value && "font-medium text-primary",
        className,
      )}
      onClick={() => {
        onValueChange?.(value);
        setOpen(false);
      }}
      data-value={value}
    >
      {children}
    </div>
  );
}

export { Select, SelectTrigger, SelectValue, SelectContent, SelectItem };
