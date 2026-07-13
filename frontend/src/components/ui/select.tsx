import * as React from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Minimal controlled Select with open/close toggle and outside-click dismiss.
// Mirrors the shadcn/ui Select API.
// ---------------------------------------------------------------------------

interface SelectContextValue {
  value?: string;
  onValueChange?: (value: string) => void;
  open: boolean;
  setOpen: (v: boolean) => void;
  /** value -> rendered label, populated by each SelectItem on mount. */
  labels: Map<string, React.ReactNode>;
  /** SelectItem calls this on mount to register its label; returns an unregister fn. */
  registerLabel: (value: string, label: React.ReactNode) => () => void;
}

const SelectContext = React.createContext<SelectContextValue>({
  open: false,
  setOpen: () => {},
  labels: new Map(),
  registerLabel: () => () => {},
});

interface SelectProps {
  value?: string;
  onValueChange?: (value: string) => void;
  children?: React.ReactNode;
}

function Select({ value, onValueChange, children }: SelectProps) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef<HTMLDivElement>(null);
  const labelsRef = React.useRef(new Map<string, React.ReactNode>());
  const [, forceTick] = React.useState(0);

  const registerLabel = React.useCallback(
    (val: string, label: React.ReactNode) => {
      labelsRef.current.set(val, label);
      forceTick((n) => n + 1);
      return () => {
        labelsRef.current.delete(val);
        forceTick((n) => n + 1);
      };
    },
    [],
  );

  // Close on outside click or Escape — respect the keyboard: a user who opened the
  // menu must be able to dismiss it without reaching for the mouse.
  React.useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <SelectContext.Provider
      value={{ value, onValueChange, open, setOpen, labels: labelsRef.current, registerLabel }}
    >
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
      aria-haspopup="listbox"
      aria-expanded={open}
      onClick={() => setOpen(!open)}
      className={cn(
        "flex items-center justify-between gap-1 rounded-md border border-input bg-background px-3 py-1.5 text-sm shadow-sm transition-colors ease-smooth hover:bg-accent focus:outline-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        className,
      )}
    >
      {children}
      <ChevronDown className={cn("h-3.5 w-3.5 text-muted-foreground transition-transform", open && "rotate-180")} />
    </button>
  );
}

function SelectValue({ placeholder }: { placeholder?: string }) {
  const { value, labels } = React.useContext(SelectContext);
  if (value !== undefined && labels.has(value)) {
    return <span className="truncate">{labels.get(value)}</span>;
  }
  return <span className="truncate">{value ?? placeholder}</span>;
}

function SelectContent({ children }: { children?: React.ReactNode }) {
  const { open } = React.useContext(SelectContext);
  if (!open) return null;
  return (
    <div
      role="listbox"
      className="absolute left-0 top-full z-[200] mt-1 max-h-60 min-w-full overflow-y-auto rounded-md border border-border bg-popover py-1 text-popover-foreground shadow-elevated animate-scale-in origin-top"
    >
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
  const { value: selected, onValueChange, setOpen, registerLabel } =
    React.useContext(SelectContext);

  // Register this item's label with the parent Select so <SelectValue/> can
  // render the human label (e.g. "Ollama (local, free)") instead of the raw
  // value ("ollama"). Unregisters on unmount.
  React.useEffect(() => registerLabel(value, children), [value, children, registerLabel]);

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
