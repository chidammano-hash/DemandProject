import * as React from "react";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Minimal native-select wrappers that match the shadcn/ui Select API surface
// used in AIPlannerTab. Using a native <select> keeps the bundle lean and
// avoids a Radix-UI dependency for this specific use case.
// ---------------------------------------------------------------------------

interface SelectProps {
  value?: string;
  onValueChange?: (value: string) => void;
  children?: React.ReactNode;
}

function Select({ value, onValueChange, children }: SelectProps) {
  return (
    <SelectContext.Provider value={{ value, onValueChange }}>
      <div className="relative">{children}</div>
    </SelectContext.Provider>
  );
}

interface SelectContextValue {
  value?: string;
  onValueChange?: (value: string) => void;
}

const SelectContext = React.createContext<SelectContextValue>({});

function SelectTrigger({ className, children }: { className?: string; children?: React.ReactNode }) {
  const ctx = React.useContext(SelectContext);
  return (
    <div className={cn("flex items-center gap-1 cursor-pointer", className)} data-value={ctx.value}>
      {children}
    </div>
  );
}

function SelectValue({ placeholder }: { placeholder?: string }) {
  const ctx = React.useContext(SelectContext);
  return <span>{ctx.value ?? placeholder}</span>;
}

function SelectContent({ children }: { children?: React.ReactNode }) {
  return <div className="absolute z-50 mt-1 rounded-md border bg-popover shadow-md">{children}</div>;
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
  const ctx = React.useContext(SelectContext);
  return (
    <div
      className={cn("cursor-pointer px-3 py-1.5 text-sm hover:bg-accent", className)}
      onClick={() => ctx.onValueChange?.(value)}
      data-value={value}
    >
      {children}
    </div>
  );
}

export { Select, SelectTrigger, SelectValue, SelectContent, SelectItem };
