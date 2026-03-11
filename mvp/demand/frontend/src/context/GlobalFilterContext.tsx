import { createContext, useContext } from "react";
import type { GlobalFilters } from "@/types/theme";

export interface GlobalFilterContextValue {
  filters: GlobalFilters;
  setFilters: (partial: Partial<GlobalFilters>) => void;
  resetFilters: () => void;
  hasActiveFilters: boolean;
  planningDate: string | null;
}

const GlobalFilterContext = createContext<GlobalFilterContextValue | null>(null);

export function GlobalFilterProvider({
  value,
  children,
}: {
  value: GlobalFilterContextValue;
  children: React.ReactNode;
}) {
  return (
    <GlobalFilterContext.Provider value={value}>
      {children}
    </GlobalFilterContext.Provider>
  );
}

export function useGlobalFilterContext(): GlobalFilterContextValue {
  const ctx = useContext(GlobalFilterContext);
  if (!ctx) throw new Error("useGlobalFilterContext must be used within GlobalFilterProvider");
  return ctx;
}
