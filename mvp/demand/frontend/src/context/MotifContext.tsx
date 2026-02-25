import { createContext, useContext, type ReactNode } from "react";
import type { UseMotifThemeReturn } from "@/hooks/useMotifTheme";

const MotifContext = createContext<UseMotifThemeReturn | null>(null);

export function MotifProvider({ children, value }: { children: ReactNode; value: UseMotifThemeReturn }) {
  return <MotifContext.Provider value={value}>{children}</MotifContext.Provider>;
}

export function useMotif(): UseMotifThemeReturn {
  const ctx = useContext(MotifContext);
  if (!ctx) throw new Error("useMotif must be used inside MotifProvider");
  return ctx;
}

/** Safe variant — returns null outside MotifProvider instead of throwing. */
export function useMotifOptional(): UseMotifThemeReturn | null {
  return useContext(MotifContext);
}
