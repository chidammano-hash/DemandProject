import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { ThemeProvider } from "@/context/ThemeContext";
import { GlobalFilterProvider } from "@/context/GlobalFilterContext";
import { vi } from "vitest";

const DEFAULT_FILTER_VALUE = {
  filters: {
    brand: [] as string[],
    category: [] as string[],
    market: [] as string[],
    channel: [] as string[],
    item: [] as string[],
    location: [] as string[],
    timeGrain: "month" as const,
  },
  setFilters: vi.fn(),
  resetFilters: vi.fn(),
  hasActiveFilters: false,
  planningDate: null,
};

export function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });
}

export function TestQueryWrapper({ children }: { children: ReactNode }) {
  const client = createTestQueryClient();
  return (
    <ThemeProvider value={{ theme: "light" }}>
      <QueryClientProvider client={client}>
        <GlobalFilterProvider value={DEFAULT_FILTER_VALUE}>
          {children}
        </GlobalFilterProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
}
