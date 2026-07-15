import React, { lazy, Suspense } from "react";
import ReactDOM from "react-dom/client";
import { ErrorBoundary } from "react-error-boundary";
import {
  MutationCache,
  QueryCache,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";
import App from "./App";
import "./index.css";
import { toast } from "./components/Toaster";
import { formatApiError, extractStatus } from "./lib/formatApiError";
import { AuthGate } from "./components/AuthGate";
import { AuthProvider } from "./context/AuthContext";

// Devtools are dev-only and ~80KB. Lazy + dev-gated so the prod bundle
// doesn't ship them. Vite strips this branch entirely in `npm run build`.
const ReactQueryDevtools = import.meta.env.DEV
  ? lazy(() =>
      import("@tanstack/react-query-devtools").then((m) => ({ default: m.ReactQueryDevtools })),
    )
  : null;

/**
 * Retry at most twice on network / 5xx errors, never on 4xx (client errors
 * won't fix themselves on retry).
 */
function shouldRetry(failureCount: number, error: unknown): boolean {
  const status = extractStatus(error);
  if (status != null && status >= 400 && status < 500) return false;
  return failureCount < 2;
}

/**
 * Global error handler — sanitizes and surfaces a non-blocking toast so one
 * failing query doesn't crash the UI or silently fail.
 */
function handleGlobalError(error: unknown): void {
  toast.error(formatApiError(error));
}

const queryClient = new QueryClient({
  queryCache: new QueryCache({ onError: handleGlobalError }),
  mutationCache: new MutationCache({ onError: handleGlobalError }),
  defaultOptions: {
    queries: {
      staleTime: 120_000,
      gcTime: 5 * 60_000,
      retry: shouldRetry,
      refetchOnWindowFocus: false,
      refetchIntervalInBackground: false,
    },
    mutations: {
      retry: false,
    },
  },
});

// Renders before ThemeProvider/QueryClientProvider even mount (this is the
// top-level crash boundary), so it can't read theme context — it relies on
// Tailwind utility classes resolving against the `:root`/`.dark` CSS var
// fallbacks in index.css instead, same as everywhere else.
function CrashFallback({ error, resetErrorBoundary }: { error: unknown; resetErrorBoundary: () => void }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-8 text-center">
      <div className="max-w-[480px] rounded-xl border border-destructive/30 bg-destructive/10 p-8">
        <h1 className="mb-2 text-xl font-bold text-foreground">Something went wrong</h1>
        <p className="mb-4 text-sm text-muted-foreground">{message}</p>
        <button
          onClick={resetErrorBoundary}
          className="rounded-lg bg-primary px-6 py-2 text-sm font-semibold text-primary-foreground hover:opacity-90"
        >
          Reload App
        </button>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary FallbackComponent={CrashFallback} onReset={() => window.location.reload()}>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <AuthGate><App /></AuthGate>
        </AuthProvider>
        {ReactQueryDevtools && (
          <Suspense fallback={null}>
            {/* bottom-right is the Assistant FAB's spot (GlobalChatDrawer) — keep
                the devtools toggle out of its way. */}
            <ReactQueryDevtools initialIsOpen={false} buttonPosition="bottom-left" />
          </Suspense>
        )}
      </QueryClientProvider>
    </ErrorBoundary>
  </React.StrictMode>,
);
