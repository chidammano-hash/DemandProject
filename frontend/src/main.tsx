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

function CrashFallback({ error, resetErrorBoundary }: { error: unknown; resetErrorBoundary: () => void }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "100vh", fontFamily: "Inter, system-ui, sans-serif", padding: "2rem", textAlign: "center" }}>
      <div style={{ border: "2px solid #e11d48", borderRadius: "12px", padding: "2rem", maxWidth: "480px", background: "#fff1f2" }}>
        <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>Ac</div>
        <h1 style={{ fontSize: "1.25rem", fontWeight: 700, color: "#1e293b", marginBottom: "0.5rem" }}>Something went wrong</h1>
        <p style={{ fontSize: "0.875rem", color: "#64748b", marginBottom: "1rem" }}>{message}</p>
        <button
          onClick={resetErrorBoundary}
          style={{ background: "#4f46e5", color: "#fff", border: "none", borderRadius: "8px", padding: "0.5rem 1.5rem", cursor: "pointer", fontSize: "0.875rem", fontWeight: 600 }}
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
            <ReactQueryDevtools initialIsOpen={false} />
          </Suspense>
        )}
      </QueryClientProvider>
    </ErrorBoundary>
  </React.StrictMode>,
);
