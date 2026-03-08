import React from "react";
import ReactDOM from "react-dom/client";
import { ErrorBoundary } from "react-error-boundary";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import App from "./App";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      gcTime: 5 * 60_000,
      retry: 2,
      refetchOnWindowFocus: true,
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
        <App />
        <ReactQueryDevtools initialIsOpen={false} />
      </QueryClientProvider>
    </ErrorBoundary>
  </React.StrictMode>,
);
