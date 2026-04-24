import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Gen-4 Roadmap Coder-2 P0: array-driven proxy loop.
// Add a new API path prefix by appending one line below. `make audit-routers`
// compares this list against FastAPI mounts in api/main.py.
const API_PATH_PREFIXES: readonly string[] = [
  "/domains",
  "/health",
  "/chat",
  "/forecast",
  "/sku",
  "/sku-features",
  "/competition",
  "/market-intelligence",
  "/inventory",
  "/dashboard",
  "/clustering",
  "/jobs",
  "/inv-planning",
  "/fill-rate",
  "/ai-planner",
  "/control-tower",
  "/storyboard",
  "/supply",
  "/analytics",
  "/finance",
  "/sop",
  "/events",
  "/scenarios",
  "/auth",
  "/users",
  "/data-quality",
  "/notifications",
  "/collaboration",
  "/demand-signals",
  "/demand-history",
  "/fva",
  "/reports",
  "/webhooks",
  "/cache",
  "/config",
  "/sql-runner",
  "/sourcing",
  "/purchase-orders",
  "/lgbm-tuning",
  "/catboost-tuning",
  "/xgboost-tuning",
  "/cluster-eda",
  "/cluster-experiments",
  "/champion-experiments",
  "/feature-lab",
  "/accuracy-budget",
  "/model-tuning",
  "/expsys",
  "/customer-analytics",
  "/backtest-management",
];

const API_TARGET = "http://127.0.0.1:8000";

const apiProxy = Object.fromEntries(
  API_PATH_PREFIXES.map((prefix) => [
    prefix,
    { target: API_TARGET, changeOrigin: true },
  ]),
);

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": "/src",
    },
  },
  server: {
    host: true,
    port: 5173,
    proxy: apiProxy,
  },
  build: {
    outDir: "dist",
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom"],
          charts: ["recharts"],
          icons: ["lucide-react"],
          radix: ["@radix-ui/react-checkbox", "@radix-ui/react-slot"],
          query: ["@tanstack/react-query"],
          "table-grid": ["@tanstack/react-table", "@tanstack/react-virtual"],
          echarts: ["echarts", "echarts-for-react"],
          leaflet: ["leaflet", "react-leaflet"],
        },
      },
    },
  },
});
