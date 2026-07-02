import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Gen-4 Roadmap Coder-2 P0: array-driven proxy loop.
// Add a new API path prefix by appending one line below. `make audit-routers`
// compares this list against FastAPI mounts in api/main.py.
const API_PATH_PREFIXES: readonly string[] = [
  "/domains",
  "/health",
  "/forecast",
  "/sku",
  "/sku-features",
  "/sku-chat",
  "/ai-champion",
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
  "/champion-sweeps",
  "/feature-lab",
  "/accuracy-budget",
  "/model-tuning",
  "/expsys",
  "/customer-analytics",
  "/backtest-management",
  "/integration",
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
        // Function-form so vendor packages with many sub-packages (e.g. all
        // 37 @radix-ui/* modules) collapse into a single shared chunk
        // instead of leaking into the main bundle.
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("/react/") || id.includes("/react-dom/") || id.includes("/scheduler/")) {
            return "vendor";
          }
          if (id.includes("/@radix-ui/")) return "radix";
          if (id.includes("/recharts/") || id.includes("/d3-")) return "charts";
          if (id.includes("/echarts/") || id.includes("/echarts-for-react/") || id.includes("/zrender/")) {
            return "echarts";
          }
          if (id.includes("/leaflet") || id.includes("/react-leaflet")) return "leaflet";
          if (id.includes("/lucide-react/")) return "icons";
          if (id.includes("/@tanstack/react-query")) return "query";
          if (id.includes("/@tanstack/react-table") || id.includes("/@tanstack/react-virtual")) {
            return "table-grid";
          }
          if (id.includes("/papaparse/")) return "papaparse";
          return undefined;
        },
      },
    },
  },
});
