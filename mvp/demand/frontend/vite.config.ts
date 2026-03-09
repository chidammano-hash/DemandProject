import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

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
    proxy: {
      "/domains": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/chat": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/forecast": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/dfu": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/competition": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/market-intelligence": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/inventory": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/dashboard": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/clustering": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/jobs": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/inv-planning": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/fill-rate": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/ai-planner": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/control-tower": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/storyboard": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/supply": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/analytics": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/finance": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/sop": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/events": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/scenarios": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
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
        },
      },
    },
  },
});
