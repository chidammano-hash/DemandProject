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
        },
    },
    build: {
        outDir: "dist",
    },
});
