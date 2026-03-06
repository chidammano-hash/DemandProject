# 05 — UI, Automation & Cross-Cutting

This group covers all user-facing interface design, UX architecture, cross-cutting platform concerns, and automation infrastructure within Demand Studio. It includes the chatbot, market intelligence, theming system, data explorer enhancements, job scheduling, and the comprehensive testing strategy.

## Files

| File | Feature | Summary |
|---|---|---|
| `feature10.md` | Multi-Dimensional Accuracy Slicing | Accuracy analysis sliced by cluster, supplier, ABC class, region, brand, lag, and model — powered by materialized views and a collapsible Accuracy tab panel. |
| `feature11.md` | Chatbot / Natural Language Queries | OpenAI-powered NL→SQL chatbot (`POST /chat`) with pgvector context retrieval, read-only execution, 5s timeout, and 500-row limit. |
| `feature16.md` | Data Explorer Performance & UX | Fast interactive filtering on 60M+ row tables: type-aware column filters, GIN trigram indexes, column typeahead suggestions, and chemistry-themed loading overlay. |
| `feature17.md` | DFU Analysis Tab | Unified chart overlaying sales history and multi-model forecast predictions per DFU at three scope levels: single DFU, item across locations, or all items at a location. |
| `feature18.md` | Market Intelligence | AI-powered tab combining Google Custom Search results with GPT-4o narrative synthesis to deliver contextual demand insights for any product + location pair. |
| `feature22.md` | UI Theming (Dark Mode) | Global theme system with Light, Dark, and Midnight modes; `localStorage` persistence and instant application across all tabs via CSS variable palettes. |
| `feature28.md` | UI Architecture & Performance | Decomposition of the monolithic `App.tsx` into lazy-loaded tab components with error boundaries, TanStack Query caching, keyboard shortcuts, and Vitest testing infrastructure. |
| `feature31.md` | Comprehensive Testing Strategy | Full-stack mandatory testing spec: backend pytest patterns (mock DB pool, ASGITransport), frontend Vitest + RTL patterns, coverage targets, and per-feature test requirements. |
| `feature35.md` | Configurable Multi-Theme System | (Archived) Multi-motif theme system; subsequently simplified to a single professional theme with light/dark modes in feature36. |
| `feature36.md` | Product-Grade UI Overhaul | Collapsible sidebar navigation, global filter bar, dashboard landing page with KPI sparklines and heatmap, single Demand Studio theme with light/dark support. |
| `feature39.md` | Job Scheduler/Monitor | APScheduler 3.11 automation engine: 12 REST endpoints, per-group concurrency with FIFO queueing, cron/interval scheduling, job pipelines, retry logic, and automation dashboard UI. |
| `feature40.md` | Demand Planner Storyboard | UX workflow design for an exception-based demand planning experience: exception surfacing, recommended actions, champion model insights, and guided planner interventions. |
| `theme-testing-strategy.md` | Multi-Theme Testing Strategy | Testing specification for theme correctness: unit tests for color tokens and CSS variables; integration, accessibility, and performance test plans (unit tests implemented). |
