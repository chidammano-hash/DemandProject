# Design Specs — Index

This directory contains all feature design specifications for Demand Studio, organised into five thematic groups. Each group has its own subfolder with a dedicated `README.md` listing the specs it contains.

## Groups

| Group | Folder | Specs | Description |
|---|---|---|---|
| Platform & Infrastructure | `01-platform-infrastructure/` | 6 | Foundational infrastructure (Docker, Postgres, Iceberg), core data architecture and contracts, dimension and fact table DDL, and cross-platform utilities such as benchmarking and Figma tooling. |
| Forecasting & Models | `02-forecasting-models/` | 18 | All tree-based and deep-learning backtest implementations (LGBM, CatBoost, XGBoost, and archived models), the shared backtesting framework, champion model selection, hyperparameter tuning, SHAP feature selection, and recursive forecasting. |
| Clustering & Seasonality | `03-clustering-seasonality/` | 5 | KMeans-based DFU clustering pipeline, seasonality detection and profile assignment, what-if scenario tooling, and scenario chart enhancements. |
| Inventory Planning | `04-inventory-planning/` | 19 | End-to-end inventory planning module: demand variability profiling, safety stock simulation, EOQ calculation, replenishment policy management, health scoring, exception queue, and all supporting IP features. |
| UI, Automation & Cross-Cutting | `05-ui-automation/` | 13 | User-facing interface design and UX architecture, chatbot, market intelligence, data explorer performance, theming, job scheduler/monitor, comprehensive testing strategy, and demand planner workflow storyboard. |
