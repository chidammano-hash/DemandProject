# 03 — Clustering & Seasonality

This group covers the DFU clustering framework, seasonality detection and profile assignment, and the what-if scenario tooling that lets analysts experiment with clustering parameters without affecting the production model.

## Files

| File | Feature | Summary |
|---|---|---|
| `feature7.md` | DFU Clustering Framework | KMeans-based pipeline that groups DFUs by demand pattern; feature engineering, optimal-K selection, business label assignment, and MLflow tracking. |
| `feature29.md` | What-If / Scenario UI for Clustering | Frontend panel for running trial clustering scenarios with custom parameters, viewing results (elbow, silhouette, radar, pie, gap charts), and promoting a winning scenario to production. |
| `feature30.md` | DFU Seasonality Detection & Profile Assignment | Pipeline (`detect_seasonality.py` + `update_seasonality_profiles.py`) that computes seasonality strength, profile label, peak/trough month, and peak-to-trough ratio per DFU and writes results back to `dim_dfu`. |
| `feature32.md` | Seasonality Profile Filtering | Backend router extension exposing seasonality profile filters on the accuracy and DFU analysis endpoints; frontend UI pending. |
| `feature38.md` | Clustering What-If Scenario Enhancements | Background execution, runtime estimation, scenario queueing, dashboard completion alerts, enhanced scenario charts (optimal-K ReferenceLine, silhouette quality zones), "View Results" navigation from JobsTab, and Past Scenarios history. |
