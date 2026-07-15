export const CUSTOMER_FORECAST_ROUTE_ORDER = [
  "moving_average_3",
  "trailing_average_6",
  "seasonal_repeat_12",
  "tsb",
  "adida",
  "croston",
  "ses",
  "holt_damped",
] as const;

export type CustomerForecastRouteModelId = (typeof CUSTOMER_FORECAST_ROUTE_ORDER)[number];

const CUSTOMER_MODEL_LABELS: Record<string, string> = {
  customer_rule_router: "Customer Rule Router",
  customer_rule_router_v2: "Customer Rule Router v2",
  moving_average_3: "3-Month Moving Average",
  trailing_average_6: "6-Month Trailing Average",
  seasonal_repeat_12: "12-Month Seasonal Repeat",
  tsb: "TSB",
  adida: "ADIDA",
  croston: "Croston/SBA",
  ses: "Simple Exponential Smoothing",
  holt_damped: "Damped Holt",
  chronos2_enriched: "Chronos 2E",
};

export function customerForecastModelLabel(modelId: string): string {
  return (
    CUSTOMER_MODEL_LABELS[modelId] ??
    modelId
      .split("_")
      .filter(Boolean)
      .map((part) => part[0]?.toUpperCase() + part.slice(1))
      .join(" ")
  );
}
