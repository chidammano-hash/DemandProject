export const CUSTOMER_FORECAST_ROUTE_ORDER = [
  "moving_average_3",
  "seasonal_repeat_12",
  "croston",
] as const;

export type CustomerForecastRouteModelId = (typeof CUSTOMER_FORECAST_ROUTE_ORDER)[number];

const CUSTOMER_MODEL_LABELS: Record<string, string> = {
  customer_rule_router: "Customer Rule Router",
  moving_average_3: "3-Month Moving Average",
  seasonal_repeat_12: "12-Month Seasonal Repeat",
  croston: "Croston/SBA",
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
