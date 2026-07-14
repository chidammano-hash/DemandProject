import type { CustomerBlendSeriesMonth } from "@/api/queries/customerForecast";
import { Badge } from "@/components/ui/badge";

interface CustomerBlendSeriesTableProps {
  months: CustomerBlendSeriesMonth[];
  formatMonth: (value: string) => string;
  formatPercent: (value: number | null) => string;
  formatQuantity: (value: number | null) => string;
}

export function CustomerBlendSeriesTable({
  months,
  formatMonth,
  formatPercent,
  formatQuantity,
}: CustomerBlendSeriesTableProps) {
  return (
    <div className="overflow-x-auto rounded-md border">
      <table className="w-full text-sm">
        <caption className="sr-only">Monthly customer blend components and coverage</caption>
        <thead className="bg-muted/50 text-left">
          <tr>
            <th scope="col" className="px-3 py-2">
              Month
            </th>
            <th scope="col" className="px-3 py-2 text-right">
              Raw Customer Demand
            </th>
            <th scope="col" className="px-3 py-2 text-right">
              Normalized Customer Bottom-Up
            </th>
            <th scope="col" className="px-3 py-2 text-right">
              Source Champion
            </th>
            <th scope="col" className="px-3 py-2 text-right">
              Customer Blend
            </th>
            <th scope="col" className="px-3 py-2 text-right">
              Fulfillment
            </th>
            <th scope="col" className="px-3 py-2 text-right">
              Customer weight
            </th>
            <th scope="col" className="px-3 py-2">
              Coverage
            </th>
          </tr>
        </thead>
        <tbody>
          {months.map((month) => (
            <tr key={month.forecast_month} className="border-t">
              <th scope="row" className="px-3 py-2 text-left font-normal">
                {formatMonth(month.forecast_month)}
              </th>
              <td className="px-3 py-2 text-right">
                {formatQuantity(month.raw_customer_demand_qty)}
              </td>
              <td className="px-3 py-2 text-right">
                {formatQuantity(month.normalized_customer_qty)}
              </td>
              <td className="px-3 py-2 text-right">{formatQuantity(month.champion_qty)}</td>
              <td className="px-3 py-2 text-right">{formatQuantity(month.blended_qty)}</td>
              <td className="px-3 py-2 text-right">
                {month.fulfillment_ratio == null
                  ? "—"
                  : formatPercent(month.fulfillment_ratio * 100)}
              </td>
              <td className="px-3 py-2 text-right">
                {formatPercent(month.effective_customer_weight * 100)}
              </td>
              <td className="px-3 py-2">
                <Badge variant={month.coverage_status === "blended" ? "success" : "outline"}>
                  {month.coverage_status === "blended" ? "Blended" : "Champion fallback"}
                </Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
