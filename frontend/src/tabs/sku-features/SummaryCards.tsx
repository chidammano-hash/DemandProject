/**
 * SKU Features — Summary cards row (Total SKUs, Last Computed, Avg CV, Avg Amplitude).
 */
import { Database, Clock, TrendingUp, Waves } from "lucide-react";
import type { SkuFeaturesSummary } from "@/api/queries/sku-features";
import { Skeleton } from "@/components/Skeleton";
import { formatNumber, relativeTime } from "./utils";

interface SummaryCardProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  subtitle?: string;
  isLoading: boolean;
}

function SummaryCard({ icon: Icon, label, value, subtitle, isLoading }: SummaryCardProps) {
  if (isLoading) {
    return (
      <div className="rounded-lg border border-border bg-card p-4 space-y-2">
        <Skeleton className="h-3 w-24" />
        <Skeleton className="h-7 w-16" />
        <Skeleton className="h-2.5 w-32" />
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2 text-muted-foreground">
        <Icon className="h-4 w-4" />
        <span className="text-xs font-medium">{label}</span>
      </div>
      <p className="mt-1 text-2xl font-bold tabular-nums">{value}</p>
      {subtitle && (
        <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>
      )}
    </div>
  );
}

interface SummaryCardsProps {
  summary: SkuFeaturesSummary | undefined;
  isLoading: boolean;
}

export function SummaryCards({ summary, isLoading }: SummaryCardsProps) {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      <SummaryCard
        icon={Database}
        label="Total SKUs"
        value={summary ? summary.total_skus.toLocaleString() : "—"}
        isLoading={isLoading}
      />
      <SummaryCard
        icon={Clock}
        label="Last Computed"
        value={summary?.last_computed ? relativeTime(summary.last_computed) : "Never"}
        subtitle={summary?.last_computed ? new Date(summary.last_computed).toLocaleString() : undefined}
        isLoading={isLoading}
      />
      <SummaryCard
        icon={Waves}
        label="Avg CV Demand"
        value={formatNumber(summary?.averages?.cv_demand)}
        subtitle="Coefficient of variation"
        isLoading={isLoading}
      />
      <SummaryCard
        icon={TrendingUp}
        label="Avg Seasonal Amplitude"
        value={formatNumber(summary?.averages?.seasonal_amplitude)}
        subtitle="Peak-to-trough ratio"
        isLoading={isLoading}
      />
    </div>
  );
}
