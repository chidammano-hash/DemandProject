import { cn } from "@/lib/utils";

type KpiCardProps = {
  label: string;
  value: string;
  sublabel?: string;
  colorClass?: string;
  borderClass?: string;
};

export function KpiCard({ label, value, sublabel, colorClass, borderClass }: KpiCardProps) {
  return (
    <div className={cn("rounded-md border bg-card px-3 py-2", borderClass)}>
      <p className="text-xs text-muted-foreground">
        {label}
        {sublabel && <span className="text-xs text-muted-foreground ml-1">{sublabel}</span>}
      </p>
      <p className={cn("text-lg font-bold tabular-nums", colorClass)}>{value}</p>
    </div>
  );
}
