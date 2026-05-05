import type { JSX } from "react";
import type { DomainInfo } from "../../api/queries/integration";

interface DomainSelectorProps {
  value: string;
  onChange: (domain: string) => void;
  domains: DomainInfo[];
  disabled?: boolean;
  label?: string;
  id?: string;
}

export function DomainSelector({
  value,
  onChange,
  domains,
  disabled = false,
  label = "Domain",
  id = "domain-selector",
}: DomainSelectorProps): JSX.Element {
  return (
    <div className="flex flex-col gap-1">
      <label
        htmlFor={id}
        className="text-xs font-medium uppercase tracking-wide text-muted-foreground"
      >
        {label}
      </label>
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded border border-border bg-background px-2 py-1 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
      >
        <option value="">— Select domain —</option>
        {domains.map((domain) => {
          const optionText = `${domain.name}${
            domain.partitioned ? "  •  partitioned" : ""
          }`;
          return (
            <option key={domain.name} value={domain.name}>
              {optionText}
            </option>
          );
        })}
      </select>
    </div>
  );
}
