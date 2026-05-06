/**
 * Header bar: title, domain selector, search, page size, and Fields toggle.
 */
import { ChevronsUpDown } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import type { DomainMeta } from "@/types";
import { titleCase, formatNumber } from "@/lib/formatters";

import { DIMENSION_DOMAINS } from "./types";

export interface ExplorerHeaderProps {
  domain: string;
  meta: DomainMeta | undefined;
  total: number;
  totalApproximate: boolean;
  search: string;
  limit: number;
  onDomainChange: (domain: string) => void;
  onSearchChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onLimitChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  onToggleFieldPanel: () => void;
  children?: React.ReactNode;
}

export function ExplorerHeader({
  domain,
  meta,
  total,
  totalApproximate,
  search,
  limit,
  onDomainChange,
  onSearchChange,
  onLimitChange,
  onToggleFieldPanel,
  children,
}: ExplorerHeaderProps) {
  return (
    <CardHeader className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <CardTitle className="text-base">Data Explorer</CardTitle>
          <CardDescription className="max-w-2xl">
            Browse raw data across all domains. Select a <strong>Domain</strong> (item, location, DFU, sales,
            forecast, inventory) to explore its records. Use column filters to narrow results — prefix
            with <code className="text-[10px] bg-muted px-1 rounded">=</code> for exact match, or type
            plain text for fuzzy substring search.
          </CardDescription>
        </div>
        <div className="flex items-center gap-3">
          <select
            className="h-9 w-[180px] rounded-md border border-input bg-background px-3 text-sm"
            value={domain}
            onChange={(e) => onDomainChange(e.target.value)}
          >
            {DIMENSION_DOMAINS.map((d) => (
              <option key={d} value={d}>
                {titleCase(d)}
              </option>
            ))}
          </select>
          <Badge variant="secondary">
            {meta
              ? `${titleCase(meta.name)} (${totalApproximate ? `${formatNumber(total - 1)}+` : formatNumber(total)})`
              : "Loading"}
          </Badge>
        </div>
      </div>

      <div className="grid gap-2 md:grid-cols-[2fr_120px_1fr]">
        <Input
          placeholder="Search across configured fields"
          value={search}
          onChange={onSearchChange}
          disabled={!meta}
        />
        <select
          className="h-9 rounded-md border border-input bg-background px-3 text-sm"
          value={limit}
          onChange={onLimitChange}
        >
          {[50, 100, 250, 500].map((v) => (
            <option key={v} value={v}>
              {v}/page
            </option>
          ))}
        </select>
        <Button variant="outline" onClick={onToggleFieldPanel}>
          <ChevronsUpDown className="mr-2 h-4 w-4" /> Fields
        </Button>
      </div>

      {children}
    </CardHeader>
  );
}
