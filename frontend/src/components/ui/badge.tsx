import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary text-primary-foreground",
        secondary: "border-transparent bg-secondary text-secondary-foreground",
        outline: "text-foreground",
        // Semantic severity/status variants (U5.1). Each carries a Light + dark:
        // tint pair so the pill stays legible in Dark theme. Keep these in sync
        // with `severityBadgeClass()` in lib/severityBadge.ts.
        critical: "border-red-200 bg-red-100 text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300",
        high: "border-orange-200 bg-orange-100 text-orange-700 dark:border-orange-900/50 dark:bg-orange-950/40 dark:text-orange-300",
        warning: "border-amber-200 bg-amber-100 text-amber-700 dark:border-amber-900/50 dark:bg-amber-950/40 dark:text-amber-300",
        info: "border-blue-200 bg-blue-100 text-blue-700 dark:border-blue-900/50 dark:bg-blue-950/40 dark:text-blue-300",
        success: "border-green-200 bg-green-100 text-green-700 dark:border-green-900/50 dark:bg-green-950/40 dark:text-green-300",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
