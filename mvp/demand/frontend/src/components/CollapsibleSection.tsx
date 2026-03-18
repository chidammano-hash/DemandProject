import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface CollapsibleSectionProps {
  title: string;
  /** Start expanded (default true) */
  defaultOpen?: boolean;
  /** Extra elements rendered in the header row (right side) */
  headerRight?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

export function CollapsibleSection({
  title,
  defaultOpen = true,
  headerRight,
  children,
  className,
}: CollapsibleSectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <Card className={cn("animate-fade-in", className)}>
      <CardHeader
        className="cursor-pointer select-none flex flex-row items-center justify-between pb-2"
        onClick={() => setOpen((v) => !v)}
      >
        <div className="flex items-center gap-2">
          {open ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}
          <CardTitle className="text-sm font-medium">{title}</CardTitle>
        </div>
        {headerRight && (
          <div onClick={(e) => e.stopPropagation()}>{headerRight}</div>
        )}
      </CardHeader>
      {open && <CardContent>{children}</CardContent>}
    </Card>
  );
}
