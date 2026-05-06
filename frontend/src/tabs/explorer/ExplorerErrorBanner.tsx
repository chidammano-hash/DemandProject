/**
 * Inline error banner with a Retry button for the Data Explorer tab.
 */
import { RefreshCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export interface ExplorerErrorBannerProps {
  message: string;
  onRetry: () => void;
}

export function ExplorerErrorBanner({
  message,
  onRetry,
}: ExplorerErrorBannerProps) {
  return (
    <Card className="mb-4 border-destructive/30 bg-destructive/10">
      <CardContent className="pt-4 flex items-center justify-between gap-2">
        <span className="text-sm text-destructive">{message}</span>
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RefreshCcw className="mr-1 h-3.5 w-3.5" /> Retry
        </Button>
      </CardContent>
    </Card>
  );
}
