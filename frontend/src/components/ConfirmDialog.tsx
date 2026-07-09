import type { ReactNode } from "react";
import { AlertTriangle } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

export interface ConfirmDialogDetail {
  label: string;
  value: ReactNode;
}

export interface ConfirmDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  details?: ConfirmDialogDetail[];
  confirmLabel: string;
  pendingLabel?: string;
  cancelLabel?: string;
  tone?: "default" | "destructive";
  isPending?: boolean;
  confirmDisabled?: boolean;
  onConfirm: () => void;
}

export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  details,
  confirmLabel,
  pendingLabel = "Working...",
  cancelLabel = "Cancel",
  tone = "default",
  isPending = false,
  confirmDisabled = false,
  onConfirm,
}: ConfirmDialogProps): JSX.Element {
  const isDestructive = tone === "destructive";

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!isPending || nextOpen) onOpenChange(nextOpen);
      }}
    >
      <DialogContent size="lg" className="max-w-[calc(100vw-2rem)]" hideCloseButton={isPending}>
        <DialogHeader>
          <div className="flex items-start gap-3">
            <span
              className={cn(
                "mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md border",
                isDestructive
                  ? "border-destructive/25 bg-destructive/10 text-destructive"
                  : "border-primary/20 bg-primary/10 text-primary",
              )}
              aria-hidden="true"
            >
              <AlertTriangle className="h-4 w-4" strokeWidth={1.7} />
            </span>
            <div className="min-w-0">
              <DialogTitle>{title}</DialogTitle>
              <DialogDescription className="mt-1 leading-5">{description}</DialogDescription>
            </div>
          </div>
        </DialogHeader>

        {details && details.length > 0 && (
          <dl
            className={cn(
              "mx-5 my-4 grid gap-3 rounded-md border p-3 text-xs",
              isDestructive
                ? "border-destructive/20 bg-destructive/5"
                : "border-border bg-muted/35",
            )}
          >
            {details.map((detail) => (
              <div key={detail.label} className="grid gap-1 sm:grid-cols-[8rem_minmax(0,1fr)]">
                <dt className="font-medium text-muted-foreground">{detail.label}</dt>
                <dd className="min-w-0 break-words text-foreground">{detail.value}</dd>
              </div>
            ))}
          </dl>
        )}

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => onOpenChange(false)}
            disabled={isPending}
          >
            {cancelLabel}
          </Button>
          <Button
            type="button"
            size="sm"
            onClick={onConfirm}
            disabled={isPending || confirmDisabled}
            className={cn(
              isDestructive &&
                "bg-destructive text-destructive-foreground hover:bg-destructive/90",
            )}
          >
            {isPending ? pendingLabel : confirmLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
