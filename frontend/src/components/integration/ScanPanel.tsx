/**
 * ScanPanel — triggers a `/integration/scan` and reports the result upward.
 *
 * Renders a section card with a "Scan Now" button. While scanning the button
 * is disabled and shows "Scanning…". On success, the parent receives the
 * `ScanResult` via `onScanned` and the panel displays the timestamp inline.
 * On error, an inline error message is shown.
 */

import { useMutation } from "@tanstack/react-query";
import { scanInputs, type ScanResult } from "../../api/queries/integration_chain";
import { CollapsibleSection } from "@/components/CollapsibleSection";

interface ScanPanelProps {
  onScanned: (result: ScanResult) => void;
}

const TIME_FMT = new Intl.DateTimeFormat(undefined, {
  hour: "numeric",
  minute: "2-digit",
  second: "2-digit",
  month: "short",
  day: "numeric",
});

function formatScannedAt(value: string): string {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return TIME_FMT.format(d);
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

export function ScanPanel(props: ScanPanelProps): JSX.Element {
  const { onScanned } = props;

  const mutation = useMutation({
    mutationFn: scanInputs,
    onSuccess: (result) => {
      onScanned(result);
    },
  });

  const isScanning = mutation.isPending;
  const lastScannedAt = mutation.data?.scanned_at;

  const headerRight = (
    <div className="flex flex-col items-end gap-1">
      <button
        type="button"
        onClick={() => mutation.mutate()}
        disabled={isScanning}
        aria-label="Scan data/input/ for changed files"
        className="inline-flex items-center gap-2 rounded-md border border-border bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {isScanning ? "Scanning…" : "Scan Now"}
      </button>
      {lastScannedAt && !isScanning && (
        <span className="text-xs text-muted-foreground tabular-nums" aria-live="polite">
          Last scan: {formatScannedAt(lastScannedAt)}
        </span>
      )}
    </div>
  );

  return (
    <CollapsibleSection
      title="Detect Changes"
      subtitle="Scan data/input/ for changed source files."
      storageKey="integration.scan"
      headerRight={headerRight}
    >
      {mutation.isError ? (
        <p
          role="alert"
          className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300"
        >
          Scan failed: {getErrorMessage(mutation.error)}
        </p>
      ) : (
        <p className="text-xs text-muted-foreground">
          Click <strong>Scan Now</strong> to compare every source file under{" "}
          <code>data/input/</code> with what was last loaded into the warehouse.
        </p>
      )}
    </CollapsibleSection>
  );
}
