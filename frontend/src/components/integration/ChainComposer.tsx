/**
 * ChainComposer — visualizes scan results and submits a sequential chain.
 *
 * Layout: summary line, MASTERS (dim) + DETAILS (fact) groups with distinct
 * tints, vertical proposed-chain pipeline, and a Run Chain button that POSTs
 * to /integration/chains and reports the new chain id via `onSubmitted`.
 */

import { useMutation } from "@tanstack/react-query";
import {
  submitChain,
  type ChainStep,
  type DomainChange,
  type LoadMode,
  type ScanResult,
  type SubmitChainRequest,
} from "../../api/queries/integration_chain";
import { CollapsibleSection } from "@/components/CollapsibleSection";

interface ChainComposerProps {
  scan: ScanResult;
  onSubmitted: (chainId: string) => void;
}

const MODE_PILL: Record<LoadMode, string> = {
  onetime: "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
  delta: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  file: "bg-sky-100 text-sky-700 dark:bg-sky-950 dark:text-sky-300",
};
const KIND_TINT = {
  dim: "border-blue-200 bg-blue-50/60 dark:border-blue-900 dark:bg-blue-950/30",
  fact: "border-amber-200 bg-amber-50/60 dark:border-amber-900 dark:bg-amber-950/30",
} as const;
const KIND_TEXT = {
  dim: "text-blue-700 dark:text-blue-300",
  fact: "text-amber-700 dark:text-amber-300",
} as const;
const PILL_BASE =
  "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium uppercase tracking-wide";

function ModePill({ mode }: { mode: LoadMode }): JSX.Element {
  return (
    <span className={`${PILL_BASE} ${MODE_PILL[mode]}`} aria-label={`mode: ${mode}`}>
      {mode}
    </span>
  );
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function ChangeGroup({
  title,
  kind,
  changes,
}: {
  title: string;
  kind: "dim" | "fact";
  changes: DomainChange[];
}): JSX.Element {
  return (
    <div className={`rounded-lg border ${KIND_TINT[kind]} p-3`}>
      <h4 className={`text-xs font-semibold uppercase tracking-wide ${KIND_TEXT[kind]}`}>
        {title}
      </h4>
      {changes.length === 0 ? (
        <p className="mt-2 text-sm text-muted-foreground">No changes.</p>
      ) : (
        <ul className="mt-2 space-y-1.5">
          {changes.map((c) => (
            <li key={c.domain} className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
              <span className="font-medium text-foreground">{c.domain}</span>
              <span className="text-muted-foreground">·</span>
              <span className="text-muted-foreground">{c.reason}</span>
              <span className="text-muted-foreground">·</span>
              <ModePill mode={c.proposed_mode} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ProposedChain({ steps }: { steps: ChainStep[] }): JSX.Element {
  if (steps.length === 0) {
    return <p className="text-sm text-muted-foreground">No steps in proposed chain.</p>;
  }
  return (
    <ol className="space-y-2" aria-label="Proposed chain steps">
      {steps.map((s, idx) => (
        <li key={`${s.step}-${s.domain}`} className="flex items-start gap-3">
          <span
            aria-hidden="true"
            className="mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-border bg-muted text-xs font-semibold tabular-nums text-foreground"
          >
            {s.step}
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
              <span className="font-medium text-foreground">{s.domain}</span>
              <ModePill mode={s.mode} />
              {s.slice && (
                <span className="text-xs text-muted-foreground tabular-nums">
                  slice {s.slice}
                </span>
              )}
            </div>
            {idx < steps.length - 1 && (
              <span aria-hidden="true" className="mt-1 ml-3 block h-3 w-px bg-border" />
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}

export function ChainComposer(props: ChainComposerProps): JSX.Element {
  const { scan, onSubmitted } = props;
  const detected = scan.changes.filter((c) => c.changed);
  const dimChanges = detected.filter((c) => c.kind === "dim");
  const factChanges = detected.filter((c) => c.kind === "fact");
  const total = detected.length;
  const stepCount = scan.proposed_chain.length;

  const mutation = useMutation({
    mutationFn: submitChain,
    onSuccess: (resp) => onSubmitted(resp.chain_id),
  });

  const handleRun = (): void => {
    const body: SubmitChainRequest = {
      jobs: scan.proposed_chain.map((s) => ({
        domain: s.domain,
        mode: s.mode,
        ...(s.slice ? { slice: s.slice } : {}),
      })),
    };
    mutation.mutate(body);
  };

  const isSubmitting = mutation.isPending;
  const runDisabled = stepCount === 0 || isSubmitting;
  const buttonLabel = isSubmitting
    ? "Submitting…"
    : stepCount > 0
      ? `Run Chain (${stepCount} step${stepCount === 1 ? "" : "s"})`
      : "Run Chain";

  const headerRight = (
    <span className="text-sm text-muted-foreground">
      {total === 0 ? "All files up to date" : `${total} change${total === 1 ? "" : "s"} detected`}
    </span>
  );

  return (
    <CollapsibleSection
      title="Proposed Load Chain"
      storageKey="integration.chain_composer"
      headerRight={headerRight}
    >
      <div className="space-y-4">
        <div className="grid gap-3 md:grid-cols-2">
          <ChangeGroup title="Masters (dimensions)" kind="dim" changes={dimChanges} />
          <ChangeGroup title="Details (facts)" kind="fact" changes={factChanges} />
        </div>

        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Execution Order
          </h4>
          <ProposedChain steps={scan.proposed_chain} />
        </div>

        <div className="flex flex-col items-start gap-2">
          <button
            type="button"
            onClick={handleRun}
            disabled={runDisabled}
            aria-label={buttonLabel}
            className="inline-flex items-center gap-2 rounded-md border border-border bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {buttonLabel}
          </button>
          {mutation.isError && (
            <p
              role="alert"
              className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300"
            >
              Submit failed: {getErrorMessage(mutation.error)}
            </p>
          )}
        </div>
      </div>
    </CollapsibleSection>
  );
}
