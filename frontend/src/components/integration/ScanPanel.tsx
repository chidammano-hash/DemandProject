/**
 * ScanPanel — asks the AI planner to inspect the latest inputs and propose a
 * safe execution sequence.
 *
 * The first click runs the scan + planning pass. If the model needs more
 * context, the panel renders follow-up questions inline and lets the user
 * refine the plan before handing the final sequence upward.
 */

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  planScan,
  type PlannerAnswer,
  type ScanPlanResult,
} from "../../api/queries/integration_chain";
import { CollapsibleSection } from "@/components/CollapsibleSection";

interface ScanPanelProps {
  onPlanned: (result: ScanPlanResult) => void;
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
  const { onPlanned } = props;
  const [plan, setPlan] = useState<ScanPlanResult | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});

  const mutation = useMutation({
    mutationFn: planScan,
    onSuccess: (result) => {
      setPlan(result);
      if (result.status !== "questions") {
        onPlanned(result);
      }
    },
  });

  const isScanning = mutation.isPending;
  const lastScannedAt = mutation.data?.scanned_at;
  const activePlan = plan ?? mutation.data ?? null;
  const questions = activePlan?.questions ?? [];
  const hasQuestions = questions.length > 0;
  const canSubmitAnswers = questions.every(
    (q) => !q.required || (answers[q.id] ?? "").trim() !== ""
  );

  const submitPlan = (nextAnswers: PlannerAnswer[] = []): void => {
    mutation.mutate({ answers: nextAnswers });
  };

  const handleQuestionRun = (): void => {
    const nextAnswers = questions
      .map((q) => ({ question_id: q.id, answer: (answers[q.id] ?? "").trim() }))
      .filter((row) => row.answer.length > 0);
    submitPlan(nextAnswers);
  };

  const headerRight = (
    <div className="flex flex-col items-end gap-1">
      <button
        type="button"
        onClick={() => submitPlan()}
        disabled={isScanning}
        aria-label="Scan data/input/ for changed files"
        className="inline-flex items-center gap-2 rounded-md border border-border bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {isScanning ? "Scanning…" : hasQuestions ? "Ask Again" : "Scan Now"}
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
      subtitle="Scan data/input/ and let the AI planner choose the safest next sequence."
      storageKey="integration.scan"
      headerRight={headerRight}
      className="border-primary/25 bg-gradient-to-br from-primary/[0.04] to-transparent"
    >
      {mutation.isError ? (
        <p
          role="alert"
          className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300"
        >
          Planner failed: {getErrorMessage(mutation.error)}
        </p>
      ) : (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground">
            Click <strong>Scan Now</strong> to compare every source file under{" "}
            <code>data/input/</code>, check the current job queue, and ask the AI for the safest
            load sequence.
          </p>

          {activePlan && (
            <div className="space-y-3 rounded-lg border border-border bg-muted/25 p-3">
              <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <span className="rounded-full border border-border px-2 py-0.5 font-medium text-foreground">
                  {activePlan.provider} · {activePlan.model}
                </span>
                <span>confidence {(activePlan.confidence * 100).toFixed(0)}%</span>
                <span className="rounded-full border border-border px-2 py-0.5">
                  {activePlan.status}
                </span>
              </div>

              {activePlan.risk_flags.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {activePlan.risk_flags.map((flag) => (
                    <span
                      key={flag}
                      className="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[11px] text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300"
                    >
                      {flag}
                    </span>
                  ))}
                </div>
              )}

              <p className="text-sm text-foreground">{activePlan.explanation}</p>

              {activePlan.status === "questions" && hasQuestions && (
                <div className="space-y-3 rounded-md border border-blue-200 bg-blue-50/60 p-3 dark:border-blue-900 dark:bg-blue-950/30">
                  <p className="text-xs font-semibold uppercase tracking-wide text-blue-700 dark:text-blue-300">
                    The planner needs a little more context
                  </p>
                  <div className="space-y-3">
                    {questions.map((question) => (
                      <label key={question.id} className="block space-y-1">
                        <span className="text-sm font-medium text-foreground">
                          {question.prompt}
                        </span>
                        {question.reason && (
                          <span className="block text-xs text-muted-foreground">
                            {question.reason}
                          </span>
                        )}
                        {question.answer_type === "choice" && question.options.length > 0 ? (
                          <select
                            value={answers[question.id] ?? ""}
                            onChange={(e) =>
                              setAnswers((current) => ({
                                ...current,
                                [question.id]: e.target.value,
                              }))
                            }
                            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                          >
                            <option value="">Select an option</option>
                            {question.options.map((option) => (
                              <option key={option} value={option}>
                                {option}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <textarea
                            value={answers[question.id] ?? ""}
                            onChange={(e) =>
                              setAnswers((current) => ({
                                ...current,
                                [question.id]: e.target.value,
                              }))
                            }
                            rows={2}
                            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                            placeholder={
                              question.answer_type === "boolean" ? "yes / no" : "Type your answer"
                            }
                          />
                        )}
                      </label>
                    ))}
                  </div>
                  <button
                    type="button"
                    onClick={handleQuestionRun}
                    disabled={isScanning || !canSubmitAnswers}
                    className="inline-flex items-center gap-2 rounded-md border border-border bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {isScanning ? "Refining…" : "Refine Plan"}
                  </button>
                </div>
              )}

              {activePlan.evidence.length > 0 && (
                <div className="space-y-1">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Evidence
                  </p>
                  <ul className="space-y-1">
                    {activePlan.evidence.map((item) => (
                      <li
                        key={`${item.kind}-${item.label}-${item.value}`}
                        className="text-xs text-muted-foreground"
                      >
                        <span className="font-medium text-foreground">{item.label}</span>
                        <span className="mx-1">·</span>
                        <span>{item.value}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </CollapsibleSection>
  );
}
