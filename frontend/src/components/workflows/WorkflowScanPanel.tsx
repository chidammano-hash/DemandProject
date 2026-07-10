import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Loader2,
  Play,
  ScanSearch,
  ShieldCheck,
} from "lucide-react";

import {
  planOperationalWorkflows,
  runNamedPipeline,
  type WorkflowPlanAnswer,
} from "@/api/queries/jobs";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const PRIORITY_STYLE = {
  critical: "border-destructive/30 bg-destructive/5 text-destructive",
  high: "border-amber-500/30 bg-amber-500/5 text-amber-700 dark:text-amber-300",
  medium: "border-sky-500/30 bg-sky-500/5 text-sky-700 dark:text-sky-300",
  low: "border-border bg-muted/30 text-muted-foreground",
} as const;

export function WorkflowScanPanel(): JSX.Element {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [startedPipeline, setStartedPipeline] = useState<string | null>(null);

  const planMutation = useMutation({
    mutationFn: (submittedAnswers: WorkflowPlanAnswer[]) =>
      planOperationalWorkflows(submittedAnswers),
  });
  const runMutation = useMutation({
    mutationFn: runNamedPipeline,
    onSuccess: (result) => setStartedPipeline(result.pipeline_id),
  });

  const plan = planMutation.data;
  const questionAnswers =
    plan?.questions.map((question) => ({
      question_id: question.id,
      answer: answers[question.id] ?? "",
    })) ?? [];
  const questionsComplete = questionAnswers.every((answer) => answer.answer.trim());

  return (
    <Card className="overflow-hidden border-primary/25 bg-gradient-to-br from-primary/[0.06] via-card to-card">
      <CardHeader className="gap-4 border-b border-border/60 md:flex-row md:items-center md:justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <BrainCircuit className="h-5 w-5" aria-hidden="true" />
            </span>
            <div>
              <CardTitle className="text-base">AI Operations Guide</CardTitle>
              <p className="text-xs text-muted-foreground">
                Evidence first. System guarded. AI verified when available.
              </p>
            </div>
          </div>
        </div>
        <Button
          type="button"
          onClick={() => planMutation.mutate([])}
          disabled={planMutation.isPending || runMutation.isPending}
          className="min-h-11 gap-2"
        >
          {planMutation.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          ) : (
            <ScanSearch className="h-4 w-4" aria-hidden="true" />
          )}
          {plan ? "Analyze again" : "Analyze workflows"}
        </Button>
      </CardHeader>

      <CardContent className="space-y-5 pt-5">
        {!plan && !planMutation.isError && (
          <div className="grid gap-3 md:grid-cols-3">
            {[
              [
                "1",
                "Inspect",
                "Inputs, active jobs, lineage, clustering, release and archive state",
              ],
              ["2", "Clarify", "Ask only when your answer changes safety or ordering"],
              ["3", "Execute", "Run a registered pipeline and monitor every step"],
            ].map(([number, title, description]) => (
              <div key={number} className="rounded-lg border border-border/70 bg-background/60 p-3">
                <div className="mb-2 flex items-center gap-2">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
                    {number}
                  </span>
                  <span className="text-sm font-medium">{title}</span>
                </div>
                <p className="text-xs leading-relaxed text-muted-foreground">{description}</p>
              </div>
            ))}
          </div>
        )}

        {planMutation.isError && (
          <div
            role="alert"
            className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
          >
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            {(planMutation.error as Error).message || "Workflow analysis failed."}
          </div>
        )}

        {plan && (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <span
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium",
                  plan.ai_verified
                    ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                    : "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300"
                )}
              >
                {plan.ai_verified ? (
                  <ShieldCheck className="h-3.5 w-3.5" />
                ) : (
                  <AlertTriangle className="h-3.5 w-3.5" />
                )}
                {plan.ai_verified ? "AI verified" : "System verified"}
              </span>
              <span className="text-xs text-muted-foreground">
                {plan.provider} · {plan.model} · {Math.round(plan.confidence * 100)}% confidence
              </span>
              <span className="text-xs text-muted-foreground">
                Planning month {plan.evidence.planning_month.slice(0, 7)}
              </span>
            </div>

            <p className="text-sm leading-relaxed text-foreground">{plan.explanation}</p>

            {plan.questions.length > 0 && (
              <section
                aria-labelledby="workflow-questions-heading"
                className="space-y-3 rounded-lg border border-sky-500/25 bg-sky-500/5 p-4"
              >
                <h3 id="workflow-questions-heading" className="text-sm font-semibold">
                  One decision before execution
                </h3>
                {plan.questions.map((question) => (
                  <label key={question.id} className="block space-y-1.5">
                    <span className="text-sm text-foreground">{question.prompt}</span>
                    {question.options.length > 0 ? (
                      <select
                        value={answers[question.id] ?? ""}
                        onChange={(event) =>
                          setAnswers((current) => ({
                            ...current,
                            [question.id]: event.target.value,
                          }))
                        }
                        className="min-h-11 w-full rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      >
                        <option value="">Choose an option</option>
                        {question.options.map((option) => (
                          <option key={option}>{option}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        value={answers[question.id] ?? ""}
                        onChange={(event) =>
                          setAnswers((current) => ({
                            ...current,
                            [question.id]: event.target.value,
                          }))
                        }
                        className="min-h-11 w-full rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      />
                    )}
                    {question.reason && (
                      <span className="text-xs text-muted-foreground">{question.reason}</span>
                    )}
                  </label>
                ))}
                <Button
                  type="button"
                  variant="outline"
                  disabled={!questionsComplete || planMutation.isPending}
                  onClick={() => planMutation.mutate(questionAnswers)}
                >
                  Refine recommendation
                </Button>
              </section>
            )}

            {plan.recommendations.length === 0 ? (
              <div className="flex items-center gap-3 rounded-lg border border-emerald-500/25 bg-emerald-500/5 p-4">
                <CheckCircle2 className="h-5 w-5 text-emerald-600" aria-hidden="true" />
                <div>
                  <p className="text-sm font-medium">Everything is current</p>
                  <p className="text-xs text-muted-foreground">
                    No workflow needs to run right now.
                  </p>
                </div>
              </div>
            ) : (
              <section aria-labelledby="workflow-plan-heading" className="space-y-3">
                <h3
                  id="workflow-plan-heading"
                  className="text-xs font-semibold uppercase tracking-wide text-muted-foreground"
                >
                  Recommended sequence
                </h3>
                <p className="text-xs text-muted-foreground">
                  Run the next workflow, then analyze again. Readiness is rechecked before each
                  downstream stage.
                </p>
                {plan.recommendations.map((recommendation, index) => (
                  <article
                    key={recommendation.pipeline_name}
                    className="rounded-xl border border-border bg-background/70 p-4 shadow-sm"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="flex min-w-0 gap-3">
                        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border bg-muted text-sm font-semibold tabular-nums">
                          {index + 1}
                        </span>
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            <h4 className="font-semibold">{recommendation.title}</h4>
                            <span
                              className={cn(
                                "rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
                                PRIORITY_STYLE[recommendation.priority]
                              )}
                            >
                              {recommendation.priority}
                            </span>
                          </div>
                          <p className="mt-1 text-sm text-foreground/85">{recommendation.reason}</p>
                          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                            {recommendation.description}
                          </p>
                        </div>
                      </div>
                      {index === 0 && plan.status === "planned" && (
                        <Button
                          type="button"
                          onClick={() => runMutation.mutate(recommendation.pipeline_name)}
                          disabled={runMutation.isPending || recommendation.blockers.length > 0}
                          className="min-h-11 gap-2"
                          aria-label={`Run ${recommendation.title}`}
                        >
                          {runMutation.isPending ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Play className="h-4 w-4" />
                          )}
                          Run next
                        </Button>
                      )}
                    </div>
                    <ol
                      className="mt-4 flex flex-wrap items-center gap-2"
                      aria-label={`${recommendation.title} steps`}
                    >
                      {recommendation.steps.map((step, stepIndex) => (
                        <li
                          key={`${step.position}-${step.job_type}`}
                          className="flex items-center gap-2"
                        >
                          <span className="rounded-md border border-border bg-muted/50 px-2 py-1 font-mono text-[11px] text-foreground/80">
                            {step.job_type.replace(/_/g, " ")}
                          </span>
                          {stepIndex < recommendation.steps.length - 1 && (
                            <ChevronRight
                              className="h-3.5 w-3.5 text-muted-foreground"
                              aria-hidden="true"
                            />
                          )}
                        </li>
                      ))}
                    </ol>
                    {recommendation.blockers.length > 0 && (
                      <p className="mt-3 text-xs text-amber-700 dark:text-amber-300">
                        Prerequisite: {recommendation.blockers.join(" ")}
                      </p>
                    )}
                  </article>
                ))}
              </section>
            )}

            {startedPipeline && (
              <div
                role="status"
                className="rounded-lg border border-emerald-500/25 bg-emerald-500/5 p-3 text-sm text-emerald-700 dark:text-emerald-300"
              >
                Pipeline {startedPipeline} started. Open Workflow Library to monitor every step.
              </div>
            )}
            {runMutation.isError && (
              <div
                role="alert"
                className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
              >
                {(runMutation.error as Error).message || "Workflow submission failed."}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
