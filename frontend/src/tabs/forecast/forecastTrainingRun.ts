import type { JobStatus } from "@/types/jobs";

export const PENDING_TRAINING_STORAGE_KEY = "demand:forecast:pending-training";

export interface PendingTrainingRun {
  jobId: string;
  modelId: string;
}

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): unknown;
  removeItem(key: string): unknown;
}

export type TrainingTerminalOutcome = "completed" | "failed" | "cancelled";

function browserStorage(): StorageLike | null {
  try {
    return typeof globalThis.localStorage === "undefined" ? null : globalThis.localStorage;
  } catch {
    return null;
  }
}

export function loadPendingTrainingRun(storage?: StorageLike): PendingTrainingRun | null {
  const target = storage ?? browserStorage();
  if (!target) return null;
  try {
    const raw = target.getItem(PENDING_TRAINING_STORAGE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    const value = parsed as Record<string, unknown>;
    return typeof value.jobId === "string" &&
      value.jobId.length > 0 &&
      typeof value.modelId === "string" &&
      value.modelId.length > 0
      ? { jobId: value.jobId, modelId: value.modelId }
      : null;
  } catch {
    return null;
  }
}

export function persistPendingTrainingRun(
  run: PendingTrainingRun | null,
  storage?: StorageLike
): void {
  const target = storage ?? browserStorage();
  if (!target) return;
  try {
    if (run) {
      target.setItem(PENDING_TRAINING_STORAGE_KEY, JSON.stringify(run));
    } else {
      target.removeItem(PENDING_TRAINING_STORAGE_KEY);
    }
  } catch {
    // Storage can be unavailable in private browsing or at quota; live state
    // still tracks the job for the current mount.
  }
}

export function resolveTrainingTerminalOutcome(
  status: JobStatus | undefined
): TrainingTerminalOutcome | null {
  return status === "completed" || status === "failed" || status === "cancelled" ? status : null;
}
