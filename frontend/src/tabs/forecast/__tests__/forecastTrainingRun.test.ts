import { beforeEach, describe, expect, it } from "vitest";

import {
  loadPendingTrainingRun,
  persistPendingTrainingRun,
  resolveTrainingTerminalOutcome,
  type StorageLike,
} from "../forecastTrainingRun";

function memoryStorage(): StorageLike {
  const values = new Map<string, string>();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
}

describe("forecast training run persistence", () => {
  let storage: StorageLike;

  beforeEach(() => {
    storage = memoryStorage();
  });

  it("restores the exact submitted job and model after a remount", () => {
    persistPendingTrainingRun({ jobId: "job-train-42", modelId: "nhits" }, storage);
    expect(loadPendingTrainingRun(storage)).toEqual({
      jobId: "job-train-42",
      modelId: "nhits",
    });
  });

  it("clears terminal runs and ignores corrupt persisted state", () => {
    storage.setItem("demand:forecast:pending-training", "not-json");
    expect(loadPendingTrainingRun(storage)).toBeNull();

    persistPendingTrainingRun({ jobId: "job-train-42", modelId: "nhits" }, storage);
    persistPendingTrainingRun(null, storage);
    expect(loadPendingTrainingRun(storage)).toBeNull();
  });
});

describe("resolveTrainingTerminalOutcome", () => {
  it("waits through queued and running states", () => {
    expect(resolveTrainingTerminalOutcome("queued")).toBeNull();
    expect(resolveTrainingTerminalOutcome("running")).toBeNull();
  });

  it("distinguishes completion, failure, and cancellation", () => {
    expect(resolveTrainingTerminalOutcome("completed")).toBe("completed");
    expect(resolveTrainingTerminalOutcome("failed")).toBe("failed");
    expect(resolveTrainingTerminalOutcome("cancelled")).toBe("cancelled");
  });
});
